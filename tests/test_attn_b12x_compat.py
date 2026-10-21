#!/usr/bin/env python3
"""Feasibility test: DeepInfer-written V4.1 KV caches -> b12x 1.5.1 compressed sparse MLA on SM120.

What this proves (each step prints its own [PASS]/[FAIL]/[INFO] line):

  1. DeepInfer's own writer (`store_kv_cache`, TileLang kernel, cross-checked against
     `store_kv_cache_ref`) writes an SWA cache (V41_FORMAT, 528 B/token: 512 x E4M3 + 16 x UE8M0)
     and a main/compressed cache (V41_FP4_FORMAT, 288 B/token: 256 B packed E2M1 + 32 x E4M3 per-16
     scales) into memory laid out the way DeepInfer's cache_engine does it: strided views into one
     multi-layer block buffer (main cache) / per-request state ring (SWA), block 0 reserved,
     shuffled physical blocks.
  2. Byte-compat: DeepInfer's dequant (`load_paged_kv_cache_ref`) and b12x's independent V4.1
     decoder (`unpack_deepseek_v41_cache_reference`) see bit-identical values for every written
     slot, reading the *same* bytes (b12x gets zero-copy 2-D [pages, page_bytes] views).
  3. Index-compat: DeepInfer's index builders (`get_kv_swa_index_decode`, `convert_kv_index`)
     produce the indices handed to b12x (SWA: global ring slots; main: either DeepInfer physical
     slots directly, or LIL-style logical indices + page table = block table).
  4. Compute: b12x `compressed_sparse_mla` (cache_format="deepseek_v41", decode and extend modes,
     default FP8-internal and BF16-internal compute) output matches a pure-torch fp32 sparse
     attention with sinks (cos > 0.999), and b12x's own torch reference agrees with ours.

Runs inside the SM120 container
(PYTHONPATH=/opt/deepinfer, b12x==1.5.1, tilelang, tile-kernels installed; FlashMLA/DeepGEMM/
DeepEP/DeepSelect absent and stubbed only if something imports them).

Env knobs: TEST_SWA_BLOCK (default 134 = window 128 + DeepInfer default backtracking 6),
TEST_SEED (default 1234), TEST_SKIP_EXTEND=1, TEST_SKIP_BF16=1.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import math
import os
import pkgutil
import sys
import time
import traceback
import types

# b12x: no multi-candidate autotune race for a correctness test (session.py:271).
os.environ.setdefault("B12X_AUTOTUNE", "0")
os.environ.setdefault("DEEPINFER_ACCELERATOR", "nvidia")

for _p in (os.environ.get("DEEPINFER_ROOT"), "/opt/deepinfer"):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# --------------------------------------------------------------------------------------
# Result bookkeeping
# --------------------------------------------------------------------------------------
RESULTS: list[tuple[str, str, str]] = []  # (status, name, detail)


def record(name: str, ok: bool | None, detail: str = "") -> bool:
    status = "INFO" if ok is None else ("PASS" if ok else "FAIL")
    RESULTS.append((status, name, detail))
    print(f"[{status}] {name}: {detail}", flush=True)
    return bool(ok)


@contextlib.contextmanager
def stage(name: str):
    print(f"\n=== {name} ===", flush=True)
    t0 = time.time()
    try:
        yield
    except Exception as exc:  # noqa: BLE001 - keep going, report everything
        traceback.print_exc()
        record(f"stage:{name}", False, f"exception {type(exc).__name__}: {exc}")
    finally:
        print(f"    ({name} took {time.time() - t0:.1f}s)", flush=True)


# --------------------------------------------------------------------------------------
# Stubs for modules absent in the SM120 container. Only installed if NOT importable, so a
# real install always wins. None of the DeepInfer modules used here import them directly
# (tilelang_ops/{cache_common,store_kv_cache,load_paged_kv_cache,flash_mla_utils,get_kv_index}
# and tilelang_ops/cuda/*), but layers/attention_backend.py does (`from flash_mla import ...`),
# so stub defensively in case some transitive import pulls it in.
# --------------------------------------------------------------------------------------
STUBBED: list[str] = []


def _missing(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is None
    except (ImportError, ValueError):
        return True


class _Dummy:
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        raise RuntimeError("stubbed module function called (module not installed in container)")

    def __getattr__(self, item):
        if item.startswith("__"):
            raise AttributeError(item)
        return _Dummy()


def _stub_getattr(item: str):
    # Dunders (__file__, __path__, ...) must stay missing, or inspect/torch import machinery breaks.
    if item.startswith("__"):
        raise AttributeError(item)
    return _Dummy()


def _stub_module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    mod.__getattr__ = _stub_getattr  # type: ignore[attr-defined]
    sys.modules[name] = mod
    parent, _, child = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], child, mod)
    STUBBED.append(name)
    return mod


def install_stubs() -> None:
    if _missing("flash_mla"):
        _stub_module("flash_mla", flash_mla_sparse_fwd=_Dummy(), flash_mla_with_kvcache=_Dummy(),
                     fused_norm_rope_attn_rope_cast=_Dummy(), get_mla_metadata=_Dummy())
        _stub_module("flash_mla.flash_mla_interface", FlashMLASchedMeta=_Dummy)
    for name in ("deep_gemm", "deep_ep", "deep_select"):
        if _missing(name):
            _stub_module(name)
    # deepinfer.platform.get_platform() needs deep_cluster.hardware.accelerator_kind
    # (shipped in the DeepInfer tree at /opt/deepinfer/deep_cluster; stub only if absent).
    if _missing("deep_cluster"):
        _stub_module("deep_cluster")
        _stub_module("deep_cluster.hardware", accelerator_kind=lambda **_: "nvidia")
    # DeepInfer tilelang_ops modules import pytest at module level (Dockerfile installs it).
    if _missing("pytest"):
        class _Mark:
            def __getattr__(self, item):
                def factory(*a, **k):
                    if len(a) == 1 and callable(a[0]) and not k:
                        return a[0]
                    return lambda f: f
                return factory
        _stub_module("pytest", mark=_Mark(), fixture=lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f)),
                     param=lambda *a, **k: a, raises=_Dummy())


install_stubs()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# --------------------------------------------------------------------------------------
# Geometry (DeepSeek V4.1, ratio-2 layer, decode)
# --------------------------------------------------------------------------------------
H = 64                    # local q heads (TP1 on one 5090)
D = 512                   # latent head dim incl. 64 rope
WINDOW = 128              # SWA window
TOPK = 512                # index top-k (b12x indexed_width)
RATIO = 2                 # compression ratio (layer-2 style)
BLOCK = 128               # DeepInfer block_size in *uncompressed* tokens
MAIN_PAGE = BLOCK // RATIO  # 64 compressed tokens per block = b12x indexed_page_size
SWA_BLOCK = int(os.environ.get("TEST_SWA_BLOCK", str(WINDOW + 6)))  # get_inference_swa_block_size
SEQ_LENS = [300, 1000, 2500, 5000]
EXTEND_ROWS = 4           # q rows per sequence in extend-mode test (last 4 positions)
SM_SCALE = D ** -0.5      # DeepInfer: softmax_scale = qk_head_dim ** -0.5 (v4_modules/attn.py:78)
SEED = int(os.environ.get("TEST_SEED", "1234"))
SWA_REC, MAIN_REC = 528, 288
INDEX_K_BYTES = 128 // 2 + 128 // 32  # DeepInfer _get_qk_topk_bytes for index_head_dim=128 -> 68
COS_THRESHOLD = 0.999

assert SWA_BLOCK >= WINDOW + EXTEND_ROWS - 1, "SWA ring must hold the extend rows' windows"


def align(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def metrics(out: torch.Tensor, ref: torch.Tensor) -> dict:
    a = out.float().flatten()
    b = ref.float().flatten()
    diff = (a - b).abs()
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "cos": F.cosine_similarity(a, b, dim=0).item(),
        "ref_absmax": b.abs().max().item(),
        "finite": bool(torch.isfinite(a).all().item()),
    }


def fmt(m: dict) -> str:
    return (f"cos={m['cos']:.6f} max_abs={m['max_abs']:.4e} mean_abs={m['mean_abs']:.4e} "
            f"ref_absmax={m['ref_absmax']:.3f} finite={m['finite']}")


def check_close(name: str, out: torch.Tensor, ref: torch.Tensor, cos_thr: float = COS_THRESHOLD,
                rel_max: float = 0.1) -> bool:
    m = metrics(out, ref)
    ok = m["finite"] and m["cos"] > cos_thr and m["max_abs"] <= rel_max * m["ref_absmax"] + 1e-2
    return record(name, ok, fmt(m) + f" (thr cos>{cos_thr}, max_abs<={rel_max}*ref_absmax+1e-2)")


# --------------------------------------------------------------------------------------
# Imports (DeepInfer + b12x), each guarded
# --------------------------------------------------------------------------------------
DI: dict = {}
B12X: dict = {}
NOTES: dict = {"stubbed": STUBBED}

TL_PKG = "deepinfer.model_executor.layers.tilelang_ops"


def import_deepinfer() -> None:
    cc = importlib.import_module(f"{TL_PKG}.cache_common")
    DI["cc"] = cc
    try:
        ops = importlib.import_module(f"{TL_PKG}.ops")
        NOTES["ops"] = "full ops dispatcher imported"
    except Exception as exc:  # noqa: BLE001
        # ops.py imports *every* tilelang_ops/cuda/*.py; if one breaks on this machine, build a
        # replacement exposing whatever *_tl factories do import, so the ones we need survive.
        traceback.print_exc()
        ops = types.ModuleType(f"{TL_PKG}.ops")
        cuda_pkg = importlib.import_module(f"{TL_PKG}.cuda")
        bad = []
        for info in pkgutil.iter_modules(cuda_pkg.__path__):
            try:
                mod = importlib.import_module(f"{cuda_pkg.__name__}.{info.name}")
            except Exception as sub_exc:  # noqa: BLE001
                bad.append(f"{info.name}: {type(sub_exc).__name__}")
                continue
            for k, v in vars(mod).items():
                if k.endswith("_tl") and not k.startswith("_") and not isinstance(v, types.ModuleType):
                    setattr(ops, k, v)

        def _missing_tl(name):
            raise AttributeError(f"TL factory {name} unavailable (cuda module import failed)")

        ops.__getattr__ = _missing_tl  # type: ignore[attr-defined]
        sys.modules[f"{TL_PKG}.ops"] = ops
        setattr(importlib.import_module(TL_PKG), "ops", ops)
        NOTES["ops"] = f"ops import failed ({type(exc).__name__}: {exc}); partial ops, broken: {bad}"
    print(f"    deepinfer ops: {NOTES['ops']}")
    skv = importlib.import_module(f"{TL_PKG}.store_kv_cache")
    lpk = importlib.import_module(f"{TL_PKG}.load_paged_kv_cache")
    fmu = importlib.import_module(f"{TL_PKG}.flash_mla_utils")
    gki = importlib.import_module(f"{TL_PKG}.get_kv_index")
    DI.update(
        V41_FORMAT=cc.V41_FORMAT, V41_FP4_FORMAT=cc.V41_FP4_FORMAT,
        store_kv_cache=skv.store_kv_cache, store_kv_cache_ref=skv.store_kv_cache_ref,
        load_paged_kv_cache=lpk.load_paged_kv_cache, load_paged_kv_cache_ref=lpk.load_paged_kv_cache_ref,
        convert_kv_index=fmu.convert_kv_index, convert_kv_index_ref=fmu.convert_kv_index_ref,
        get_kv_swa_index_decode=gki.get_kv_swa_index_decode,
        get_kv_swa_index_decode_ref=gki.get_kv_swa_index_decode_ref,
    )
    assert cc.V41_FORMAT.bytes_per_token == SWA_REC, cc.V41_FORMAT.bytes_per_token
    assert cc.V41_FP4_FORMAT.bytes_per_token == MAIN_REC, cc.V41_FP4_FORMAT.bytes_per_token
    record("deepinfer formats", True,
           f"V41_FORMAT={cc.V41_FORMAT.bytes_per_token}B/tok, V41_FP4_FORMAT={cc.V41_FP4_FORMAT.bytes_per_token}B/tok")


def import_b12x() -> None:
    import b12x
    from b12x.attention import compressed_sparse_mla as mla
    from b12x.attention._shared.mla import compressed_reference as cref
    from b12x.attention.compressed_sparse_mla import _tuning as mla_tuning
    from b12x import preparation as prep

    B12X.update(mla=mla, cref=cref, tuning=mla_tuning, prep=prep)
    ver = getattr(b12x, "__version__", "?")
    sup = mla.is_supported(torch.device("cuda", 0))
    record("b12x import", sup, f"b12x {ver} from {os.path.dirname(b12x.__file__)}; is_supported(SM12x)={sup}")
    # page geometry contract
    swa_w = mla.page_nbytes(SWA_BLOCK, cache_kind="swa", cache_format="deepseek_v41")
    main_w = mla.page_nbytes(MAIN_PAGE, cache_kind="indexed", cache_format="deepseek_v41")
    record("b12x page_nbytes == DeepInfer block bytes", swa_w == SWA_BLOCK * SWA_REC and main_w == MAIN_PAGE * MAIN_REC,
           f"swa {swa_w} vs {SWA_BLOCK * SWA_REC}, indexed {main_w} vs {MAIN_PAGE * MAIN_REC}")


# --------------------------------------------------------------------------------------
# Emulated DeepInfer cache memory (worker/cache_engine.py:400-475)
# --------------------------------------------------------------------------------------
def finite_garbage(shape, gen, device) -> torch.Tensor:
    # Random bytes & 0x77: never an E4M3 NaN (S.1111.111), never UE8M0 0xFF, finite E2M1/E4M3.
    # Any mis-strided / mis-indexed read by b12x shows up as a numeric mismatch, not as NaN noise.
    return (torch.randint(0, 256, shape, dtype=torch.uint8, device=device, generator=gen) & 0x77)


class Caches:
    """One set of SWA + main caches, laid out like DeepInfer's KVCacheManager."""

    def __init__(self, num_blocks: int, num_state_blocks: int, gen, device):
        # --- main/compressed cache: a region inside a multi-layer per-block buffer ---
        index_delta = align(MAIN_PAGE * INDEX_K_BYTES, 32)     # this layer's index-K region first
        other_layers = 1024                                       # pretend bytes of other layers
        self.main_offset = index_delta + other_layers
        main_delta = MAIN_PAGE * MAIN_REC
        self.block_stride = align(self.main_offset + main_delta + 96, 32)
        self.gpu_blocks = finite_garbage((num_blocks, self.block_stride), gen, device)
        gb = self.gpu_blocks
        self.ca_cache = gb[:, self.main_offset:self.main_offset + main_delta].as_strided(
            (num_blocks, MAIN_PAGE, 1, MAIN_REC), (gb.stride(0), MAIN_REC, MAIN_REC, 1))
        # --- SWA cache: per-request ring in the state tensor ---
        self.swa_offset = 32                                       # nonzero storage offset
        swa_size = SWA_BLOCK * SWA_REC
        self.state_stride = align(self.swa_offset + swa_size + 512, 32)
        self.state = finite_garbage((num_state_blocks, self.state_stride), gen, device)
        st = self.state
        self.swa_cache = st[:, self.swa_offset:self.swa_offset + swa_size].as_strided(
            (num_state_blocks, SWA_BLOCK, 1, SWA_REC), (st.stride(0), SWA_REC, SWA_REC, 1))

    def clone(self) -> "Caches":
        new = object.__new__(Caches)
        new.__dict__.update(self.__dict__)
        new.gpu_blocks = self.gpu_blocks.clone()
        new.state = self.state.clone()
        gb, st = new.gpu_blocks, new.state
        new.ca_cache = gb[:, self.main_offset:self.main_offset + MAIN_PAGE * MAIN_REC].as_strided(
            self.ca_cache.shape, (gb.stride(0), MAIN_REC, MAIN_REC, 1))
        new.swa_cache = st[:, self.swa_offset:self.swa_offset + SWA_BLOCK * SWA_REC].as_strided(
            self.swa_cache.shape, (st.stride(0), SWA_REC, SWA_REC, 1))
        return new

    # b12x views: SAME memory, [pages, page_bytes] with the allocator page stride (no repack).
    def b12x_swa(self) -> torch.Tensor:
        c = self.swa_cache
        v = c.as_strided((c.shape[0], SWA_BLOCK * SWA_REC), (c.stride(0), 1), c.storage_offset())
        assert v.data_ptr() == c.data_ptr()
        return v

    def b12x_main(self) -> torch.Tensor:
        c = self.ca_cache
        v = c.as_strided((c.shape[0], MAIN_PAGE * MAIN_REC), (c.stride(0), 1), c.storage_offset())
        assert v.data_ptr() == c.data_ptr()
        return v


# --------------------------------------------------------------------------------------
# Test state
# --------------------------------------------------------------------------------------
S: dict = {}


def build_workload(device) -> None:
    gen = torch.Generator(device=device).manual_seed(SEED)
    cpu_gen = torch.Generator().manual_seed(SEED)
    nseq = len(SEQ_LENS)
    blocks_per_seq = [math.ceil(L / BLOCK) for L in SEQ_LENS]
    max_blocks = max(blocks_per_seq)
    num_blocks = 1 + sum(blocks_per_seq) + 16          # block 0 reserved + spare blocks
    num_state_blocks = 1 + nseq + 3                     # state block 0 reserved too
    perm = (torch.randperm(num_blocks - 1, generator=cpu_gen) + 1).tolist()
    sperm = (torch.randperm(num_state_blocks - 1, generator=cpu_gen) + 1).tolist()

    block_tables, state_blocks = [], []
    for i in range(nseq):
        bt = [perm.pop() for _ in range(blocks_per_seq[i])]
        block_tables.append(bt + [0] * (max_blocks - len(bt)))   # DeepInfer-style 0 padding
        state_blocks.append(sperm.pop())

    def latent(n):
        x = torch.randn((n, D), generator=gen, device=device, dtype=torch.float32)
        tok_amp = torch.exp2(torch.empty((n, 1), device=device).uniform_(-3, 3, generator=gen))
        grp_amp = torch.exp2(torch.empty((n, D // 16, 1), device=device).uniform_(-2, 2, generator=gen))
        return (x.view(n, D // 16, 16) * grp_amp).view(n, D).mul(tok_amp).to(torch.bfloat16)

    swa_kv, swa_slots, main_kv, main_slots = [], [], [], []
    for i, L in enumerate(SEQ_LENS):
        start = max(0, L - SWA_BLOCK)                    # ring keeps the last SWA_BLOCK tokens
        pos = torch.arange(start, L, device=device)
        swa_kv.append(latent(L - start))
        swa_slots.append(state_blocks[i] * SWA_BLOCK + pos % SWA_BLOCK)
        C = L // RATIO
        c = torch.arange(C, device=device)
        bt = torch.tensor(block_tables[i], dtype=torch.int64, device=device)
        main_kv.append(latent(C))
        main_slots.append(bt[c // MAIN_PAGE] * MAIN_PAGE + c % MAIN_PAGE)

    S.update(
        gen=gen, cpu_gen=cpu_gen, num_blocks=num_blocks, num_state_blocks=num_state_blocks,
        max_blocks=max_blocks, block_tables=block_tables, state_blocks=state_blocks,
        swa_kv=torch.cat(swa_kv), swa_slots=torch.cat(swa_slots).to(torch.int32).contiguous(),
        main_kv=torch.cat(main_kv), main_slots=torch.cat(main_slots).to(torch.int32).contiguous(),
        caches0=Caches(num_blocks, num_state_blocks, gen, device),
        sink=(torch.randn((H,), generator=gen, device=device) * 2.0).float().contiguous(),
    )
    print(f"    seqs={SEQ_LENS} blocks={num_blocks} (block0 reserved) state_blocks={num_state_blocks} "
          f"SWA_BLOCK={SWA_BLOCK} main page={MAIN_PAGE} block_stride={S['caches0'].block_stride}B "
          f"state_stride={S['caches0'].state_stride}B swa_tokens={S['swa_kv'].shape[0]} "
          f"main_tokens={S['main_kv'].shape[0]}")
    print(f"    block_tables={[bt[:blocks_per_seq[i]] for i, bt in enumerate(block_tables)]}")
    print(f"    state_blocks={state_blocks}")


def write_caches() -> None:
    """Write both caches with DeepInfer's writer: TL kernel and torch ref into twin buffers."""
    V41, FP4 = DI["V41_FORMAT"], DI["V41_FP4_FORMAT"]
    ref = S["caches0"].clone()
    tl = S["caches0"].clone()
    ref_ok = tl_ok = False
    try:
        DI["store_kv_cache_ref"](S["swa_kv"], ref.swa_cache, S["swa_slots"], layout=V41)
        DI["store_kv_cache_ref"](S["main_kv"], ref.ca_cache, S["main_slots"], layout=FP4)
        torch.cuda.synchronize()
        ref_ok = record("DeepInfer store_kv_cache_ref (torch)", True, "SWA FP8 + main FP4 written")
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        record("DeepInfer store_kv_cache_ref (torch)", False, f"{type(exc).__name__}: {exc}")
    try:
        DI["store_kv_cache"](S["swa_kv"], tl.swa_cache, S["swa_slots"], layout=V41)
        DI["store_kv_cache"](S["main_kv"], tl.ca_cache, S["main_slots"], layout=FP4)
        torch.cuda.synchronize()
        tl_ok = record("DeepInfer store_kv_cache (TileLang store_kv_cache_tl)", True, "SWA FP8 + main FP4 written")
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        record("DeepInfer store_kv_cache (TileLang store_kv_cache_tl)", False, f"{type(exc).__name__}: {exc}")
    if ref_ok and tl_ok:
        eq_main = torch.equal(ref.gpu_blocks, tl.gpu_blocks)
        eq_swa = torch.equal(ref.state, tl.state)
        record("DeepInfer TL writer bytes == torch ref writer bytes (whole backing incl. untouched bytes)",
               eq_main and eq_swa, f"main_backing_equal={eq_main} swa_backing_equal={eq_swa}")
    if tl_ok:
        S["caches"], NOTES["writer"] = tl, "DeepInfer TileLang store_kv_cache_tl"
    elif ref_ok:
        S["caches"], NOTES["writer"] = ref, "DeepInfer store_kv_cache_ref (torch)"
    else:
        # Last resort so the b12x compute stages still run: b12x's own record encoder written
        # into the DeepInfer-layout memory. This does NOT prove DeepInfer-writer compat.
        fb = S["caches0"].clone()
        pack = B12X["cref"].pack_deepseek_v41_cache_reference
        for cache, kv, slots, kind, page in ((fb.swa_cache, S["swa_kv"], S["swa_slots"], "swa", SWA_BLOCK),
                                            (fb.ca_cache, S["main_kv"], S["main_slots"], "indexed", MAIN_PAGE)):
            rec = pack(kv, page_size=1, cache_kind=kind)
            s = slots.long()
            cache[s // page, s % page, 0, :] = rec
        S["caches"], NOTES["writer"] = fb, "FALLBACK b12x pack_deepseek_v41_cache_reference (NOT DeepInfer)"
        record("writer", False, NOTES["writer"])
    record("writer used", None, NOTES["writer"])


def _full_segments(num_blocks: int, block_size: int, device) -> torch.Tensor:
    b = torch.arange(num_blocks, dtype=torch.int32, device=device)
    seg = torch.stack([b, torch.zeros_like(b), torch.full_like(b, block_size), b * block_size], dim=1)
    return seg.contiguous()


def dequant_and_byte_compat(device) -> None:
    c = S["caches"]
    V41, FP4 = DI["V41_FORMAT"], DI["V41_FP4_FORMAT"]
    unpack = B12X["cref"].unpack_deepseek_v41_cache_reference
    pack = B12X["cref"].pack_deepseek_v41_cache_reference
    for kind, cache4d, view2d, page, layout, slots, kv, rec_bytes in (
        ("swa", c.swa_cache, c.b12x_swa(), SWA_BLOCK, V41, S["swa_slots"], S["swa_kv"], SWA_REC),
        ("indexed", c.ca_cache, c.b12x_main(), MAIN_PAGE, FP4, S["main_slots"], S["main_kv"], MAIN_REC),
    ):
        nb = cache4d.shape[0]
        segs = _full_segments(nb, page, device)
        di = DI["load_paged_kv_cache_ref"](cache4d, segs, nb * page, layout=layout).bf16_data.float()
        S[f"deq_{kind}"] = di
        try:
            di_tl = DI["load_paged_kv_cache"](cache4d, segs, nb * page, layout=layout).bf16_data.float()
            s = slots.long()
            record(f"[{kind}] DeepInfer TL load_paged_kv_cache == _ref (written slots)",
                   torch.equal(di_tl[s], di[s]), "")
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            record(f"[{kind}] DeepInfer TL load_paged_kv_cache", None, f"unavailable: {type(exc).__name__}: {exc}")
        bx = unpack(view2d, page_size=page, cache_kind=kind).float()
        S[f"deqb_{kind}"] = bx
        s = slots.long()
        same = torch.equal(di[s], bx[s])
        nmis = int((di[s] != bx[s]).sum().item())
        record(f"[{kind}] BYTE-COMPAT: DeepInfer dequant == b12x unpack_deepseek_v41 (bit-exact, written slots)",
               same, f"{s.numel()} tokens x {D}, mismatching elems={nmis}")
        qm = metrics(di[s], kv.float())
        record(f"[{kind}] quantization sanity (dequant vs original bf16)", qm["cos"] > (0.99 if kind == "swa" else 0.95),
               fmt(qm))
        # Encoder agreement (informational): DeepInfer bytes vs b12x's independent encoder bytes.
        rec_b = pack(kv, page_size=1, cache_kind=kind)
        rec_d = cache4d[s // page, s % page, 0, :]
        byte_eq = (rec_b == rec_d).float().mean().item()
        tok_eq = (rec_b == rec_d).all(dim=1).float().mean().item()
        record(f"[{kind}] encoder agreement DeepInfer vs b12x pack_deepseek_v41_cache_reference", None,
               f"byte match {byte_eq * 100:.4f}%, whole-record match {tok_eq * 100:.3f}% (rounding-tie diffs OK)")


def build_indices(mode: str, device) -> dict:
    """Per q row: SWA ring slots (DeepInfer), logical top-k, physical top-k (DeepInfer)."""
    rows_seq, rows_pos = [], []
    for i, L in enumerate(SEQ_LENS):
        positions = [L - 1] if mode == "decode" else list(range(L - EXTEND_ROWS, L))
        for p in positions:
            rows_seq.append(i)
            rows_pos.append(p)
    R = len(rows_seq)
    # --- SWA: kv_swa_index_offset_len = [start slot, len] (input_metadata.py:313-316) ---
    off_len = []
    for i, p in zip(rows_seq, rows_pos):
        w = min(p + 1, WINDOW)
        start = p - w + 1
        off_len.append([S["state_blocks"][i] * SWA_BLOCK + start % SWA_BLOCK, w])
    off_len = torch.tensor(off_len, dtype=torch.int32, device=device)
    swa_idx = DI["get_kv_swa_index_decode_ref"](off_len, WINDOW, SWA_BLOCK).contiguous()
    try:
        swa_idx_tl = DI["get_kv_swa_index_decode"](off_len, WINDOW, SWA_BLOCK)
        record(f"[{mode}] DeepInfer TL get_kv_swa_index_decode == _ref", torch.equal(swa_idx_tl, swa_idx), "")
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        record(f"[{mode}] DeepInfer TL get_kv_swa_index_decode", None, f"unavailable: {type(exc).__name__}: {exc}")
    swa_len = off_len[:, 1].contiguous()
    # independent expectation
    exp = torch.full((R, WINDOW), -1, dtype=torch.int32)
    for r, (i, p) in enumerate(zip(rows_seq, rows_pos)):
        w = min(p + 1, WINDOW)
        pos = torch.arange(p - w + 1, p + 1)
        exp[r, :w] = (S["state_blocks"][i] * SWA_BLOCK + pos % SWA_BLOCK).to(torch.int32)
    record(f"[{mode}] SWA indices == expected ring slots", torch.equal(exp.to(device), swa_idx), f"rows={R}")

    # --- main: top-k logical (sorted, -1 padded like DeepInfer topk(sorted_index=True)) ---
    logical = torch.full((R, TOPK), -1, dtype=torch.int32)
    klen = []
    for r, p in enumerate(rows_pos):
        cvis = (p + 1) // RATIO
        klen.append(cvis)
        if cvis <= TOPK:
            sel = torch.randperm(cvis, generator=S["cpu_gen"])  # all of them, shuffled then sorted
        else:
            sel = torch.randperm(cvis, generator=S["cpu_gen"])[:TOPK]
        sel = sel.sort().values
        logical[r, :sel.numel()] = sel.to(torch.int32)
    logical = logical.to(device).contiguous()
    klen_t = torch.tensor(klen, dtype=torch.int32, device=device)
    main_len = klen_t.clamp(max=TOPK).contiguous()
    bt_rows = torch.tensor([S["block_tables"][i] for i in rows_seq], dtype=torch.int32, device=device).contiguous()
    # DeepInfer conversion logical -> physical slot (index_attn.py:508; context_lens = klen)
    phys_ref = DI["convert_kv_index_ref"](logical.clamp(min=0).unsqueeze(1), MAIN_PAGE, klen_t.unsqueeze(1), bt_rows)
    phys = phys_ref.squeeze(1).contiguous()
    try:
        phys_tl = logical.clone().unsqueeze(1).contiguous()
        DI["convert_kv_index"](phys_tl, MAIN_PAGE, klen_t.unsqueeze(1).contiguous(), bt_rows)
        record(f"[{mode}] DeepInfer TL convert_kv_index == _ref", torch.equal(phys_tl.squeeze(1), phys), "")
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        record(f"[{mode}] DeepInfer TL convert_kv_index", None, f"unavailable: {type(exc).__name__}: {exc}")
    lg = logical.long()
    exp_phys = torch.where(lg >= 0, bt_rows.long().gather(1, (lg.clamp(min=0) // MAIN_PAGE)) * MAIN_PAGE
                           + lg.clamp(min=0) % MAIN_PAGE, torch.full_like(lg, -1)).to(torch.int32)
    record(f"[{mode}] main physical indices == block_table mapping", torch.equal(exp_phys, phys), "")
    # b12x page table (LIL convention attention.py:244: vLLM block 0 reserved -> -1 holes)
    page_table = torch.where(bt_rows > 0, bt_rows, torch.full_like(bt_rows, -1)).contiguous()
    return dict(R=R, rows_seq=rows_seq, rows_pos=rows_pos, swa_idx=swa_idx, swa_len=swa_len,
                logical=logical, phys=phys, main_len=main_len, page_table=page_table)


def torch_reference(q: torch.Tensor, ix: dict, k_swa: torch.Tensor, k_main: torch.Tensor,
                    sink: torch.Tensor) -> torch.Tensor:
    """fp32 sparse MLA with attention sink: softmax over [SWA | top-k], K == V (512-d latent)."""
    swa_idx, main_idx = ix["swa_idx"].long(), ix["phys"].long()
    idx_valid = torch.cat([swa_idx >= 0, main_idx >= 0], dim=1)                     # [R, W]
    K = torch.cat([k_swa[swa_idx.clamp(min=0)], k_main[main_idx.clamp(min=0)]], dim=1)  # [R, W, D]
    scores = torch.einsum("rhd,rwd->rhw", q.float(), K) * SM_SCALE
    scores = scores.masked_fill(~idx_valid[:, None, :], float("-inf"))
    m = torch.maximum(scores.amax(dim=-1), sink[None, :])
    w = torch.exp(scores - m[..., None])
    denom = w.sum(dim=-1) + torch.exp(sink[None, :] - m)
    return torch.einsum("rhw,rwd->rhd", w, K) / denom[..., None]


def unquantized_kv_tables(device):
    k_swa = torch.zeros_like(S["deq_swa"])
    k_swa[S["swa_slots"].long()] = S["swa_kv"].float()
    k_main = torch.zeros_like(S["deq_indexed"])
    k_main[S["main_slots"].long()] = S["main_kv"].float()
    return k_swa, k_main


def run_mode(mode: str, device) -> None:
    mla, cref, tuning = B12X["mla"], B12X["cref"], B12X["tuning"]
    prep = B12X["prep"]
    c = S["caches"]
    ix = build_indices(mode, device)
    R = ix["R"]
    gen = S["gen"]
    q = torch.randn((R, H, D), generator=gen, device=device, dtype=torch.float32).to(torch.bfloat16).contiguous()
    sink = S["sink"]
    swa2d, main2d = c.b12x_swa(), c.b12x_main()

    ref = torch_reference(q, ix, S["deq_swa"], S["deq_indexed"], sink)
    S[f"ref_{mode}"] = ref
    # b12x's own torch reference on the same bytes (DeepInfer physical slots; it has no page table)
    try:
        bref = cref.compressed_sparse_mla_reference(
            q, swa2d, ix["swa_idx"], ix["swa_len"], sm_scale=SM_SCALE, attn_sink=sink,
            extra_k_cache=main2d, extra_indices=ix["phys"], extra_topk_lengths=ix["main_len"],
            swa_page_size=SWA_BLOCK, extra_page_size=MAIN_PAGE, cache_format="deepseek_v41")
        check_close(f"[{mode}] b12x compressed_sparse_mla_reference vs our fp32 ref", bref, ref, cos_thr=0.99999,
                    rel_max=0.02)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        record(f"[{mode}] b12x compressed_sparse_mla_reference", False, f"{type(exc).__name__}: {exc}")
    k_swa_u, k_main_u = unquantized_kv_tables(device)
    ref_unq = torch_reference(q, ix, k_swa_u, k_main_u, sink)
    record(f"[{mode}] quantized-KV ref vs unquantized-KV ref (pure FP8/FP4 KV error)", None, fmt(metrics(ref, ref_unq)))

    compute_modes = ["fp8"] + ([] if os.environ.get("TEST_SKIP_BF16") == "1" else ["bf16"])
    for compute in compute_modes:
        tag = f"[{mode}/{compute}]"
        with stage(f"b12x {mode} compute={compute}"):
            out = torch.empty((R, H, D), dtype=torch.bfloat16, device=device)
            caps = mla.Caps(
                device=device, num_q_heads=H, max_q_rows=R, max_width=WINDOW + TOPK,
                swa_width=WINDOW, indexed_width=TOPK, swa_page_size=SWA_BLOCK, indexed_page_size=MAIN_PAGE,
                max_page_table_width=S["max_blocks"], mode=mode,
                decode_row_capacity=R if mode == "decode" else None,
                cache_format="deepseek_v41", use_cuda_graph=False,
            )
            invocation = mla.invocation_from_tensors(q=q, swa_k_cache=swa2d, indexed_k_cache=main2d,
                                                     attn_sink=sink, out=out)
            plan0 = mla.plan(caps, invocation=invocation)
            override = None
            default_cfg = tuning._default_config(plan0.query, None)
            if compute != default_cfg.v41_compute_mode:
                from dataclasses import replace
                override = replace(default_cfg, v41_compute_mode=compute, v41_heads_per_block=16)
            plan = plan0 if override is None else mla.plan(caps, invocation=invocation, override=override)
            print(f"    query: mode={mode} rows={R} swa_page={SWA_BLOCK} idx_page={MAIN_PAGE} "
                  f"swa_cache {tuple(swa2d.shape)} stride {tuple(swa2d.stride())}, "
                  f"indexed_cache {tuple(main2d.shape)} stride {tuple(main2d.stride())}; "
                  f"default cfg={default_cfg}; override={override}")
            holder: dict = {}

            def prepare_call(state, holder=holder):
                holder["state"] = state
                return prep.PreparedCall(run=lambda: None)

            t0 = time.time()
            prepare_default = getattr(prep, "prepare_default", None)
            if prepare_default is not None:
                prepare_default(plan.request(name=f"feas.{mode}.{compute}", prepare_call=prepare_call))
            else:
                session = prep.PreparationSession(device=device, autotune=False)
                session.__enter__()
                S.setdefault("sessions", []).append(session)
                session.prepare((plan.request(name=f"feas.{mode}.{compute}", prepare_call=prepare_call),))
            print(f"    prepared (compile) in {time.time() - t0:.1f}s; selection={getattr(plan, 'selection', None)}")
            try:
                specs = plan.memory_requirements().scratch
            except Exception:  # noqa: BLE001
                specs = holder["state"].scratch_plan.scratch_specs()
            scratch = [torch.empty(spec.shape, dtype=spec.dtype, device=device) for spec in specs]

            variants = (
                ("LIL logical+page_table", dict(indexed_indices=ix["logical"], indexed_lengths=ix["main_len"],
                                                indexed_page_table=ix["page_table"])),
                ("DeepInfer physical slots", dict(indexed_indices=ix["phys"], indexed_lengths=ix["main_len"])),
            )
            for vname, vkw in variants:
                try:
                    binding = mla.bind(plan, scratch=scratch, q=q, swa_indices=ix["swa_idx"],
                                       swa_lengths=ix["swa_len"], **vkw)
                    out.fill_(float("nan"))
                    res = mla.run(plan=plan, binding=binding, swa_k_cache=swa2d, indexed_k_cache=main2d,
                                  swa_page_size=SWA_BLOCK, indexed_page_size=MAIN_PAGE, sm_scale=SM_SCALE,
                                  attn_sink=sink, out=out, cache_format="deepseek_v41")
                    torch.cuda.synchronize()
                    res = res[0] if isinstance(res, tuple) else res
                    if res.data_ptr() != out.data_ptr():
                        print("    note: run returned a tensor different from provided out")
                    check_close(f"{tag} b12x ({vname}) vs fp32 torch ref", res, ref)
                    m_unq = metrics(res, ref_unq)
                    record(f"{tag} b12x ({vname}) vs unquantized-KV ref", None, fmt(m_unq))
                    per_seq = []
                    for i in range(len(SEQ_LENS)):
                        rr = [r for r, s in enumerate(ix["rows_seq"]) if s == i]
                        per_seq.append(f"L={SEQ_LENS[i]}:cos={metrics(res[rr], ref[rr])['cos']:.5f}")
                    record(f"{tag} b12x ({vname}) per-seq", None, " ".join(per_seq))
                    S[f"out_{mode}_{compute}_{vname}"] = res.clone()
                except Exception as exc:  # noqa: BLE001
                    traceback.print_exc()
                    record(f"{tag} b12x ({vname})", False, f"{type(exc).__name__}: {exc}")
            a = S.get(f"out_{mode}_{compute}_LIL logical+page_table")
            b = S.get(f"out_{mode}_{compute}_DeepInfer physical slots")
            if a is not None and b is not None:
                record(f"{tag} page-table path == physical-slot path (bitwise)", torch.equal(a, b),
                       fmt(metrics(a, b)))


def main() -> int:
    print(f"python {sys.version.split()[0]} torch {torch.__version__} cuda_avail={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        record("cuda", False, "no CUDA device")
        return summary()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(0.6, device)
    props = torch.cuda.get_device_properties(device)
    record("device", props.major == 12, f"{props.name} sm_{props.major}{props.minor} {props.total_memory / 2**30:.1f} GiB")

    with stage("import DeepInfer"):
        import_deepinfer()
    with stage("import b12x"):
        import_b12x()
    if not DI.get("store_kv_cache_ref") or not B12X.get("mla"):
        record("prerequisites", False, "DeepInfer or b12x import failed; cannot continue")
        return summary()
    print(f"    stubbed modules: {STUBBED or 'none'}")

    with stage("build workload"):
        build_workload(device)
    with stage("write caches with DeepInfer writer"):
        write_caches()
    if "caches" not in S:
        return summary()
    with stage("dequant + byte compat"):
        dequant_and_byte_compat(device)
    if "deq_swa" not in S or "deq_indexed" not in S:
        return summary()
    with stage("decode mode"):
        run_mode("decode", device)
    if os.environ.get("TEST_SKIP_EXTEND") != "1":
        with stage("extend mode"):
            run_mode("extend", device)
    print(f"    peak GPU memory allocated: {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB")
    return summary()


def summary() -> int:
    for sess in S.get("sessions", []):
        with contextlib.suppress(Exception):
            sess.__exit__(None, None, None)
    print("\n" + "=" * 100)
    print(f"writer: {NOTES.get('writer')} | ops: {NOTES.get('ops')} | stubbed: {STUBBED or 'none'}")
    n_pass = sum(1 for s, _, _ in RESULTS if s == "PASS")
    n_fail = sum(1 for s, _, _ in RESULTS if s == "FAIL")
    for status, name, detail in RESULTS:
        if status != "INFO":
            print(f"  {status}  {name}")
    verdict = "PASS" if n_fail == 0 and n_pass > 0 else "FAIL"
    print(f"SUMMARY: {verdict}  ({n_pass} passed, {n_fail} failed)")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
