"""Minimal runtime patches that let DeepInfer run on SM120 (RTX 5090 / RTX PRO 6000), single GPU, EP=1, CP=1.

Imported by sm120/sitecustomize.py at interpreter start-up. Nothing heavy happens at import: every patch
is a post-import hook that fires when DeepInfer imports the target module (no torch import here).
Every patch is listed and justified in patches/PATCHES.md.

    SM120_PATCHES=0                  disable all patches
    SM120_FORCE_SM120=1|0            override the device check (default: detect compute capability 12.x)
    SM120_BF16_TFLOPS=<int>          dense BF16 peak used for DeepInfer's MFU statistics
    SM120_FUSED_NORM_ROPE_ATTN=auto|0|1   DeepInfer's FlashMLA fused norm/rope/attention path (attn.py:87)
    SM120_MEGA_MHC=0|1               keep DeepInfer's mega_mhc fusion (default 0 -> tile_kernels unfused path)
    SM120_MAX_GPU_BLOCKS=<int>       cap the KV-cache block count found by memory profiling
    SM120_MOE_IMPL=b12x|torch|deepep see sm120_moe.py
    SM120_MOE_TINY_DECODE=0|1|auto   b12x w4a8 tiny-decode MoE path for T<=4 (sets B12X_W4A8_TINY_DECODE unless
                                     that is set explicitly). 0 (default): dynamic path, measured faster on the
                                     5090 (tiny 0.90-0.95x). 1/auto: tiny-decode iff the installed b12x carries
                                     patches/b12x-tiny-decode-swiglu.patch (clamp fix), else still off.
    SM120_REMAP_SMEM_BYTES=<int>     shared-memory budget of the patched sampler.remap_draft_topk kernel (P8)
    SM120_CACHE_ROOT=/cache          persistent cache root: B12X_COMPILE_CACHE_DIR, TRITON_CACHE_DIR defaults
    SM120_RETAIN_GRAPH_PROGRAMS=1|0  keep the b12x programs launched by graph warm-up/capture loaded (P9, default 1)
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.abc
import importlib.util
import os
import sys

_HOOKS: dict[str, list] = {}
_DONE: set[str] = set()
_BUSY: set[str] = set()
APPLIED: list[str] = []

def b12x_tiny_decode_patched() -> bool | None:
    """True if the installed b12x tiny-decode kernel applies swiglu_limit (our patch), None if b12x is absent.
    Reads the source file instead of importing b12x (no torch import at interpreter start-up)."""
    try:
        spec = importlib.util.find_spec("b12x")
    except Exception:  # noqa: BLE001
        return None
    for base in (spec.submodule_search_locations or []) if spec is not None else []:
        path = os.path.join(base, "moe", "_shared", "kernels", "tiny_decode.py")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                return "SWIGLU_LIMIT_SUPPORTED = True" in f.read()
    return None


def _tiny_decode_policy() -> str:
    """Decide B12X_W4A8_TINY_DECODE (read by b12x at plan time). The b12x 1.5.1 tiny-decode path (T<=4,
    n % 128 == 0) ignores swiglu_limit (FlashInfer issue #6866), V4.1 always clamps at 10."""
    if "B12X_W4A8_TINY_DECODE" in os.environ:
        return f"B12X_W4A8_TINY_DECODE={os.environ['B12X_W4A8_TINY_DECODE']} (set by the user)"
    # Default off even on a patched b12x: CUDA-graph replay on the 5090 (V4.1 expert shape, T=1..4) measured
    # tiny-decode at 0.90-0.95x the dynamic path (T=1 71.4 vs 67.6 us, T=4 305.8 vs 287.8 us).
    mode = os.environ.get("SM120_MOE_TINY_DECODE", "0").strip().lower()
    patched = b12x_tiny_decode_patched()
    if mode in ("0", "off", "false"):
        on, why = False, "SM120_MOE_TINY_DECODE=0: dynamic path, faster than tiny-decode on the 5090"
    else:
        on = bool(patched)
        why = f"SM120_MOE_TINY_DECODE={mode}, b12x tiny-decode swiglu patch {'present' if patched else 'ABSENT -> kept off'}"
    os.environ["B12X_W4A8_TINY_DECODE"] = "1" if on else "0"
    return f"B12X_W4A8_TINY_DECODE={'1' if on else '0'} ({why})"


def _cache_defaults() -> None:
    root = os.environ.get("SM120_CACHE_ROOT", "/cache")
    if os.path.isdir(root) and os.access(root, os.W_OK):
        os.environ.setdefault("B12X_COMPILE_CACHE_DIR", os.path.join(root, "b12x", "compile"))
        os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(root, "triton"))


# b12x env defaults (read by b12x at plan time)
TINY_DECODE_POLICY = _tiny_decode_policy()
# Our plans use b12x's lazy default preparation (heuristic configs), which ignores B12X_AUTOTUNE. Only
# SM120_B12X_AUTOTUNE=1 (deep_gemm shim, block_fp8_linear) runs PreparationSessions, and those stop tuning
# when B12X_AUTOTUNE=0.
os.environ.setdefault("B12X_AUTOTUNE", "1" if os.environ.get("SM120_B12X_AUTOTUNE", "0") == "1" else "0")
_cache_defaults()


# ------------------------------------------------------------------------------------------------ hook machinery
class _Loader(importlib.abc.Loader):
    def __init__(self, inner, name):
        self._inner = inner
        self._name = name

    def create_module(self, spec):
        return self._inner.create_module(spec) if hasattr(self._inner, "create_module") else None

    def exec_module(self, module):
        self._inner.exec_module(module)
        _run_hooks(self._name, module)

    def __getattr__(self, item):
        return getattr(self._inner, item)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname not in _HOOKS or fullname in _BUSY or fullname in _DONE:
            return None
        _BUSY.add(fullname)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            _BUSY.discard(fullname)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _Loader(spec.loader, fullname)
        return spec


def _run_hooks(name, module) -> None:
    if name in _DONE:
        return
    _DONE.add(name)
    for fn in _HOOKS.get(name, ()):
        try:
            fn(module)
        except Exception as exc:  # noqa: BLE001
            print(f"[sm120_patches] patch {fn.__name__} on {name} FAILED: {type(exc).__name__}: {exc}", flush=True)
            raise


def enabled() -> bool:
    return os.environ.get("SM120_PATCHES", "1").strip().lower() not in ("0", "false", "no", "off")


def when_imported(name: str):
    """Register ``fn(module)`` to run right after ``name`` is first imported (public: the attention
    shim may use it from ``sm120_attn.patches``)."""

    def deco(fn):
        _HOOKS.setdefault(name, []).append(fn)
        mod = sys.modules.get(name)
        if enabled() and mod is not None and name not in _DONE:
            _run_hooks(name, mod)
        return fn

    return deco


def _note(msg: str) -> None:
    APPLIED.append(msg)
    if os.environ.get("SM120_PATCH_VERBOSE", "1") not in ("0", "false"):
        print(f"[sm120_patches] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ device check
def is_sm12x() -> bool:
    forced = os.environ.get("SM120_FORCE_SM120")
    if forced is not None:
        return forced not in ("0", "false", "no", "off")
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_capability()[0] == 12
    except Exception:  # noqa: BLE001
        pass
    try:
        from deep_cluster.hardware import nvidia

        cc = nvidia.compute_capability()
        return cc is not None and cc[0] == 12
    except Exception:  # noqa: BLE001
        return False


def _gpu_name() -> str:
    try:
        import torch

        return torch.cuda.get_device_name()
    except Exception:  # noqa: BLE001
        return ""


def bf16_tflops_sm120() -> int:
    v = os.environ.get("SM120_BF16_TFLOPS")
    if v:
        return int(float(v))
    name = _gpu_name().upper()
    if "5090" in name:
        return 210  # GB202 GeForce: ~209.5 TFLOPS dense BF16 (fp32 accumulate)
    if "PRO 6000" in name and "MAX-Q" in name:
        return 190
    if "PRO 6000" in name:
        return 250
    return 200


# ------------------------------------------------------------------------------------------------ patches
@when_imported("deep_cluster.hardware")
def _patch_tflops(mod):
    """P1: get_bf16_tflops_limit raises for non-SM10x (llm_engine.py:304, MFU statistics only)."""
    dev = importlib.import_module("deep_cluster.hardware.device")
    orig = dev.get_bf16_tflops_limit
    if getattr(orig, "_sm120", False):
        return

    def get_bf16_tflops_limit():
        try:
            return orig()
        except RuntimeError:
            if is_sm12x():
                return bf16_tflops_sm120()
            raise

    get_bf16_tflops_limit._sm120 = True
    dev.get_bf16_tflops_limit = get_bf16_tflops_limit
    mod.get_bf16_tflops_limit = get_bf16_tflops_limit
    _note("P1 deep_cluster.hardware.get_bf16_tflops_limit: SM12x value for MFU stats")


@when_imported("deepinfer.model_executor.parallel_utils.parallel_state")
def _patch_nvl(mod):
    """P2: no NVLink on SM120 -> never "within NVLink domain" (mega-MoE / symm-mem EPLB shuffle off)."""
    cls = mod.ParallelProcessGroup
    orig = cls.__dict__["is_within_nvl_domain"].__func__

    def is_within_nvl_domain(group):
        from deepinfer import PLATFORM

        if PLATFORM.is_gpu and is_sm12x():
            return False
        return orig(group)

    cls.is_within_nvl_domain = staticmethod(is_within_nvl_domain)
    _note("P2 ParallelProcessGroup.is_within_nvl_domain -> False on SM12x (non-mega MoE, no symm-mem shuffle)")


@when_imported("deepinfer.model_executor.models.v4_modules.mlp")
def _patch_moe(mod):
    """P3: DeepseekMoE -- no mega gate (deep_gemm.bf16_mega_gate) + SM120 routed-expert backend."""
    import sm120_moe

    cls = mod.DeepseekMoE
    orig_init, orig_forward = cls.__init__, cls.forward

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        if is_sm12x():
            self.use_mega_gate = False
            sm120_moe.setup(self)

    def forward(self, hidden_states, hidden_states_bf16, input_metadata):
        return sm120_moe.forward(self, orig_forward, hidden_states, hidden_states_bf16, input_metadata)

    cls.__init__ = __init__
    cls.forward = forward
    _note(f"P3 DeepseekMoE: use_mega_gate=False, routed experts via SM120_MOE_IMPL={sm120_moe.moe_impl()}")


@when_imported("deepinfer.model_executor.models.v4_modules.hc")
def _patch_hc(mod):
    """P4: HCLayerMixin.use_mega_mhc (hc.py:99) -> False: DeepInfer's own tile_kernels mHC path."""
    cls = mod.HCLayerMixin
    orig_init = cls.__init__
    keep = os.environ.get("SM120_MEGA_MHC", "0") in ("1", "true", "yes", "on")

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        if not keep and hasattr(self, "use_mega_mhc") and is_sm12x():
            self.use_mega_mhc = False

    cls.__init__ = __init__
    _note(f"P4 HCLayerMixin.use_mega_mhc -> {'unchanged (torch mega_mhc reference)' if keep else 'False'}")


def _fused_attn_available() -> bool:
    try:
        import flash_mla

        f = getattr(flash_mla, "fused_norm_rope_attn_rope_cast", None)
        return f is not None and all(hasattr(f, n) for n in ("prefill", "decode", "permute_q_b_proj", "permute_wv_proj"))
    except Exception:  # noqa: BLE001
        return False


@when_imported("deepinfer.model_executor.models.v4_modules.attn")
def _patch_attn(mod):
    """P5: V4AttentionBase.use_fused_norm_rope_attn_rope_cast (attn.py:87-92) gate.

    auto (default): keep DeepInfer's decision iff the flash_mla shim provides the fused API
    (prefill/decode/permute_q_b_proj/permute_wv_proj); 0: force the unfused path; 1: never touch.
    Must be decided in __init__ because the flag also selects the weight permutations (attn.py:114,143).
    """
    mode = os.environ.get("SM120_FUSED_NORM_ROPE_ATTN", "auto").strip().lower()
    cls = mod.V4AttentionBase
    orig_init = cls.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        if not is_sm12x() or mode in ("1", "true", "on", "keep"):
            return
        if mode in ("0", "false", "off") or not _fused_attn_available():
            self.use_fused_norm_rope_attn_rope_cast = False

    cls.__init__ = __init__
    _note(f"P5 V4AttentionBase.use_fused_norm_rope_attn_rope_cast gate mode={mode} "
          f"(fused API available: {_fused_attn_available()})")


@when_imported("deepinfer.model_executor.layers.sampler")
def _patch_sampler(mod):
    """P6 (instrumentation, inert unless run_deepinfer arms it): logit capture."""
    import sm120_capture

    cls = mod.Sampler
    orig_forward = cls.forward
    orig_get_logits = mod.get_logits

    def forward(self, mtp_module, embedding, hidden_states, sampler_inputs, main_input_metadata, *args, **kwargs):
        sm120_capture.on_sampler_forward(embedding, hidden_states, main_input_metadata,
                                         kwargs.get("seq_metadata_list", args[1] if len(args) > 1 else None))
        return orig_forward(self, mtp_module, embedding, hidden_states, sampler_inputs, main_input_metadata,
                            *args, **kwargs)

    def get_logits(hidden_states, embedding, vocab_size):
        logits = orig_get_logits(hidden_states, embedding, vocab_size)
        sm120_capture.on_get_logits(logits)
        return logits

    cls.forward = forward
    mod.get_logits = get_logits
    _note("P6 Sampler.forward / sampler.get_logits wrapped for logit capture (inert until armed)")


@when_imported("deepinfer.worker.base")
def _patch_blocks(mod):
    """P7 (optional): cap the profiled KV block count (SM120_MAX_GPU_BLOCKS) to keep headroom."""
    cap = os.environ.get("SM120_MAX_GPU_BLOCKS")
    if not cap:
        return
    cap = int(cap)
    cls = mod.WorkerBase
    orig = cls.profile_num_available_blocks

    def profile_num_available_blocks(self):
        n = orig(self)
        if n > cap:
            n = cap // 2 * 2
            self.cache_config.num_gpu_blocks = n
        return n

    cls.profile_num_available_blocks = profile_num_available_blocks
    _note(f"P7 WorkerBase.profile_num_available_blocks capped at {cap} blocks")


_REMAP_MOD = "deepinfer.model_executor.layers.tilelang_ops.cuda.sampler"
_REMAP_POOL_LINE = "pool_size = 25856 if force_topk <= 1024 else 44032"


def remap_pool_size(vocab_size: int) -> int:
    """fp32 entries of the shared vocab window of remap_draft_topk on SM12x: the fewest equal windows
    that fit SM120_REMAP_SMEM_BYTES (default 96 KiB < the 99 KiB opt-in limit), rounded up to 64.
    The kernel loops over ceil(vocab / pool) windows, so any pool size gives the same result."""
    budget = int(os.environ.get("SM120_REMAP_SMEM_BYTES", str(96 * 1024))) // 4
    windows = -(-int(vocab_size) // budget)
    pool = -(-int(vocab_size) // windows)
    return min(budget // 64 * 64, -(-pool // 64) * 64)


@when_imported(_REMAP_MOD)
def _patch_remap_draft_topk(mod):
    """P8: sampler.remap_draft_topk (MTP rejection with DEEPINFER_SAMPLER_FORCE_TOPK>0) allocates a
    25856/44032-float shared window (103/176 KB); SM120 allows 99 KB. Re-define the TileLang factory from
    its own source with an SM12x pool size. Runs while tilelang_ops/ops.py imports the backend, i.e.
    before ops re-exports ``remap_draft_topk_tl``."""
    if not is_sm12x():
        return
    import ast
    import linecache

    path = mod.__file__
    with open(path, encoding="utf-8") as f:
        src = f.read()
    node = next((n for n in ast.parse(src).body
                 if isinstance(n, ast.FunctionDef) and n.name == "remap_draft_topk_tl"), None)
    if node is None or src.count(_REMAP_POOL_LINE) != 1:
        _note("P8 remap_draft_topk: source layout changed, NOT patched (kernel may exceed SM120 shared memory)")
        return
    start = min([d.lineno for d in node.decorator_list] + [node.lineno]) - 1
    lines = src.splitlines(keepends=True)
    func_src = "".join(lines[start:node.end_lineno]).replace(
        _REMAP_POOL_LINE, "pool_size = _sm120_remap_pool_size(vocab_size)")
    # TileLang/TVMScript re-read the prim_func source through inspect/linecache: register the patched text
    # under its own name (line numbers kept) so parser, tracebacks and source-keyed caches see it.
    fake = f"<sm120_patches:{path}>"
    text = "\n" * start + func_src
    linecache.cache[fake] = (len(text), None, text.splitlines(keepends=True), fake)
    mod.__dict__["_sm120_remap_pool_size"] = remap_pool_size
    # dont_inherit: this file's `from __future__ import annotations` must not leak into the rebuilt code, or the
    # prim_func annotations (T.Tensor[(pool_rows, ...)]) become strings TileLang can't resolve (closure locals)
    exec(compile(text, fake, "exec", dont_inherit=True), mod.__dict__)  # noqa: S102  re-binds remap_draft_topk_tl
    _note(f"P8 tilelang sampler.remap_draft_topk_tl: shared vocab window -> {remap_pool_size(129280)} floats "
          f"for vocab 129280 (was 25856/44032)")


@when_imported("deepinfer.model_executor.layers.tilelang_ops.ops")
def _patch_remap_reexport(mod):
    """P8 (cont.): if the backend module was imported before ops.py, refresh the re-export."""
    backend = sys.modules.get(_REMAP_MOD)
    fn = getattr(backend, "remap_draft_topk_tl", None) if backend is not None else None
    if fn is not None and "_sm120_remap_pool_size" in backend.__dict__ and getattr(mod, "remap_draft_topk_tl", None) is not fn:
        mod.remap_draft_topk_tl = fn


_RETAINED_B12X: list = []


@contextlib.contextmanager
def _retain_b12x_programs():
    """Keep every b12x program launched by the warm-up + capture loaded for the life of the process.

    Every later lazy b12x preparation (a prefill / indexer shape first seen while serving) ends with
    evict_unretained(), which drops each program that is neither in that session's plans nor retained,
    and the collected CuTe executable unloads its CUDA library. A captured graph that launches such a
    program then fails on replay with "illegal instruction". b12x's retain_compiled_programs() records
    the programs launched inside it; the record only counts while the object is alive, so it is kept."""
    if os.environ.get("SM120_RETAIN_GRAPH_PROGRAMS", "1") == "0":
        yield
        return
    try:
        from b12x._lib.compile_plan import retain_compiled_programs
    except Exception:  # noqa: BLE001 - no b12x / no such API: nothing to retain
        yield
        return
    with retain_compiled_programs() as retained:
        _RETAINED_B12X.append(retained)
        yield


def _wrap_graph_capture(cls, label: str) -> None:
    orig = cls.capture
    if getattr(orig, "_sm120", False):
        return

    def capture(self, *args, **kwargs):
        import time

        import sm120_moe
        from deep_gemm import _b12x

        sizes = list(getattr(self, "batch_sizes", []))
        if label == "CudaGraphRunner":
            sm120_moe.add_graph_buckets(sizes)  # main graphs: batch size == token count (incl. MTP rows)
        m0, g0 = sm120_moe.stats(), _b12x.stats()
        t0 = time.time()
        print(f"[sm120_patches] {label} capture start: sizes={sizes} moe={m0} gemm={g0}", flush=True)
        try:
            with _retain_b12x_programs():
                return orig(self, *args, **kwargs)
        except Exception as exc:
            print(f"[sm120_patches] {label} capture FAILED ({type(exc).__name__}: {exc}). Every shim op inside "
                  "the forward must be capturable: b12x MoE/linears are (plans are warmed by DeepInfer's "
                  "per-size warm-up run); SM120_MOE_IMPL=torch|deepep are not. "
                  "Set DEEPINFER_ENFORCE_EAGER=1 to bisect.", flush=True)
            raise
        finally:
            print(f"[sm120_patches] {label} capture done in {time.time() - t0:.1f}s: moe={sm120_moe.stats()} "
                  f"gemm={_b12x.stats()}", flush=True)

    capture._sm120 = True
    cls.capture = capture


@when_imported("deepinfer.hugging_face.config")
def _patch_yoco_boundary(mod):
    """P13 (EP + YOCO compact): record n_prefill_layer, the first YOCO consumer layer, for sm120_ep (its consumer
    layers exchange the kept-row counts once per prefill step instead of padding to the step's tokens)."""
    orig = getattr(mod, "_n_prefill_layer", None)
    if orig is None or getattr(orig, "_sm120", False):
        return

    def _n_prefill_layer(*args, **kwargs):
        n = orig(*args, **kwargs)
        try:
            from sm120_ep import dp_sync
            dp_sync._STATE["yoco_boundary"] = int(n) if n is not None else None
        except Exception:  # noqa: BLE001
            pass
        return n

    _n_prefill_layer._sm120 = True
    mod._n_prefill_layer = _n_prefill_layer
    _note("P13 hugging_face.config._n_prefill_layer: YOCO boundary recorded for sm120_ep")


@when_imported("deepinfer.worker.cuda_graph_runner")
def _patch_graph_runner(mod):
    """P9 (logging only): CudaGraphRunner.capture reports the b12x plans created by the warm-up runs."""
    _wrap_graph_capture(mod.CudaGraphRunner, "CudaGraphRunner")
    _note("P9 CudaGraphRunner.capture wrapped (b12x plan summary, failure hint)")


@when_imported("deepinfer.worker.dflash_cuda_graph_runner")
def _patch_dflash_graph_runner(mod):
    """P9 (cont.): same for the DFlash/DSpark draft graphs."""
    _wrap_graph_capture(mod.DFlashCudaGraphRunner, "DFlashCudaGraphRunner")
    _note("P9 DFlashCudaGraphRunner.capture wrapped")


def numa_node_count() -> int:
    """NUMA nodes the kernel reports (/sys/devices/system/node/node<N>); 1 if unknown."""
    try:
        return max(1, sum(1 for d in os.listdir("/sys/devices/system/node")
                          if d.startswith("node") and d[4:].isdigit()))
    except OSError:
        return 1


@when_imported("deep_cluster.topology")
def _patch_bind_numa(mod):
    """P10: deep_cluster.topology.bind_numa(which_numa(gpu)) segfaults when the GPU has no NUMA affinity (sysfs
    numa_node = -1, the normal value on single-socket hosts): libnuma is called with node -1 and
    numa_bitmask_setbit(mask, (unsigned)-1) writes out of bounds. Without libnuma it "only" logs a long traceback
    (worker/base.py _bind_numa, ve/multimodal_processor.py). Binding is meaningless with one node, so skip it there
    and for node < 0; multi-node hosts with a valid node keep DeepInfer's binding (needs libnuma1)."""
    from deep_cluster.topology import bind as bind_mod

    orig = mod.bind_numa
    if getattr(orig, "_sm120", False):
        return

    def bind_numa(node: int) -> int:
        nodes = numa_node_count()
        if int(node) < 0 or nodes <= 1:
            if not _NUMA_LOGGED:
                _NUMA_LOGGED.append(True)
                print(f"[sm120_patches] NUMA binding skipped (GPU NUMA node {int(node)}, {nodes} node(s) on this host)",
                      flush=True)
            return 0
        return orig(node)

    bind_numa._sm120 = True
    mod.bind_numa = bind_numa
    bind_mod.bind_numa = bind_numa
    _note("P10 deep_cluster.topology.bind_numa: skipped on single-node hosts / node < 0 (libnuma segfault)")


_NUMA_LOGGED: list = []


@when_imported("deepinfer.ve.multimodal_processor")
def _patch_mm_pool_unlink(mod):
    """P11: unlink the multimodal pool's /dev/shm name (multiprocessing.shared_memory, psm_*) as soon as the worker
    processes exist. They are forked (ForkProcessWithCrashHeader) and inherit the mapping, nothing reopens the
    segment by name, so the memory now lives exactly as long as a process maps it: no psm_* is left behind after
    SIGTERM/SIGKILL/OOM (DeepInfer only unlinks in LLMEngine.close(), which a killed server never reaches)."""
    cls = mod.MultimodalProcessor
    orig = cls._start_workers
    if getattr(orig, "_sm120", False):
        return

    def _start_workers(self, num_workers):
        orig(self, num_workers)
        pool = mod.get_multi_modal_pool()
        shm = getattr(pool, "_buffer_shm", None) if pool is not None else None
        if shm is None:
            return
        try:
            shm.unlink()  # shm_unlink + resource_tracker unregister
        except FileNotFoundError:
            pass
        shm.unlink = lambda: None  # SharedTorchMemoryPool.close() unlinks again
        print(f"[sm120_patches] multimodal pool /dev/shm/{shm.name} unlinked after fork "
              f"({pool.total_bytes >> 20} MiB stay mapped until exit)", flush=True)

    _start_workers._sm120 = True
    cls._start_workers = _start_workers
    _note("P11 MultimodalProcessor._start_workers: multimodal pool shm name unlinked after the worker fork")


@when_imported("deepinfer")
def _patch_attn_agent(mod):
    """Hand-off point for the attention shim's own DeepInfer patches (optional module)."""
    try:
        spec = importlib.util.find_spec("sm120_attn.patches")
    except Exception:  # noqa: BLE001
        spec = None
    if spec is not None:
        p = importlib.import_module("sm120_attn.patches")
        if hasattr(p, "install"):
            p.install()
            _note("sm120_attn.patches.install() called")


@when_imported("deepinfer.model_executor.layers.sampler_utils")
def _patch_lmhead_agent(mod):
    """Hand-off point for the FP8 lm_head (sm120_lmhead.py): wraps sampler_utils.bf16_gemm_nt,
    inert unless SM120_LMHEAD_IMPL=fp8. P6's get_logits wrapper calls the original get_logits, which looks
    bf16_gemm_nt up at call time, so both compose."""
    try:
        spec = importlib.util.find_spec("sm120_lmhead")
    except Exception:  # noqa: BLE001
        spec = None
    if spec is not None:
        import sm120_lmhead

        sm120_lmhead.install()
        _note("sm120_lmhead.install() called")


@when_imported("deepinfer")
def _patch_engram_agent(mod):
    """Hand-off point for the engram patches (sm120_patches_engram.py; no-op for models without engram)."""
    try:
        spec = importlib.util.find_spec("sm120_patches_engram")
    except Exception:  # noqa: BLE001
        spec = None
    if spec is not None:
        import sm120_patches_engram

        sm120_patches_engram.install()
        _note("sm120_patches_engram.install() called")


def install() -> None:
    if not enabled():
        return
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
