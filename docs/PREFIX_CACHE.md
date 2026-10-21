# Prefix caching for DeepInfer V4.1 on SM120

`prefix_cache/prefix_cache.py` reuses the KV cache and the per-sequence state of prompt prefixes that were already computed. It is exact: with a
deterministic MoE, a request served from the cache produces bit-identical logits and tokens to the same request without a hit, when both use the same prefill chunk boundaries (see "Exactness").

Checkpoints are the truncated test models described in [VALIDATION.md](VALIDATION.md); numbers are from one RTX 5090.

The cache itself is Python: it wraps `LLMEngine.add_request`, `step`, `flush_inflight_reqs` and `scheduler.schedule`. It needs
**one C++ line**, a read-only `Sequence.block_table` binding:
[`patches/deepinfer-sequence-block-table.patch`](../patches/deepinfer-sequence-block-table.patch).

## Why this is small: DeepInfer already has the hit mechanics

DeepSeek's KV store is not in the open-source release, but its hooks are in the C++ scheduler:
- `Scheduler.fork_seq(parent, blocks, "_<branch>", ...)` creates a sequence with the parent's tokens, pins the given KV blocks, and pins the parent's **state slot** (`scheduler.hpp:1060-1120`).
- On scheduling, a waiting sequence whose `branch_id` starts with `_` takes the **prefix-hit path** (`scheduler.hpp:398-433`):
  - it keeps `num_tokens // 128` blocks;
  - `_clone_state` copies the state slot copy-on-write;
  - a partially filled tail block is copied;
  - prefill continues at token `num_tokens`, exactly like a later chunk of a chunked prefill.
- `multimodal_processor.add(..., num_cached_tokens)`, `Sequence.cached_hit_tokens` and `chunk_checkpoints` exist for the same purpose.

The state slot holds everything that is not in the paged KV blocks:
- the SWA ring of every layer (consumer layers included, in the "dropped" region),
- the ratio-2 compressor and indexer partial groups,
- the DFlash/DSpark SWA.

The compressed KV (FP4, 288 B/token) and the index K (68 B/token) of the kv/index source layers are paged and append-only, so a prefix's blocks
never change once written.

## What the module adds

| piece | how |
|---|---|
| snapshot points | every request gets `chunk_checkpoints` at multiples of `grain` (default 128 in tests, 512 in the server), so prefill chunks end there; plus the end of every prompt; plus the **end of every generation** (prompt + answer, for the next chat turn) |
| generation end | the step wrapper reads finished request ids from the step's outputs (`RequestOutput.get_finished`). The sequence (remembered from `schedule`) is snapshotted after `step()` returns, before the next `schedule()` frees it, at k = its `num_cached_tokens` read together with its state. With ZOS the engine post-processes step N inside step N+1 *after* launching N+1 (`worker.execute_model_step`), so a finished sequence may already have run one more forward; reading k and copying the state after everything launched keeps the pair consistent |
| snapshot | after a step whose prefill chunk ended at k: `BlockManager.pin_blocks(seq.block_table[:ceil(k/128)])`, and the state slot is copied into a cache-owned slot (`clone_and_zero_states_and_blocks_on_default_stream`, ordered on the stream after the step's forward and before the sequence's next step) |
| index | key = (k, blake2b of the k token ids), entries in an LRU. Eviction unpins the blocks and frees the slot |
| capacity | `max_entries` snapshots (state slots reserved at engine start via `DEEPINFER_NUM_STATE_SLOTS_RATIO`, about 3 MB each for 40 layers); at most `max_block_frac` of the KV blocks pinned by the cache; `max_num_seqs + 1` state slots always left free |
| hit | in `add_request`: the longest cached k ≤ len - 2 with a matching hash. The engine builds the Sequence (`enqueue=False`), gets `num_cached_tokens = k` and the snapshot's `state_id`, then goes through `fork_seq(seq, cached blocks, "_pc", ...)`. For vision models the fork is held unschedulable until the multimodal processor (`add(..., num_cached_tokens=k)`) has produced its token-type tensors, exactly as an enqueued request |
| engram | nothing special: n-gram hashes are computed from the sequence's token ids, which the fork carries |
| not cached | prompts with images, `prompt_logprobs` (fork_seq refuses), structured output |

`k ≤ len - 2` keeps at least the last prompt token computed, which is needed for the first logits. It also keeps `fork_seq` on its "prompt incomplete" path.

## Build and run

The serving image (`docker/Dockerfile`) builds DeepInfer with the patch already applied, so `--prefix-cache` works there directly:

```bash
python3 /opt/sm120/server/server.py --ckpt /work/ckpt3 --prefix-cache [--prefix-cache-grain 512 --prefix-cache-entries 64 \
    --prefix-cache-max-block-frac 0.5]                      # /health shows "prefix_cache": hits, hit_tokens, entries ...
SM120_MOE_IMPL=torch python3 /opt/sm120/tests/test_prefix_cache.py --ckpt /work/ckpt3 --json /work/pc_test_ckpt3.json
```

In an image whose `/opt/deepinfer` was built without the patch (e.g. the dev image), build a patched copy once:

```bash
/opt/sm120/prefix_cache/build_patched_c.sh                  # about 36 s, incremental -> /work/deepinfer_pc (DST=...)
export PYTHONPATH=/opt/sm120/sm120:/work/deepinfer_pc DEEPINFER_ROOT=/work/deepinfer_pc
```

## Exactness (RTX 5090 test machine, `SM120_MOE_IMPL=torch`, eager)

`tests/test_prefix_cache.py` builds two prompts from a seeded random token stream:
- **shared:** A = P[:X] + a and B = P[:X] + b.
- **extend (multi-turn):** B = A + c.

It runs B three ways and compares the logits DeepInfer samples from (prefill last position plus every decode step) and the greedy tokens:

| run of B | what it is |
|---|---|
| nocache | plain engine, one-pass prefill |
| chunked | cache off, prefill split at the same boundaries the hit run uses |
| cached | A first (records snapshots), then B with a prefix hit |

**PASS = cached is bit-identical to chunked, and the hit lands at the expected k.**

| checkpoint | cases | result |
|---|---|---|
| `ckpt3` (3-layer test model: SWA, C2 source, C1 candidate source) | shared X = 127, 128, 129, 255, 256, 257, 300, 383, 384, 385, 640; extend LA = 128, 200, 333, 511 | **15/15 bit-identical**. Hits at k = 128, 256, 384, 640 (block boundaries), 200 (mid-block: tail-block copy), 333 and 511 (odd: compressor partial group in the state); X = 127 correctly misses. SWA boundaries (128, 256, 384) are crossed both inside the prefix and inside the recomputed suffix |
| `ckpt_full` (full-depth test model: 40 layers, 16/384 routed experts, engram on 1 and 14, MTP heads present, nextn 0) | X = 129, 257, 300, 640; LA = 200, 333 | **6/6 bit-identical** |

**Generation-end hits** (B = A + A's answer + a new turn). The answer's KV came from one-token decode steps in the cached run, but from a
prefill chunk in any non-cached run. Those are different kernels, so the comparison is rounding-level, not bitwise:

| ckpt_full, torch MoE | hit at | vs chunk-aligned baseline | vs one-pass baseline |
|---|---|---|---|
| nextn 0, LA = 200 / 333, 12 generated | k = 211 / 344 (= prompt + answer - 1) | min cos 0.9965 / 0.9960, top-1 100 %, tokens identical | 0.9952 / 0.9960, tokens identical |
| nextn 1 | k = 211 / 344 | 0.9964 / 0.9963, top-1 100 %, tokens identical | 0.9954 / 0.9951, tokens identical |
| nextn 5 | k = 211 / 344 | 0.9871 / 0.9972, top-1 100 %, tokens identical | 0.9913 / (0.47; the one-pass run itself drifts, tokens identical) |

On the chaotic 3-layer `ckpt3` the same hits are not interpretable: the rounding noise there is as large as the signal.

**With MTP on** (ckpt_full, torch MoE): prompt-prefix hits stay **bit-identical** to the chunk-aligned baseline.
- nextn 5 with prompt-end snapshots: shared X=300 (k=256) and extend LA=333 (k=333).
- nextn 1, rerun with prompt-end snapshots on: shared X=300 (k=256), extend LA=200 and LA=333 (k=200, 333), 5/5 including the gen-end cases.
Prompt-end snapshots are therefore on in every mode.

**Chunked vs one-pass prefill.** Splitting a prefill into chunks changes the rounding: the GEMM and attention problem sizes differ, and
`DEEPINFER_BATCH_INVARIANT=1` does not remove it. So a hit equals a miss *with the same chunking*, not necessarily the one-pass engine.
- `ckpt3`: the 3-layer model is chaotic (ref-vs-ref top-1 is about 0.7), and the logits differ by up to 10.7.
- `ckpt_full`: min cos 0.9895-0.9967, top-1 100%, identical tokens. That is ordinary rounding.

Within a server running with `--prefix-cache`, every request is split at the same grain multiples, hit or not. A request's output
therefore does not depend on what is in the cache.

## TTFT

| setup | plain engine (no cache, one pass) | cache miss (chunked at grain) | cache hit |
|---|---|---|---|
| ckpt3, 1920-token cached prefix + 30 new, grain 128 | 0.84 s | 5.71 s | **0.24 s** (3.5x faster than plain) |
| ckpt3, same, grain 512 (hit at 1536) | 0.85 s | 2.24 s | 0.58 s (1.5x) |
| ckpt_full, 1024 + 30, grain 128 | 3.49 s | 8.79 s | **1.52 s** (2.3x) |
| server, ckpt3, 1503-token prompt repeated (grain 512, hit at 1024) | | 1.82 s (first request) | 0.56 s |

Each forced chunk is a separate prefill step, and eager prefill steps are expensive on SM120 (torch prefill attention / indexer paths). So a
small grain makes misses slower. The server therefore defaults to grain 512 plus prompt-end snapshots. The multi-turn case, where the next
prompt starts with the previous whole prompt, always hits at the previous prompt end, whatever the grain.

## Limits and open points

- A generation-end hit equals the non-cached result only to rounding, not bit for bit (decode-computed vs prefill-computed KV of the answer, see above).
- A prompt that is a strict prefix-hit of exactly `len - 1` tokens uses the next lower snapshot (`k ≤ len - 2`).
- Eviction is driven by the cache's own limits (entries, pinned-block fraction, free state slots). The cache does not watch the scheduler's queue. If many long requests wait while the cache holds 50% of the blocks, they wait for running requests to finish rather than evicting. `--prefix-cache-max-block-frac` bounds this.
- Data parallel / multiple engines: one cache per engine, nothing shared.
