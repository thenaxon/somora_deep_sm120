"""CPU unit test of prefix_cache's bookkeeping (no GPU, no DeepInfer): fake engine with block / state managers.

    python3 tests/test_prefix_cache_unit.py

Checks: a failed pin / clone gives the blocks and the state slot back; the pinned-block counter equals the set of
blocks the entries hold, shared prefix blocks count once against max_blocks; an oversize snapshot evicts nothing;
_live trimming drops the least recently scheduled requests; the one-pass lookup digests equal _digest."""
import os
import sys
from collections import Counter
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prefix_cache"))
import prefix_cache as pc  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        FAILS.append(name)


class Blocks:
    def __init__(self):
        self.pins = Counter()
        self.fail = False

    def pin_blocks(self, blocks):
        if self.fail:
            raise RuntimeError("pin failed (test)")
        self.pins.update(blocks)

    def unpin_blocks(self, blocks):
        self.pins.subtract(blocks)
        self.pins = +self.pins


class States:
    def __init__(self, n):
        self.free = list(range(n))
        self.used = set()

    def get_num_free_blocks(self):
        return len(self.free)

    def lock(self):
        pass

    def unlock(self):
        pass

    def allocate_nolock(self, n):
        out = [self.free.pop() for _ in range(n)]
        self.used.update(out)
        return out

    def unpin_blocks(self, ids):
        for i in ids:
            self.used.discard(i)
            self.free.append(i)


class Engine:
    def __init__(self, states=64):
        self.block_manager, self.state_manager = Blocks(), States(states)
        self.scheduler_config = SimpleNamespace(max_num_seqs=2)
        self.clone_fail = False

    def _run_workers(self, name, **kw):
        if self.clone_fail:
            raise RuntimeError("clone failed (test)")


def cache(max_blocks, states=64):
    c = pc.PrefixCache.__new__(pc.PrefixCache)  # no DeepInfer: set the fields __init__ would
    c.engine = Engine(states)
    c.max_blocks, c.max_entries, c.grain, c.snapshot = max_blocks, 64, pc.BLOCK, True
    c.entries = pc.OrderedDict()
    c._live, c._pin_refs, c._pending = pc.OrderedDict(), {}, []
    c.snapshot_gen_end = True
    c.stats = dict(snapshots=0, gen_end_snapshots=0, evictions=0, snapshot_skipped=0, snapshot_errors=0)
    return c


def seq(ids, blocks, state_id=0):
    return SimpleNamespace(data=SimpleNamespace(get_token_ids=lambda: list(ids)), block_table=list(blocks),
                           state_id=state_id, num_cached_tokens=len(ids))


def snap(c, ids, blocks):
    c._pending.append((seq(ids, blocks), len(ids)))
    c._take_snapshots()


def held(c):
    return {b for e in c.entries.values() for b in e.blocks}


B = pc.BLOCK
rng = np.random.default_rng(0)

# 1. failed pin / clone: no leaked pins, no leaked state slot
for what in ("pin", "clone"):
    c = cache(100)
    free0 = c.engine.state_manager.get_num_free_blocks()
    setattr(c.engine.block_manager if what == "pin" else c.engine, "fail" if what == "pin" else "clone_fail", True)
    snap(c, rng.integers(0, 1000, 2 * B), [1, 2])
    check(f"1 {what} failure releases", not c.entries and not +c.engine.block_manager.pins
          and c.engine.state_manager.get_num_free_blocks() == free0 and c.stats["snapshot_errors"] == 1
          and not c._pin_refs, f"entries={len(c.entries)} pins={dict(c.engine.block_manager.pins)} "
                               f"free={c.engine.state_manager.get_num_free_blocks()}/{free0}")

# 2. counter == set of held blocks; shared prefix blocks count once against max_blocks
c = cache(4)
base = rng.integers(0, 1000, 4 * B)
snap(c, base[: 2 * B], [10, 11])
snap(c, base[: 3 * B], [10, 11, 12])
snap(c, base[: 4 * B], [10, 11, 12, 13])  # 4 distinct blocks: fits max_blocks=4 without eviction
check("2 shared blocks count once", len(c.entries) == 3 and c.stats["evictions"] == 0 and c._pinned_blocks() == 4,
      f"entries={len(c.entries)} evictions={c.stats['evictions']} pinned={c._pinned_blocks()}")
snap(c, rng.integers(0, 1000, 2 * B), [20, 21])  # needs 2 new blocks: evicts the oldest entries until it fits
check("2 counter == held set after evictions", c._pinned_blocks() == len(held(c)) and set(c._pin_refs) == held(c)
      and c._pinned_blocks() <= 4, f"pinned={c._pinned_blocks()} held={sorted(held(c))}")
c.clear()
check("2 clear", not c._pin_refs and not c.entries and not +c.engine.block_manager.pins)

# 3. an oversize snapshot evicts nothing
c = cache(2)
snap(c, rng.integers(0, 1000, B), [1])
snap(c, rng.integers(0, 1000, 4 * B), [5, 6, 7, 8])
check("3 oversize snapshot keeps the cache", len(c.entries) == 1 and c.stats["evictions"] == 0
      and c.stats["snapshot_skipped"] == 1, f"entries={len(c.entries)} evictions={c.stats['evictions']}")

# 4. _live trimming drops the least recently scheduled
c = cache(100)
c.snapshot_prompt_end = True
outs = lambda seqs: SimpleNamespace(scheduled_seqs=seqs, prompt_run=False)  # noqa: E731
mk = lambda i: SimpleNamespace(request_id=f"r{i}")  # noqa: E731
c._on_schedule([], outs([mk(i) for i in range(4096)]))
c._on_schedule([], outs([mk(0)]))  # the longest-running request is scheduled again
c._on_schedule([], outs([mk(i) for i in range(4096, 4100)]))
check("4 _live keeps recently scheduled", "r0" in c._live and "r1" not in c._live and "r4099" in c._live
      and len(c._live) <= 4096, f"len={len(c._live)}")

# 5. one-pass digests == _digest
ok = True
for _ in range(50):
    ids = rng.integers(-5, 130000, int(rng.integers(1, 5000))).astype(np.int64)
    ks = sorted(set(int(x) for x in rng.integers(0, ids.size + 1, 10)))
    d = pc._prefix_digests(ids, ks)
    ok &= all(d[k] == pc._digest(ids, k) for k in ks)
check("5 one-pass lookup digests bit-equal", ok)
c = cache(100)
ids = rng.integers(0, 1000, 5 * B)
snap(c, ids[: 2 * B], [1, 2])
snap(c, ids[: 4 * B], [1, 2, 3, 4])
hit = c._lookup(ids)
check("5 lookup finds the longest prefix", hit is not None and hit.k == 4 * B and c._lookup(ids[: 3 * B]).k == 2 * B)

print("ALL PASS" if not FAILS else f"FAILURES: {FAILS}")
sys.exit(1 if FAILS else 0)
