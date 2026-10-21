"""Profile the b12x prefill path (where does the time go?)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("B12X_AUTOTUNE", "0")
import torch
import test_attn_shims as T
from torch.profiler import profile, ProfilerActivity
s_q = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
c = T.build_prefill_case(seed=7, s_q=s_q, ratio=4, holes=0.0)
os.environ["SM120_ATTN_PREFILL_IMPL"] = "b12x"
os.environ["SM120_ATTN_STRICT"] = "1"
f = lambda: T.flash_mla.flash_mla_sparse_fwd(c["q"], c["kv"], c["idx"], c["sm_scale"], 512, c["sink"], None)
for _ in range(3): f()
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
    for _ in range(3): f()
    torch.cuda.synchronize()
print(p.key_averages().table(sort_by="cuda_time_total", row_limit=25, max_name_column_width=60))
