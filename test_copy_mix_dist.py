"""Multi-process CPU test for the one part of the copy mixture that only does work with more than one rank:
copy_fit() all-reduces its sums so that every rank ends with the same weights.

Runs copy_fit() out of train_gpt.py in N real processes over torch.distributed (gloo), each holding its own
slice of the fit tokens, the way each GPU holds its own slice of the train shard. Every rank must end with
identical weights, and they must equal a single-process fit on all the tokens.

Not covered here, because gloo cannot do it: ReduceOp.AVG (NCCL only), which validate() uses for the
mixture loss exactly as upstream already uses it for val_loss.

Run:  python test_copy_mix_dist.py
"""
import os
import re
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor

HERE = os.path.dirname(os.path.abspath(__file__))


def load_block():
    src = open(os.path.join(HERE, "train_gpt.py")).read()
    block = re.search(r"^# Document-local copy mixture.*?(?=^# -{20,}\n# Training Management)", src, re.M | re.S).group(0)
    ns = {"torch": torch, "Tensor": Tensor, "os": type("o", (), {"environ": {}})(), "BOS_ID": 50256, "dist": dist}
    exec(block, ns)
    return ns


def fit_data(n=400_000, seed=11):
    """Tokens whose hit rate and model probability vary by bucket, so every bucket gets a different weight."""
    ns = load_block()
    nb = ns["COPY_NUM_BUCKETS"]
    g = torch.Generator().manual_seed(seed)
    bucket = torch.randint(0, nb, (n,), generator=g)
    hit = torch.rand(n, generator=g) < torch.linspace(0.05, 0.95, nb)[bucket]
    p = torch.where(hit, torch.linspace(0.9, 0.05, nb)[bucket], torch.tensor(0.2)) * (0.5 + torch.rand(n, generator=g))
    return -torch.log(p.clamp(max=0.999)).float(), hit, bucket


def worker(rank, world, store, out_dir):
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world)
    ns = load_block()
    loss, hit, bucket = fit_data()
    # Uneven slices on purpose: ranks never hold the same number of hits, only the same number of buckets.
    edges = [0] + [int(len(loss) * (i + 1) ** 1.3 / world ** 1.3) for i in range(world)]
    sl = slice(edges[rank], edges[rank + 1])
    lam = ns["copy_fit"](loss[sl], hit[sl], bucket[sl])
    torch.save(lam, os.path.join(out_dir, f"lam{rank}.pt"))
    dist.destroy_process_group()


if __name__ == "__main__":
    failures = []
    for world in (2, 8):
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(worker, args=(world, os.path.join(tmp, "store"), tmp), nprocs=world, join=True)
            lams = [torch.load(os.path.join(tmp, f"lam{r}.pt")) for r in range(world)]
        single = load_block()["copy_fit"](*fit_data())  # dist is not initialised in this process: no all-reduce
        same = all(torch.equal(lams[0], l) for l in lams[1:])
        gap = float((lams[0] - single).abs().max())
        for name, ok, detail in ((f"{world} ranks end with bit-identical weights", same, ""),
                                 (f"{world} ranks: weights equal a single-process fit on all the tokens", gap < 1e-9, f"max gap {gap:.2e}"),
                                 (f"{world} ranks: the fit is not trivial (weights span {float(single.min()):.2f}..{float(single.max()):.2f})",
                                  float(single.max()) > 0.5 and float(single[0]) == 0.0, "")):
            print(f"  {'PASS' if ok else 'FAIL'}  {name} {detail}")
            if not ok:
                failures.append(name)
    print()
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("all distributed copy-mixture tests passed")
