"""CPU tests for the MEAP-style input masking added to train_gpt.py.

train_gpt.py cannot be imported off-GPU (CUDA + torch.distributed at module scope), so
these tests exec the masking block straight out of the source file. That keeps the test
honest: it exercises the shipped text, not a copy of it.

Run:  python test_masking.py
"""
import os
import re
import sys
import torch
from torch import Tensor

SRC = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_gpt.py")).read()

# ---- Load the masking block out of train_gpt.py ------------------------------------
BLOCK = re.search(r"^MASK_ID = 50257$.*?^def get_bigram_hash", SRC, re.M | re.S)
assert BLOCK, "masking block not found in train_gpt.py"

def load(env):
    ns = {"torch": torch, "Tensor": Tensor, "os": type("o", (), {"environ": env})(), "BOS_ID": 50256}
    exec(BLOCK.group(0).rsplit("def get_bigram_hash", 1)[0], ns)
    return ns

FAILURES = []

def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name} {detail}")
        FAILURES.append(name)

# Representative token stream: real GPT-2 ids with BOS at document starts.
g = torch.Generator().manual_seed(7)
toks = torch.randint(0, 50256, (200_000,), generator=g, dtype=torch.int32)
toks[::997] = 50256  # BOS every ~997 tokens

print("mask ratio and value")
ns = load({"MASK_P_START": "0.15", "MASK_SEED": "3"})
out = ns["apply_token_mask"](toks, 0.15, rank=0)
maskable = toks != 50256
changed = out != toks
rate = changed[maskable].float().mean().item()
check("empirical rate matches p", abs(rate - 0.15) < 0.005, f"got {rate:.4f}")
check("all changed positions are MASK_ID", bool((out[changed] == 50257).all()))
check("unchanged positions are untouched", bool((out[~changed] == toks[~changed]).all()))
check("BOS is never masked", bool((out[toks == 50256] == 50256).all()))
check("dtype preserved", out.dtype == torch.int32, f"got {out.dtype}")
check("MASK_ID is inside the padded vocab and outside real tokens", 50256 < ns["MASK_ID"] < 50304)

print("p = 0 is exactly the baseline")
ns0 = load({})
check("p=0 returns input unchanged", bool((ns0["apply_token_mask"](toks, 0.0, rank=0) == toks).all()))
check("default env gives p=0 at every step",
      all(ns0["mask_p_for_step"](s, 1285) == 0.0 for s in (0, 400, 1285)))

print("reproducibility and rank independence")
a = load({"MASK_P_START": "0.15", "MASK_SEED": "3"})["apply_token_mask"](toks, 0.15, rank=0)
b = load({"MASK_P_START": "0.15", "MASK_SEED": "3"})["apply_token_mask"](toks, 0.15, rank=0)
c = load({"MASK_P_START": "0.15", "MASK_SEED": "3"})["apply_token_mask"](toks, 0.15, rank=1)
d = load({"MASK_P_START": "0.15", "MASK_SEED": "4"})["apply_token_mask"](toks, 0.15, rank=0)
check("same seed and rank reproduces the mask", bool((a == b).all()))
check("different rank gives a different mask", not bool((a == c).all()))
check("different MASK_SEED gives a different mask", not bool((a == d).all()))

print("the mask RNG does not disturb the global RNG")
torch.manual_seed(11)
before = torch.rand(4)
torch.manual_seed(11)
load({"MASK_P_START": "0.15"})["apply_token_mask"](toks, 0.15, rank=0)
after = torch.rand(4)
check("global RNG stream is unaffected by masking", bool(torch.equal(before, after)))

print("anneal schedule")
ann = load({"MASK_P_START": "0.15", "MASK_P_END": "0.0"})
f = ann["mask_p_for_step"]
check("starts at MASK_P_START", abs(f(0, 1285) - 0.15) < 1e-9)
check("reaches MASK_P_END at the last step", abs(f(1285, 1285) - 0.0) < 1e-9)
check("is monotone and halves at the midpoint", abs(f(642, 1285) - 0.075) < 1e-3)
check("clamps past the end", f(2000, 1285) == 0.0)
flat = load({"MASK_P_START": "0.15"})
check("MASK_P_END defaults to MASK_P_START (flat)", abs(flat["mask_p_for_step"](1285, 1285) - 0.15) < 1e-9)

print("mask is applied before every input-derived feature")
gen = SRC.split("def distributed_data_generator", 1)[1]
i_mask, i_bigram = gen.find("apply_token_mask"), gen.find("get_bigram_hash")
check("masking precedes get_bigram_hash in the generator", -1 < i_mask < i_bigram,
      f"mask@{i_mask} bigram@{i_bigram}")
check("targets are never masked", "apply_token_mask(_targets" not in SRC)
check("val loader does not mask (apply_mask defaults to False)",
      "align_to_bos=False, apply_mask=True" not in SRC and "apply_mask: bool = False" in SRC)
check("both train loaders mask", SRC.count("args.train_files") == SRC.count("apply_mask=True") == 2)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
    sys.exit(1)
print("all masking tests passed")
