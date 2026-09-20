"""CPU tests for the document-local copy mixture added to train_gpt.py.

train_gpt.py cannot be imported off-GPU (CUDA + torch.distributed at module scope), so
these tests exec the copy-mixture block straight out of the source file, the same way
test_masking.py does. The matcher is checked against a brute-force search that shares no
code with it.

Run:  python test_copy_mix.py
"""
import os
import re
import sys
import torch
from torch import Tensor

SRC = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_gpt.py")).read()
BLOCK = re.search(r"^# Document-local copy mixture.*?(?=^# -{20,}\n# Training Management)", SRC, re.M | re.S)
assert BLOCK, "copy-mixture block not found in train_gpt.py"
BOS = 50256

def load(env=None, dists=None):
    ns = {"torch": torch, "Tensor": Tensor, "os": type("o", (), {"environ": env or {}})(), "BOS_ID": BOS,
          "dist": torch.distributed}
    exec(BLOCK.group(0), ns)
    if dists is not None:  # small streams need small distance bins to reach every bucket
        ns["COPY_DISTS"] = dists
    return ns

FAILURES = []

def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name} {detail}")
        FAILURES.append(name)

def brute(x, splits, dists):
    """Longest ladder-length suffix match, most recent occurrence, same document. O(N^2)."""
    cand, bucket, src = [], [], []
    start = 0
    for t in range(len(x)):
        if x[t] == BOS:
            start = t
        best = None
        for k in reversed(range(len(splits))):
            L = splits[k][0]
            if t - L + 1 < start:
                continue
            gram = x[t - L + 1:t + 1]
            for s in range(t - 1, start + L - 2, -1):
                if x[s - L + 1:s + 1] == gram:
                    best = (k, s)
                    break
            if best:
                break
        if best is None:
            cand.append(None); bucket.append(0); src.append(-1)
        else:
            k, s = best
            d = sum(t - s > b for b in dists)
            cand.append(x[s + 1]); bucket.append(1 + k * (len(dists) + 1) + d); src.append(s)
    return cand, bucket, src

def stream(n, vocab, mean_doc, seed, open_mid_doc):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, vocab, (n,), generator=g)
    x[torch.rand(n, generator=g) < 1 / mean_doc] = BOS
    x[0] = 7 if open_mid_doc else BOS
    return x.to(torch.int32)

print("matcher agrees with brute force")
DISTS = (5, 40)
ns = load(dists=DISTS)
for seed, vocab, mean_doc, mid in ((1, 3, 150, True), (2, 2, 400, False), (3, 6, 60, True), (4, 2, 1200, True)):
    x = stream(1200, vocab, mean_doc, seed, mid)
    cand, bucket = ns["copy_candidates"](x)
    b_cand, b_bucket, _ = brute(x.tolist(), ns["COPY_SPLITS"], DISTS)
    check(f"buckets match (vocab {vocab}, docs ~{mean_doc}, opens mid-document: {mid})", bucket.tolist() == b_bucket,
          f"{sum(a != b for a, b in zip(bucket.tolist(), b_bucket))} differ")
    m = bucket > 0
    check("  candidates match wherever there is a match",
          cand[m].tolist() == [c for c in b_cand if c is not None])
    check("  every bucket id is in range", int(bucket.max()) < ns["COPY_NUM_BUCKETS"] and int(bucket.min()) == 0)
# Random streams never repeat 32 tokens, so plant repeats at a near, a middle and a far distance.
g = torch.Generator().manual_seed(9)
seg = lambda n: torch.randint(100, 40000, (n,), generator=g).tolist()
s1, s2 = seg(35), seg(35)
planted = [BOS] + [1, 2] * 30 + seg(10) + s1 + s1 + seg(10) + s2 + seg(100) + s2 + [BOS] + s2
x = torch.tensor(planted, dtype=torch.int32)
cand, bucket = ns["copy_candidates"](x)
b_cand, b_bucket, _ = brute(planted, ns["COPY_SPLITS"], DISTS)
check("planted long repeats match brute force", bucket.tolist() == b_bucket and
      cand[bucket > 0].tolist() == [c for c in b_cand if c is not None])
check("the longest ladder length (32) is reached in all three distance bins",
      {int(b) for b in bucket.unique()} >= {1 + 9 * 3, 1 + 9 * 3 + 1, 1 + 9 * 3 + 2}, str(sorted(bucket.unique().tolist())))

print("causal, and blind to the targets")
x = stream(1200, 3, 150, 5, True)
cand, bucket = ns["copy_candidates"](x)
ok = True
for t in (0, 1, 17, 300, 777, 1198):
    y = x.clone()
    y[t + 1:] = torch.randint(0, 3, (len(y) - t - 1,), generator=torch.Generator().manual_seed(t)).to(torch.int32)
    c2, b2 = ns["copy_candidates"](y)
    ok &= bool((c2[:t + 1] == cand[:t + 1]).all() and (b2[:t + 1] == bucket[:t + 1]).all())
check("rewriting every input after t leaves positions <= t unchanged", ok)
_, _, src = brute(x.tolist(), ns["COPY_SPLITS"], DISTS)
check("the copied token always sits at or before the current position",
      all(s + 1 <= t for t, s in enumerate(src) if s >= 0))
check("copy_candidates takes the inputs and nothing else",
      ns["copy_candidates"].__code__.co_varnames[:ns["copy_candidates"].__code__.co_argcount] == ("inputs",))
check("a BOS is never proposed (it never follows a match inside a document)", bool((cand[bucket > 0] != BOS).all()))

print("never matches across a document boundary")
a = [11, 12, 13, 14, 15, 16]
x = torch.tensor([BOS] + a + [BOS] + a + [BOS] + a[:3], dtype=torch.int32)
_, bucket = ns["copy_candidates"](x)
check("a repeat of the previous document finds no match", int(bucket.sum()) == 0, str(bucket.tolist()))
x = torch.tensor(a + [BOS] + a, dtype=torch.int32)
_, bucket = ns["copy_candidates"](x)
check("the fragment that opens a chunk is a document of its own", int(bucket.sum()) == 0, str(bucket.tolist()))
x = torch.tensor([BOS] + a + a + [BOS, 99], dtype=torch.int32)
cand, bucket = ns["copy_candidates"](x)
# Lengths 1,2,3,4,5,6 land on ladder rungs 1,2,3,4,4,6; the source is 6 back, the middle bin of (5, 40).
check("the same repeat inside one document does match", bucket[7:13].tolist() == [1 + 3 * k + 1 for k in (0, 1, 2, 3, 3, 4)] and
      cand[7:13].tolist() == a[1:] + a[:1], f"{bucket.tolist()} {cand.tolist()}")
check("and the match does not leak into the next document", bucket[13:].tolist() == [0, 0])

print("production constants")
real = load()
check("on by default, and COPY_MIX=0 turns it off", real["COPY_MIX"] is True and load({"COPY_MIX": "0"})["COPY_MIX"] is False)
check("distance bins sit at the final eval windows (6 and 20 blocks of 128)", real["COPY_DISTS"] == (6 * 128, 20 * 128))
ladder = [s[0] for s in real["COPY_SPLITS"]]
check("every ladder length is the sum of two earlier ones",
      all(L == 1 or (l + r == L and l in ladder[:i] and r in ladder[:i]) for i, (L, l, r) in enumerate(real["COPY_SPLITS"])))
x = stream(262144, 50000, 800, 6, True)
x[100000:100900] = x[50000:50900]  # planted repeats, one of them across a BOS
cand, bucket = real["copy_candidates"](x)
check("runs at the validation chunk size with int64 keys intact", bucket.shape == (262144,) and int(bucket.max()) > 0)

print("the mixture is a probability distribution")
V = 40
g = torch.Generator().manual_seed(21)
logp = torch.log_softmax(3 * torch.randn(64, V, generator=g, dtype=torch.float64), -1)  # 64 positions, full model distributions
cand = torch.randint(0, V, (64,), generator=g)
for lam_value in (0.0, 0.3, 0.98):
    lam = torch.full((64,), lam_value, dtype=torch.float64)
    # Score every possible target v at every position through the shipped function, then add up the probabilities.
    total = sum(torch.exp(-real["copy_mix_loss"](-logp[:, v], cand == v, lam)) for v in range(V))
    check(f"sums to one over the vocabulary at lam={lam_value}", bool(torch.allclose(total, torch.ones(64, dtype=torch.float64), atol=1e-12)),
          f"max error {float((total - 1).abs().max()):.2e}")
loss = -logp[torch.arange(64), torch.randint(0, V, (64,), generator=g)]
hit = torch.rand(64, generator=g) < 0.5
lam = torch.rand(64, generator=g, dtype=torch.float64) * 0.9
direct = -torch.log((1 - lam) * torch.exp(-loss) + lam * hit)
check("matches -log((1-lam) p + lam [target == c]) computed directly",
      bool(torch.allclose(real["copy_mix_loss"](loss, hit, lam), direct, atol=1e-12)))

print("a weight of zero reproduces the standard loss exactly")
for dtype in (torch.float32, torch.float64):
    loss = (torch.rand(100_000, generator=g) * 12).to(dtype)
    loss[:3] = torch.tensor([0.0, 1e-7, 34.0], dtype=dtype)
    hit = torch.rand(100_000, generator=g) < 0.3
    out = real["copy_mix_loss"](loss, hit, torch.zeros(100_000, dtype=dtype))
    check(f"bit-identical per token, hits included ({str(dtype).split('.')[1]})", bool(torch.equal(out, loss)))
    check(f"and so is the mean ({str(dtype).split('.')[1]})", bool(torch.equal(out.mean(), loss.mean())))

print("weight fit")
NB = real["COPY_NUM_BUCKETS"]
n = 400_000
bucket = torch.randint(0, NB, (n,), generator=g)
# A model that is good where the copy rule is bad and the reverse, bucket by bucket; buckets 1-3 are ones copying cannot help.
hit_rate = torch.linspace(0.05, 0.95, NB)
model_p_on_hit = torch.linspace(0.9, 0.05, NB)
model_p_on_hit[1:4] = 0.99
hit = torch.rand(n, generator=g) < hit_rate[bucket]
p = torch.where(hit, model_p_on_hit[bucket], torch.tensor(0.2)) * (0.5 + torch.rand(n, generator=g))
loss = -torch.log(p.clamp(max=0.999)).float()
lam = real["copy_fit"](loss, hit, bucket)
nll = lambda lam_b, b: float(real["copy_mix_loss"](loss[bucket == b].double(), hit[bucket == b], torch.full((int((bucket == b).sum()),), lam_b, dtype=torch.float64)).sum())
grid = torch.linspace(0, 0.98, 491).tolist()
worst = max(abs(float(lam[b]) - min(grid, key=lambda v: nll(v, b))) for b in range(1, NB, 3))
check("agrees with a grid search of the likelihood, to the grid's resolution", worst <= 0.002 + 1e-9, f"worst gap {worst:.4f}")
check("buckets the copy rule cannot help get exactly 0", all(float(lam[b]) == 0.0 for b in (1, 2, 3)), str(lam[:4].tolist()))
check("the no-match bucket is always 0", float(lam[0]) == 0.0)
check("float64, one weight per bucket, inside [0, 0.98]", lam.dtype == torch.float64 and lam.shape == (NB,) and
      bool(((lam >= 0) & (lam <= 0.98)).all()))
fitted = float(real["copy_mix_loss"](loss.double(), hit, lam[bucket]).mean())
check("the fitted mixture is never worse than the model on the tokens it was fitted on", fitted <= float(loss.double().mean()),
      f"{fitted:.5f} vs {float(loss.double().mean()):.5f}")
few = torch.tensor([7, 7, 7])
lam_few = real["copy_fit"](torch.tensor([3.0, 3.0, 3.0]), torch.tensor([True, True, True]), few)
lam_many = real["copy_fit"](torch.full((3000,), 3.0), torch.ones(3000, dtype=torch.bool), few.repeat(1000))
check("three lucky hits are shrunk toward 0; three thousand are not", 0 < float(lam_few[7]) < 0.4 and float(lam_many[7]) > 0.95,
      f"{float(lam_few[7]):.3f} {float(lam_many[7]):.3f}")

print("the fit runs on training tokens, in eval mode, without gradients")
class FakeModel:
    """Per-token loss from a fixed unigram table. Records how it was called."""
    def __init__(self):
        self.mode, self.calls = "train", []
        self.table = torch.rand(50304, generator=torch.Generator().manual_seed(3)) * 8
    def eval(self): self.mode = "eval"
    def train(self): self.mode = "train"
    def __call__(self, inputs, targets, cum_seqlens, bigram_inputs, forward_args):
        self.calls.append((self.mode, torch.is_grad_enabled(), forward_args, inputs.numel()))
        return self.table[targets]
opened = []
def fake_loader(pattern, num_tokens, max_seq_len, grad_accum_steps=1, align_to_bos=True, apply_mask=False):
    opened.append((pattern, num_tokens, max_seq_len, grad_accum_steps, align_to_bos, apply_mask))
    gg = torch.Generator().manual_seed(5)
    while True:
        x = torch.randint(0, 300, (num_tokens // grad_accum_steps + 1,), generator=gg)
        x[::211] = BOS
        yield x[:-1].int(), x[1:], torch.zeros(4, dtype=torch.int32), x[:-1].int(), None
import glob as real_glob, time as real_time, tempfile
tmp = tempfile.mkdtemp()
for name in ("fineweb_train_000001.bin", "fineweb_train_000002.bin", "fineweb_train_000009.bin", "fineweb_val_000000.bin"):
    open(os.path.join(tmp, name), "w").close()
fit_ns = load()
fit_ns.update(glob=real_glob, time=real_time, distributed_data_generator=fake_loader, grad_accum_steps=8,
              args=type("a", (), {"train_files": os.path.join(tmp, "fineweb_train_*.bin"), "val_files": os.path.join(tmp, "fineweb_val_*.bin"),
                                  "val_batch_size": 8 * 4096})())
torch.cuda.synchronize = lambda *a, **k: None  # no CUDA on this machine; the call only closes the timing window
fake = FakeModel()
lam, ms = fit_ns["copy_fit_on_train"](fake, "FORWARD_ARGS")
check("reads the last train shard and never the validation files",
      len(opened) == 1 and opened[0][0].endswith("fineweb_train_000009.bin"), str(opened))
check("batches are cut like validation batches (unaligned, full length, unmasked)", opened[0][1:] == (8 * 4096, -1, 8, False, False))
check("one forward per accumulation step, in eval mode, under no_grad, with the caller's forward args",
      fake.calls == [("eval", False, "FORWARD_ARGS", 4096)] * 8, str(fake.calls[:2]))
check("the model is handed back in train mode", fake.mode == "train")
check("returns one weight per bucket and the time it took", lam.shape == (NB,) and lam.dtype == torch.float64 and ms >= 0)
fit_ns["copy_fit_on_train"](fake, "FORWARD_ARGS")
check("a second fit reuses the cached tokens", len(opened) == 1 and len(fake.calls) == 16)

print("validate(): the model's loss is untouched, the mixture rides on the same forwards")
VALIDATE = re.search(r"^def validate\(.*?^    return val_loss, \(copy_loss if copy_lam is not None else None\)\n", SRC, re.M | re.S)
assert VALIDATE, "validate() not found in train_gpt.py"
val_ns = load()
val_ns.update(time=real_time, distributed_data_generator=fake_loader, grad_accum_steps=8,
              model=FakeModel(), training_manager=type("tm", (), {"get_forward_args": lambda self: "ARGS"})(),
              args=type("a", (), {"val_files": "VAL", "val_tokens": 5 * 8 * 4096, "val_batch_size": 8 * 4096})(),
              dist=type("d", (), {"reduce": staticmethod(lambda *a, **k: None), "all_reduce": staticmethod(lambda *a, **k: None),
                                  "is_initialized": staticmethod(lambda: False), "ReduceOp": torch.distributed.ReduceOp})())
exec(VALIDATE.group(0), val_ns)
plain, nothing = val_ns["validate"]()
with_mix, mix_loss = val_ns["validate"](lam)
zero_base, zero_mix = val_ns["validate"](torch.zeros(NB, dtype=torch.float64))
check("without weights there is no mixture loss", nothing is None)
check("the model's val loss is bit-identical with the mixture on", bool(torch.equal(plain, with_mix)))
check("the mixture loss is a finite scalar, scored over all 40 forwards",
      mix_loss.dim() == 0 and bool(torch.isfinite(mix_loss)) and len(val_ns["model"].calls) == 3 * 40)
check("weights of zero give the model's loss back", abs(float(zero_mix) - float(zero_base)) < 1e-6,
      f"{float(zero_mix):.8f} vs {float(zero_base):.8f}")
check("validation leaves the model in train mode and never refits the weights",
      val_ns["model"].mode == "train" and "copy_fit" not in VALIDATE.group(0))

print("where the fit sits in the training loop")
loop = SRC.split("# --------------- VALIDATION SECTION -----------------", 1)[1].split("# --------------- TRAINING SECTION", 1)[0]
i_ext, i_mask, i_fit, i_stop, i_val = (loop.find(s) for s in ("apply_final_ws_ext()", "canon_mask_builder.collect(",
                                                              "copy_fit_on_train(", "# stop the clock", "validate(copy_lam"))
check("final windows, then the canonical mask, then the fit, then the clock stops, then validation",
      -1 < i_ext < i_mask < i_fit < i_stop < i_val, f"{i_ext} {i_mask} {i_fit} {i_stop} {i_val}")
check("the clock is read after the fit", loop.find("training_time_ms += 1000 * (time.perf_counter() - t0)") > i_fit)
check("with COPY_MIX off no weights exist, so validate() scores the model alone; on, only the final validation is fitted",
      "copy_lam = None\n        if COPY_MIX and last_step:" in loop)
check("the mixture's loss is what the run reports as val_loss, in the record's own line format, with the model's own on a second line",
      loop.find("val_loss, val_loss_no_copy = copy_loss, val_loss") < loop.find('print0(f"step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:')
      < loop.find("val_loss_no_copy:{val_loss_no_copy:.6f}") and loop.count("val_loss:{val_loss:.4f}") == 1)
block = BLOCK.group(0)
check("the block never touches the validation files and never runs a backward pass",
      "val_files" not in block and ".backward(" not in block and "requires_grad" not in block)
check("the fit tokens are not masked", "apply_mask" not in block)
warm = SRC.split("Warmup kernels", 1)[1].split('print0("Resetting Model"', 1)[0]
check("the final-window eval graph is compiled during warmup, and the warmup's tokens are dropped",
      warm.find("apply_final_ws_ext()") < warm.find("copy_fit_on_train(") < warm.find("_copy_fit_batches.clear()") and "if COPY_MIX:" in warm)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
    sys.exit(1)
print("all copy-mixture tests passed")
