"""Paired analysis for the masking experiment. No third-party dependencies.

  python experiments/analyze.py experiments/logs

Reads every run directory, pairs baseline vs. masked runs by seed, and reports the
paired mean difference with an exact permutation p-value (sign-flip test -- the right
null for a paired design, and it does not assume normality at n=5).

Reported in the repo's own format so the numbers drop straight into a PR table:
    loss  3.27836 +/- 0.00208 (n=5)

Convention: a NEGATIVE mean difference means the masked arm reached a LOWER (better)
validation loss than its paired baseline.
"""
import re
import sys
from itertools import product
from math import ceil, exp, lgamma, pi
from pathlib import Path
from statistics import mean, stdev

TARGET = 3.28
# Rough exchange rate between val loss and training steps near the end of the run, from the
# 1xH100 baselines upstream published for record #88 (records/track_1_short/
# 2026-07-13_PrefixTokenPrediction): 1375 steps -> 3.28091, 1390 steps -> 3.27894.
LOSS_PER_STEP = (3.28091 - 3.27894) / 15

VAL_RE = re.compile(r"step:(\d+)/(\d+) val_loss:([0-9.]+)")
TIME_RE = re.compile(r"train_time:([0-9.]+)ms")


def read_run(d: Path):
    log = d / "train.log"
    if not log.exists():
        return None
    text = log.read_text()
    vals = VAL_RE.findall(text)
    # A run only counts once it has logged val loss at its last step. Without this check a
    # crashed or preempted run would have its last intermediate val loss read as the result.
    if not vals or vals[-1][0] != vals[-1][1]:
        print(f"skipping {d.name}: no final val_loss (crashed, preempted, or still running)")
        return None
    curve = {int(step): float(loss) for step, _, loss in vals}
    cfg = dict(
        re.findall(r"^(\w+):\s+(.*)$", (d / "config.txt").read_text(), re.M)
    ) if (d / "config.txt").exists() else {}
    times = TIME_RE.findall(text)
    return dict(
        name=d.name,
        arm=cfg.get("arm", "masked" if d.name.startswith("masked") else "baseline"),
        seed=cfg.get("seed", "?").split()[0],
        p=cfg.get("p", "?"),
        steps=cfg.get("steps", "default"),
        rep=cfg.get("rep", "1"),
        loss=float(vals[-1][2]),
        curve=curve,
        time_ms=float(times[-1]) if times else None,
    )


def t_cdf(t, df):
    """P(T <= t) for Student's t, by Simpson's rule on the density (keeps this file scipy-free)."""
    c = exp(lgamma((df + 1) / 2) - lgamma(df / 2)) / (df * pi) ** 0.5
    pdf = lambda x: c * (1 + x * x / df) ** (-(df + 1) / 2)
    n, h = 4000, abs(t) / 4000
    area = h / 3 * (pdf(0) + pdf(abs(t)) + sum((4 if i % 2 else 2) * pdf(i * h) for i in range(1, n)))
    return 0.5 - area if t < 0 else 0.5 + area


def summarize(label, xs):
    s = stdev(xs) if len(xs) > 1 else 0.0
    line = f"{label:<36} {mean(xs):.5f} +/- {s:.5f} (n={len(xs)})"
    # The leaderboard's own bar (README rule 2): one-sided one-sample t-test that the mean
    # val loss is <= 3.28, at p < 0.01. Same statistic as scipy.stats.ttest_1samp(xs, 3.28,
    # alternative="less"), which is what the record PRs report.
    if len(xs) > 1 and s > 0:
        p = t_cdf((mean(xs) - TARGET) / (s / len(xs) ** 0.5), len(xs) - 1)
        line += f"   p(mean<={TARGET})={p:.4f}" + ("  <- clears the record bar" if p < 0.01 else "")
    return line


def merge(reps):
    """Collapse same-seed replicates of one arm into a single run by averaging, so that a
    seed still counts once in the paired test however many times it was rerun."""
    if len(reps) == 1:
        return reps[0]
    shared = set.intersection(*(set(r["curve"]) for r in reps))
    times = [r["time_ms"] for r in reps if r["time_ms"]]
    return dict(reps[0], loss=mean(r["loss"] for r in reps), time_ms=mean(times) if times else None,
                curve={k: mean(r["curve"][k] for r in reps) for k in shared})


def noise_floor(groups):
    """Pooled SD of same-seed, same-arm replicates: the noise pairing cannot cancel."""
    reps = {k: v for k, v in groups.items() if len(v) > 1}
    if not reps:
        print("no same-seed replicates yet, so the paired noise floor is unmeasured.\n"
              "  run.sh --arm baseline --seed S --rep 2   (before spending the budget on screening)\n")
        return
    ss = sum((r["loss"] - mean(x["loss"] for x in v)) ** 2 for v in reps.values() for r in v)
    n, g = sum(len(v) for v in reps.values()), len(reps)
    sd = (ss / (n - g)) ** 0.5
    print(f"replicate spread (same seed, same arm): pooled SD {sd:.5f}  ({n} runs in {g} group(s))")
    for (label, seed), v in sorted(reps.items()):
        print(f"    {label} seed {seed}: " + ", ".join(f"{r['loss']:.5f}" for r in v))
    sd_pair = sd * 2 ** 0.5
    print(f"  => one paired difference carries ~{sd_pair:.5f} of noise that pairing cannot remove.\n"
          f"     Published unseeded spread is ~0.002: if this is not clearly smaller, seeding is\n"
          f"     buying little and the plan needs more pairs, not more arms.")
    # Normal-approximation sample size for 80% power at two-sided alpha 0.05. It ignores any
    # seed-by-treatment interaction, so it is a lower bound; 6 is the sign-flip test's own floor.
    need = lambda delta: max(6, ceil((2.8 * sd_pair / delta) ** 2))
    print("     pairs needed to detect a loss difference of "
          + ", ".join(f"{d} (~{d / LOSS_PER_STEP:.0f} steps): {need(d)}" for d in (0.001, 0.002, 0.003)) + "\n")


def permutation_p(diffs):
    """Exact two-sided sign-flip test: under H0 each paired difference is equally
    likely to have either sign. Exact while 2**n is small; that covers n <= 20."""
    n = len(diffs)
    if n == 0:
        return float("nan")
    observed = abs(mean(diffs))
    if n > 20:
        return float("nan")
    hits = sum(
        1 for signs in product((1, -1), repeat=n)
        if abs(mean(s * d for s, d in zip(signs, diffs))) >= observed - 1e-15
    )
    return hits / 2 ** n


def main(root):
    runs = [r for r in (read_run(d) for d in sorted(Path(root).iterdir()) if d.is_dir()) if r]
    if not runs:
        sys.exit(f"no completed runs found under {root}")

    # The reference is the full-length baseline. Everything else is an arm, and a --steps
    # override is part of the arm's identity: a 1200-step run must not overwrite, or be
    # averaged with, the full-length run at the same p and seed.
    groups = {}
    for r in runs:
        label = ("baseline" if r["arm"] == "baseline" else f"masked p={r['p']}") \
            + ("" if r["steps"] == "default" else f" steps={r['steps']}")
        groups.setdefault((label, r["seed"]), []).append(r)
    base, treat = {}, {}
    for (label, seed), reps in groups.items():
        (base if label == "baseline" else treat.setdefault(label, {}))[seed] = merge(reps)

    print(f"{len(runs)} run(s)\n")
    noise_floor(groups)
    if base:
        print(summarize("baseline", [r["loss"] for r in base.values()]))

    for label, arm in sorted(treat.items()):
        print(summarize(label, [r["loss"] for r in arm.values()]))
        seeds = sorted(set(base) & set(arm))
        unpaired = sorted(set(arm) - set(base))
        if not seeds:
            print(f"  no seed overlap with baseline -- cannot pair (arm seeds: {sorted(arm)})\n")
            continue
        diffs = [arm[s]["loss"] - base[s]["loss"] for s in seeds]
        print(f"  paired on {len(seeds)} seed(s): {', '.join(seeds)}")
        for s, d in zip(seeds, diffs):
            print(f"    seed {s}: {base[s]['loss']:.5f} -> {arm[s]['loss']:.5f}  ({d:+.5f})")
        sd = stdev(diffs) if len(diffs) > 1 else 0.0
        floor = 2 / 2 ** len(diffs)
        print(f"  mean diff {mean(diffs):+.5f} +/- {sd:.5f}   p={permutation_p(diffs):.4f}"
              f"  (floor {floor:.4f} at n={len(diffs)})")
        if all(arm[s]["steps"] == "default" for s in seeds):
            print(f"  worth roughly {-mean(diffs) / LOSS_PER_STEP:+.0f} steps at ~{LOSS_PER_STEP:.5f} loss/step "
                  f"(upstream's 1375-vs-1390 baselines; use it to pick --steps, not as a result)")
        # Same GPU type only. 1xH100 times are not leaderboard times, but the RATIO says what
        # masking costs per step, which a step-count win has to beat to be a wall-clock win.
        ts = [(arm[s]["time_ms"], base[s]["time_ms"]) for s in seeds
              if arm[s]["time_ms"] and base[s]["time_ms"] and arm[s]["steps"] == "default"]
        if ts:
            print(f"  train_time vs baseline: {mean(a / b - 1 for a, b in ts):+.2%}")
        if floor > 0.05:
            print(f"  NOTE: a sign-flip test on {len(diffs)} pairs cannot go below p={floor:.4f}, "
                  f"however large the effect. n>=6 to clear 0.05, n>=8 for headroom.")
        # Every run already logs val loss every 250 steps; pairing those too shows WHEN the
        # arms separate (behind early and catching up, or ahead early and fading) at no
        # extra GPU cost. Only steps shared by every paired run are comparable, and only
        # full-length arms: under a --steps override, step N sits at a different point of
        # the LR schedule than step N of the baseline.
        shared = set.intersection(*(set(arm[s]["curve"]) & set(base[s]["curve"]) for s in seeds))
        if len(shared) > 1 and all(arm[s]["steps"] == "default" for s in seeds):
            # In upstream's logs the run-to-run SD at steps 250 and 500 is 0.008 to 0.03, five to
            # twenty times the final SD, and those checkpoints correlate with the final loss at
            # only r = -0.2 to +0.5 (0.8+ from step 1000). Early rows describe; they do not predict.
            print("  paired diff by checkpoint (rows before step 1000 are mostly noise; do not decide on them):")
            for step in sorted(shared):
                ds = [arm[s]["curve"][step] - base[s]["curve"][step] for s in seeds]
                spread = f" +/- {stdev(ds):.5f}" if len(ds) > 1 else ""
                print(f"    step {step:>5}: {mean(ds):+.5f}{spread}")
        if unpaired:
            print(f"  ignored (no baseline at that seed): {', '.join(unpaired)}")
        if len(diffs) < 5:
            print(f"  WARNING: n={len(diffs)} is too small to conclude anything. Published "
                  f"baseline spread is ~0.002; keep going.")
        print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "experiments/logs")
