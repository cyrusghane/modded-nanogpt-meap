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
from pathlib import Path
from statistics import mean, stdev

VAL_RE = re.compile(r"val_loss:([0-9.]+)")
TIME_RE = re.compile(r"train_time:([0-9.]+)ms")


def read_run(d: Path):
    log = d / "train.log"
    if not log.exists():
        return None
    text = log.read_text()
    losses = VAL_RE.findall(text)
    if not losses:
        return None
    cfg = dict(
        re.findall(r"^(\w+):\s+(.*)$", (d / "config.txt").read_text(), re.M)
    ) if (d / "config.txt").exists() else {}
    times = TIME_RE.findall(text)
    return dict(
        name=d.name,
        arm=cfg.get("arm", "masked" if d.name.startswith("masked") else "baseline"),
        seed=cfg.get("seed", "?").split()[0],
        p=cfg.get("p", "?"),
        loss=float(losses[-1]),
        time_ms=float(times[-1]) if times else None,
    )


def summarize(label, xs):
    s = stdev(xs) if len(xs) > 1 else 0.0
    return f"{label:<28} {mean(xs):.5f} +/- {s:.5f} (n={len(xs)})"


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

    base = {r["seed"]: r for r in runs if r["arm"] == "baseline"}
    treat = {}
    for r in runs:
        if r["arm"] != "baseline":
            treat.setdefault(r["p"], {})[r["seed"]] = r

    print(f"{len(runs)} run(s)\n")
    if base:
        print(summarize("baseline", [r["loss"] for r in base.values()]))

    for p, arm in sorted(treat.items()):
        print(summarize(f"masked p={p}", [r["loss"] for r in arm.values()]))
        seeds = sorted(set(base) & set(arm))
        unpaired = sorted(set(arm) - set(base))
        if not seeds:
            print(f"  no seed overlap with baseline -- cannot pair (masked seeds: {sorted(arm)})\n")
            continue
        diffs = [arm[s]["loss"] - base[s]["loss"] for s in seeds]
        print(f"  paired on {len(seeds)} seed(s): {', '.join(seeds)}")
        for s, d in zip(seeds, diffs):
            print(f"    seed {s}: {base[s]['loss']:.5f} -> {arm[s]['loss']:.5f}  ({d:+.5f})")
        sd = stdev(diffs) if len(diffs) > 1 else 0.0
        floor = 2 / 2 ** len(diffs)
        print(f"  mean diff {mean(diffs):+.5f} +/- {sd:.5f}   p={permutation_p(diffs):.4f}"
              f"  (floor {floor:.4f} at n={len(diffs)})")
        if floor > 0.05:
            print(f"  NOTE: a sign-flip test on {len(diffs)} pairs cannot go below p={floor:.4f}, "
                  f"however large the effect. n>=6 to clear 0.05, n>=8 for headroom.")
        if unpaired:
            print(f"  ignored (no baseline at that seed): {', '.join(unpaired)}")
        if len(diffs) < 5:
            print(f"  WARNING: n={len(diffs)} is too small to conclude anything. Published "
                  f"baseline spread is ~0.002; keep going.")
        print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "experiments/logs")
