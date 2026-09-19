"""Run the paired masking experiment on a single Modal H100 (fits in the $30/month free credits).

One-time setup, from the repo root:
    pip install modal && modal setup
    modal run experiments/modal_run.py --setup            # data -> Volume, CPU only, no GPU billed

Preflight (about 2 cents, on a T4): checks the image, the compiler toolchain, the compile
cache on the Volume and the flash-attn3 kernel download, before any H100 time is spent:
    modal run experiments/modal_run.py --preflight

Port check. Do this once before the paired blocks: it proves the stack runs and fills the compile
cache, so the ~7 minute torch.compile cost is paid once rather than on every run:
    modal run experiments/modal_run.py --arm baseline --seeds 1

Paired block (baseline + masked at each seed, fanned out in parallel):
    modal run experiments/modal_run.py --p 0.15 --seeds 1,2,3,4
    modal run experiments/modal_run.py --p 0.15 --p-end 0.0 --seeds 1,2,3,4      # annealed arm
    modal run experiments/modal_run.py --p 0.15 --seeds 1,2,3,4 --steps 1255      # 15 steps removed
    modal run experiments/modal_run.py --arm baseline --seeds 1,2 --rep 2         # noise floor

A closed laptop kills a plain `modal run` and the credit spent so far is wasted. Either keep
the Mac awake (`caffeinate -i modal run ...`) or use `modal run --detach ...`: every run also
saves its logs to the Volume, and any later invocation (or `--fetch` alone) pulls them down.

Each result is written to experiments/logs/<tag>/ on THIS machine, in the layout run.sh
produces, so `python experiments/analyze.py` works unchanged. A run whose train.log already
exists locally is skipped. That makes an interrupted sweep resumable, and it means the
baseline at a given seed is paid for once and reused across every p you screen.

--steps sets num_scheduled_iterations (default 1270; the run is that plus 15 extension steps).
In pair mode only the masked arm is shortened: the comparison that matters is a SHORT masked
run against the FULL baseline. Upstream's 1xH100 baselines put 15 steps at about 0.002 val loss.

Spend: every attempt, failed ones included, leaves a record under experiments/logs/_spend/.
The launcher adds up this calendar month's records and refuses a batch that would pass
--budget (default 30). It only sees runs made through this file, and only function time, so
the Modal dashboard stays the authority; set a workspace spend limit there as the hard stop.

This wraps experiments/run.sh rather than reimplementing it, so tags and config.txt cannot
drift between local and remote runs.
"""
import hashlib
import os
import shutil
import subprocess
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
VOL = "/vol"

app = modal.App("nanogpt-masking")
vol = modal.Volume.from_name("nanogpt-masking", create_if_missing=True)

# Only the four files a run needs, baked into the image (copy=True): an explicit list keeps
# the 280MB records/ directory out, and an image layer is an ordinary writable directory,
# which train_gpt.py needs for its own logs/. Editing train_gpt.py rebuilds this layer only.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential")  # Triton compiles a small C launcher stub at runtime
    .pip_install_from_requirements(str(REPO / "requirements.txt"))
    .env({
        # Compile caches live on the Volume so only the first run pays for compilation.
        # The compiled graph is identical across arms (masking happens in the CPU loader).
        "TORCHINDUCTOR_CACHE_DIR": f"{VOL}/cache/inductor",
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
        "TRITON_CACHE_DIR": f"{VOL}/cache/triton",
        "TORCHINDUCTOR_AUTOGRAD_CACHE": "1",
        "HF_HOME": f"{VOL}/cache/hf",  # the prebuilt flash-attn3 kernel from `kernels`
        # train_gpt.py calls tiktoken.get_encoding("gpt2") at import, which downloads the vocab
        # unless cached: without this every container depends on a third-party host being up
        # while the H100 is already billing.
        "TIKTOKEN_CACHE_DIR": f"{VOL}/cache/tiktoken",
        "DATA_PATH": VOL,              # train_gpt.py reads $DATA_PATH/data/fineweb10B/*.bin
    })
    .add_local_file(str(REPO / "train_gpt.py"), "/repo/train_gpt.py", copy=True)
    .add_local_file(str(REPO / "triton_kernels.py"), "/repo/triton_kernels.py", copy=True)
    .add_local_file(str(REPO / "dc_triton_kernels.py"), "/repo/dc_triton_kernels.py", copy=True)
    .add_local_file(str(REPO / "experiments" / "run.sh"), "/repo/experiments/run.sh", copy=True)
)


@app.function(image=image, volumes={VOL: vol}, timeout=2 * 60 * 60)
def download_data(num_chunks: int = 9):
    """data/cached_fineweb10B.py, pointed at the Volume. 9 chunks (900M tokens) is what the
    README uses for the current record. Runs on CPU so no GPU time is spent on downloading."""
    from huggingface_hub import hf_hub_download

    local_dir = f"{VOL}/data/fineweb10B"
    names = ["fineweb_val_%06d.bin" % 0] + ["fineweb_train_%06d.bin" % i for i in range(1, num_chunks + 1)]
    for name in names:
        if not os.path.exists(os.path.join(local_dir, name)):
            print("downloading", name)
            hf_hub_download(repo_id="kjj0/fineweb10B-gpt2", filename=name, repo_type="dataset", local_dir=local_dir)
    vol.commit()
    print(f"{len(names)} shard(s) ready in {local_dir}")


# "H100!" and not "H100": Modal may silently upgrade a plain H100 request to an H200, which
# would put different hardware under different runs of the same paired comparison.
# max_containers stays under the Volume's limit of ~5 concurrent commits.
# timeout caps a hung run at about $4 of credit. Upstream's own 1xH100 logs put training at
# ~10 min and a cold compile at ~7, so 60 leaves a slow first run room to finish, because a
# run killed at 95% costs the same as a finished one and yields nothing.
# MODAL_GPU exists for plumbing checks on a cheaper card only. Anything that is not an H100
# is a different experiment (pre-Hopper cards also need DISABLE_FP8=1), so never mix GPU
# types inside one paired comparison.
GPU = os.environ.get("MODAL_GPU", "H100!")
USD_PER_RUN = 1.50  # planning figure with headroom: ~19 min cold / ~13 min warm at USD_PER_SEC. Replace once measured.
USD_PER_SEC = 0.001097 + 4 * 0.0000131 + 16 * 0.00000222  # H100 + 4 cores + 16 GiB, modal.com/pricing 2026-09
KEEP = ("config.txt", "train.log", "train.failed.log")
RUN_LOGS = "/tmp/run_logs"  # inside the container


def _cache_key() -> str:
    """Names the compile cache this code would hit: GPU type plus the files whose contents
    decide what gets compiled. An edit to any of them is treated as a cold cache."""
    h = hashlib.sha256()
    for name in ("train_gpt.py", "triton_kernels.py", "dc_triton_kernels.py"):
        h.update((REPO / name).read_bytes())
    return f"warm-{GPU.strip('!')}-{h.hexdigest()[:12]}"


# cpu=4: Modal's default request is 0.125 cores. torch.compile is CPU-bound, and every minute
# it waits on CPU is a minute of idle H100 billed at 20x the price of the cores.
# memory=16 GiB: the default request is 128 MiB, which lets the scheduler place the run on a
# host with nothing to spare. Upstream's 1xH100 logs show 36 GB of GPU memory in use, and the
# host side holds two pinned 200 MB shards plus the compile workers. A request is not a cap.
# scaledown_window: an idle container is still billed, and the default keeps it for 60 s.
@app.function(image=image, gpu=GPU, cpu=4.0, memory=16384, volumes={VOL: vol}, timeout=60 * 60,
              max_containers=4, scaledown_window=5)
def train(arm: str, p: str, p_end: str, seed: str, steps: str, rep: str, git_commit: str, git_dirty: str,
          cache_key: str):
    t0 = time.time()
    if not os.path.isdir(f"{VOL}/data/fineweb10B"):
        raise RuntimeError("no data on the Volume -- run `modal run experiments/modal_run.py --setup` first")

    log_root = RUN_LOGS  # emptied per call, so the one directory in it afterwards is this run's
    shutil.rmtree(log_root, ignore_errors=True)
    cmd = ["bash", "experiments/run.sh", "--arm", arm, "--seed", seed]
    if arm != "baseline":
        cmd += ["--p", p] + (["--p-end", p_end] if p_end else [])
    if steps:
        cmd += ["--steps", steps]
    if rep != "1":
        cmd += ["--rep", rep]
    env = {**os.environ, "GPUS": "1", "LOG_ROOT": log_root, "GIT_COMMIT": git_commit, "GIT_DIRTY": git_dirty}
    proc = subprocess.run(cmd, cwd="/repo", env=env)

    (tag,) = os.listdir(log_root)
    out = Path(log_root) / tag
    # train.log means "this run finished", to analyze.py and to the skip check alike. A failed
    # run keeps its log for debugging, under a name neither of them reads.
    if proc.returncode != 0 and (out / "train.log").exists():
        (out / "train.log").rename(out / "train.failed.log")
    files = {f"{tag}/{name}": (out / name).read_text() for name in KEEP if (out / name).exists()}
    # One spend record per ATTEMPT, under a unique name, so a rerun or a parallel container can
    # never overwrite another attempt's cost.
    secs = time.time() - t0
    files[f"_spend/{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-{tag}.txt"] = (
        f"date={time.strftime('%Y-%m-%d', time.gmtime())} secs={secs:.0f} usd={secs * USD_PER_SEC:.2f} "
        f"gpu={GPU} exit={proc.returncode}\n")
    for rel, text in files.items():  # the Volume copy survives a dropped connection; _fetch reads it back
        dest = Path(f"{VOL}/logs/{rel}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)
    if proc.returncode == 0:
        os.makedirs(f"{VOL}/cache", exist_ok=True)
        Path(f"{VOL}/cache/{cache_key}").touch()
    vol.commit()  # also persists the compile cache
    return {"tag": tag, "code": proc.returncode, "secs": secs, "files": files}


@app.function(image=image, gpu=os.environ.get("MODAL_PREFLIGHT_GPU", "T4"), volumes={VOL: vol}, timeout=10 * 60,
              scaledown_window=5)
def run_preflight():
    """Everything that can break a run except the H100 itself, for about 2 cents instead of
    about $1.40. Run it twice: the second run should report cache files found at start."""
    import torch, triton
    print(f"torch {torch.__version__}  cuda {torch.version.cuda}  triton {triton.__version__}  {torch.cuda.get_device_name(0)}")
    shards = sorted(os.listdir(f"{VOL}/data/fineweb10B")) if os.path.isdir(f"{VOL}/data/fineweb10B") else []
    print(f"data: {sum(n.endswith('.bin') for n in shards)} shard(s) on the Volume")

    count = lambda d: sum(len(fs) for _, _, fs in os.walk(d))
    caches = (os.environ["TORCHINDUCTOR_CACHE_DIR"], os.environ["TRITON_CACHE_DIR"])
    before = [count(d) for d in caches]
    # Compiling anything exercises gcc + Python headers (Triton's launcher stub) and the
    # cache directories' file locking on the Volume, which is where a network mount would fail.
    t = time.time()
    fn = torch.compile(lambda x: (x.sin() * 2).relu().sum())
    fn(torch.randn(4096, device="cuda"))
    print(f"torch.compile ok in {time.time() - t:.1f}s; cache files inductor/triton: {before} -> {[count(d) for d in caches]}"
          f"  ({'warm: the Volume cache persisted' if any(before) else 'cold: run preflight again to confirm it persists'})")

    import tiktoken  # train_gpt.py does this at import; doing it here puts the vocab on the Volume
    print("tiktoken gpt2 vocab:", tiktoken.get_encoding("gpt2").n_vocab, "cached in", os.environ["TIKTOKEN_CACHE_DIR"],
          os.listdir(os.environ["TIKTOKEN_CACHE_DIR"]))

    from kernels import get_kernel  # same call as train_gpt.py: fails here if no build matches this torch/CUDA
    try:
        fa3 = get_kernel("kernels-community/flash-attn3", version=1).flash_attn_interface
        print("flash-attn3 kernel loads:", hasattr(fa3, "flash_attn_varlen_func"))
    except RuntimeError as e:
        # The build targets sm_80 and sm_90 (A100/H100). On the T4 the arch check refuses to
        # load it, but only AFTER a build matching this torch/CUDA was found, which is the
        # part that could not be known in advance. Any other error is a real failure.
        if "does not support the current device" not in str(e):
            raise
        print("flash-attn3: matching build found;", str(e).split("RuntimeError:")[-1].split(". The declared")[0].strip())
    vol.commit()


def _fetch(logs: Path) -> int:
    """Pull finished logs off the Volume. Makes `modal run --detach` and dropped connections
    harmless: the skip check below then sees runs that finished while nobody was watching."""
    from modal.volume import FileEntryType
    try:
        entries = vol.listdir("logs", recursive=True)
    except Exception:
        return 0  # nothing has been logged to the Volume yet
    n = 0
    for e in entries:
        dest = logs / Path(e.path).relative_to("logs")
        wanted = dest.name in KEEP or dest.parent.name == "_spend"
        if e.type == FileEntryType.FILE and wanted and not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"".join(vol.read_file(e.path)))
            n += 1
    return n


def _spent_this_month(logs: Path):
    """Sum of this calendar month's spend records (UTC). A floor, not the bill."""
    month, usd, n = time.strftime("%Y-%m", time.gmtime()), 0.0, 0
    for f in (logs / "_spend").glob("*.txt"):
        kv = dict(x.split("=", 1) for x in f.read_text().split())
        if kv.get("date", "").startswith(month):
            usd, n = usd + float(kv["usd"]), n + 1
    return usd, n


def _cache_is_warm(key: str) -> bool:
    try:
        return any(Path(e.path).name == key for e in vol.listdir("cache"))
    except Exception:
        return False


def _tag(arm: str, p: str, p_end: str, seed: str, steps: str, rep: str = "1") -> str:
    """Mirror of TAG in run.sh. Only used to skip finished runs; the directory actually
    written uses the tag run.sh reports, so a drift here costs a rerun, not a wrong result."""
    if arm == "baseline":
        p, p_end = "0.0", ""
    return f"{arm}_p{p}-{p_end or p}_seed{seed}" + (f"_steps{steps}" if steps else "") + (f"_rep{rep}" if rep != "1" else "")


@app.local_entrypoint()
def main(arm: str = "pair", p: str = "0.15", p_end: str = "", seeds: str = "1", steps: str = "", rep: str = "1",
         setup: bool = False, preflight: bool = False, fetch: bool = False, max_runs: int = 8,
         budget: float = 30.0):
    if setup:
        download_data.remote()
        return
    if preflight:
        run_preflight.remote()
        return
    logs = REPO / "experiments" / "logs"
    pulled = _fetch(logs)
    if pulled or fetch:
        print(f"fetched {pulled} new log file(s) from the Volume")
    if fetch:
        return
    if arm not in ("pair", "baseline", "masked"):
        raise SystemExit(f"--arm must be pair, baseline or masked (got {arm})")

    git = lambda *a: subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout
    commit = git("rev-parse", "HEAD").strip()
    dirty = str(len(git("status", "--porcelain", "--", "train_gpt.py").splitlines()))

    key = _cache_key()
    jobs = []
    for seed in seeds.split(","):
        for a in (("baseline", "masked") if arm == "pair" else (arm,)):
            # A step-removal test compares a SHORT masked run against the FULL baseline.
            s = "" if (arm == "pair" and a == "baseline") else steps
            if (logs / _tag(a, p, p_end, seed, s, rep) / "train.log").exists():
                print(f"skip  {_tag(a, p, p_end, seed, s, rep)} (already have it)")
            else:
                jobs.append((a, p, p_end, seed, s, rep, commit, dirty, key))
    if not jobs:
        print("nothing to run")
        return
    # H100s need a card on file, so an oversized sweep would bill real money once the $30 of
    # monthly credit is gone. Make going past a small batch a deliberate act.
    if len(jobs) > max_runs:
        raise SystemExit(f"{len(jobs)} runs is roughly ${len(jobs) * USD_PER_RUN:.0f} of the $30 monthly credit and "
                         f"--max-runs is {max_runs}. Pass fewer seeds, or raise --max-runs on purpose.")
    spent, n_spent = _spent_this_month(logs)
    planned = len(jobs) * USD_PER_RUN
    if spent + planned > budget:
        raise SystemExit(f"this month's {n_spent} recorded attempt(s) come to ~${spent:.2f}; {len(jobs)} more at "
                         f"~${USD_PER_RUN:.2f} would reach ~${spent + planned:.2f}, past --budget {budget:.0f}. "
                         f"Run fewer, or raise --budget on purpose.")
    print(f"{len(jobs)} run(s) on 1x{GPU}, roughly ${planned:.0f}; ~${spent:.2f} already recorded this month "
          f"(launcher's count only: the Modal dashboard is the authority).")

    # Parallel cold starts would each pay the full compile (~7 min of H100 apiece). When this
    # code has never finished a run on this GPU type, one job goes first and fills the cache.
    batches = [jobs]
    if len(jobs) > 1 and not _cache_is_warm(key):
        print("compile cache is cold for this code: running the first job alone to fill it, then the rest in parallel")
        batches = [jobs[:1], jobs[1:]]

    def run_batch(batch) -> int:
        ok = 0
        for result in train.starmap(batch, order_outputs=False, return_exceptions=True):
            if isinstance(result, Exception):
                print(f"FAILED: {result!r}")
                if isinstance(result, modal.exception.FunctionTimeoutError):
                    # The container was killed, so it left no spend record. It still billed the full timeout.
                    rec = logs / "_spend" / f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-timeout.txt"
                    rec.parent.mkdir(parents=True, exist_ok=True)
                    rec.write_text(f"date={time.strftime('%Y-%m-%d', time.gmtime())} secs=3600 usd={3600 * USD_PER_SEC:.2f} gpu={GPU} exit=timeout\n")
                continue
            tag, code, files, secs = result["tag"], result["code"], result["files"], result["secs"]
            for rel, text in files.items():
                dest = logs / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(text)
            last = [line for line in files.get(f"{tag}/train.log", "").splitlines() if "val_loss:" in line]
            print(f"{'done ' if code == 0 else f'EXIT {code}'} {tag}: {last[-1] if last else 'no val_loss logged'}"
                  f"  [{secs / 60:.1f} min, ~${secs * USD_PER_SEC:.2f}]")
            ok += code == 0
        return ok

    done = 0
    for i, batch in enumerate(batches):
        ok = run_batch(batch)
        done += ok
        # Whatever broke the first job would very likely break the others too, and they would
        # each pay for a compile before finding out.
        if i == 0 and len(batches) > 1 and not ok:
            print(f"the first job failed, so the remaining {len(batches[1])} were NOT started. "
                  f"See experiments/logs/<tag>/train.failed.log")
            break
    if done:
        print("\nnext: python experiments/analyze.py")
