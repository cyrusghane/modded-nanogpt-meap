"""One GPU per rank: launch train_gpt.py so that every rank sees its own GPU as cuda:0.

    torchrun --standalone --nproc_per_node=8 experiments/one_gpu_per_rank.py train_gpt.py

train_gpt.py picks its device from LOCAL_RANK (cuda:{LOCAL_RANK}), and torch.compile's cache keys carry
the device of the example inputs, so without this every rank writes its own copy of every compiled graph.
Restricting each rank to its own GPU with CUDA_VISIBLE_DEVICES makes that GPU cuda:0 for the rank, so all
ranks of a run share one set of cache entries: an N-GPU run's cache serves every rank of the next N-GPU run.
It does NOT let a 1-GPU run warm an 8-GPU one: the graphs themselves differ with the GPU count (parameter
banks are padded to a multiple of world_size, and grad_scale = 1/grad_accum_steps is a baked-in constant).
The computation is identical (one GPU per process either way); only the numbering changes. Harness only:
train_gpt.py is unmodified.
"""
import os
import runpy
import sys

local_rank = os.environ["LOCAL_RANK"]
os.environ["CUDA_VISIBLE_DEVICES"] = local_rank  # before torch is imported anywhere
os.environ["LOCAL_RANK"] = "0"
script = os.path.abspath(sys.argv[1])
sys.argv = [script] + sys.argv[2:]            # train_gpt.py reads its own source from sys.argv[0]
sys.path.insert(0, os.path.dirname(script))   # and imports triton_kernels from its own directory
runpy.run_path(script, run_name="__main__")
