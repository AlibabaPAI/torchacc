#!/bin/bash
set -exo pipefail

export PYTHONPATH=$PYTHONPATH:.

function test_standalone() {
    torchrun --nproc_per_node=2 standalone/mnist_dp.py
    torchrun --nproc_per_node=4 standalone/pipeline.py --pp_num 4 --gc --bf16
    torchrun --nproc_per_node=4 standalone/pipeline.py --pp_num 4 --test_skip
    torchrun --nproc_per_node=4 standalone/ta_accelerate.py --gc
    torchrun --nproc_per_node=4 standalone/consolidate_and_reshard_ckpts.py --fsdp_num 4 --ckpt_dir standalone/ckpt --reshard_num 4
    # PyTorch DDP
    torchrun --nproc_per_node=4 standalone/ta_accelerate.py --backend eager
    # PyTorch FSDP
    torchrun --nproc_per_node=4 standalone/ta_accelerate.py --backend eager --fsdp_num 4
    # PyTorch FSDP + DP (HYBRID_SHARD)
    torchrun --nproc_per_node=4 standalone/ta_accelerate.py --backend eager --fsdp_num 2
    torchrun --nproc_per_node=2 standalone/default_config.py
    torchrun --nproc_per_node=2 standalone/offload.py
}

test_standalone
