#!/bin/bash
set -exo pipefail

export PYTHONPATH=$PYTHONPATH:.

function test_standalone() {
    torchrun --nproc_per_node=2 standalone/mnist_dp.py
    torchrun --nproc_per_node=4 standalone/pipeline.py --pp_num 4 --gc --bf16
    torchrun --nproc_per_node=4 standalone/pipeline.py --pp_num 4 --test_skip
    torchrun --nproc_per_node=4 standalone/ta_accelerate.py --gc
    torchrun --nproc_per_node=2 standalone/default_config.py
    torchrun --nproc_per_node=2 standalone/offload.py
}

test_standalone
