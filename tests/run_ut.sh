#!/bin/bash
set -exo pipefail

export PYTHONPATH=$PYTHONPATH:.


function test_unittests() {
    pytest tests/core/test_bucketing_loader.py
    pytest tests/core/test_dynamic.py
    pytest tests/distributed/test_dist_ops.py
    pytest tests/distributed/test_fsdp_optim_state.py
    pytest tests/ops/test_flash_attn.py
    pytest tests/ops/test_context_parallel.py
    pytest tests/core/test_without_torch_xla.py
}

test_unittests
