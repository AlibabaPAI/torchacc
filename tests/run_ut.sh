#!/bin/bash
set -exo pipefail

export PYTHONPATH=$PYTHONPATH:.


function test_unittests() {
    pytest tests/core/test_bucketing_loader.py
    pytest tests/core/test_dynamic.py
    pytest tests/distributed/test_dist_ops.py
    pytest tests/ops/test_flash_attn.py
    pytest tests/ops/test_context_parallel.py
}

test_unittests
