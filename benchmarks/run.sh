#!/bin/bash
set -exo pipefail


export XLA_ALLOCATOR_FRACTION=0.8


GPU_NUM=4
LOG_FOLDER_BASE=./log/tb
FILE_NAME=transformer.py


function run_baseline {
    local precision=$1

    torchrun --nproc_per_node=1        $FILE_NAME --tb_folder="${LOG_FOLDER_BASE}_${precision}/" --"$precision"
    torchrun --nproc_per_node=$GPU_NUM $FILE_NAME --tb_folder="${LOG_FOLDER_BASE}_${precision}/" --"$precision"
}

function test_accuracy {
    local precision=$1
    torchrun --nproc_per_node=1        $FILE_NAME --tb_folder="${LOG_FOLDER_BASE}_${precision}/" --"$precision" --acc
    torchrun --nproc_per_node=$GPU_NUM $FILE_NAME --tb_folder="${LOG_FOLDER_BASE}_${precision}/" --"$precision" --acc --dp_size=$GPU_NUM
    torchrun --nproc_per_node=$GPU_NUM $FILE_NAME --tb_folder="${LOG_FOLDER_BASE}_${precision}/" --"$precision" --acc --fsdp_size=$GPU_NUM --gc
}

function test_performance {
    local precision=$1

    torchrun --nproc_per_node=1        $FILE_NAME --tb_folder="${LOG_FOLDER_BASE}_${precision}/" --"$precision" --acc --disable_loss_print
    torchrun --nproc_per_node=$GPU_NUM $FILE_NAME --tb_folder="${LOG_FOLDER_BASE}_${precision}/" --"$precision" --acc --disable_loss_print --dp_size=$GPU_NUM
    torchrun --nproc_per_node=$GPU_NUM $FILE_NAME --tb_folder="${LOG_FOLDER_BASE}_${precision}/" --"$precision" --acc --disable_loss_print --fsdp_size=$GPU_NUM --gc
}

function run_tests {
    local precision=$1

    run_baseline $precision
    test_accuracy $precision
    test_performance $precision
}


# BF16 Tests
run_tests "bf16"

# FP16 Tests
run_tests "fp16"
