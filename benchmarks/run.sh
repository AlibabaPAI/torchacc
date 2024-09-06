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
# run_tests "bf16"

# FP16 Tests
# run_tests "fp16"

# export TF_CPP_MIN_LOG_LEVEL=0
# export TF_CPP_VMODULE=bfc_allocator=10
rm -rf log/profile/*
# torchrun --nproc_per_node=8 $FILE_NAME --bf16 --acc --fsdp_size=8 --dp_size=1 --batch_size 1 --max_seq_length 4096 --profile --model_name "../../AlibabaPAI/transformers/examples/pytorch/torchacc/llama3/Meta-Llama-3-8B" --model_block "LlamaDecoderLayer" --backend "eager" 
export CUDA_VISIBLE_DEVICES=4,5
torchrun --nproc_per_node=2 $FILE_NAME --bf16 --acc --fsdp_size=2 --dp_size=1 --batch_size 2 --profile
