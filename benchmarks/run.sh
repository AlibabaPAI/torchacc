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

export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890 

# export PJRT_USE_TORCH_ALLOCATOR=1
# export TF_CPP_MIN_LOG_LEVEL=0
# export TF_CPP_VMODULE=bfc_allocator=3
rm -rf log/profile/*
export CUDA_VISIBLE_DEVICES=4,5,6,7

# export TORCH_LOGS="+graph_breaks,+graph"
export TORCH_LOGS="+recompiles"
export TORCH_LOGS="+graph_breaks"
# export TORCHDYNAMO_VERBOSE=1
# export TORCH_COMPILE_DEBUG=1
export USE_TORCH_XLA=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export XLA_DYNAMO_DEBUG=1
export PT_XLA_DEBUG=1

export PJRT_USE_TORCH_ALLOCATOR=1

DP=1
FSDP=1
GPU_NUM=$((DP*FSDP))

export XLA_FLAGS="${XLA_FLAGS} --xla_dump_hlo_as_text --xla_dump_to=./hlo"
rm -rf hlo

torchrun --nproc_per_node=$GPU_NUM $FILE_NAME --bf16 --dp_size=$DP --fsdp_size=$FSDP --batch_size 1 --max_seq_length 512 --model_name "/root/hyt/FlashModels/hf_models/config/llama-3-1b" --model_block "LlamaDecoderLayer" --disable_loss_print --profile --backend "eager" --acc |& tee log.txt

# --use_flash_attn

# "../../AlibabaPAI/transformers/examples/pytorch/torchacc/llama3/Meta-Llama-3-8B"
# --backend "eager"
# export CUDA_VISIBLE_DEVICES=4,5
# torchrun --nproc_per_node=2 $FILE_NAME --bf16 --acc --fsdp_size=2 --dp_size=1 --profile
