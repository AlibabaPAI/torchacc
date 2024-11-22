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

rm -rf log/profile/*
export CUDA_VISIBLE_DEVICES=4,5,6,7

# export TORCH_LOGS="+graph_breaks,+graph"
# export TORCH_LOGS="+recompiles"
# export TORCH_LOGS="+graph_breaks,+recompiles,+graph"
# export TORCHDYNAMO_VERBOSE=1
# export TORCH_COMPILE_DEBUG=1
export USE_TORCH_XLA=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export XLA_ALLOCATOR_FRACTION=0.5
export PJRT_USE_TORCH_ALLOCATOR=1
export ZERO_COPY_ENABLED=1

# export XLA_IR_DEBUG=1 XLA_HLO_DEBUG=1

DP=1
FSDP=4
GPU_NUM=$((DP*FSDP))
MBS=1
MODEL="llama-3-1b"

# export TF_CPP_MIN_LOG_LEVEL=0
# export TF_CPP_VMODULE=torch_allocator=10,dl_convertor=10,pjrt_stream_executor_client=10,tracked_device_buffer=10,xla_graph_executor=10

# export XLA_DYNAMO_DEBUG=1
# export PT_XLA_DEBUG=1

# export XLA_FLAGS="${XLA_FLAGS} --xla_dump_hlo_as_text --xla_dump_to=./hlo"
# rm -rf hlo

# export XLA_HLO_DEBUG_VERBOSE_STACK=1 USE_TORCHACC=1 XLA_DUMP_FATAL_STACK=1 XLA_DUMP_HLO_GRAPH=1 XLA_SAVE_TENSORS_FILE=XLA_SAVE_TENSORS_FILE.txt XLA_SAVE_HLO_FILE=XLA_SAVE_HLO_FILE.txt XLA_SAVE_TENSORS_FMT=hlo XLA_METRICS_FILE=XLA_METRICS_FILE.txt XLA_IR_DEBUG=1 PT_XLA_DEBUG=1 XLA_HLO_DEBUG=1 
# rm -f XLA_SAVE_TENSORS_FILE*
# rm -f XLA_METRICS_FILE*


BACKEND="eager"
# BACKEND="lazy"

JOB_NAME=${MODEL}-${BACKEND}-FSDP${FSDP}-mbs${MBS}

torchrun --nproc_per_node=$GPU_NUM --master_port=9082 $FILE_NAME --num_train_epochs 2 --dp_size=$DP --fsdp_size=$FSDP --batch_size ${MBS} --max_seq_length 512 --model_name "/root/hyt/FlashModels/hf_models/config/${MODEL}" --model_block "LlamaDecoderLayer" --backend ${BACKEND} --acc --bf16 --profile --use_flash_attn \
    |& tee log/${JOB_NAME}.txt
