#!/bin/bash
set -exo pipefail

# export XLA_DYNAMO_DEBUG=1
# export TORCH_LOGS="+recompiles"
# export TF_CPP_MIN_LOG_LEVEL=0
# export TF_CPP_VMODULE=xla_graph_executor=10
# export TF_CPP_VMODULE=hlo_pass_pipeline=10
# export TF_CPP_VMODULE=xla_graph_executor=10,pjrt_stream_executor_client=10,tensor=10

# export XLA_IR_DEBUG=1 XLA_HLO_DEBUG=1 XLA_HLO_DEBUG_VERBOSE_STACK=1 XLA_DUMP_FATAL_STACK=1

# export PT_XLA_DEBUG=1

# rm -rf hlo
# rm -rf ./log/torchrun

export CUDA_VISIBLE_DEVICES=4,5,6,7

DATASET_PATH=data/wikitext-2-raw-v1.json


MODELS=("Qwen2-7B" "llama-3-8b" "Yi-1.5-9B" "gemma-2-9b-it" "glm-4-9b")
MODELS=("Qwen2.5-3B-Instruct" "Llama-3.2-3B-Instruct" "gemma-2-2b-it")
MODELS=("Qwen2.5-3B-Instruct")

timestamp=$(date +%Y%m%d_%H%M%S)
# timestamp=benchmark
LOG_DIR="./log/benchmark/${timestamp}"

mkdir -p $LOG_DIR

MODEL_DIR="/root/hyt/models"

export USE_TORCH_XLA=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export PJRT_USE_TORCH_ALLOCATOR=1
export ZERO_COPY_ENABLED=1
export XLA_DYNAMO_CALL_COMPUTATION=1

# export XLA_EXPERIMENTAL=early_sync

FSDP_SIZE=4

declare -A BACKAND_PARAMS=(
    ["xla"]="--backend lazy"
    ["xla+dynamo"]="--backend lazy --use_dynamo"
    ["cuda"]="--backend eager"
)

function run_benchmark() {
    local model_name=$1
    local backend=$2
    local fsdp=$3

    if [ -z "$fsdp" ]; then
        fsdp=1
    fi

    local dp=1
    local gpu_num=$((dp*fsdp))
    local mbs=1

    local params=${BACKAND_PARAMS[$backend]}

    if [ -z "$params" ]; then
        echo "Error: Backend '$backend' not found."
        return 1
    fi

    local log_file="$LOG_DIR/${backend}_${model_name}.log"

    echo "Running benchmark for model: $model_name with backend: $backend"
    rm -rf log/profile
    torchrun --nproc_per_node=$gpu_num --master_port=9082 \
        transformer.py --dataset ${DATASET_PATH} --num_train_epochs 2 --dp_size $dp --fsdp_size $fsdp --batch_size $mbs --max_seq_length 4096 --model_name "${MODEL_DIR}/${model_name}" --model_block "auto" --acc --bf16 --use_flash_attn --benchmark --benchmark_steps 10 --profile --disable_loss_print ${params} \
         |& tee $log_file

}


for MODEL in "${MODELS[@]}"; do
    run_benchmark "$MODEL" "xla+dynamo" ${FSDP_SIZE}
    run_benchmark "$MODEL" "xla" ${FSDP_SIZE}
    run_benchmark "$MODEL" "cuda" ${FSDP_SIZE}
done

