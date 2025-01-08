#!/bin/bash
set -exo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3

DATASET_PATH=data/wikitext-2-raw-v1.json

timestamp=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./log/benchmark/${timestamp}"

mkdir -p $LOG_DIR

MODEL_DIR="./models"

MODELS=($(find "$MODEL_DIR" -maxdepth 1 -type d -mindepth 1 -exec basename {} \;))

if [ ${#MODELS[@]} -eq 0 ]; then
    echo "No models found in $MODEL_DIR. Please run \`bash prepare.sh\` first."
    exit 1
fi

export USE_TORCH_XLA=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PJRT_USE_TORCH_ALLOCATOR=1

export XLA_EXPERIMENTAL=early_sync

FSDP_SIZE=4

declare -A BACKAND_PARAMS=(
    ["torchacc"]="--backend lazy"
    ["hybridtrace"]="--backend lazy --hybrid_trace"
    ["cuda"]="--backend eager"
)

function run_benchmark() {
    local model_name=$1
    local backend=$2
    local fsdp=$3

    if [ "$backend" == "hybridtrace" ]; then
        export TORCHACC_PATCH_FA=0
    else
        export TORCHACC_PATCH_FA=1
    fi

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
    torchrun --nproc_per_node=$gpu_num --master_port=9082 \
        transformer.py --dataset ${DATASET_PATH} --num_train_epochs 2 --dp_size $dp --fsdp_size $fsdp --batch_size $mbs --max_seq_length 4096 --model_name "${MODEL_DIR}/${model_name}" --model_block "auto" --acc --bf16 --use_flash_attn --benchmark --benchmark_steps 100 --disable_loss_print ${params} \
         |& tee $log_file

}


for MODEL in "${MODELS[@]}"; do
    run_benchmark "$MODEL" "hybridtrace" ${FSDP_SIZE}
    run_benchmark "$MODEL" "torchacc" ${FSDP_SIZE}
    run_benchmark "$MODEL" "cuda" ${FSDP_SIZE}
done

python parse.py $LOG_DIR --outdir $LOG_DIR
