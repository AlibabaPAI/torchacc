#!/bin/bash

export USE_TORCH_XLA=0

RANK="${RANK:-0}"
WORLD_SIZE="${WORLD_SIZE:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-9010}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
BS="${BS:-4}"
SEQLEN="${SEQLEN:-4096}"
TASK_TAG="${TASK_TAG:-0000}"

PRECISION="bf16=true"
JOB_NAME="LLAMA_FSDP_TORCH_GPU${NPROC_PER_NODE}_BS${BS}_SEQLEN${SEQLEN}_BF16"
FSDP_CONFIG="llama_fsdp_torch.json"
CLS_TO_WRAP="LlamaDecoderLayer"

# $1: the run_clm.py path
# $2: local model directory
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <the run_clm.py path> <local model dir>"
    echo "You must provide exactly 2 parameters."
    exit 1
fi
TRANSFORMERS_DIR=$(realpath "$1")
MODEL_DIR=$(realpath "$2")
OUTPUTS_DIR=$(basename "$MODEL_DIR")_torch
RUN_CLM=$TRANSFORMERS_DIR/examples/pytorch/language-modeling/run_clm.py


# This is the training config. You can change it as you need.
cat >"$FSDP_CONFIG" <<EOF
{
    "fsdp_transformer_layer_cls_to_wrap": [
        "${CLS_TO_WRAP}"
    ],
    "activation_checkpointing": false
}
EOF

# Launch the job
torchrun --nproc_per_node "$NPROC_PER_NODE" \
    --nnodes "$WORLD_SIZE" \
    --node_rank "$RANK" \
    --master_port "$MASTER_PORT" \
    --master_addr "$MASTER_ADDR" \
    "$RUN_CLM" \
    --num_train_epochs 2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --use_fast_tokenizer false \
    --per_device_train_batch_size "$BS" \
    --per_device_eval_batch_size "$BS" \
    --do_train \
    --output_dir "$OUTPUTS_DIR" \
    --overwrite_output_dir \
    --model_name_or_path "$MODEL_DIR" \
    --tokenizer_name "$MODEL_DIR" \
    --trust_remote_code true \
    --low_cpu_mem_usage true \
    --cache_dir ./cache \
    --block_size "$SEQLEN" \
    --optim adamw_torch \
    --save_strategy no \
    --logging_strategy steps \
    --gradient_checkpointing no \
    --logging_steps 100 \
    --max_train_samples 100 \
    --"$PRECISION" \
    --fsdp "auto_wrap" \
    --fsdp_config "$FSDP_CONFIG" 2>&1 | tee ./${JOB_NAME}_${RANK}_${TASK_TAG}.log
