#!/bin/bash

# $1: the HF transformers dir
# $2: local model directory
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <the HF transformers dir> <local model dir>"
    echo "You must provide exactly 2 parameters."
    exit 1
fi

export PJRT_DEVICE=CUDA
export XLA_FLAGS='--xla_gpu_memory_limit_slop_factor=500 --xla_multiheap_size_constraint_per_heap=15032385536'
export ACCELERATE_USE_FSDP=true
export PJRT_USE_TORCH_ALLOCATOR=true
# export LOW_CPU_MEM_USAGE=1
# export XLA_PERSISTENT_CACHE_PATH=./compiled_cache # uncomment this line to cache the compile results and speed up initialization.

RANK="${RANK:-0}"
WORLD_SIZE="${WORLD_SIZE:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-9010}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
BS="${BS:-1}"
SEQLEN="${SEQLEN:-4096}"
TASK_TAG="${TASK_TAG:-0000}"

PRECISION="bf16=true"
JOB_NAME="qwen_FSDP_TORCHACC_GPU${NPROC_PER_NODE}_BS${BS}_SEQLEN${SEQLEN}_BF16"
FSDP_CONFIG="qwen_fsdp_acc.json"
CLS_TO_WRAP="Qwen2DecoderLayer"

TRANSFORMERS_DIR=$(realpath "$1")
MODEL_DIR=$(realpath "$2")
OUTPUTS_DIR=$(basename "$MODEL_DIR")_acc
RUN_CLM=$TRANSFORMERS_DIR/examples/pytorch/language-modeling/run_clm.py

# Patch the run_clm.py
PATCH_FILE=$(realpath ./run_clm.py.acc.patch)
git config --global --add safe.directory $TRANSFORMERS_DIR
pushd $TRANSFORMERS_DIR
git checkout .
patch -p1 < $PATCH_FILE
popd

# This is the training config. You can change it as you need.
cat >"$FSDP_CONFIG" <<EOF
{
    "fsdp_transformer_layer_cls_to_wrap": [
        "${CLS_TO_WRAP}"
    ],
    "xla": true,
    "xla_fsdp_settings": {
        "compute_dtype": "bfloat16",
        "buffer_dtype": "bfloat16",
        "opt_flatten_overlap": true,
        "pin_layout_in_collective_ops": false,
        "flatten_parameters": true
    },
    "xla_fsdp_grad_ckpt": true
}
EOF

# Launch the job
torchrun --nproc_per_node "$NPROC_PER_NODE" \
    --nnodes "$WORLD_SIZE" \
    --node_rank "$RANK" \
    --master_port "$MASTER_PORT" \
    --master_addr "$MASTER_ADDR" \
    "$RUN_CLM" \
    --num_train_epochs 1 \
    --dataset_name Salesforce/wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
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
    --fsdp_config "$FSDP_CONFIG"
