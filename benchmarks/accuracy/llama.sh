#!/bin/bash

if [[ $# -ne 2 && $# -ne 3 ]]; then
  echo "Usage: $0 <local_model_dir> <use_torchacc> [checkpiont_output_dir]"
  echo "  local_model_dir: Path to the local directory where the model will be saved."
  echo "  use_torchacc: 0 or 1 to indicate whether to use TorchAcc."
  echo "  checkpoint_output_dir: Optional. Default is the model name in <local_model_dir>."
  exit 1
fi

MODEL_DIR=$(realpath "$1")
USE_TORCHACC=$2
RANK="${RANK:-0}"
WORLD_SIZE="${WORLD_SIZE:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-9010}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SEQLEN="${SEQLEN:-1024}"
DATASET_NAME="${DATASET_NAME:-Salesforce/wikitext}"
DATASET_CONFIG_NAME="${DATASET_CONFIG_NAME:-wikitext-2-raw-v1}"
PRECISION="bf16=true"
RUN_CLM=./run_clm.py


if [ "$USE_TORCHACC" -eq 0 ]; then
  export USE_TORCH_XLA=0
  FSDP_CONFIG="llama_fsdp_torch.json"
  TEMP_OUTPUT_DIR=$(basename "$MODEL_DIR")_torch
  OUTPUTS_DIR=${3:-$TEMP_OUTPUT_DIR}
elif [ "$USE_TORCHACC" -eq 1 ]; then
  export PJRT_DEVICE=CUDA
  export XLA_FLAGS='--xla_gpu_memory_limit_slop_factor=500 --xla_multiheap_size_constraint_per_heap=15032385536'
  export ACCELERATE_USE_FSDP=true
  export PJRT_USE_TORCH_ALLOCATOR=true
  export LOW_CPU_MEM_USAGE=1
  export XLA_PERSISTENT_CACHE_PATH=./compiled_cache
  FSDP_CONFIG="llama_fsdp_acc.json"
  TEMP_OUTPUT_DIR=$(basename "$MODEL_DIR")_acc
  OUTPUTS_DIR=${3:-$TEMP_OUTPUT_DIR}
else
  echo "The third argument must be 0 or 1"
  exit 1
fi

# Launch the job
torchrun --nproc_per_node "$NPROC_PER_NODE" \
  --nnodes "$WORLD_SIZE" \
  --node_rank "$RANK" \
  --master_port "$MASTER_PORT" \
  --master_addr "$MASTER_ADDR" \
  "$RUN_CLM" \
  --num_train_epochs 2 \
  --dataset_name $DATASET_NAME \
  --dataset_config_name $DATASET_CONFIG_NAME \
  --use_fast_tokenizer false \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --per_device_eval_batch_size "$BATCH_SIZE" \
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
  --"$PRECISION" \
  --fsdp "auto_wrap" \
  --fsdp_config "$FSDP_CONFIG"
