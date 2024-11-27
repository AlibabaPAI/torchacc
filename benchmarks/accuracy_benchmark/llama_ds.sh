set -ex

[ -z "$RANK" ] && RANK=0
[ -z "$WORLD_SIZE" ] && WORLD_SIZE=1
[ -z "$MASTER_ADDR" ] && MASTER_ADDR=127.0.0.1
[ -z "$MASTER_PORT" ] && MASTER_PORT=9010
[ -z "$TASK_TAG" ] && TASK_TAG=0000
[ -z "$BS" ] && BS=4
[ -z "$SEQLEN" ] && SEQLEN=4096

RUN_CLM=/home/wangang.wa/transformers/examples/pytorch/language-modeling/run_clm.py
MODEL_NAME_OR_PATH=/home/wangang.wa/open_source_models/model_scope_models/Llama-3.2-1B-Instruct/

# This is the training config. You can change it as you need.
FSDP_CONFIG="llama_fsdp_ds.json"
cat <<EOF > "$FSDP_CONFIG"
{
    "train_batch_size": $((BS*8)),
    "train_micro_batch_size_per_gpu": $BS,
    "optimizer": {
        "type": "AdamW"
    },
    "zero_optimization": {
        "stage": 3
    },
    "bf16": {
        "enabled": true
    }
}
EOF


echo "Running a deepspeed job ..."
export USE_TORCH_XLA=0

NPROC_PER_NODE=8
PRECISION="bf16=true"
JOB_NAME="QWEN_FSDP_DEEPSPEED_GPU${NPROC_PER_NODE}_BS${BS}_SEQLEN${SEQLEN}_BF16"


torchrun --nproc_per_node $NPROC_PER_NODE \
    --nnodes $WORLD_SIZE \
    --node_rank $RANK \
    --master_port $MASTER_PORT \
    --master_addr $MASTER_ADDR \
    $RUN_CLM \
    --num_train_epochs 2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --use_fast_tokenizer false \
    --per_device_train_batch_size $BS \
    --per_device_eval_batch_size $BS \
    --do_train \
    --output_dir ./outputs_ds \
    --overwrite_output_dir \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --tokenizer_name $MODEL_NAME_OR_PATH \
    --trust_remote_code true \
    --cache_dir ./cache \
    --block_size $SEQLEN \
    --optim adamw_torch \
    --save_strategy no \
    --logging_strategy steps \
    --gradient_checkpointing no \
    --logging_steps 100 \
    --max_train_samples 100 \
    --$PRECISION \
    --deepspeed $FSDP_CONFIG 2>&1 | tee ./$JOB_NAME.log
