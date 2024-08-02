# HuggingFace Transformers

This is a best Practices for Accelerating Training with HuggingFace Transformers. TorchAcc supports acceleration of native HuggingFace Transformers. For users using the HF `Trainer` interface, it is very convenient to accelerate Transformers model training through TorchAcc.

The following will use the `run_clm.py` example script from the Transformers library to train `Llama3-8B` tasks as examples, demonstrating how to use `TorchAcc` to accelerate Transformers training. We will also compare the methods of training Transformers models using `native PyTorch` and `DeepSpeed` with the `run_clm.py` script.


## Environment Preparation

### Start a container

Refer to the `"install"` section to obtain the latest image:

```bash
sudo docker run --gpus all --net host --ipc host --shm-size 10G -it --rm --cap-add=SYS_PTRACE registry.cn-wulanchabu.aliyuncs.com/pai-dlc/acc:r2.3.0-cuda12.1.0-py3.10-nightly bash
```

### Environment Configuration

Since we are running the `run_clm.py` script built into Transformers, it requires source code installation of the Transformers library:

```bash
# Uninstall Transformers already installed in the image
pip uninstall transformers -y

# Clone and install Transformers
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .

# Install related dependencies
pip install evaluate scikit-learn

# If DeepSpeed training is needed
pip install deepspeed
```

### Preparation

> If your network cannot access HuggingFace, please refer to the following instructions to manually download the model. Otherwise, you can skip the `Model Preparation` and `Dataset Preparation` sections.

#### Model Preparation

You can download the `Meta-Llama-3-8B` model from the model repositories of `HuggingFace` or `ModelScope`. Here's an example using `ModelScope`, with the model repository link: [https://modelscope.cn/models/LLM-Research/Meta-Llama-3-8B](https://modelscope.cn/models/LLM-Research/Meta-Llama-3-8B/files).

You can clone the model configuration and weights to your local machine using `git clone`, and store them in a specified directory (such as Meta-Llama-3-8B):

```bash
apt-get update && apt-get install git-lfs
git clone https://www.modelscope.cn/LLM-Research/Meta-Llama-3-8B.git
```

### Dataset Preparation

We use the wikitext dataset for training the model. For detailed information about the dataset, visit: [https://www.modelscope.cn/datasets/modelscope/wikitext/](https://www.modelscope.cn/datasets/modelscope/wikitext/)

You can download the dataset and place it in the specified directory.

### Enabling FlashAttention

In the `run_clm.py` file, find the `AutoModelForCausalLM.from_pretrained()` and `AutoModelForCausalLM.from_config()`, and add `attn_implementation="flash_attention_2"` to enable FlashAttention computation.

```diff
diff --git a/examples/pytorch/language-modeling/run_clm.py b/examples/pytorch/language-modeling/run_clm.py
index c0db57037..dc8e3040a 100755
--- a/examples/pytorch/language-modeling/run_clm.py
+++ b/examples/pytorch/language-modeling/run_clm.py
@@ -434,9 +436,10 @@ def main():
                trust_remote_code=model_args.trust_remote_code,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=model_args.low_cpu_mem_usage,
+               attn_implementation='flash_attention_2'
            )
        else:
-        model = AutoModelForCausalLM.from_config(config, trust_remote_code=model_args.trust_remote_code)
+        model = AutoModelForCausalLM.from_config(config, trust_remote_code=model_args.trust_remote_code, attn_implementation='flash_attention_2')
            n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
            logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
```



## PyTorch Native Training

You can follow these steps to conduct Transformers Llama3-8B PyTorch native FSDP training.

### Configure FSDP config file

When running FSDP training tasks using `run_clm.py`, you need to configure this file:

**Note: Due to a bug in Transformers, activation_checkpoint cannot be enabled for this task.**

```json
{
    "fsdp_transformer_layer_cls_to_wrap": [
        "LlamaDecoderLayer"
    ],
    "activation_checkpointing": false
}
```

### Training command

We utilize the `run_clm.py` file from the Transformers library to run directly by specifying parameters without needing additional code. The `run_clm.py` file encapsulates the logic of the Transformers library's Trainer and provides various parameter configurations. For specific parameter information, you can run `python run_clm.py --help`.

Use `torchrun` to start the training. The specific command is as follows:


```bash
set -ex

echo "Running a native torch job ..."

export USE_TORCH_XLA=0

[ -z "$RANK" ] && RANK=0
[ -z "$WORLD_SIZE" ] && WORLD_SIZE=1
[ -z "$MASTER_ADDR" ] && MASTER_ADDR=127.0.0.1
[ -z "$MASTER_PORT" ] && MASTER_PORT=9010

BS=1
SEQLEN=4096
NPROC_PER_NODE=8
PRECISION="bf16=true"
FSDP_CONFIG="llama3_fsdp_native.json"
JOB_NAME="LLAMA3_FSDP_NATIVE_GPU${NPROC_PER_NODE}_BS${BS}_SEQLEN${SEQLEN}_BF16_FA"


torchrun --nproc_per_node $NPROC_PER_NODE \
    --nnodes $WORLD_SIZE \
    --node_rank $RANK \
    --master_port $MASTER_PORT \
    --master_addr $MASTER_ADDR \
    ../examples/pytorch/language-modeling/run_clm.py \
    --num_train_epochs 2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --use_fast_tokenizer false \
    --per_device_train_batch_size $BS \
    --per_device_eval_batch_size $BS \
    --do_train \
    --output_dir /tmp/test-clm \
    --overwrite_output_dir \
    --config_name ./Meta-Llama-3-8B/ \
    --tokenizer_name ./Meta-Llama-3-8B/ \
    --trust_remote_code true \
    --cache_dir ./cache \
    --block_size $SEQLEN \
    --optim adamw_torch \
    --save_strategy no \
    --logging_strategy steps \
    --gradient_checkpointing no \
    --logging_steps 100 \
    --$PRECISION \
    --fsdp "auto_wrap" \
    --fsdp_config $FSDP_CONFIG 2>&1 | tee ./$JOB_NAME.log
```


## DeepSpeed Training

### Configuring DeepSpeed Config

Specific configuration details can be found in the official documentation: [https://www.deepspeed.ai/docs/config-json/](https://www.deepspeed.ai/docs/config-json/). To align with the Transformers native and torchacc's FSDP training tasks, we configure the DeepSpeed file as follows:

* The zero3 training strategy is the same as FSDP.
* train_batch_size = train_micro_batch_size_per_gpu * number of training cards, should align with `batch_size` in the training script.

> The Transformers library does not recognize activation checkpointing configured in DeepSpeed config, so activation checkpointing configuration is unnecessary.

```json
{
    "train_batch_size": 8,
    "train_micro_batch_size_per_gpu": 1,
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
```

### Training Script


```bash
set -ex

echo "Running a deepspeed job ..."

export USE_TORCH_XLA=0

[ -z "$RANK" ] && RANK=0
[ -z "$WORLD_SIZE" ] && WORLD_SIZE=1
[ -z "$MASTER_ADDR" ] && MASTER_ADDR=127.0.0.1
[ -z "$MASTER_PORT" ] && MASTER_PORT=9010

BS=1
SEQLEN=4096
NPROC_PER_NODE=8
PRECISION="bf16=true"
FSDP_CONFIG="llama3_fsdp_ds.json"
JOB_NAME="LLAMA3_FSDP_DEEPSPEED_GPU${NPROC_PER_NODE}_BS${BS}_SEQLEN${SEQLEN}_BF16_FA"


torchrun --nproc_per_node $NPROC_PER_NODE \
    --nnodes $WORLD_SIZE \
    --node_rank $RANK \
    --master_port $MASTER_PORT \
    --master_addr $MASTER_ADDR \
    ./examples/pytorch/language-modeling/run_clm.py \
    --num_train_epochs 2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --use_fast_tokenizer false \
    --per_device_train_batch_size $BS \
    --per_device_eval_batch_size $BS \
    --do_train \
    --output_dir /tmp/test-clm \
    --overwrite_output_dir \
    --config_name ./Meta-Llama-3-8B/ \
    --tokenizer_name ./Meta-Llama-3-8B/ \
    --trust_remote_code true \
    --cache_dir ./cache \
    --block_size $SEQLEN \
    --optim adamw_torch \
    --save_strategy no \
    --logging_strategy steps \
    --gradient_checkpointing no \
    --logging_steps 100 \
    --$PRECISION \
    --deepspeed $FSDP_CONFIG 2>&1 | tee ./$JOB_NAME.log
```

## TorchAcc Training

If you want to accelerate the training of `Transformers llama3-8b` with Torchacc, you need to make the following changes:

Open the `examples/pytorch/language-modeling/run_clm.py` file and insert the following code at the very top:

```python
import torchacc
torchacc.utils.patch.patch_llama(True)
```

### Configure TorchAcc FSDP Config

You can control the `xla_fsdp_grad_ckpt` parameter to enable or disable gradient checkpointing.

```json
{
    "fsdp_transformer_layer_cls_to_wrap": [
        "LlamaDecoderLayer"
    ],
    "xla": true,
    "xla_fsdp_settings": {
        "compute_dtype": "bfloat16",
        "buffer_dtype": "bfloat16",
        "opt_flatten_overlap": true,
        "pin_layout_in_collective_ops": false,
        "flatten_parameters": true
    },
    "xla_fsdp_grad_ckpt": false
}
```

### Training Script


```bash
set -ex

echo "Running a torch job with torchacc ..."

export PJRT_ALLOCATOR_FRACTION=0.97
export PJRT_DEVICE=CUDA
#export XLA_PERSISTENT_CACHE_PATH=./compiled_cache # uncomment this line to cache the compile results and speed up initialization.

[ -z "$RANK" ] && RANK=0
[ -z "$WORLD_SIZE" ] && WORLD_SIZE=1
[ -z "$MASTER_ADDR" ] && MASTER_ADDR=127.0.0.1
[ -z "$MASTER_PORT" ] && MASTER_PORT=9010

BS=3
SEQLEN=4096
NPROC_PER_NODE=8
PRECISION="bf16=true"
FSDP_CONFIG="llama3_fsdp_acc.json"
JOB_NAME="LLAMA3_FSDP_TORCHACC_GPU${NPROC_PER_NODE}_BS${BS}_SEQLEN${SEQLEN}_BF16_FA"


torchrun --nproc_per_node $NPROC_PER_NODE \
    --nnodes $WORLD_SIZE \
    --node_rank $RANK \
    --master_port $MASTER_PORT \
    --master_addr $MASTER_ADDR \
    ./examples/pytorch/language-modeling/run_clm.py \
    --num_train_epochs 2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --use_fast_tokenizer false \
    --per_device_train_batch_size $BS \
    --per_device_eval_batch_size $BS \
    --do_train \
    --output_dir /tmp/test-clm \
    --overwrite_output_dir \
    --config_name ./Meta-Llama-3-8B/ \
    --tokenizer_name ./Meta-Llama-3-8B/ \
    --trust_remote_code true \
    --cache_dir ./cache \
    --block_size $SEQLEN \
    --optim adamw_torch \
    --save_strategy no \
    --logging_strategy steps \
    --gradient_checkpointing no \
    --logging_steps 100 \
    --$PRECISION \
    --fsdp "auto_wrap" \
    --fsdp_config $FSDP_CONFIG 2>&1 | tee ./$JOB_NAME.log
```

## Performance

The following is a comparison of various configurations tested for Torch native, DeepSpeed, and TorchAcc, aiming to select the optimal performance configuration for each framework.

Experimental Parameters:

1. `flash_attn==2.5.6`
2. Sequence Length = 4096
3. Compute Resources: 8 * 80G A100
4. transformers commit hash: [f91c16d270e5e3ff32fdb32ccf286d05c03dfa66](https://github.com/huggingface/transformers/tree/f91c16d270e5e3ff32fdb32ccf286d05c03dfa66)


Here is the table with the last two rows removed:

| Global Batch Size | PyTorch | DeepSpeed | TorchAcc |
| --- | --- | --- | --- |
| 8 | 2945.0 tokens/s/GPU | 3123.2 tokens/s/GPU | 3276.8 tokens/s/GPU |
| 16 | OOM | OOM | 3737.6 token/s/GPU |
| 24 | OOM | OOM | 4044.8 tokens/s/GPU |

- Optimal PyTorch Configuration: BS=8+FA+noGC, Throughput: 2945.0 tokens/perGPU/s
- Optimal DeepSpeed Configuration: BS=8+FA+noGC, Throughput: 3123.2 tokens/perGPU/s, showing a **6%** improvement over
PyTorch's optimal performance.
- Optimal TorchAcc Configuration: BS=24+FA+noGC, Throughput: 4044.8 tokens/perGPU/s, showing a **37%** improvement over PyTorch's optimal performance and a **30%** improvement over DeepSpeed's optimal performance.
