# FSDP (Fully Sharded Data Parallel)


FSDP (Fully Sharded Data Parallel) splits the model parameters, gradients, and optimizer states on top of data parallelism. It breaks down the all-reduce communication operation into reduce-scatter and all-gather, thereby reducing the peak memory usage of individual parallel workers. This allows training on larger models or with larger micro-batch sizes.

Let’s demonstrate how to accelerate FSDP training using TorchAcc optimization with a simple example.

## Torch Native Task

Below is the code for `GPT2` in Torch which training with `bfloat16`:

```python
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Model and tokenizer setup
model_name = 'gpt2'
max_length = 512
batch_size = 16

model_config = AutoConfig.from_pretrained(model_name, cache_dir='./log/model_cache')
model = AutoModelForCausalLM.from_config(model_config)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.model_max_length = max_length
tokenizer.pad_token = tokenizer.eos_token

# Dataset and dataloader
def preprocess_function(examples):
    examples['text'] = [text for text in examples['text'] if len(text) > 0]
    tokenized = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)
    tokenized['labels'] = tokenized['input_ids'].copy()
    return tokenized

dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train').map(preprocess_function, batched=True)
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = 'cuda:0'
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

model.train()
for step, batch in enumerate(tqdm(train_dataloader, unit='batch')):
    optimizer.zero_grad()
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        loss = model(**batch)['loss']
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print(f'step: {step}, loss: {loss.item():.4f}')
```

## FSDP

You only need to configure the TorchAcc `Config` and pass it to the `torchacc.accelerate` function to easily achieve FSDP training.

```diff
  import torch
+ import torchacc
  from datasets import load_dataset
  from tqdm import tqdm
  from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

  # Model and tokenizer setup
  model_name = 'gpt2'
  max_length = 512
  batch_size = 16

  model_config = AutoConfig.from_pretrained(model_name, cache_dir='./log/model_cache')
  model = AutoModelForCausalLM.from_config(model_config)
  tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
  tokenizer.model_max_length = max_length
  tokenizer.pad_token = tokenizer.eos_token

  # Dataset and dataloader
  def preprocess_function(examples):
      examples['text'] = [text for text in examples['text'] if len(text) > 0]
      tokenized = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)
      tokenized['labels'] = tokenized['input_ids'].copy()
      return tokenized

  dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train').map(preprocess_function, batched=True)
  dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
  train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

- device = 'cuda:0'
- model.to(device)
+ config = torchacc.Config()
+ config.compute.bf16 = True
+ config.dist.fsdp.size = 4
+ config.dist.fsdp.wrap_layer_cls = {'GPT2Block'}
+ model, train_dataloader = torchacc.accelerate(model, train_dataloader, config)
  optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

  model.train()
  for step, batch in enumerate(tqdm(train_dataloader, unit='batch')):
      optimizer.zero_grad()
-     batch = {k: v.to(device) for k, v in batch.items()}
      with torch.cuda.amp.autocast(dtype=torch.bfloat16):
          loss = model(**batch)['loss']
      loss.backward()
      optimizer.step()
      if step % 100 == 0:
          print(f'step: {step}, loss: {loss.item():.4f}')
```


The main changes:

* Removed the code for moving the model to the CUDA device.
* Configured FSDP and compute information through torchacc `Config`.
* Used the `torchacc.accelerate` interface to wrap and accelerate the `model` and `train_dataloader`.

The shell command for running FSDP tasks is the same as data parallelism:

```bash
$ torchrun --nproc_per_node=4 gpt2_acc.py
```




## Checkpoint Save/Load

### Save Checkpoint

Save the model parameters and optimizer states for each FSDP shard and LR scheduler. Note that you need to save ``shard_metadata`` to restore the correct shard information.
```python
import torch_xla.core.xla_model as xm
shard_meta_data = model.model.model.get_shard_metadata()

# 1) Save model shards
xm.rendezvous("saving_model")
model_ckpt = {
    'model': model.state_dict(),
    'shard_metadata': shard_meta_data,
}

torchacc.save(model_ckpt, CKPT_DIR + MODEL_NAME_PATTERN, master_only=False)

# 2) Save optimizer shards
xm.rendezvous("saving_optimizer_states")
optim_ckpt = {
    'optimizer': optimizer.state_dict(),
    'shard_metadata': shard_meta_data,
}
torchacc.save(optim_ckpt, CKPT_DIR + OPTIM_NAME_PATTERN, master_only=False)

# 3) Save lr_scheduler
torchacc.save(lr_scheduler.state_dict(), LR_SCHEDULER_DIR)
```

### offline consolidation
We now support offline consolidate and reshard fsdp model and optimizer ckpts. You can run ``consolidate_and_reshard_fsdp_ckpts --help`` to refer to the instruction.
```shell
# consolidate model and optimizer
consolidate_and_reshard_fsdp_ckpts --ckpt_dir CKPT_DIR --model_ckpt_name_pattern MODEL_NAME_PATTERN --optimizer_ckpt_name_pattern OPTIM_NAME_PATTERN
# you can use --reshard_num to reshard the fsdp checkpoints
```

### Load from Checkpoint
```python
# 1) Load model
model_consolidated = torch.load("model_consolidated.pth")
model.load_state_dict(model_consolidated)

# 2) Load optimizer
optimizer_consolidated = torch.load("optimizer_consolidated.pth")
optimizer.load_state_dict(optimizer_consolidated)

# 3) Load LR scheduler
lr_scheduler_state = torch.load(LR_SCHEDULER_DIR)
lr_scheduler.load_state_dict(lr_scheduler_state)
```

## Configurable parameters

The configurable parameters of FSDP are as follows:

| Parameter                  | Type          | Description                                                                                                                                                                                                                                                       |
|----------------------------|---------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| size                       | int           | Number of fully sharded data parallel.                                                                                                                                                                                                                            |
| wrap_layer_cls             | Set[str]      | Submodules with one of the `wrap_layer_cls` names will be wrapped as separated FSDP units.                                                                                                                                                                        |
| flatten_parameters         | bool          | If ``True``, flatten parameters into a single contiguous tensor for all_gather and reduce_scatter, which could potentially improve speed. In this case, one cannot apply separate optimizer groups to different original parameters in the wrapped module (e.g. setting bias terms or any BatchNorm submodules to have zero weight decay) since all the original parameters now become a single concatenated vector. |
| sync_module_states         | bool          | If ``True``, then each FSDP module will broadcast module parameters and buffers from rank 0 to ensure that they are replicated across ranks (adding communication overhead and more GPU memory overhead during initialization).                                     |
| use_spmd                   | bool          | If ``True``, use SPMD based FSDP.                                                                                                                                                                                                                                 |
| shard_output_callable      | callable      | A callable to shard the output of the forward pass. The callable should have the signature (output, mesh) -> None. If None, the default implementation will shard the first tensor in the output. If the output is a tuple, only the first tensor will be sharded.   |