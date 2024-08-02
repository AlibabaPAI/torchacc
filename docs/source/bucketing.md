# Data Bucketing

## Introduction
The output shape of the data loader can be dynamic; for example, in LLM training,
when the padding_strategy is set to 'longest', Each batch from the data loader
will have a variable last dimension, corresponding to differing sequence lengths.
This variability can cause multiple compilations in TorchAcc, which may slow down
the overall training process. On the other hand, padding each batch to a uniform
fixed length results in numerous needless computations. One approach to balancing
these two issues is to implement data bucketing, which reduces the number of
compilations and decreases the amount of padding required. Instead of padding
each batch to a uniform fixed length, we could select the nearest bucket size
based on the batch's shape and pad the batch to that bucket size.


## How to use

TorchAcc offers a BucketingDataLoader for managing data bucketing. It can be
utilized within `torchacc.accelerate` or directly via the `torchacc.AsyncLoader`.


Note: TorchAcc currently only supports bucketing for the final dimension of the data.


### 1. Use in torchacc.accelerate

You can directly specify bucket sizes based on the data distribution.

```python
config.dataloader.buckets=[128, 256, 512, 1024]
config.dataloader.pad_value_dict={'input_ids': 0, 'attention_mask': 0, 'labels': -100}

model, loader = torchacc.accelerate(model, loader, config)
```

or alternatively, employ the default strategy of uniform bucketing.
```python
# config.dataloader.buckets = [256, 512, 768, 1024]
config.dataloader.max_length=1024
config.dataloader.num_buckets=4
config.dataloader.pad_value_dict={'input_ids': 0, 'attention_mask': 0, 'labels': -100}

model, loader = torchacc.accelerate(model, loader, config)
```

For both options, you are required to specify the padding values for each category
in the data loader's output by utilizing the `pad_value_dict`.



### 2. Directly use via torchacc.AsyncLoader

You can use default uniform bucketing,
```python
torchacc.AsyncLoader(loader, device,
    max_length=1024,
    num_buckets=8,
    pad_value_dict={'input_ids': 0, 'attention_mask': 0, 'labels': -100})
```
or specify buckets,
```python
torchacc.AsyncLoader(loader, device,
    buckets=[128, 256, 512, 1024]
    pad_value_dict={'input_ids': 0, 'attention_mask': 0, 'labels': -100})
```

Additionally, by setting the environment variable `ACC_LOG_LEVEL=debug`, you can
activate logging to track the original data's last dimension size, the bucket length,
and the padding length. The details of the log are as follows:

```text
[2024-07-10 16:24:46,811 - DEBUG] original length: 179, bucket length: 192, padding length: 13
[2024-07-10 16:24:46,811 - DEBUG] original length: 367, bucket length: 384, padding length: 17
[2024-07-10 16:24:46,812 - DEBUG] original length: 248, bucket length: 256, padding length: 8
[2024-07-10 16:24:46,812 - DEBUG] original length: 63, bucket length: 64, padding length: 1
```