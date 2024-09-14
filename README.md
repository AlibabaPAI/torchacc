[![docs](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://torchacc.readthedocs.io/en/latest/)
[![CI](https://github.com/alibabapai/torchacc/actions/workflows/unit_test.yml/badge.svg)](https://github.com/alibabapai/torchacc/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/alibabapai/torchacc/blob/main/LICENSE)

# TorchAcc
**TorchAcc** is an AI training acceleration framework developed by Alibaba Cloud’s PAI team.

TorchAcc is built on [PyTorch/XLA](https://github.com/pytorch/xla) and provides an easy-to-use interface to accelerate the training of PyTorch models. At the same time, TorchAcc has implemented extensive optimizations for distributed training, memory management, and computation specifically for GPUs, ultimately achieving improved ease of use, better GPU training performance, and enhanced scalability for distributed training.

[**Documentation**](https://torchacc.readthedocs.io/en/latest/)

## Highlighted Features

* Rich distributed parallelism strategies
    * Data Parallelism
    * Fully Sharded Data Parallelism
    * Tensor Parallelism
    * Pipeline Parallelism
    * Context Parallelism
      * [Ulysess](https://arxiv.org/abs/2309.14509)
      * [Ring Attention](https://arxiv.org/abs/2310.01889)
      * FlashSequence (2D Sequence Parallelism)
* Memory efficient
* High Performance
* Easy-to-use API

  You can accelerate your transformer models with just a few lines of code using TorchAcc.

<p align="center">
  <img width="80%" src=docs/figures/api.gif />
</p>


## Architecture Overview
The main goal of TorchAcc is to provide a high-performance AI training framework.
It utilizes IR abstractions at different layers and employs static graph compilation optimization like XLA and dynamic graph compilation optimization like BladeDISC, as well as distributed optimization techniques, to offer a comprehensive end-to-end optimization solution from the underlying operators to the upper-level models.


<p align="center">
  <img width="80%" src=docs/figures/arch.png />
</p>


## Installation

### Docker
```
sudo docker run  --gpus all --net host --ipc host --shm-size 10G -it --rm --cap-add=SYS_PTRACE registry.cn-hangzhou.aliyuncs.com/pai-dlc/acc:r2.3.0-cuda12.1.0-py3.10 bash
```

### Build from source

see the [contribution guide](docs/source/contributing.md).


## Getting Started

We present a straightforward example for training a Transformer model using TorchAcc, illustrating the usage of the TorchAcc API.
You can quickly initiate training a Transformer model with TorchAcc by executing the following command:
``` shell
torchrun --nproc_per_node=4 benchmarks/transformer.py --bf16 --acc --disable_loss_print --fsdp_size=4 --gc
```

## LLMs training examples

### Utilizing HuggingFace Transformers

If you are familiar with HuggingFace Transformers's Trainer, you can easily accelerate a Transformer model using TorchAcc, see the [huggingface transformers](docs/source/tutorials/hf_transformers.md)

### LLMs training acceleration with FlashModels

If you want to try the latest features of Torchacc or want to use the TorchAcc interface more flexibly for model acceleration, you can use our LLM acceleration library, FlashModels. FlashModels integrates various distributed implementations of commonly used open-source LLMs and provides a wealth of examples and benchmarks.

https://github.com/AlibabaPAI/FlashModels

### SFT using modelscope/swift
coming soon..


## Contributing
see the [contribution guide](docs/source/contributing.md).


## Contact Us

You can contact us by adding our DingTalk group:

<p align="center">
  <img width="30%" src=docs/figures/group.png />
</p>

## License
[Apache License 2.0](LICENSE)
