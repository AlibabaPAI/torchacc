[![docs](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://torchacc.readthedocs.io/en/latest/)
[![CI](https://github.com/alibabapai/torchacc/actions/workflows/unit_test.yml/badge.svg)](https://github.com/alibabapai/torchacc/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/alibabapai/torchacc/blob/main/LICENSE)

**TorchAcc** is an AI training acceleration framework developed by Alibaba Cloud’s PAI.

TorchAcc is built on [PyTorch/XLA](https://github.com/pytorch/xla) and provides an easy-to-use interface to accelerate the training of PyTorch models. At the same time, TorchAcc has implemented extensive optimizations for distributed training, memory management, and computation specifically for GPUs, ultimately achieving improved ease of use, better GPU training performance, and enhanced scalability for distributed systems.


## Highlighted Features

The key features of TorchAcc:

* Rich distributed Parallelism
    * Data Parallelism
    * Fully Sharded Data Parallelism
    * Tensor Parallelism
    * Pipeline Parallelism
    * Context Parallelism
      * [Ulysess](https://arxiv.org/abs/2309.14509)
      * [Ring Attention](https://arxiv.org/abs/2310.01889)
      * Flash Sequence (Solution for Long Sequence)
* Low Memory Cost
* High Performance
* Ease use

## Architecture Overview
The main goal of TorchAcc is to create a high-performance AI training framework. It utilizes IR abstractions at different layers and employs static graph compilation optimization like XLA and dynamic graph compilation optimization like BladeDISC, as well as distributed optimization techniques, to offer a comprehensive end-to-end optimization solution from the underlying operators to the upper-level models.


<p align="center">
  <img width="80%" src=docs/figures/arch.png />
</p>


## Installation

### Docker
```
sudo docker run  --gpus all --net host --ipc host --shm-size 10G -it --rm --cap-add=SYS_PTRACE registry.cn-hangzhou.aliyuncs.com/pai-dlc/acc:r2.3.0-cuda12.1.0-py3.10 bash
```

### Build from source
1. Requirements
```
torch==2.3.0
torch_xla==2.3.0
transformers>=4.41.2
```
Note: you should use the alibabapai/torch and alibabapai/xla to ensure GPU compatibility and performance.
see the [contribution guide](docs/source/contributing.md).


2. Build
```
python setup.py install
```

3. UT
```
sh tests/run_ut.sh
```

## LLMs training examples

### Use LLMs acceleration libray FlashModels
https://github.com/AlibabaPAI/FlashModels

### Getting Started with huggingface Transformers
```
torchrun --nproc_per_node=4 benchmarks/transformer.py --bf16 --acc --disable_loss_print --fsdp_size=4 --gc
```

## License
[Apache License 2.0](LICENSE)
