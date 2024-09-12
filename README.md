# TorchAcc

**TorchAcc** is a PyTorch distributed training acceleration framework provided by Alibaba Cloud's PAI platform.

TorchAcc leverages the work of the [PyTorch/XLA](https://github.com/pytorch/xla) to provide users with training acceleration capabilities. At the same time, we have conducted a considerable amount of targeted optimization based on GPU. TorchAcc offers better usability, superior performance, and richer functionality.

## Highlighted Features

The key features of TorchAcc:

* Rich distributed Parallelism
    * Data Parallelism
    * Fully Sharded Data Parallelism
    * Tensor Parallelism
    * Pipeline Parallelism
    * [Ulysess](https://arxiv.org/abs/2309.14509)
    * [Ring Attention](https://arxiv.org/abs/2310.01889)
    * Flash Sequence (Solution for Long Sequence)
* Low Memory Cost
* High Performance
* Ease use

## Installation

### Build from source
1. Build
```
python setup.py install
```

2. UT
```
sh tests/run_ut.sh
```

## License
[Apache License 2.0](LICENSE)
