# Contribute To TorchAcc


TorchAcc is built on top of PyTorch/XLA, and it requires a specific version of PyTorch/XLA to
to ensure GPU compatibility and performance.
We highly recommend you to use our prebuilt Docker image to start your development work.

## Building from source
If you want to build from source, you first need to build PyTorch and torch_xla from source.

1. build PyTorch
```shell
git clone --recursive -b v2.3.0 git@github.com:AlibabaPAI/pytorch.git
cd pytorch
TORCH_CUDA_ARCH_LIST="8.0" TF_CUDA_COMPUTE_CAPABILITIES="8.0" python setup.py develop
```


2. build torch_xla
```shell
git clone --recursive -b acc git@github.com:AlibabaPAI/xla.git
cd xla
TORCH_CUDA_ARCH_LIST="8.0" TF_CUDA_COMPUTE_CAPABILITIES="8.0" USE_CUDA=1 XLA_CUDA=1 python setup.py develop
```

3. build torchacc
```shell
python setup.py develop
```