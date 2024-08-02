# Installation

## Docker images

It is recommended to use the existing release image directly. The image address is:

```bash
registry.<region>.aliyuncs.com/pai-dlc/acc:r2.3.0-cuda12.1.0-py3.10-nightly
```

Replace `<region>` with one of the following as needed:

* cn-hangzhou
* cn-wulanchabu


## Building from Source

Building from source requires compiling three code repositories: pytorch, torch_xla and torchacc.

1. Set up the environment

You need to use a CUDA-enabled image such as `nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04`. Install C++, Bazel, and Python dependencies as specified in the `docker/Dodckerfile.base` file.

Alternatively, you can build directly using our release image mentioned earlier.

2. Clone the code

Clone the following three repositories and organize them as shown:

```bash
# the code structure
# pytorch/
# |---xla/
# torchacc/

git clone https://github.com/AlibabaPAI/torchacc.git
git clone https://github.com/AlibabaPAI/pytorch.git
cd pytorch
git clone https://github.com/AlibabaPAI/xla.git
```

3. Compile pytorch

```bash
cd pytorch && python setup.py develop
```

4. Compile torch_xla

```bash
cd xla && python setup.py develop
```

5. Compile torchacc

```bash
cd torchacc && python setup.py develop
```