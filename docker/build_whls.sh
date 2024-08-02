#!/bin/bash
set -eux

SCRIPTPATH=$(realpath $0 | xargs -n1 dirname)
source $SCRIPTPATH/config.ini

cd ${work_dir}
mkdir -p whls

proxy=${use_proxy:+'https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890'}

git config --global --add safe.directory '*'

function build_pytorch {
    pushd ${work_dir}/pytorch
    export PYTORCH_BUILD_VERSION=2.2.2
    export PYTORCH_BUILD_NUMBER=1
    USE_CCACHE=1 TORCH_CUDA_ARCH_LIST=${cuda_compute_capabilities} \
        TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
        _GLIBCXX_USE_CXX11_ABI=0 \
        USE_DISTRIBUTED=1 \
        USE_LMDB=1 \
        USE_STATIC_NCCL=OFF \
        USE_MKLDNN=1 \
        CMAKE_BUILD_PARALLEL_LEVEL=96 \
        python setup.py ${build_mode}
    if [ "${build_mode}" = "bdist_wheel" ]; then
        env ${proxy} pip install dist/*.whl
        cp dist/*.whl ${work_dir}/whls/
    fi
    popd
}


function build_torch_xla {
    pushd pytorch/xla
    python setup.py clean
    env ${proxy} TF_CUDA_COMPUTE_CAPABILITIES="${cuda_compute_capabilities//\ /,}" \
        TORCH_CUDA_ARCH_LIST="${cuda_compute_capabilities}" \
        BUILD_CPP_TESTS=0 \
        TF_NEED_CUDA=1 \
        CXX_ABI=0 \
        cuda=1 \
        USE_CUDA=1 \
        XLA_CUDA=1 \
        XLA_BAZEL_VERBOSE=0 \
        GLIBCXX_USE_CXX11_ABI=0 \
        CFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" \
        CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" \
        python setup.py ${build_mode}
    if [ "${build_mode}" = "bdist_wheel" ]; then
        cp dist/*.whl ${work_dir}/whls/
        cp third_party/flash-attention/dist/*.whl ${work_dir}/whls/
    fi
    popd
}


function build_torchacc {
    pushd torchacc
    python setup.py ${build_mode}
    if [ "${build_mode}" = "bdist_wheel" ]; then
        cp dist/*.whl ${work_dir}/whls/
    fi
    popd
}


function build_torch_distx {
    pushd torchdistx
    TORCH_CUDA_ARCH_LIST="${cuda_compute_capabilities}" cmake -DTORCHDIST_INSTALL_STANDALONE=ON -B build
    TORCH_CUDA_ARCH_LIST="${cuda_compute_capabilities}" cmake --build build
    TORCH_CUDA_ARCH_LIST="${cuda_compute_capabilities}" python setup.py ${build_mode}
    if [ "${build_mode}" = "bdist_wheel" ]; then
        cp dist/*.whl ${work_dir}/whls/
    fi
    popd
}


time build_pytorch
echo "build_pytorch DONE."

time build_torch_xla
echo "build_torch_xla DONE."

time build_torchacc
echo "build_torchacc DONE."

time build_torch_distx
echo "build_torch_distx DONE."
