#!/bin/bash

set -eux

SCRIPTPATH=$(realpath $0 | xargs -n1 dirname)
source $SCRIPTPATH/config.ini

proxy=${use_proxy:+'https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890'}


function build_base_docker {
    docker build -t $base_docker                            \
                 -f $SCRIPTPATH/$base_dockerfile            \
                 --network=host                             \
                 --build-arg base_docker=$cuda_base_docker  \
                 --build-arg bazel_version=$bazel_version   \
                 --build-arg use_proxy=$use_proxy .

    echo "build base docker finished: ", ${base_docker}
}


function clone_code {
    if [ ! -d pytorch ]; then
        env ${proxy} git clone --recursive -b ${torch_branch} ${torch_git} --depth=1
        pushd pytorch
        if [ ! -d xla ]; then
          env ${proxy} git clone --recursive -b ${xla_branch} ${xla_git} --depth=1
        fi
        popd
    fi

    if [ ! -d torchacc ]; then
        env ${proxy} git clone --recursive -b ${torchacc_branch} ${torchacc_git}
    fi

    if [ ! -d torchdistx ]; then
        env ${proxy} git clone --recursive https://github.com/Seventeen17/torchdistx.git
    fi
}


function build_whls {
    mkdir -p ${cache_path}/.cache ${cache_path}/.ccache
    docker run -t --rm --net=host -v $PWD:/workspace -v ${cache_path}/.cache:/root/.cache/ -v ${cache_path}/.ccache:/root/.ccache -w /workspace $base_docker bash -c "bash /workspace/build_whls.sh"
}


function build_release_docker {
    docker build -t $docker_with_torchacc             \
                 -f $SCRIPTPATH/$release_dockerfile   \
                 --network=host                       \
                 --build-arg work_dir=$work_dir       \
                 --build-arg base_docker=$base_docker \
                 --build-arg use_proxy=$use_proxy .
}


function upload_to_oss {
    if [ -n "${OSS_ENDPOINT+x}" ] && [ -n "${OSS_AK_ID+x}" ] && [ -n "${OSS_AK_ID+x}" ]; then
        ossutil config -e ${OSS_ENDPOINT} -i ${OSS_AK_ID} -k ${OSS_AK_SECRET}
        ossutil cp -r -f -j 10 whls/ oss://pai-devel/docker_build/whls/latest/
        ossutil cp -r -f -j 10 whls/ oss://pai-devel/docker_build/whls/$(date +"%Y%m")/$(date +"%Y%m%d-%H-%M-%S")/
    else
        echo "No oss information found. Skip uploading to oss."
    fi
}


function push_to_hub {
    docker_with_torchacc_suffix="-"$(date +"%y%m%d")

    if [ "$push_to_hub" = true ]; then
        for region in ${regions}
        do
            if [ -n "${DOCKER_USERNAME+x}" ] && [ -n "${DOCKER_PASSWORD+x}" ]; then
                docker login --username=$DOCKER_USERNAME --password=$DOCKER_PASSWORD registry.cn-${region}.aliyuncs.com

                # Push to pai-dlc as latest
                docker tag ${docker_with_torchacc} registry.cn-${region}.aliyuncs.com/pai-dlc/${docker_with_torchacc}
                docker push registry.cn-${region}.aliyuncs.com/pai-dlc/${docker_with_torchacc}

                # Push to pai-dlc with date
                docker tag ${docker_with_torchacc} registry.cn-${region}.aliyuncs.com/pai-dlc/${docker_with_torchacc}${docker_with_torchacc_suffix}
                docker push registry.cn-${region}.aliyuncs.com/pai-dlc/${docker_with_torchacc}${docker_with_torchacc_suffix}
            else
                echo "No docker login information found. Skip uploading to docker hub."
            fi
        done
    fi
}


function clean_all {
    if [ "$do_clean" = true ]; then
        rm -rf pytorch torchacc torchdistx whls
    else
        echo "Did not clean up the environment"
    fi
}


time build_base_docker
echo "build_base_docker DONE."

time clone_code
echo "clone_code DONE."

time build_whls
echo "build_whls DONE."

time build_release_docker
echo "build_release_docker DONE."

time upload_to_oss
echo "upload_to_oss DONE."

time push_to_hub
echo "push_to_hub DONE."

time clean_all
echo "clean_all DONE."
