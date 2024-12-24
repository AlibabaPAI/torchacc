#!/bin/bash

set -e

if [ "$#" -ne 1 ]; then
  echo "Usage: MIT_SPIDER_TOKEN=*** MIT_SPIDER_URL=*** M6_TENANT=*** $0 <local_model_dir>"
  echo "You must provide exactly 1 parameters."
  exit 1
fi

if [[ -z "${MIT_SPIDER_TOKEN}" || -z "${MIT_SPIDER_URL}" || -z "${M6_TENANT}" ]]; then
  echo "Error: One or more required environment variables are not set."
  echo "Required variables:"
  [[ -z "${MIT_SPIDER_TOKEN}" ]] && echo "  - MIT_SPIDER_TOKEN"
  [[ -z "${MIT_SPIDER_URL}" ]] && echo "  - MIT_SPIDER_URL"
  [[ -z "${M6_TENANT}" ]] && echo "  - M6_TENANT"
  exit 1
fi

MODEL_DIR=$(realpath $1)
MODEL_ID=$(basename "$MODEL_DIR")_$(date +"%Y%m%d_%H%M%S")
NUM_GPUS_TOTAL=4
JUDGMENT_PARALLEL=4

function install_fastchat {
  if [[ ! -d "FastChat_TorchAcc" ]]; then
    git clone https://github.com/AlibabaPAI/FastChat_TorchAcc.git
  fi

  if python -m pip list | grep -q fschat; then
    echo "All requirements are installed."
  else
    echo "Install requirements ..."
    pushd ./FastChat_TorchAcc
    pip install --use-pep517 -e ".[model_worker,llm_judge]"
    pip install gradio
    popd
  fi
}

function run_bench {
  SCRIPT_DIR=./FastChat_TorchAcc/fastchat/llm_judge/
  if [[ ! -d "$SCRIPT_DIR" ]]; then
    echo "Directory $SCRIPT_DIR is not exist."
    exit 1
  fi
  if [[ ! -d "$MODEL_DIR" ]]; then
    echo "Directory $MODEL_DIR is not exist."
    exit 1
  fi

  cd $SCRIPT_DIR

  echo "====gen start===="
  python gen_model_answer.py --model-path $MODEL_DIR --model-id $MODEL_ID --num-gpus-total $NUM_GPUS_TOTAL
  echo "====gen done===="

  echo "====judge start===="
  python gen_judgment.py --model-list $MODEL_ID --parallel $JUDGMENT_PARALLEL
  echo "====judge done===="

  echo "====show score===="
  # python show_result.py --model-list $MODEL_ID
  python show_result_by_category.py --model-list $MODEL_ID

}

install_fastchat
run_bench
