#!/bin/bash

# $1: local model directory
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <local_model_dir>"
  echo "You must provide exactly 1 parameters."
  exit 1
fi

if [[ -z "${MIT_SPIDER_TOKEN}" ]]; then
  echo "Error: Environment variable MIT_SPIDER_TOKEN is not set." >&2
  exit 1
fi

if [[ -z "${MIT_SPIDER_URL}" ]]; then
  echo "Error: Environment variable MIT_SPIDER_URL is not set." >&2
  exit 1
fi

MODEL_DIR=$(realpath $1)
MODEL_ID=$(basename "$MODEL_DIR")_$(date +"%Y%m%d_%H%M%S")
NUM_GPUS_TOTAL=1
JUDGMENT_PARALLEL=4
export M6_TENANT=M6

function instal_fastchat {
  if [[ ! -d "FastChat" ]]; then
    git clone https://github.com/lm-sys/FastChat.git
  fi

  # if python -c "import fschat" &>/dev/null; then
  #   echo "All requirements are installed."
  # else
  #   echo "Install requirements ..."
  #   pushd ./FastChat
  #   pip install -e ".[model_worker,llm_judge]"
  #   pip install gradio
  #   popd
  # fi
}

function run_bench {
  SCRIPT_DIR=./FastChat/fastchat/llm_judge/
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

instal_fastchat
run_bench
