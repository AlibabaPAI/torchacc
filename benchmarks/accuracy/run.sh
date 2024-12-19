#!/bin/bash

set -e

if [ "$#" -eq 1 ]; then
  MODEL_DIR=$(realpath "$1")
elif [ "$#" -eq 0 ]; then
  MODEL_DIR="./Llama-3.2-1B"
  if [[ ! -d "$MODEL_DIR" ]]; then
    MS_CKPT_URL="https://www.modelscope.cn/models/LLM-Research/Llama-3.2-1B.git"
    git clone $MS_CKPT_URL
  fi
else
  echo "Usage: $0 <local model dir>"
  exit 1
fi


MODEL_NAME=$(basename "$MODEL_DIR")
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RES_FOLDER="./result/$TIMESTAMP"
MODEL_NAME_TORCH="$RES_FOLDER/torch_ckpt"
MODEL_NAME_ACC="$RES_FOLDER/acc_ckpt"
TORCH_TRAIN_LOG="$RES_FOLDER/torch_training.log"
ACC_TRAIN_LOG="$RES_FOLDER/acc_training.log"
ORIG_MODEL_EVAL_LOG="$RES_FOLDER/original_model_eval.log"
TORCH_MODEL_EVAL_LOG="$RES_FOLDER/torch_model_eval.log"
ACC_MODEL_EVAL_LOG="$RES_FOLDER/acc_model_eval.log"
RES_LOG_FILE="$RES_FOLDER/result.log"

mkdir -p $RES_FOLDER


function do_train {
  # Run the torch native job
  bash ./llama.sh "$MODEL_DIR" 0 $MODEL_NAME_TORCH 2>&1 | tee $TORCH_TRAIN_LOG

  # Run the torchacc job
  bash ./llama.sh "$MODEL_DIR" 1 $MODEL_NAME_ACC 2>&1 | tee $ACC_TRAIN_LOG
}

function do_evaluation {
  # Evaluate original checkpoint
  bash ./fastchat.sh "$MODEL_DIR" 2>&1 | tee $ORIG_MODEL_EVAL_LOG

  # Evaluate Torch job
  bash ./fastchat.sh "$MODEL_NAME_TORCH" 2>&1 | tee $TORCH_MODEL_EVAL_LOG

  # Evaluate TorchAcc job
  bash ./fastchat.sh "$MODEL_NAME_ACC" 2>&1 | tee $ACC_MODEL_EVAL_LOG
}


function upload_to_oss {
  if [ -n "${OSS_ENDPOINT+x}" ] && [ -n "${OSS_AK_ID+x}" ] && [ -n "${OSS_AK_ID+x}" ]; then
    if ! command -v ossutil >/dev/null 2>&1; then
      curl https://gosspublic.alicdn.com/ossutil/install.sh | bash
    fi
    ossutil config -e ${OSS_ENDPOINT} -i ${OSS_AK_ID} -k ${OSS_AK_SECRET}
    ossutil cp -r -f -j 10 --exclude "*.safetensors" --exclude="*.bin" $RES_FOLDER oss://pai-devel/benchmark/accuracy/"$TIMESTAMP"
  else
    echo "No oss information found. Skip uploading to oss."
  fi
}


function collect_and_show_results {
  # Collect and compare the results
  ORIG_SCORE=$(tail -1 $ORIG_MODEL_EVAL_LOG | awk '{print $NF}')
  TORCH_SCORE=$(tail -1 $TORCH_MODEL_EVAL_LOG | awk '{print $NF}')
  ACC_SCORE=$(tail -1 $ACC_MODEL_EVAL_LOG | awk '{print $NF}')

  TORCH_TRAIN_LOSS=$(grep -oP "'train_loss': \K[0-9.]*" $TORCH_TRAIN_LOG)
  TORCH_TRAIN_RUNTIME=$(grep -oP "'train_runtime': \K[0-9.]*" $TORCH_TRAIN_LOG)
  TORCH_TRAIN_STEPS_PER_SECOND=$(grep -oP "'train_steps_per_second': \K[0-9.]*" $TORCH_TRAIN_LOG)
  ACC_TRAIN_LOSS=$(grep -oP "'train_loss': \K[0-9.]*" $ACC_TRAIN_LOG)
  ACC_TRAIN_RUNTIME=$(grep -oP "'train_runtime': \K[0-9.]*" $ACC_TRAIN_LOG)
  ACC_TRAIN_STEPS_PER_SECOND=$(grep -oP "'train_steps_per_second': \K[0-9.]*" $ACC_TRAIN_LOG)


  RESET='\033[0m'
  RED='\033[31m'
  GREEN='\033[32m'
  YELLOW='\033[33m'
  BLUE='\033[34m'
  CYAN='\033[36m'

  {
    echo -e "\n${BLUE}==================== Training Results ====================${RESET}"
    echo -e "${YELLOW}Torch    train loss             = ${GREEN}${TORCH_TRAIN_LOSS}${RESET}"
    echo -e "${YELLOW}TorchAcc train loss             = ${GREEN}${ACC_TRAIN_LOSS}${RESET}"
    echo -e "${YELLOW}Torch    train runtime (s)      = ${GREEN}${TORCH_TRAIN_RUNTIME}${RESET}"
    echo -e "${YELLOW}TorchAcc train runtime (s)      = ${GREEN}${ACC_TRAIN_RUNTIME}${RESET}"
    echo -e "${YELLOW}Torch    train steps per second = ${GREEN}${TORCH_TRAIN_STEPS_PER_SECOND}${RESET}"
    echo -e "${YELLOW}TorchAcc train steps per second = ${GREEN}${ACC_TRAIN_STEPS_PER_SECOND}${RESET}"

    echo -e "\n${BLUE}=================== Evaluation Results ===================${RESET}"
    echo -e "${YELLOW}Original Model Score            = ${GREEN}${ORIG_SCORE}${RESET}"
    echo -e "${YELLOW}Torch    Model Score            = ${GREEN}${TORCH_SCORE}${RESET}"
    echo -e "${YELLOW}TorchAcc Model Score            = ${GREEN}${ACC_SCORE}${RESET}"

    echo -e "\n${CYAN}More details can be found in    = ${RESET}${RES_FOLDER}"
    echo -e "${BLUE}==========================================================${RESET}"
  } | tee >(sed 's/\x1b\[[0-9;]*m//g' > $RES_LOG_FILE)
}

do_train
do_evaluation
collect_and_show_results
upload_to_oss
