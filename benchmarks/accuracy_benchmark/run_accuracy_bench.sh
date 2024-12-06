#!/bin/bash

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <local model dir>"
  echo "You must provide exactly 1 parameters."
  exit 1
fi

MODEL_DIR=$(realpath "$1")
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


# Run the torch native job
bash ./llama.sh "$MODEL_DIR" 0 $MODEL_NAME_TORCH 2>&1 | tee $TORCH_TRAIN_LOG

# Run the torchacc job
bash ./llama.sh "$MODEL_DIR" 1 $MODEL_NAME_ACC 2>&1 | tee $ACC_TRAIN_LOG

# Evaluate original checkpoint
bash ./mtbench.sh "$MODEL_DIR" 2>&1 | tee $ORIG_MODEL_EVAL_LOG

# Evaluate Torch job
bash ./mtbench.sh "$MODEL_NAME_TORCH" 2>&1 | tee $TORCH_MODEL_EVAL_LOG

# Evaluate TorchAcc job
bash ./mtbench.sh "$MODEL_NAME_ACC" 2>&1 | tee $ACC_MODEL_EVAL_LOG

# Collect and compare the results
ORIG_SCORE=$(tail -1 $ORIG_MODEL_EVAL_LOG | awk '{print $NF}')
TORCH_SCORE=$(tail -1 $TORCH_MODEL_EVAL_LOG | awk '{print $NF}')
ACC_SCORE=$(tail -1 $ACC_MODEL_EVAL_LOG | awk '{print $NF}')

torch_train_loss=$(grep -oP 'train_loss\s*=\s*\K[0-9.]*' $TORCH_TRAIN_LOG)
torch_train_runtime=$(grep -oP 'train_runtime\s*=\s*\K[0-9:.]*' $TORCH_TRAIN_LOG)
torch_train_samples_per_second=$(grep -oP 'train_samples_per_second\s*=\s*\K[0-9.]*' $TORCH_TRAIN_LOG)
acc_train_loss=$(grep -oP 'train_loss\s*=\s*\K[0-9.]*' $ACC_TRAIN_LOG)
acc_train_runtime=$(grep -oP 'train_runtime\s*=\s*\K[0-9:.]*' $ACC_TRAIN_LOG)
acc_train_samples_per_second=$(grep -oP 'train_samples_per_second\s*=\s*\K[0-9.]*' $ACC_TRAIN_LOG)


RESET='\033[0m'
RED='\033[31m'
GREEN='\033[32m'
YELLOW='\033[33m'
BLUE='\033[34m'
CYAN='\033[36m'

{
  echo -e "${BLUE}==================== Training Results ====================${RESET}"
  echo -e "${YELLOW}Torch train loss                  = ${GREEN}${torch_train_loss}${RESET}"
  echo -e "${YELLOW}TorchAcc train loss               = ${GREEN}${acc_train_loss}${RESET}"
  echo -e "${YELLOW}Torch train runtime               = ${GREEN}${torch_train_runtime}${RESET}"
  echo -e "${YELLOW}TorchAcc train runtime            = ${GREEN}${acc_train_runtime}${RESET}"
  echo -e "${YELLOW}Torch train samples per second    = ${GREEN}${torch_train_samples_per_second}${RESET}"
  echo -e "${YELLOW}TorchAcc train samples per second = ${GREEN}${acc_train_samples_per_second}${RESET}"


  echo -e "${BLUE}==================== Evaluate Results ====================${RESET}"
  echo -e "${YELLOW}Original Model Score         = ${GREEN}${ORIG_SCORE}${RESET}"
  echo -e "${YELLOW}Torch Model Score            = ${GREEN}${TORCH_SCORE}${RESET}"
  echo -e "${YELLOW}TorchAcc Model Score         = ${GREEN}${ACC_SCORE}${RESET}"

  echo -e "\n${CYAN}More details can be found in = ${RESET}${RES_FOLDER}"
  echo -e "${BLUE}==========================================================${RESET}"
} | tee >(sed 's/\x1b\[[0-9;]*m//g' > $RES_LOG_FILE)
