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
MODEL_NAME_TORCH="$MODEL_NAME"_torch
MODEL_NAME_ACC="$MODEL_NAME"_acc
TORCH_TRAIN_LOG="$RES_FOLDER/torch_training.log"
ACC_TRAIN_LOG="$RES_FOLDER/acc_training.log"
ORIG_MODEL_EVAL_LOG="$RES_FOLDER/original_model_eval.log"
TORCH_MODEL_EVAL_LOG="$RES_FOLDER/torch_model_eval.log"
ACC_MODEL_EVAL_LOG="$RES_FOLDER/acc_model_eval.log"

mkdir -p $RES_FOLDER

##### Reinstall transformers #####
git clone https://github.com/huggingface/transformers.git 2>/dev/null || echo "Directory already exists, skipping clone."
pushd transformers
TRANSFORMERS_DIR=$(realpath "./")
if ! pip list 2>/dev/null | grep -q "$TRANSFORMERS_DIR"; then
  pip uninstall -y transformers
  pip install -e .
fi
popd

##### Run the torch native job ######
bash ./llama_torch.sh "$TRANSFORMERS_DIR" "$MODEL_DIR" 2>&1 | tee $TORCH_TRAIN_LOG

##### Run the torchacc job ######
bash ./llama_acc.sh "$TRANSFORMERS_DIR" "$MODEL_DIR" 2>&1 | tee $ACC_TRAIN_LOG

##### Evaluate original job ######
bash ./mtbench.sh "$MODEL_DIR" 2>&1 | tee $ORIG_MODEL_EVAL_LOG

##### Evaluate Torch job ######
bash ./mtbench.sh "./$MODEL_NAME_TORCH" 2>&1 | tee $TORCH_MODEL_EVAL_LOG

##### Evaluate TorchAcc job ######
bash ./mtbench.sh "./$MODEL_NAME_ACC" 2>&1 | tee $ACC_MODEL_EVAL_LOG

##### Collect and compre the result ######
ORIG_SCORE=$(tail -1 $ORIG_MODEL_EVAL_LOG | awk '{print $NF}')
TORCH_SCORE=$(tail -1 $TORCH_MODEL_EVAL_LOG | awk '{print $NF}')
ACC_SCORE=$(tail -1 $ACC_MODEL_EVAL_LOG | awk '{print $NF}')

RESET='\033[0m'
RED='\033[31m'
GREEN='\033[32m'
YELLOW='\033[33m'
BLUE='\033[34m'
CYAN='\033[36m'

# Print colored output
echo -e "${BLUE}==================== Final Results ====================${RESET}"
echo -e "${YELLOW}Original Model Score        : ${GREEN}${ORIG_SCORE}${RESET}"
echo -e "${YELLOW}Torch Model Score           : ${GREEN}${TORCH_SCORE}${RESET}"
echo -e "${YELLOW}TorchAcc Model Score        : ${GREEN}${ACC_SCORE}${RESET}"
echo -e "\n${CYAN}More details can be found in: ${RESET}${RES_FOLDER}"
echo -e "${BLUE}=======================================================${RESET}"
