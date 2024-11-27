#!/bin/bash

# 0. reinstall transformers
# 1. Run torch native job
# 2. Run torchacc job
# 3. evaluate original model
# 4. evaluate torch native model
# 5. evaluate torchacc model
# 6. Collect and compare the result


# $1: local model directory
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
git clone https://github.com/huggingface/transformers.git
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
ORIG_SCORE=$(tail -1 $TORCH_MODEL_EVAL_LOG | awk {print $2})
