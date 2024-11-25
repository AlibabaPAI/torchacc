#!/bin/bash

# $1: mtbench url or local directory
# $2: local model directory
# $3: an unique benchmark id

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <url_or_directory> <local_model_dir> <unique_bench_id>"
  echo "You must provide exactly 3 parameters."
  exit 1
fi

INPUT="$1"
MODEL_DIR="$2"
BENCH_ID="$3"
MT_BENCH=None
TEMP_DIR="./temp"
mkdir -p "$TEMP_DIR"

function get_mtbench {
  if [[ "$INPUT" == http* ]] && [[ "$INPUT" == *.tar.gz ]]; then
    FILE_NAME=$(basename "$INPUT")
    DOWNLOAD_PATH="$TEMP_DIR/$FILE_NAME"

    if [ -f "$DOWNLOAD_PATH" ]; then
      echo "File already downloaded: $DOWNLOAD_PATH"
    else
      echo "Downloading tar.gz file with wget: $INPUT"
      wget -q -P "$TEMP_DIR" "$INPUT"
    fi

    EXTRACTED_DIR=$(find "$TEMP_DIR" -mindepth 1 -maxdepth 1 -type d | head -n 1)
    if [ -n "$EXTRACTED_DIR" ]; then
      echo "File already extracted: $EXTRACTED_DIR"
    else
      echo "Extracting file to $TEMP_DIR"
      tar -xzf "$DOWNLOAD_PATH" -C "$TEMP_DIR"
      EXTRACTED_DIR=$(find "$TEMP_DIR" -mindepth 1 -maxdepth 1 -type d | head -n 1)
    fi
    MT_BENCH="$EXTRACTED_DIR"
  elif [ -d "$INPUT" ]; then
    echo "Listing directory: $INPUT"
    MT_BENCH="$INPUT"
  else
    echo "Invalid input. Provide a valid URL or an existing directory."
    exit 2
  fi

  echo "------------"
  echo $MT_BENCH
  ls $MT_BENCH
}


function install_requirements {
  if python -c "import gradio" &> /dev/null; then
    echo "All requirements are installed."
  else
    echo "Install requirements ..."
    pushd "$MT_BENCH/FastChat/"
    pip install -e ".[model_worker,llm_judge]"
    pip install gradio
    popd
  fi
}

function run_bench {
  echo
  SCRIPT_DIR="$MT_BENCH"/FastChat/fastchat/llm_judge/
  if [[ ! -d "$SCRIPT_DIR" ]]; then
    echo "Directory $SCRIPT_DIR is not exist."
    exit 1
  fi
  if [[ ! -d "$MODEL_DIR" ]]; then
    echo "Directory $MODEL_DIR is not exist."
    exit 1
  fi

  pushd $SCRIPT_DIR
  bash run_judge.sh $MODEL_DIR $BENCH_ID
  popd
}

get_mtbench
install_requirements
run_bench
