# Accuracy Benchmark

## Overview

The Accuracy Benchmark evaluates the performance of TorchAcc using [FastChat](https://github.com/AlibabaPAI/FastChat_TorchAcc) against a baseline established by Torch native. The benchmark aims to ensure that TorchAcc maintains comparable accuracy levels with Torch native.

## Evaluation Process

To perform the evaluation, follow these steps:

1. Set Baseline

    ```bash
    bash ./llama.sh <ORIGINAL_MODEL_DIR> 0
    ```

    Run the Torch native job using `run_clm.py`, a script copied from HuggingFace Transformers. `ORIGINAL_MODEL_DIR` is the path to the original model checkpoint downloaded from HuggingFace or ModelScope. `0` indicates that this training job does not use `torchacc`.

2. Run TorchAcc

    ```bash
    bash ./llama.sh <ORIGINAL_MODEL_DIR> 1
    ```

    Run the TorchAcc job using the same script as used for Torch native. `ORIGINAL_MODEL_DIR` is the path to the original model checkpoint downloaded from HuggingFace or ModelScope. `1` indicates that this training job uses `torchacc`.


3. Evaluate Original

    ```bash
    bash ./mtbench.sh <ORIGINAL_MODEL_DIR>
    ```

    Evaluate the original checkpoint using FastChat. `ORIGINAL_MODEL_DIR` is the path to the original model checkpoint downloaded from HuggingFace or ModelScope.

4. Evaluate Outputs

    ```bash
    bash ./mtbench.sh <TORCH_NATIVE_CHECKPOINT>
    bash ./mtbench.sh <TORCHACC_CHECKPOINT>
    ```

    Evaluate the checkpoints output by Torch native job and TorchAcc. `TORCH_NATIVE_CHECKPOINT` is the path to the model checkpoint output by torch native job. `TORCHACC_CHECKPOINT` is the path to the model checkpoint output by torchacc job.

5. Compare Results

    Compare the training and evaluation results.


You can simply execute the `run.sh` script to perform all the steps.


## Main Files


All the files used in the accuracy benchmark are listed below.

* run.sh

    The script integrates all the steps.

    ```bash
    bash ./run.sh [local_model_dir]
    ```

    You could pass the local model checkpoint path to the script. If no local path is specified, it will download `llama-3.2-1B` from ModelScope.

* llama.sh

    ```bash
    # Usage: $0 <local_model_dir> <use_torchacc> [checkpiont_output_dir]
    #  local_model_dir: Path to the local directory where the model will be saved.
    #  use_torchacc: 0 or 1 to indicate whether to use TorchAcc.
    #  checkpoint_output_dir: Optional. Default is the model name in <local_model_dir>.
    bash ./llama.sh <local_model_dir> <use_torchacc> [checkpiont_output_dir]
    ```

    The script runs the llama training job using `run_clm.py` with either Torch native or TorchAcc.

* fastchat.sh

    The script runs the evaluation task on your checkpoint. The `ENV_VARIABLES` can be obtained from the maintainers of TorchAcc.

    ```bash
    ENV_VARIABLES bash ./fastchat.sh <local_model_dir>
    ```

## Evaluation Results

The evaluation results are shown as follows:

```

==================== Training Results ====================
Torch train loss                = 2.091632914827291
TorchAcc train loss             = 2.0917317353245495
Torch train runtime (s)         = 2552.8252
TorchAcc train runtime (s)      = 2272.1399
Torch train steps per second    = 5.785
TorchAcc train steps per second = 6.5

=================== Evaluation Results ===================
Original Model Score            = 1.4625
Torch Model Score               = 1.1125
TorchAcc Model Score            = 1.100629

More details can be found in    = ./result/20241205_223009
==========================================================

```