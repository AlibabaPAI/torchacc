# Accuracy Benchmark

## Overview

The Accuracy Benchmark evaluates the performance of TorchAcc using [FastChat](https://github.com/AlibabaPAI/FastChat_TorchAcc) against a baseline established by Torch native. The benchmark aims to ensure that TorchAcc maintains comparable accuracy levels with Torch native.

## Evaluation Process

To perform the evaluation, follow these steps:

1. Set Baseline

    ```bash
    bash ./llama.sh <ORIGINAL_MODEL_DIR> 0
    ```

    Run the Torch native job using `run_clm.py`, a script copied from HuggingFace Transformers.

2. Run TorchAcc

    ```bash
    bash ./llama.sh <ORIGINAL_MODEL_DIR> 1
    ```

    Run the TorchAcc job using the same script as used for Torch native.

3. Evaluate Original

    ```bash
    bash ./mtbench.sh <ORIGINAL_MODEL_DIR>
    ```

    Evaluate the original checkpoint using FastChat.

4. Evaluate Outputs

    ```bash
    bash ./mtbench.sh <TORCH_NATIVE_CHECKPOINT>
    bash ./mtbench.sh <TORCHACC_CHECKPOINT>
    ```

    Evaluate the checkpoints output by Torch native job and TorchAcc.

5. Compare Results

    Compare the training and evaluation results.


You can simply execute the `run.sh` script to perform all the steps.

## Main Files

* run.sh

    The script integrates all the steps.

    ```bash
    bash ./run.sh <local_model_dir>
    ```

* llama.sh

    The script runs llama job using `run_clm.py` with either Torch native or TorchAcc.

    ```bash
    bash ./llama.sh <local_model_dir> <use_torchacc> [checkpiont_output_dir]
    ```

* fastchat.sh

    The script runs the evaluation task on your checkpoint.

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