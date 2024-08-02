# Flash Models

FlashModels is an algorithm library integrated with TorchAcc, allowing you to quickly get started with TorchAcc. Below is an example of running an `FSDP` task for the `Llama3-8B` model, while also enabling `Gradient Checkpoint`:

```bash
git clone https://github.com/AlibabaPAI/FlashModels
cd FlashModels
bash ./examples/run.sh --model ./hf_models/config/llama-3-8b --accelerator acc --gc --mbs 2 --fsdp 8
```