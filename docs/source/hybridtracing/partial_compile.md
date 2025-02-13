# Partial Compile
## Introduction
Partial compile in TorchAcc can be employed to achieve performance acceleration over native torch cuda execution In scenarios involving complex user code (e.g., extensive tensor evaluations, custom operations, etc.) which is hard for xla to capture whole graph. Specifically, we utilize Dynamo + XLA backend for partial compilation, with enhancements and optimizations in both functionality and performance.

Note: Partial compile runs on TorchAcc eager backend(cuda device).
## How to use
It can be specified within torchacc.accelerate by setting the config.
```Python
import torchacc as ta

config = ta.config()
config.backend.mode = 'eager'
config.backend.partial_compile = True

...

ta.accelerate(model, config=config)
```

## Sceneries
Below are the sceneries we suggest to use partial compile:
- model with custom ops which xla do not support.
- model with extensive tensor evaluations.