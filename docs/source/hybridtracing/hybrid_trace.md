# Hybrid Trace
## Introduction
The Hybrid Trace approach addresses performance degradation issues that arise when XLA encounters tensor evaluation. In this solution, we combine the graph capture capabilities of Dynamo and Lazy Tensor Core (LTC). The model runs entirely on the XLA device, following LTC's execution logic, while locally employing Dynamo to reduce tracing overhead. This strategy retains the potential for full-graph optimization with XLA.

Note: Hybrid Trace runs on torchacc lazy backend(xla device).
## How to use
It can be specified within torchacc.accelerate by setting the config.
```Python
import torchacc as ta

config = ta.config()
config.backend.mode = 'lazy'
config.backend.hybrid_trace = True

...

ta.accelerate(model, config=config)
```

## Sceneries
Below are the sceneries we suggest to use hybrid trace:
- language model with tensor evaluations like qwen and llama.

## Performance
