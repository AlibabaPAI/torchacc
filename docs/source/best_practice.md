# Best Practices

## Minimize the Calls to `mark_step`

When using `AsyncLoader`, which already contains an internal `mark_step`, additional calls to `mark_step()` are generally unnecessary and can cause redundant synchronization. In other scenarios, avoid excessive calls to `mark_step` whenever possible.


## Prefer `AsyncLoader`

Use `AsyncLoader` instead of manually transferring I/O tensors to `lazy_device`.


## Avoid Evaluating Tensors
Evaluating tensors can impact performance. Operations that trigger tensor evaluation include:

- Printing tensors
- Calling the `item` method on a tensor
- Using tensor values in dynamic control flow for branch logic


## Coordinate `Gradient Accumulation` with `mark_step` and `AsyncLoader`

When using `Gradient Accumulation`, adjust the `batches_per_execution` parameter in `AsyncLoader` to match the GA minibatch count N. This ensures `mark_step` is executed once after N minibatches. Additionally, consider the memory overhead in this scenario; if it's too high, you may need to execute `mark_step` after each minibatch.


## Model Saving

For robust model reloading during continued training, save the model by first transferring it to CPU with `model.to('cpu')` before calling the save operation.
