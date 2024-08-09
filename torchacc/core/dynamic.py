from typing import List, Union

import torch
from torch._dispatch.python import enable_python_dispatcher
import torch_xla

import torchacc as ta

def mark_dynamic(x: torch.Tensor, dims: Union[List[int], int], bounds: Union[List[int], int]):
    """Mark a tensor as having dynamic dims and set corresponding upper bounds for the dims.
    Args:
        x (torch.Tensor): input tensor
        dims (Union[List[int], int]): the dim marked as dynamic, can be a single dim or
            multiple dims
        bounds (Union[List[int], int]): upper bounds corresponding to the dynamic dims
    """
    if ta.get_global_context().python_dispatcher is None:
        ta.get_global_context().python_dispatcher = enable_python_dispatcher()
    if isinstance(dims, int):
        dims = [dims]
        bounds = [bounds]
    assert isinstance(dims, list), "dims should be of int or list type"
    assert isinstance(bounds, list), "bounds should be of int or list type"
    for i, dim in enumerate(dims):
        if dim < (-x.dim()) or dim >= x.dim():
            raise ValueError(f"Dimension out of range (expected to be in range" \
                             f" of [{-x.dim()}, {x.dim()-1}], but got {dim})")
        if dim < 0:
            dims[i] = x.dim() + dim
    for dim, bound in zip(dims, bounds):
        if bound < x.size(dim):
            raise ValueError(f"The upper bound of the shape size {bound} is less" \
                             f" than the current size {x.size(dim)}")
    torch_xla._XLAC._mark_dynamic(x, dims, bounds)
