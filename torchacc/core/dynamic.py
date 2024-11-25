from typing import List, Union

import torch
from torch._dispatch.python import enable_python_dispatcher

import torchacc as ta

from torchacc.utils.import_utils import is_torch_xla_available
if is_torch_xla_available():
    import torch_xla


def mark_dynamic(x: torch.Tensor, dims: Union[List[int], int],
                 bounds: Union[List[int], int]):
    """Mark a tensor as having dynamic dims and set corresponding upper bounds for the dims.
    Args:
        x (torch.Tensor): input tensor
        dims (Union[List[int], int]): the dim marked as dynamic, can be a single dim or
            multiple dims
        bounds (Union[List[int], int]): upper bounds corresponding to the dynamic dims
    """
    if ta.get_global_context().python_dispatcher is None:
        # Some symint operators require this in order for the operator dispatch
        # to work properly
        ta.get_global_context().python_dispatcher = enable_python_dispatcher()
    if isinstance(dims, int):
        assert isinstance(bounds, int), "bounds should be of type int"
        dims = [dims]
        bounds = [bounds]
    assert isinstance(dims, list), "dims should be of type int or list of int"
    assert isinstance(bounds,
                      list), "bounds should be of type int or list of int"
    for i, dim in enumerate(dims):
        assert isinstance(dim, int), "dims should be of type int or list of int"
        if dim < (-x.dim()) or dim >= x.dim():
            raise ValueError(f"Dimension out of range (expected to be in range" \
                             f" of [{-x.dim()}, {x.dim()-1}], but got {dim})")
        if dim < 0:
            dims[i] = x.dim() + dim
    for dim, bound in zip(dims, bounds):
        assert isinstance(bound,
                          int), "bounds should be of type int or list of int"
        if bound < x.size(dim):
            raise ValueError(f"The upper bound of the shape size {bound} is less" \
                             f" than the current size {x.size(dim)}")
    torch_xla._XLAC._xla_mark_bounded_dynamic(x, dims, bounds)
