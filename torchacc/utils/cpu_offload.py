# This file is largely inspired by and partially follows the structure of
# ``transformer_engine.pytorch.cpu_offload`` in
# https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/cpu_offload.py
"""Functionality for CPU offloading of tensors saved for backward pass."""
import math
import warnings
from typing import Any

import torch

import torchacc as ta

from .utils import apply_to_tensors

__all__ = ['get_cpu_offload_context']


class CpuOffloadSavedTensorHook:
    """Contex-manager that executes a pair of pack/unpack hooks for saved tensors.

    In this context, the ``on_save_for_backward`` method will be called every time
    a tensor is saved for backward (this includes intermediary results saved using
    :func:`~torch.autograd.function._ContextMethodMixin.save_for_backward` but
    also those recorded by a PyTorch-defined operation).

    The ``on_get_saved_tensors`` method will be called when the backward function
    of this op attempts to retrieve the saved tensor from context (this includes
    :func: `torch.Tensor.backward()` or :func: `torch.autograd.grad()`. It takes the
    as input the return value of the ``on_save_for_backward``, and is meant to return
    an identical copy of the tensor being saved by ``on_save_for_backward`` in terms of
    size, device and element values.

    Example:

        >>> import torch
        >>> from typing import Any
        >>>
        >>> class DummyHook(CpuOffloadSavedTensorHook):
        ...
        ...     def on_save_for_backward(self, tensor: torch.Tensor) -> Any:
        ...         logging.info("On save", tensor)
        ...         return (tensor,)
        ...
        ...     def on_get_saved_tensor(self, saved_state: Any) -> torch.Tensor:
        ...         logging.info("On get", saved_state)
        ...         tensor, = saved_state
        ...         return tensor
        ...
        >>> a = torch.ones(5, requires_grad=True)
        >>> b = torch.ones(5, requires_grad=True) * 2
        >>> with DummyHook():
        ...     y = a * b
        ...
        On save tensor([1., 1., 1., 1., 1.], requires_grad=True)
        On save tensor([2., 2., 2., 2., 2.], grad_fn=<MulBackward0>)
        >>> y.sum().backward()
        On get (tensor([1., 1., 1., 1., 1.], requires_grad=True),)
        On get (tensor([2., 2., 2., 2., 2.], grad_fn=<MulBackward0>),)

    """

    def __init__(self) -> None:
        self.inside_context = False

    def __enter__(self):
        if not torch.is_grad_enabled():
            return

        self.inside_context = True
        torch._C._autograd._push_saved_tensors_default_hooks(
            self.on_save_for_backward, self.on_get_saved_tensor)

    def __exit__(self, *args: Any):
        if not torch.is_grad_enabled():
            return

        self.inside_context = False
        torch._C._autograd._pop_saved_tensors_default_hooks()

    def on_save_for_backward(self, tensor: torch.Tensor) -> Any:
        """On save for backward."""
        raise NotImplementedError(
            "`on_save_for_backward: Callable[[torch.Tensor], Any]`"
            "is not implemented in CpuOffloadHook class. Inherit "
            "this class and implement your custom hooks")

    def on_get_saved_tensor(self, saved_state: Any) -> torch.Tensor:
        """On get saved tensor."""
        raise NotImplementedError(
            "`on_get_saved_tensors: Callable[[Any], torch.Tensor]`"
            "is not implemented in CpuOffloadHook class. Inherit "
            "this class and implement your custom hooks")


class CpuOffloadHookWithOffloadHandler(CpuOffloadSavedTensorHook):
    """Context-manager that offloads/recovers tensors through an offload hander.

    The hook just offloads/recovers the tensor object to the handler through `tensor_push`
    and `tensor_pop` interface. How the offload-handler manages the offloading, recovering
    or prefetching timing is transparent to this hook.
    """

    def __init__(
            self,
            offload_handler,
            handler_extra_kwargs={},
            debug=False) -> None:  # pylint: disable=dangerous-default-value
        self.debug = debug
        self.offload_handler = offload_handler
        self.handler_extra_kwargs = handler_extra_kwargs
        super().__init__()

    def on_save_for_backward(self, tensor: torch.Tensor) -> Any:
        retrieve_identifier = self.offload_handler.tensor_push(
            tensor, **self.handler_extra_kwargs)
        return retrieve_identifier

    def on_get_saved_tensor(self, saved_state: Any) -> torch.Tensor:
        tensor = self.offload_handler.tensor_pop(saved_state,
                                                 **self.handler_extra_kwargs)
        return tensor


class OffloadHandler:
    """A base class for CPU offload-handler."""

    def __init__(self) -> None:
        pass

    def tensor_push(self, tensor: torch.Tensor, **kwargs) -> Any:
        """Tensor push."""
        raise NotImplementedError(
            "`tensor_push is not implented in OffloadHandler class. "
            "Inherit this class and implement your custom tensor_push.")

    def tensor_pop(self, tensor_tag: Any, **kwargs):
        """Tensor pop."""
        raise NotImplementedError(
            "`tensor_pop is not implented in OffloadHandler class. "
            "Inherit this class and implement your custom tensor_pop.")


class GroupCommitFunction(torch.autograd.Function):
    """this is a dummy op with output identical to input.
    However, it is necessary for marking a timepoint for offload handler to
    accomplish all synchronizations. Implementing it as a function is necessary
    because we need to actions in both forward and backward.
    """

    @staticmethod
    def forward(ctx, tensor, cpu_offload_handler):
        cpu_offload_handler.on_group_commit_forward()
        ctx.cpu_offload_handler = cpu_offload_handler
        # return the identical tensor
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        cpu_offload_handler = ctx.cpu_offload_handler
        cpu_offload_handler.on_group_commit_backward()
        return grad_output, None


group_prefetch_offload_commit = GroupCommitFunction.apply


class SynchronizedGroupOffloadHandler(OffloadHandler):
    """Offload Handler that offloads/reloads in a synchronized way.
    The device-to-host and host-to-device copying happen in the same stream
    as the computation kernels, thus the copying will block computation.
    """

    def __init__(self,
                 num_offload_group,
                 tensor_need_offloading_checker=(lambda _: True),
                 debug=False) -> None:
        super().__init__()

        self.num_offload_group = num_offload_group
        self.tensor_need_offloading_checker = tensor_need_offloading_checker
        self.debug = debug

        self.groupid_reset()

    def groupid_reset(self):
        """Groupid reset."""
        # Data structures to label saved tensors and book-keep their cpu copies.
        # Currently, on push, create a new cpu tensor and copies; on pop, copies
        # the tensor back to gpu and deletes the cpu tensor.
        # These will increment whenever `group_commit()` is invoked
        self.current_group, self.tensor_count_current_group = (0, 0)
        self.torch_tensor_count = 0
        self.tensor_tag_to_state = {}

    def on_group_commit_forward(self):
        """On group commit forward."""
        # finishing up with updating current group and tensor count
        self.current_group += 1  # increment
        self.tensor_count_current_group = 0  # reset

    def on_group_commit_backward(self):
        """On group commit backward."""
        self.current_group -= 1
        assert self.current_group >= 0

    @staticmethod
    def offload(src_tensor, pin_memory=True):
        """Offload."""

        cpu_backup = torch.empty(
            src_tensor.size(),
            dtype=src_tensor.dtype,
            layout=src_tensor.layout,
            device="cpu",
            pin_memory=pin_memory)

        cpu_backup.copy_(src_tensor, non_blocking=pin_memory)
        state = (src_tensor.device, cpu_backup)
        return state

    @staticmethod
    def reload(state, non_blocking=None):
        """Reload."""
        dev, cpu_backup = state
        if non_blocking is None:
            non_blocking = cpu_backup.is_pinned()
        return cpu_backup.to(dev, non_blocking=non_blocking)

    def tensor_push(self, tensor: torch.Tensor, **kwargs):
        """Tensor push."""
        # obtain a unique tensor tag
        tensor_tag = (self.current_group, self.tensor_count_current_group)
        self.tensor_count_current_group += 1
        assert tensor_tag not in self.tensor_tag_to_state
        if (self.current_group < self.num_offload_group and
                self.tensor_need_offloading_checker(tensor)):
            state = SynchronizedGroupOffloadHandler.offload(tensor)
            self.tensor_tag_to_state[tensor_tag] = state
        else:
            # will be offloaded together after group commit
            self.tensor_tag_to_state[tensor_tag] = tensor
        return tensor_tag

    def tensor_pop(self, tensor_tag, **kwargs):
        """Tensor pop."""
        assert tensor_tag in self.tensor_tag_to_state
        state = self.tensor_tag_to_state.pop(tensor_tag)
        if isinstance(state, tuple):
            tensor = SynchronizedGroupOffloadHandler.reload(state)
        else:
            tensor = state
        return tensor


class TensorState:

    def __init__(self, tensor) -> None:
        self.tensor = tensor
        self.tensor_meta = [(tensor.size(), tensor.dtype)]
        self.ref_cnt = 1
        self.device = tensor.device
        self.offloaded = False
        self.reloaded = False
        self.prefetch_buffer = None

    def add_tensor(self, tensor) -> None:
        assert tensor.data_ptr() == self.tensor.data_ptr()
        assert tensor.dtype == self.tensor.dtype
        self.ref_cnt += 1
        self.tensor_meta.append((tensor.size(), tensor.dtype))

    def get_ref_cnt(self) -> int:
        return self.ref_cnt

    def get_tensor(self) -> torch.Tensor:
        return self.tensor

    def get_reloaded_tensor(self, ref_cnt) -> torch.Tensor:
        self.ref_cnt -= 1
        assert self.ref_cnt >= 0
        assert ref_cnt >= 1 and ref_cnt <= len(self.tensor_meta)
        assert self.prefetch_buffer is not None and self.reloaded
        return self.prefetch_buffer.view(self.tensor_meta[ref_cnt - 1][0])

    def offload(self, pin_memory=True) -> None:
        assert not self.offloaded
        self.cpu_backup = torch.empty(
            self.tensor.size(),
            dtype=self.tensor.dtype,
            layout=self.tensor.layout,
            device="cpu",
            pin_memory=pin_memory)
        self.cpu_backup.copy_(self.tensor, non_blocking=pin_memory)
        self.tensor = None
        self.offloaded = True

    def create_prefetch_buffer(self) -> None:
        self.prefetch_buffer = torch.empty(
            self.cpu_backup.size(),
            dtype=self.cpu_backup.dtype,
            layout=self.cpu_backup.layout,
            device=self.device)

    def reload(self) -> None:
        assert not self.reloaded and self.offloaded
        self.prefetch_buffer.copy_(self.cpu_backup, non_blocking=True)
        self.reloaded = True


class AsyncDoubleBufferGroupOffloadHandler(SynchronizedGroupOffloadHandler):
    """Compared to synchronize, this uses more memory because of the buffer but
    achieves better performance due to the overlapping. D2h and h2d copying are
    completely hidden behind computation if computation time of a layer is longer
    than host-device communication time. Bulk offloading with delay and bulk reloading
    with prefetch are implemented. """

    def __init__(
            self,
            num_offload_group,  # must be <= actual number of groups (number of commits)
            num_prefetch_group=1,
            num_offload_sync_group=3,
            tensor_need_offloading_checker=(lambda t: True),
            debug=False) -> None:
        super().__init__(
            num_offload_group=num_offload_group,
            tensor_need_offloading_checker=tensor_need_offloading_checker,
            debug=debug)
        self.num_prefetch_group = num_prefetch_group
        self.num_offload_sync_group = num_offload_sync_group

        self.offloaded_tensor_buffers = [[] for _ in range(num_offload_group)]

        # allocate streams and events for synchronization
        self.d2h_stream = torch.cuda.Stream()
        self.h2d_stream = torch.cuda.Stream()
        self.h2d_finish_events = []
        self.d2h_finish_events = []
        self.compute_stream_bwd_start_events = []
        for _ in range(self.num_offload_group):
            self.h2d_finish_events.append(torch.cuda.Event())
            self.d2h_finish_events.append(torch.cuda.Event())
            self.compute_stream_bwd_start_events.append(torch.cuda.Event())

        self.total_offload_size = 0

    def tensor_push(self, tensor: torch.Tensor, **kwargs) -> Any:
        if self.tensor_need_offloading_checker(tensor):
            # obtain a unique tensor tag
            if self.current_group < self.num_offload_group:
                # We use the data_ptr as the tensor tag to eliminate some views of the same tensor.
                # It's worth noting that we preserve tensors, so the data pointers of different tensors
                # within the same group will not be identical.
                tensor_tag = (self.current_group, tensor.data_ptr())
                if tensor_tag not in self.tensor_tag_to_state:
                    self.tensor_tag_to_state[tensor_tag] = TensorState(tensor)
                    if self.debug and (not torch.distributed.is_initialized() or
                                       torch.distributed.get_rank() == 0):
                        import traceback
                        print(
                            f"Offloading: shape: {tensor.shape}, dtype: {tensor.dtype}, from:"
                        )
                        print(' '.join(traceback.format_stack()))
                else:
                    self.tensor_tag_to_state[tensor_tag].add_tensor(tensor)
                tensor_tag = (
                    tensor_tag,
                    self.tensor_tag_to_state[tensor_tag].get_ref_cnt())
            else:
                tensor_tag = (self.current_group,
                              self.tensor_count_current_group)
                self.tensor_count_current_group += 1
                assert tensor_tag not in self.tensor_tag_to_state
                self.tensor_tag_to_state[tensor_tag] = tensor
        else:
            tensor_tag = (-1, self.torch_tensor_count)
            self.torch_tensor_count += 1
            self.tensor_tag_to_state[tensor_tag] = tensor

        return tensor_tag

    def tensor_pop(self, tensor_tag, **kwargs):
        """Tensor pop."""
        if isinstance(tensor_tag[0], tuple):
            tensor_tag, ref_cnt = tensor_tag
        assert tensor_tag in self.tensor_tag_to_state
        tensor_or_state = self.tensor_tag_to_state.pop(tensor_tag)
        if isinstance(tensor_or_state, TensorState):
            tensor = tensor_or_state.get_reloaded_tensor(ref_cnt)
            if tensor_or_state.get_ref_cnt() > 0:
                self.tensor_tag_to_state[tensor_tag] = tensor_or_state
        else:
            tensor = tensor_or_state
        # the tensor should have been copied back in on_group_commit_backward()
        # which invokes bulk_reload_group.
        assert not isinstance(tensor, TensorState)
        return tensor

    def bulk_offload_group(self, group_to_offload):
        # the copying of this group should wait for the computation stream
        self.d2h_stream.wait_stream(torch.cuda.current_stream())
        """Bulk offload group."""
        with torch.cuda.stream(self.d2h_stream):
            for tensor_tag, state in self.tensor_tag_to_state.items():
                group_id, _ = tensor_tag
                if group_id == group_to_offload and isinstance(
                        state, TensorState):
                    tensor_on_device = state.get_tensor()
                    assert self.tensor_need_offloading_checker(tensor_on_device)
                    # if offload, return the reference to cpu copy
                    if tensor_on_device is not None:
                        self.total_offload_size += tensor_on_device.numel(
                        ) * tensor_on_device.element_size()
                        state.offload()
                        # save the tensor since this the copy of this tensor has not yet finished
                        self.offloaded_tensor_buffers[
                            self.current_group].append(tensor_on_device)
        if self.debug and (not torch.distributed.is_initialized() or
                           torch.distributed.get_rank() == 0):
            print(
                f"offloading tensor size: {self.total_offload_size / (1024.0**3):.5f} GiB"
            )

    def synchronize_on_group_commit_forward(self, current_group):
        """Synchronize on group commit forward."""
        if current_group < self.num_offload_group:
            # perform bulk offloading
            self.bulk_offload_group(current_group)
            self.d2h_stream.record_event(self.d2h_finish_events[current_group])
        for group_id in range(self.num_offload_group):
            finish_offload_group = math.ceil(
                (group_id + 1) * self.num_offload_sync_group)
            if finish_offload_group <= current_group and len(
                    self.offloaded_tensor_buffers[group_id]) > 0:
                # This is very important. The tensor needs to be released for use
                # by other streams only after the copy has finished.
                torch.cuda.current_stream().wait_event(
                    self.d2h_finish_events[group_id])
                # release tensors since the copying has finished
                self.offloaded_tensor_buffers[group_id].clear()

    def on_group_commit_forward(self):
        """This function will cause host device synchronization"""
        # handle synchronization events
        self.synchronize_on_group_commit_forward(self.current_group)

        # during forward, the next_group_to_fetch always points to the min of
        # the last commited group, and the last offloaded group
        self.next_group_to_fetch = min(self.current_group,
                                       self.num_offload_group - 1)

        super().on_group_commit_forward()

    def bulk_reload_group(self, group_to_reload):
        """Bulk reload group."""
        assert group_to_reload < self.num_offload_group
        if group_to_reload == self.num_offload_group - 1:
            self.h2d_stream.wait_event(
                self.d2h_finish_events[self.num_offload_group - 1])

        # allocating tensors in the current stream allows subsequent ops in current streams
        # to reuse the GPU memory.
        for tensor_label, state in self.tensor_tag_to_state.items():
            group_id, _ = tensor_label
            if group_id == group_to_reload:
                if isinstance(state, TensorState):
                    state.create_prefetch_buffer()
        with torch.cuda.stream(self.h2d_stream):
            # move back tensors
            for tensor_label, state in self.tensor_tag_to_state.items():
                group_id, _ = tensor_label
                if group_id == group_to_reload:
                    if isinstance(state, TensorState):
                        state.reload()

    def on_group_commit_backward(self):
        # first decrement the current group.
        # after last commit in forward, the group will +1; in backward it -1.
        # Finally it should be decremented to 0.
        self.current_group -= 1
        assert self.current_group >= 0
        if self.current_group == 0:
            self.total_offload_size = 0

        for group_id in range(self.num_offload_group):
            assert len(self.offloaded_tensor_buffers[group_id]) == 0, \
                "num_offload_sync_layers * num_offload_layers + 1 cannot be greater than" \
                " the number of all layers."

        # decide the range of group to prefetch
        should_prefetch_until_group = self.current_group - self.num_prefetch_group
        should_prefetch_until_group = max(should_prefetch_until_group, 0)

        # do prefetch
        for group_num_to_prefetch in range(self.next_group_to_fetch,
                                           should_prefetch_until_group - 1, -1):
            # record the event in the compute stream, for h2d to wait
            torch.cuda.current_stream().record_event(
                self.compute_stream_bwd_start_events[group_num_to_prefetch])

            # start of h2d should wait for the compute and the d2h
            self.h2d_stream.wait_event(
                self.compute_stream_bwd_start_events[group_num_to_prefetch])

            #recover tensors (copy back from host)
            self.bulk_reload_group(group_num_to_prefetch)

            # record an event for the backward of this layer to wait
            self.h2d_stream.record_event(
                self.h2d_finish_events[group_num_to_prefetch])

        # always is set to -1 at the end of the backward
        self.next_group_to_fetch = min(self.num_offload_group - 1,
                                       should_prefetch_until_group - 1)

        # wait for the current group
        if self.current_group < self.num_offload_group:
            torch.cuda.current_stream().wait_event(
                self.h2d_finish_events[self.current_group])


def get_cpu_offload_context(num_offload_layers: int = 1,
                            num_prefetch_layers: int = 1,
                            num_offload_sync_layers: float = 1.0,
                            debug: bool = False):
    """
    This function returns the CPU Offload context and the synchronizer function that needs to be
    used after every transformer layer.

    Usage:

    .. code-block:: python

        cpu_offload_context, cpu_offload_synchronizer = get_cpu_offload_context(num_offload_layers=3)

        for layer in layers:
            with cpu_offload_context:
                x = layer(x)
            x = cpu_offload_synchronizer(x)

    Parameters
    ----------
    num_offload_layers: int, default = 1
                        Determines the number of transformer layers
                        you want to offload activations/weights for.
    num_prefetch_layers: int, default = 1
                         Determined how many layers in advance to perform prefetching.
    num_offload_sync_layers: float, default = 1.0
                             Determined how many layers to overlap (sync) the copying of one layer.
                             num_offload_sync_layers * num_offload_layers + 1 cannot be greater than
                             the number of all layers.
    """

    def tensor_need_offloading_checker(tensor):
        if ta.is_lazy_tensor(tensor):
            warnings.warn("Offloading currently only supports CUDA tensors.")
            return False
        if not tensor.is_cuda:
            return False
        # This is a bit tricky; if a PyTorch tensor is a view tensor, then its _base attribute
        # will be the previous tensor. The judgement here will filter out tensors that are not
        # activations, such as the transposed weights (weight.T) of the linear layer.
        if tensor._base is not None:
            return not tensor._base.is_leaf
        return not tensor.is_leaf or not tensor.requires_grad

    cpu_offload_handler = AsyncDoubleBufferGroupOffloadHandler(
        num_offload_group=num_offload_layers,
        num_prefetch_group=num_prefetch_layers,
        num_offload_sync_group=num_offload_sync_layers,
        tensor_need_offloading_checker=tensor_need_offloading_checker,
        debug=debug,
    )

    def group_prefetch_offload_commit_async(outputs):
        if not torch.is_grad_enabled():
            return outputs

        _apply_tensor = None
        _applied_tensor = None

        def apply(tensor):
            nonlocal _apply_tensor, _applied_tensor
            if ta.is_lazy_tensor(tensor) or not tensor.requires_grad:
                return tensor
            if id(tensor) == id(_apply_tensor):
                return _applied_tensor
            if _apply_tensor is not None:
                warnings.warn("There are multiple tensors that need to apply sync, " \
                              "which may cause problems with the offloading.")
                return tensor
            _apply_tensor = tensor
            tensor = group_prefetch_offload_commit(tensor, cpu_offload_handler)
            _applied_tensor = tensor
            return tensor

        outputs = apply_to_tensors(apply, outputs)
        # delete the global variables to avoid memory leak
        _apply_tensor = None
        _applied_tensor = None
        return outputs

    return (
        CpuOffloadHookWithOffloadHandler(offload_handler=cpu_offload_handler),
        group_prefetch_offload_commit_async,
    )
