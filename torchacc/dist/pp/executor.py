# This file is largely inspired by and partially follows the structure of
# ``deepspeed.runtime.pipe.engine.PipelineEngine`` in
# https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/pipe/engine.py

import inspect
from types import MethodType

import torch
import torch.distributed as dist

import torchacc as ta
from torchacc.config import PPConfig
from torchacc.dist.pp import microbatch, p2p, schedule


def init_communication(mesh, device):
    ta.sync()
    # The XLA backend initializes the NCCL clique collectively at the start of the computation graph,
    # which can cause a hang in the PP scenario. Here, we initialize it in advance to avoid this.
    tmp = torch.tensor([0], device=device)
    dist.all_reduce(tmp, group=mesh.get_pp_proc_group())
    ta.sync()


class PipeExecutor:
    """ A training executor hybrid pipeline, data, and tensor parallel training.
    """
    ID_TO_DTYPE = [
        torch.float32, torch.float64, torch.complex64, torch.complex128,
        torch.float16, torch.bfloat16, torch.uint8, torch.int8, torch.int16,
        torch.int32, torch.int64, torch.bool
    ]
    DTYPE_TO_ID = {dtype: id_ for id_, dtype in enumerate(ID_TO_DTYPE)}

    def __init__(self,
                 model,
                 config: PPConfig,
                 mesh,
                 orig_forward_sig,
                 post_process,
                 input_tensor_attr,
                 device,
                 gc=False,
                 train_scheduler=None,
                 eval_scheduler=None,
                 schedule_algo=schedule.Algo.PipeDreamFlush):
        self.model = model

        self.micro_batch_num = config.num_micro_batches

        self.broadcast_loss = config.broadcast_loss

        self.device = device

        self.mesh = mesh
        self.global_rank = self.mesh.get_global_rank()

        self.micro_batch_id = 0

        #  Set Stage Info
        self.num_stages = self.mesh.get_pp_num()
        self.stage_id = self.mesh.get_stage_id()
        self.prev_stage = self.stage_id - 1
        self.next_stage = self.stage_id + 1

        self.is_pipe_parallel = self.num_stages > 1

        #initialize peer-2-peer communication
        if self.is_pipe_parallel:
            p2p.init_mesh(self.mesh)
            # initialize all-reduce communication
            if ta.is_lazy_device(device):
                init_communication(self.mesh, device)

        # Pipeline buffers
        self.num_pipe_buffers = 0
        self.pipe_buffers = {
            'inputs': [],  # batch input and received activations
            'outputs': [],  # activations
        }

        self.pipe_recv_buf = None
        self.pipe_grad_buf = None

        self.orig_forward_sig = orig_forward_sig
        self.post_process = post_process

        self.input_tensor_attr = input_tensor_attr

        self.args_split = None
        self.kwargs_split = None

        self.first_output_send = True

        # Pipeline scheduler
        self.train_scheduler = None
        self.eval_scheduler = None
        self.schedule_algo = schedule_algo
        if train_scheduler:
            assert isinstance(
                train_scheduler,
                schedule.PipeSchedule), "scheduler must base PipeSchedule"
            self.train_scheduler = train_scheduler
            self.train_cmds = self.train_scheduler.schedules()
            self._reserve_pipe_buffers(self.train_scheduler.num_pipe_buffers())
        if eval_scheduler:
            assert isinstance(
                eval_scheduler,
                schedule.PipeSchedule), "scheduler must base PipeSchedule"
            self.eval_scheduler = eval_scheduler
            self.eval_cmds = self.eval_scheduler.schedules()
            self._reserve_pipe_buffers(self.eval_scheduler.num_pipe_buffers())

        # stores the loss for the current micro batch being processed
        self.loss = None

        # stores the loss for the entire batch
        self.total_loss = None
        self.agg_loss = torch.tensor(0.0, requires_grad=False).to(self.device)

        if self.device.type == 'xla':
            self.maybe_sync = ta.sync
        else:
            self.maybe_sync = (lambda *args: None)

        self.gc_enabled = not self.is_last_stage() if gc else False
        self.preserve_rng_state = True
        if self.preserve_rng_state:
            self.rng_states = {
                'cpu_states': [],
                'gpu_devices': [],
                'gpu_states': [],
            }

        self.output_fn = None

    def _update_underlay_model(self, model: torch.nn.Module):
        self.model = model

    def _exec_reduce_tied_grads(self):
        weight_group_list = self.model.get_tied_weights_and_groups()
        for weight, group in weight_group_list:
            grad = weight.grad
            dist.all_reduce(grad, group=group)

    def _reserve_pipe_buffers(self, num_buffers):
        """Ensure that each pipeline buffer has at least ``num_buffers`` slots.

        This method only reserves slots and does not allocate tensors.

        Args:
            num_buffers (int): The number of buffers to reserve.
        """
        if self.num_pipe_buffers >= num_buffers:
            return

        num_added = num_buffers - self.num_pipe_buffers
        for key in self.pipe_buffers:
            self.pipe_buffers[key].extend([None] * num_added)
        if self.gc_enabled and self.preserve_rng_state:
            for key in self.rng_states:
                self.rng_states[key].extend([None] * num_added)
        self.num_pipe_buffers = num_buffers

    def reset_activation_shape(self):
        """Reset the buffers when the shape of activation and gradient change.
        For example, for curriculum learning that changes the seqlen of each
        sample, we need to call this whenever the seqlen is going to change.
        """
        self.first_output_send = True
        self.pipe_recv_buf = None
        self.pipe_grad_buf = None

    def forward_backward(self, *args, output_fn=None, **kwargs):
        """Execute the pipeline parallel.

        This method is equivalent to:

        .. code-block:: python

            module.train()
            output = module(*args, **kwargs)
            output = output_fn(output)
            output.backward()

        Returns:
            The arithmetic mean of the losses computed this batch.
        """
        if not torch._C.is_grad_enabled():
            raise RuntimeError(
                f'forward_backward() requires gradients enabled. Use forward() instead.'
            )

        kwargs = microbatch.bind_args_to_kwargs(args, kwargs,
                                                self.orig_forward_sig)
        args = []
        self.args_split, self.kwargs_split = microbatch.split_args_kwargs_into_chunks(
            args, kwargs, self.micro_batch_num)

        self.output_fn = output_fn

        self.model.train()

        if self.train_scheduler is None:
            self.train_scheduler = schedule.create_scheduler(
                self.schedule_algo,
                is_training=True,
                micro_batches=self.micro_batch_num,
                stages=self.num_stages,
                stage_id=self.stage_id)
            self.train_cmds = self.train_scheduler.schedules()
            self._reserve_pipe_buffers(self.train_scheduler.num_pipe_buffers())

        self._exec_schedule(self.train_cmds)
        self.agg_loss = self._aggregate_total_loss()

        return self.agg_loss

    def forward(self, *args, output_fn=None, **kwargs):
        """Execute the pipeline parallel.

        This method is equivalent to:

        .. code-block:: python

            module.eval()
            with torch.no_grad():
                output = module(*args, **kwargs)
                output = output_fn(output)

        Returns:
            The arithmetic mean of the losses computed this batch.
        """
        self.args_split, self.kwargs_split = microbatch.split_args_kwargs_into_chunks(
            args, kwargs, self.micro_batch_num)
        self.output_fn = output_fn
        self.model.eval()

        if self.eval_scheduler is None:
            self.eval_scheduler = schedule.create_scheduler(
                self.schedule_algo,
                is_training=False,
                micro_batches=self.micro_batch_num,
                stages=self.num_stages,
                stage_id=self.stage_id)
            self.eval_cmds = self.eval_scheduler.schedules()
            self._reserve_pipe_buffers(self.eval_scheduler.num_pipe_buffers())

        # prevent dead-lock with multiple evals sequence
        dist.barrier()

        with torch.no_grad():
            self._exec_schedule(self.eval_cmds)

        # TODO: for evaluation, we may need logits, not loss?
        self.agg_loss = self._aggregate_total_loss()

        return self.agg_loss

    def is_first_stage(self):
        """True if this process is in the first stage in the pipeline."""
        return self.stage_id == 0

    def is_last_stage(self):
        """True if this process is in the last stage in the pipeline."""
        return self.stage_id == self.num_stages - 1

    def _scale_loss_by_gas(self, prescaled_loss):
        if isinstance(prescaled_loss, torch.Tensor):
            scaled_loss = prescaled_loss / self.micro_batch_num
        elif isinstance(prescaled_loss, (tuple, list)):
            scaled_loss = []
            for l in prescaled_loss:
                if isinstance(l, torch.Tensor):
                    scaled_loss.append(l / self.micro_batch_num)
                else:
                    scaled_loss.append(l)
        else:
            scaled_loss = prescaled_loss

        return scaled_loss

    def _aggregate_total_loss(self):
        # Scale loss, average among DP ranks, and bcast loss to the rest of my PP group
        if self.is_last_stage():
            loss = self._scale_loss_by_gas(self.total_loss)
            agg_loss = loss.clone().detach()

            if not self.broadcast_loss:
                return agg_loss

            ## Average loss across all data-parallel groups
            if self.mesh.get_fsdp_num() > 1:
                dist.all_reduce(agg_loss, group=self.mesh.get_fsdp_proc_group())
                agg_loss /= self.mesh.get_fsdp_num()

            if self.mesh.get_dp_num() > 1:
                dist.all_reduce(agg_loss, group=self.mesh.get_dp_proc_group())
                agg_loss /= self.mesh.get_dp_num()

            assert self.global_rank in self.mesh.pp_group

            if self.is_pipe_parallel:
                dist.broadcast(
                    tensor=agg_loss,
                    src=self.global_rank,
                    group=self.mesh.get_pp_proc_group())
        else:
            # Get loss from last stage
            src_rank = self.mesh.stage_to_global(self.num_stages - 1)
            assert src_rank in self.mesh.pp_group
            agg_loss = torch.tensor(0.0).to(self.device)

            if not self.broadcast_loss:
                return agg_loss

            dist.broadcast(
                tensor=agg_loss,
                src=src_rank,
                group=self.mesh.get_pp_proc_group())
        return agg_loss

    def _get_output_fn_kwargs(self, buffer_id):
        assert self.output_fn
        kwargs = {}
        sig = inspect.signature(self.output_fn)
        # first args is loss
        parameters = list(sig.parameters.values())[1:]
        for p in parameters:
            if p.name not in self.kwargs_split and p.default != p.empty:
                kwargs[p.name] = p.default
            assert p.name in self.kwargs_split, f"args {p.name} of output_fn must " \
                "be included in the args of the forward_backward"
            kwargs[p.name] = self.kwargs_split[p.name][buffer_id]
        return kwargs

    def _exec_forward_pass(self, buffer_id):
        inputs = self.pipe_buffers['inputs'][buffer_id]

        self.micro_batch_id += 1

        if self.gc_enabled:
            import torch_xla.core.xla_model as xm
            if self.preserve_rng_state:
                self.rng_states['cpu_states'][buffer_id] = torch.get_rng_state()
                gpu_devices, gpu_states = torch.utils.checkpoint.get_device_states(
                    *inputs)
                self.rng_states['gpu_devices'][buffer_id] = gpu_devices
                self.rng_states['gpu_states'][buffer_id] = gpu_states
            xm.optimization_barrier_(inputs)
            with torch.no_grad():
                outputs = self.model(*inputs)
            if not isinstance(outputs, tuple):
                xm.optimization_barrier_([outputs])
            else:
                xm.optimization_barrier_(outputs)
            # Note [GC + AMP cache]
            # Clear the parameter cache of AMP to avoid having parameters with requires_grad set to False.
            # see https://discuss.pytorch.org/t/autocast-and-torch-no-grad-unexpected-behaviour/93475/2
            if torch.is_autocast_cache_enabled():
                torch.clear_autocast_cache()
        else:
            outputs = self.model(*inputs)

        self.pipe_buffers['outputs'][buffer_id] = outputs

        # Optionally compute loss on the last device
        if self.is_last_stage():
            if isinstance(outputs, torch.Tensor):
                outputs = [
                    outputs,
                ]
            outputs = self.post_process(*outputs)
            if self.output_fn is not None:
                kwargs = self._get_output_fn_kwargs(buffer_id)
                self.loss = self.output_fn(outputs, **kwargs)
            else:
                if isinstance(outputs, (tuple, list)):
                    self.loss = outputs[0]
                elif isinstance(outputs, dict):
                    assert 'loss' in outputs, "If the output of the model is of type dict, " \
                        "it needs to have a key 'loss' to perform backward." \
                        " Alternatively, output_fn can be used."
                    self.loss = outputs['loss']
                else:
                    raise ValueError(
                        f"Unknown outputs: {outputs}. output_fn can be used.")
            assert isinstance(self.loss, torch.Tensor), "Expected model output to be a " \
                f"tensor, but got: {self.loss}"
            if self.total_loss is None:
                self.total_loss = torch.zeros_like(
                    self.loss, requires_grad=False)
            self.total_loss += self.loss.clone().detach()
        # See Note [GC + AMP cache]
        if torch.is_autocast_cache_enabled():
            torch.clear_autocast_cache()

    def _exec_backward_pass(self, buffer_id):
        # The last stage just runs backward on the loss
        if self.is_last_stage():
            assert not self.gc_enabled
            with torch.cuda.amp.autocast(enabled=False):
                self.loss.backward()
            # Free up the memory from the output of forward()
            self.loss = None
            self.pipe_buffers['outputs'][buffer_id] = None
            return

        outputs = self.pipe_buffers['outputs'][buffer_id]

        grad_tensors = self.pipe_grad_buf

        if self.gc_enabled:
            import torch_xla.core.xla_model as xm
            inputs = self.pipe_buffers['inputs'][buffer_id]
            rng_devices = []
            if self.preserve_rng_state:
                rng_devices = self.rng_states['gpu_devices'][buffer_id]
            with torch.random.fork_rng(
                    devices=rng_devices, enabled=self.preserve_rng_state):
                if self.preserve_rng_state:
                    torch.set_rng_state(
                        self.rng_states['cpu_states'][buffer_id])
                    torch.utils.checkpoint.set_device_states(
                        self.rng_states['gpu_devices'][buffer_id],
                        self.rng_states['gpu_states'][buffer_id])
                with torch.enable_grad():
                    outputs = self.model(*inputs)
            if not isinstance(outputs, tuple):
                xm.optimization_barrier_([outputs])
            else:
                xm.optimization_barrier_(outputs)

        # This handles either a single tensor or tuple of tensors.
        if isinstance(outputs, tuple):
            out_tensors = [t for t in outputs if t.is_floating_point()]
            assert len(out_tensors) == len(grad_tensors)
            bwd_tensors = []
            bwd_grads = []
            for idx, t in enumerate(out_tensors):
                if t.requires_grad:
                    bwd_tensors.append(t)
                    bwd_grads.append(grad_tensors[idx])
            with torch.cuda.amp.autocast(enabled=False):
                torch.autograd.backward(
                    tensors=bwd_tensors, grad_tensors=bwd_grads)
        else:
            with torch.cuda.amp.autocast(enabled=False):
                torch.autograd.backward(
                    tensors=(outputs,), grad_tensors=(grad_tensors,))

        # Free up the memory from the output of forward()
        self.pipe_buffers['outputs'][buffer_id] = None
        grad_tensors = None

    def _exec_load_micro_batch(self, buffer_id):
        assert len(self.args_split) == 0
        batch = []
        # We need to convert the input tensor list to
        # the tensor list required for the current stage.
        # See Note [PP input_tensor_attr] in pp.utils
        for attr in self.input_tensor_attr:
            if isinstance(attr, int):
                x = self.pipe_buffers['inputs'][buffer_id][attr]
            elif isinstance(attr, str):
                assert attr in self.kwargs_split, f"model input {attr} not found"
                x = self.kwargs_split[attr][self.micro_batch_id]
                assert torch.is_tensor(x)
                x = x.to(self.device)
            else:
                raise ValueError(f"Unknown model input {attr}")
            batch.append(x)
        self.pipe_buffers['inputs'][buffer_id] = batch

    def _send_tensor_meta(self, buffer, recv_stage):
        """ Communicate metadata about upcoming p2p transfers.

        Metadata is communicated in this order:
            * type (0: tensor, 1: list)
            * num_tensors if type=list
            foreach tensor in buffer:
                * dtype
                * ndims
                * shape
        """

        def send_tensor_meta(tensor):
            send_dtype = torch.LongTensor(
                data=[self.DTYPE_TO_ID[tensor.dtype]]).to(self.device)
            send_ndims = torch.LongTensor(data=[len(tensor.size())]).to(
                self.device)
            send_shape = torch.LongTensor(data=tensor.size()).to(self.device)
            p2p.send(send_dtype, recv_stage)
            p2p.send(send_ndims, recv_stage)
            p2p.send(send_shape, recv_stage)

        if isinstance(buffer, torch.Tensor):
            type_tensor = torch.LongTensor(data=[0]).to(self.device)
            p2p.send(type_tensor, recv_stage)
            send_tensor_meta(buffer)
        elif isinstance(buffer, (list, tuple)):
            type_tensor = torch.LongTensor(data=[1]).to(self.device)
            p2p.send(type_tensor, recv_stage)
            count_tensor = torch.LongTensor(data=[len(buffer)]).to(self.device)
            p2p.send(count_tensor, recv_stage)
            for tensor in buffer:
                assert isinstance(tensor, torch.Tensor)
                send_tensor_meta(tensor)
        else:
            raise NotImplementedError(
                f'Could not send meta type {type(buffer)}')

    def _recv_tensor_meta(self, send_stage):
        """Receive metadata about upcoming p2p transfers and return allocated buffers.

        Metadata is communicated in this order:
            * type (0: tensor, 1: list)
            * num_tensors if type=list
            foreach tensor in buffer:
                * dtype
                * ndims
                * shape

        Returns:
            Allocated buffer for receiving from send_stage.
        """

        def recv_tensor_meta():
            recv_dtype = torch.LongTensor(data=[0]).to(self.device)
            p2p.recv(recv_dtype, send_stage)
            self.maybe_sync()
            recv_dtype = self.ID_TO_DTYPE[recv_dtype.item()]

            recv_ndims = torch.LongTensor(data=[0]).to(self.device)
            p2p.recv(recv_ndims, send_stage)
            self.maybe_sync()
            recv_ndims = recv_ndims.item()

            recv_shape = torch.LongTensor([1] * recv_ndims).to(self.device)
            p2p.recv(recv_shape, send_stage)
            self.maybe_sync()
            recv_shape = recv_shape.tolist()

            return recv_dtype, recv_shape

        type_tensor = torch.LongTensor(data=[0]).to(self.device)
        p2p.recv(type_tensor, send_stage)
        self.maybe_sync()
        recv_type = type_tensor.item()

        # A single tensor will be sent.
        if recv_type == 0:
            recv_dtype, recv_shape = recv_tensor_meta()
            return self._allocate_buffer(shape=recv_shape, dtype=recv_dtype)
        # List or tuple of tensors
        elif recv_type == 1:
            count_tensor = torch.LongTensor(data=[0]).to(self.device)
            p2p.recv(count_tensor, send_stage)
            self.maybe_sync()

            num_tensors = count_tensor.item()
            shapes = []
            dtypes = []
            for _ in range(num_tensors):
                recv_dtype, recv_shape = recv_tensor_meta()
                shapes.append(recv_shape)
                dtypes.append(recv_dtype)
            return self._allocate_buffers(shapes, dtypes)
        else:
            raise NotImplementedError(f'Could not receive type {recv_type}')

    def _exec_send_activations(self, buffer_id):
        outputs = self.pipe_buffers['outputs'][buffer_id]

        if self.first_output_send:
            self.first_output_send = False
            self._send_tensor_meta(outputs, self.next_stage)

        if isinstance(outputs, torch.Tensor):
            p2p.send(outputs, self.next_stage)
        elif isinstance(outputs, (tuple, list)):
            for buffer in outputs:
                p2p.send(buffer, self.next_stage)
        else:
            raise NotImplementedError('Could not send output of type '
                                      f'{type(outputs)}')
        self.maybe_sync()

    def _exec_send_grads(self, buffer_id):
        inputs = self.pipe_buffers['inputs'][buffer_id]

        if isinstance(inputs, torch.Tensor):
            assert inputs.grad is not None
            p2p.send(inputs.grad, self.prev_stage)
        else:
            assert len(self.input_tensor_attr) == len(inputs)
            recved_buffer = [None] * len(inputs)
            # We need to restore the input tensor list to the
            # list of tensors received from the previous stage.
            # See Note [PP input_tensor_attr] in pp.utils
            for idx, buffer in enumerate(inputs):
                real_idx = self.input_tensor_attr[idx]
                if isinstance(real_idx, int):
                    recved_buffer[real_idx] = buffer
            for buffer in recved_buffer:
                if buffer is None:
                    continue
                # Skip tensors that will not produce a grad
                if not buffer.is_floating_point():
                    assert buffer.grad is None
                    continue
                send_grad = buffer.grad
                if send_grad is None:
                    # Some propagated output tensors do not have gradients
                    send_grad = torch.zeros_like(buffer)
                p2p.send(send_grad, self.prev_stage)

        # We can free up the input buffer now
        self.pipe_buffers['inputs'][buffer_id] = None
        self.maybe_sync()

    def _exec_recv_activations(self, buffer_id):
        recvd = None

        # Allocate the buffer if necessary
        if self.pipe_recv_buf is None:
            self.pipe_recv_buf = self._recv_tensor_meta(self.prev_stage)

        if isinstance(self.pipe_recv_buf, torch.Tensor):
            p2p.recv(self.pipe_recv_buf, self.prev_stage)
            recvd = self.pipe_recv_buf.clone().detach()
            recvd.requires_grad = recvd.is_floating_point()
            recvd = [recvd]
        else:
            recvd = []
            for buffer in self.pipe_recv_buf:
                assert isinstance(buffer, torch.Tensor)
                p2p.recv(buffer, self.prev_stage)
                t = buffer.clone().detach()
                t.requires_grad = buffer.is_floating_point()
                recvd.append(t)

        self.pipe_buffers['inputs'][buffer_id] = recvd

    def _exec_recv_grads(self, buffer_id):
        outputs = self.pipe_buffers['outputs'][buffer_id]

        # Allocate gradient if necessary
        if self.pipe_grad_buf is None:
            if isinstance(outputs, torch.Tensor):
                s = list(outputs.size())
                self.pipe_grad_buf = self._allocate_buffer(
                    s, dtype=outputs.dtype)
            else:
                shapes = []
                dtypes = []
                for t in outputs:
                    if t.is_floating_point():
                        shapes.append(list(t.size()))
                        dtypes.append(t.dtype)
                self.pipe_grad_buf = self._allocate_buffers(shapes, dtypes)

        if isinstance(self.pipe_grad_buf, torch.Tensor):
            p2p.recv(self.pipe_grad_buf, self.next_stage)
        else:
            for buffer in self.pipe_grad_buf:
                p2p.recv(buffer, self.next_stage)

    def _allocate_buffer(self, shape, **kwargs):
        """ Allocate a tensor of zeros on self.device.

        Args:
            shape: the shape of the tensor to allocate
            kwargs: passed to torch.zeros()

        Returns:
            A tensor from torch.zeros() allocated on self.device.
        """
        return torch.zeros(shape, device=self.device, **kwargs)

    def _allocate_buffers(self, shapes, dtypes, requires_grads=None):
        buffer_num = len(shapes)
        assert buffer_num == len(dtypes)
        if requires_grads:
            assert buffer_num == len(requires_grads)
        buffers = []
        for idx in range(buffer_num):
            shape = shapes[idx]
            dtype = dtypes[idx]
            requires_grad = requires_grads[idx] if requires_grads else False
            buffers.append(
                self._allocate_buffer(
                    shape, dtype=dtype, requires_grad=requires_grad))
        return buffers

    # A map of PipeInstruction types to methods. Each method will be executed with the
    # kwargs provided to the PipeInstruction from the scheduler.
    _INSTRUCTION_MAP = {
        schedule.ReduceTiedGrads: _exec_reduce_tied_grads,
        schedule.LoadMicroBatch: _exec_load_micro_batch,
        schedule.ForwardPass: _exec_forward_pass,
        schedule.BackwardPass: _exec_backward_pass,
        schedule.SendActivation: _exec_send_activations,
        schedule.RecvActivation: _exec_recv_activations,
        schedule.SendGrad: _exec_send_grads,
        schedule.RecvGrad: _exec_recv_grads,
    }

    def _exec_schedule(self, pipe_schedule):
        # Reset micro batch id
        self.micro_batch_id = 0
        # Reset outputs.
        self.total_loss = None

        # For each instruction in the step
        for cmd in pipe_schedule:
            if type(cmd) not in self._INSTRUCTION_MAP:
                raise RuntimeError(
                    f'{self.__class__.__name__} does not understand instruction {repr(cmd)}'
                )

            # Equivalent to: self._exec_forward_pass(buffer_id=0)
            self._exec_instr = MethodType(self._INSTRUCTION_MAP[type(cmd)],
                                          self)
            self._exec_instr(**cmd.kwargs)
