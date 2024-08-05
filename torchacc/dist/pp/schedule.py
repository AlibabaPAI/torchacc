# This file is largely inspired by and partially follows the structure of
# ``deepspeed.runtime.pipe.schedule.PipeSchedule`` in
# https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/pipe/schedule.py

from abc import ABC, abstractmethod
from enum import auto, Enum
from torchacc.utils.utils import call_to_str


class PipeSchedule(ABC):
    """Directs the execution of a pipeline engine by generating sequences of
    :class:`PipeInstruction`.

    Schedules are list of sequences of
    :class:`PipeInstruction` to process the micro-batches in one batch.

    Below is an example schedule that implements data parallelism with gradient accumulation:

    .. code-block:: python

        class DataParallelSchedule(PipeSchedule):
            def steps(self):
                for step_id in range(self.micro_batches):
                    cmds = [
                        LoadMicroBatch(buffer_id=0),
                        ForwardPass(buffer_id=0),
                        BackwardPass(buffer_id=0),
                    ]
                    if step_id == self.micro_batches - 1:
                        cmds.extend([
                            ReduceGrads(),
                            OptimizerStep(),
                        ])
                    yield cmds

            def num_pipe_buffers(self):
                return 1

    Args:
        micro_batches (int): The number of micro-batches that comprise a batch.
        stages (int): The number of pipeline stages.
        stage_id (int): The pipe stage that will execute the generated schedule.
    """

    def __init__(self, micro_batches, stages, stage_id):
        super().__init__()
        self.micro_batches = micro_batches
        self.stages = stages
        self.stage_id = stage_id
        self.prev_stage = self.stage_id - 1
        self.next_stage = self.stage_id + 1

    @abstractmethod
    def schedules(self):
        """Create a list of :class:`PipeInstruction` for each step in the schedule.

        .. note::
            Schedules must implement ``schedules()`` to define the schedule.

        Returns:
            Instructions to be executed as one step of the pipeline
        """
        pass

    def num_pipe_buffers(self):
        """The number of pipeline buffers that will be used by this stage.

        .. note::
            Schedules should specialize ``num_pipe_buffers()`` for memory savings at scale.

        Returns:
            The number of buffers for the engine to allocate.
        """
        return self.micro_batches

    def _valid_micro_batch(self, micro_batch_id):
        return 0 <= micro_batch_id < self.micro_batches

    def _valid_stage(self, stage_id):
        return 0 <= stage_id < self.stages

    @property
    def stage(self):
        """Stage index used to configure this schedule."""
        return self.stage_id

    @property
    def num_stages(self):
        """The number of total pipeline stages used to configure this schedule."""
        return self.stages

    @property
    def num_micro_batches(self):
        """The number of total micro_batches used to configure this schedule."""
        return self.micro_batches

    @property
    def is_first_stage(self):
        """True if the configured ``stage_id`` is the first stage in the pipeline."""
        return self.stage_id == 0

    @property
    def is_last_stage(self):
        """True if the configured ``stage_id`` is the last stage in the pipeline."""
        return self.stage_id == self.stages - 1

    def _buffer_idx(self, micro_batch_id):
        """Map a micro-batch index to a pipeline buffer index.

        This method uses a cyclic allocation strategy.

        Args:
            micro_batch_id (int): The micro-batch index relative to the beginning of the schedule.

        Returns:
            int: The index of the buffer that should store data.
        """
        assert self._valid_micro_batch(micro_batch_id)
        return micro_batch_id % self.num_pipe_buffers()


class PipeDreamFlushInfer(PipeSchedule):
    """A schedule for inferencing batches using pipeline parallelism.
    """

    def _is_first_stage(self):
        """True if this process is in the first stage in the pipeline."""
        return self.stage_id == 0

    def _is_last_stage(self):
        """True if this process is in the last stage in the pipeline."""
        return self.stage_id == self.stages - 1

    def schedules(self):
        # Run warmup forward passes.
        cmds = []
        buffer_id = 0
        for _ in range(self.micro_batches):
            if not self._is_first_stage():
                cmds.append(RecvActivation(buffer_id))
            cmds.append(LoadMicroBatch(buffer_id))
            cmds.append(ForwardPass(buffer_id))
            if not self._is_last_stage():
                cmds.append(SendActivation(buffer_id))
        return cmds

    def num_pipe_buffers(self):
        """Only two pipeline buffers are required for inferencing.

        Returns:
            ``2``
        """
        return 2


class PipeDreamFlushTrain(PipeSchedule):
    """A schedule for training a batch using hybrid parallelism.

    Pipeline parallelism is extracted through gradient accumulation and thus
    convergence follows that of a data parallel approach with the same batch
    size.
    """

    def _is_first_stage(self):
        """True if this process is in the first stage in the pipeline."""
        return self.stage_id == 0

    def _is_last_stage(self):
        """True if this process is in the last stage in the pipeline."""
        return self.stage_id == self.stages - 1

    def schedules(self):
        num_buffers = self.num_pipe_buffers()
        num_warmup_microbatches = self.stages - self.stage_id - 1
        num_warmup_microbatches = min(num_warmup_microbatches,
                                      self.micro_batches)
        num_microbatches_remaining = self.micro_batches - num_warmup_microbatches
        # Run warmup forward passes.
        cmds = []
        for buffer_id in range(num_warmup_microbatches):
            if not self._is_first_stage():
                cmds.append(RecvActivation(buffer_id))
            cmds.append(LoadMicroBatch(buffer_id))
            cmds.append(ForwardPass(buffer_id))
            if not self._is_last_stage():
                cmds.append(SendActivation(buffer_id))
        # Run 1F1B in steady state.
        buffer_id = num_warmup_microbatches
        for _ in range(num_microbatches_remaining):
            if not self._is_first_stage():
                cmds.append(RecvActivation(buffer_id))
            cmds.append(LoadMicroBatch(buffer_id))
            cmds.append(ForwardPass(buffer_id))

            prev_buffer = buffer_id
            buffer_id = (buffer_id + 1) % num_buffers
            # need recv grad first
            if not self._is_last_stage():
                cmds.append(RecvGrad(buffer_id))
            if not self._is_last_stage():
                cmds.append(SendActivation(prev_buffer))

            cmds.append(BackwardPass(buffer_id))
            if not self._is_first_stage():
                cmds.append(SendGrad(buffer_id))

        # Run cooldown backward passes.
        for _ in range(num_warmup_microbatches):
            buffer_id = (buffer_id + 1) % num_buffers
            if not self._is_last_stage():
                cmds.append(RecvGrad(buffer_id))
            cmds.append(BackwardPass(buffer_id))
            if not self._is_first_stage():
                cmds.append(SendGrad(buffer_id))

        return cmds

    def num_pipe_buffers(self):
        """Return the number of pipeline buffers required for this stage.

        This is equivalent to the maximum number of in-flight forward passes,
        since we need to remember the activations of forward passes in order
        to run backpropagation. For synchronous 1F1B, this is equivalent to
        the index difference between this stage and the last stage.
        """
        buffers = min(self.stages - self.stage_id, self.micro_batches)
        return buffers


class PipeInstruction:
    """Base class for all instructions to be executed by the pipeline engine.

    All keyword arguments are stored as members similar to a ``namedtuple``. These are
    then accessible to the :class:`PipeEngine` during execution.

    Args:
        kwargs (optional): keyword arguments to store as members
    """

    def __init__(self, **kwargs):
        self.name = self.__class__.__name__
        self.kwargs = kwargs
        for key, val in kwargs.items():
            setattr(self, key, val)

    def __repr__(self):
        return call_to_str(self.name, **self.kwargs)


class OptimizerStep(PipeInstruction):
    """Performs one step with the optimizer and zeros gradients.

    .. note:: Should be issued after :class:`ReduceGrads` and :class:`ReduceTiedGrads`.

    .. note:: Can be a synchronization point among data-parallel ranks.
    """
    pass


class ReduceGrads(PipeInstruction):
    """Reduce the computed gradients among data-parallel processes within the stage.
    """
    pass


class ReduceTiedGrads(PipeInstruction):
    """Reduce the computed gradients of tied modules within a pipeline-parallel group.

    .. warning::
        The stages included in this synchronization point are not known until
        the model is partitioned among pipeline stages. In the worst case, it
        includes all pipeline stages. This instruction should be scheduled
        carefully to avoid deadlocks.
    """
    pass


class BufferOpInstruction(PipeInstruction):
    """A pipeline instruction that operates on pipeline buffer(s).

    Args:
        buffer_id (int): the index of the pipeline buffer() to modify.
    """

    def __init__(self, buffer_id, **kwargs):
        super().__init__(buffer_id=buffer_id, **kwargs)


# IO
class LoadMicroBatch(BufferOpInstruction):
    """Load a micro-batch into a buffer.

    Roughly:

    .. code-block:: python

        buffers['inputs'][buffer_id] = next(data_iter)
    """
    pass


# Compute
class ForwardPass(BufferOpInstruction):
    """Compute a forward pass.

    Roughly:

    .. code-block:: python

        buffers['outputs'][buffer_id] = forward(buffers['inputs'][buffer_id])
    """
    pass


class BackwardPass(BufferOpInstruction):
    """Compute a backward pass and accumulate gradients.

    Roughly:

    .. code-block:: python

        outputs = buffers['outputs'][buffer_id]
        gradients = buffers['gradients'][buffer_id]
        torch.autograd.backward(tensors=outputs,
                                grad_tensors=gradients)
    """
    pass


# Communication
class SendActivation(BufferOpInstruction):
    """Send activations to the next stage in the pipeline.

    Roughly:

    .. code-block:: python

        send(buffers['outputs'][buffer_id])

    .. note::
        The communication is blocking and must be paired with a :class:`RecvActivation`
        on the next pipeline stage to avoid deadlock.
    """
    pass


class RecvActivation(BufferOpInstruction):
    """Receive activations from the previous stage in the pipeline.

    Roughly:

    .. code-block:: python

        buffers['inputs'][buffer_id] = recv()

    .. note::
        The communication is blocking and must be paired with a :class:`SendActivation`
        on the previous pipeline stage to avoid deadlock.
    """
    pass


class SendGrad(BufferOpInstruction):
    """Send computed gradients to the previous pipeline stage.
    with respect to the received activations

    .. note::
        Only received tensors with ``requires_grad==True`` will produce gradients.
        Missing gradients will be replaced with ``None`` on the receiving stage.

    .. note::
        The communication is blocking and must be paired with a :class:`RecvGrad`
        on the previous pipeline stage to avoid deadlock.
    """
    pass


class RecvGrad(BufferOpInstruction):
    """Receive computed gradients the next pipeline stage.

    .. note::
        Only activations with ``requires_grad==True`` will produce gradients.
        Missing gradients will be replaced with ``None``.

    .. note::
        The communication is blocking and must be paired with a :class:`SendGrad`
        on the next pipeline stage to avoid deadlock.
    """
    pass


class Algo(Enum):
    PipeDreamFlush = auto()


def create_scheduler(algo: Algo, is_training, micro_batches, stages, stage_id):
    if is_training:
        if algo == Algo.PipeDreamFlush:
            return PipeDreamFlushTrain(micro_batches, stages, stage_id)
        else:
            raise NotImplementedError(
                f'Pipeline schedule algorithm {algo} not implemented.')
    else:
        if algo == Algo.PipeDreamFlush:
            return PipeDreamFlushInfer(micro_batches, stages, stage_id)
        else:
            raise NotImplementedError(
                f'Pipeline schedule algorithm {algo} not implemented.')
