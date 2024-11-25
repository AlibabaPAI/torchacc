import functools
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Union

import torch
import torch.distributed as dist

import torchacc as ta
import torchacc.ops.context_parallel as context_parallel

if sys.version_info >= (3, 10):
    dataclass = functools.partial(dataclass, slots=True)


class BaseConfig(ABC):

    @abstractmethod
    def validate(self):
        pass


@dataclass
class ComputeConfig(BaseConfig):
    """Configuration for computational optimization

    Args:
        fp16 (bool): Whether it is the data type of fp16.
        bf16 (bool): Whether it is the data type of bf16.
        acc_scaled_dot_attn (bool): Whether torch.nn.functional.scaled_dot_product_attention
            should be replaced with the torchacc version of flash attention.
        disable_kernel_patches (bool): Whether the kernel patches will be disabled.
    """
    fp16: bool = False
    bf16: bool = False
    acc_scaled_dot_attn: bool = False
    disable_kernel_patches: bool = False

    def validate(self):
        assert isinstance(self.fp16,
                          bool), "ComputeConfig.fp16 should be of bool type"
        assert isinstance(self.bf16,
                          bool), "ComputeConfig.bf16 should be of bool type"
        assert isinstance(
            self.acc_scaled_dot_attn,
            bool), "ComputeConfig.acc_scaled_dot_attn should be of bool type"
        assert isinstance(
            self.disable_kernel_patches,
            bool), "ComputeConfig.disable_kernel_patches should be of bool type"
        if self.fp16 and self.bf16:
            raise ValueError(f"fp16 and bf16 cannot both be True")


@dataclass
class MemoryConfig(BaseConfig):
    """Configuration for memory optimization

    Args:
        gc (bool): Whether to enable gradient checkpointing to reduce memory usage.
        gc_cls (Optional[Set[str]]): Submodules with one of the `gc_cls` names will be gradient checkpointed.
        gc_cnt (Optional[int]): Number of layers for gradient checkpointing. Currently only supported with
            FSDP combined with full gc.
    """
    gc: bool = False
    gc_cls: Optional[Set[str]] = None
    gc_cnt: Optional[int] = None

    def validate(self):
        assert isinstance(self.gc,
                          bool), "MemoryConfig.gc should be of bool type"
        if self.gc_cls is not None:
            assert isinstance(
                self.gc_cls,
                set), "MemoryConfig.gc_cls should be of set type or None"
            for cls in self.gc_cls:
                assert isinstance(
                    cls,
                    str), "cls in MemoryConfig.gc_cls should be of str type"

        if self.gc_cnt:
            assert isinstance(
                self.gc_cnt, int
            ), f"MemoryConfig.gc_cnt should be of int type or None, {self.gc_cnt}"
            if self.gc_cnt < 0:
                raise ValueError(f"MemoryConfig.gc_cnt should be >= 0")


@dataclass
class DataLoaderConfig(BaseConfig):
    """Configuration for dataloader optimization

    Args:
        buckets (list): A list of integers that records the sizes of each bucket.
            When it is not None, the following args `max_length` and `num_buckets`
            will be invalid. Default setting is None.
        max_length (int): Max last dim length used for bucketing data loader. Default
            setting is None, indicating that bucketing will not be employed.
        num_buckets (int): The total count of buckets employed within the bucketing data loader.
        pad_value_dict (dict): The default padding value for each type of element in
            bucketing dataloader's output. The default setting is as follows:
            {'input_ids': 0, 'attention_mask': 0, 'labels': -100}
    """
    buckets: Optional[List[int]] = None
    max_length: Optional[int] = None
    num_buckets: Optional[int] = None
    pad_value_dict: Optional[Dict[str, int]] = None

    def validate(self):
        if self.buckets is not None:
            assert isinstance(
                self.max_length,
                list), "DataLoaderConfig.buckets should be of list type"
        if self.max_length is not None:
            assert isinstance(
                self.max_length,
                int), "DataLoaderConfig.max_length should be of int type"
        if self.num_buckets is not None:
            assert isinstance(
                self.num_buckets,
                int), "DataLoaderConfig.num_buckets should be of int type"
        if self.pad_value_dict is not None:
            assert isinstance(
                self.pad_value_dict,
                dict), "DataLoaderConfig.pad_value_dict should be of dict type"


@dataclass
class DPConfig(BaseConfig):
    """Configuration for data parallel

    Args:
        size (Optional[int]): Number of data parallel. If None, The number of data parallel
            will be automatically inferred from the configured parallel strategy and world size.
    """
    size: Optional[int] = None

    def validate(self):
        if self.size:
            assert isinstance(
                self.size, int
            ), f"DPConfig.size should be of int type or None, {self.size}"
            if self.size < 1:
                raise ValueError(f"DPConfig.size should be >= 1")


@dataclass
class TPConfig(BaseConfig):
    """Configuration for tensor parallel

    Args:
        size (int): Number of tensor parallel.
    """
    size: int = 1

    def validate(self):
        assert isinstance(self.size, int), "TPConfig.size should be of int type"
        if self.size < 1:
            raise ValueError(f"TPConfig.size should be >= 1")


@dataclass
class PPConfig(BaseConfig):
    """Configuration for pipeline parallel

    Args:
        size (int): Number of pipeline parallel.
        num_micro_batches (int): Number of micro batches.
        input_names (Optional[List[str]]): The names of the parameters that the model needs as inputs in the
            forward. For parameters that have default values and require specific values to be passed during
            forward, the parameter name must be specified. Typically, this value corresponds to the keys in
            the batch input of the dataloader (if it is of type dict).
        split_points (Union[List[str], List[torch.nn.Module]]): A list of split points for the model.
            The number of split points should be `pp.size - 1`. The splitting point can be either the name of
            the module or the module objects in `torch.nn.Module.named_modules()`. It is worth noting
            that the model will be split `before` the corresponding modules are called.
        broadcast_loss (bool): Whether to broadcast the loss from the last stage to all devices.
    """
    size: int = 1
    num_micro_batches: int = 1
    input_names: Optional[List[str]] = None
    split_points: Union[List[str],
                        List[torch.nn.Module]] = field(default_factory=list)
    broadcast_loss: bool = True

    def validate(self):
        assert isinstance(self.size, int), "PPConfig.size should be of int type"
        assert isinstance(
            self.num_micro_batches,
            int), "PPConfig.num_micro_batches should be of int type"
        if self.input_names is not None:
            assert isinstance(
                self.input_names,
                list), "PPConfig.input_names should be of list type or None"
        assert isinstance(self.split_points,
                          list), "PPConfig.split_points should be of list type"
        assert isinstance(
            self.broadcast_loss,
            bool), "PPConfig.broadcast_loss should be of bool type"

        if self.size < 1:
            raise ValueError(f"PPConfig.size should be >= 1")
        if self.num_micro_batches < 1:
            raise ValueError(f"PPConfig.num_micro_batches should be >= 1")
        if self.input_names is not None:
            for name in self.input_names:
                assert isinstance(
                    name,
                    str), "name in PPConfig.input_names should be of str type"
        if len(self.split_points) > 0:
            assert len(self.split_points) == len(set(self.split_points)), "There should not be " \
                "any duplicate values in PPConfig.split_points"
            is_all_str = all([isinstance(p, str) for p in self.split_points])
            is_all_mod = all(
                [isinstance(p, torch.nn.Module) for p in self.split_points])
            assert is_all_str or is_all_mod, "All values in PPConfig.split_points need to be " \
                "of type str or all of type torch.nn.Module."
        assert self.size == len(self.split_points) + 1, "The number of split points" \
            " should be PPConfig.size - 1"


@dataclass
class FSDPConfig(BaseConfig):
    """Configuration for fully sharded data parallel

    Args:
        size (int): Number of fully sharded data parallel.
        wrap_layer_cls (Set[str]): Submodules with one of the `wrap_layer_cls` names
            will be wrapped as separated FSDP units.
        flatten_parameters (bool): If ``True``, flatten parameters into a single contiguous tensor for
            all_gather and reduce_scatter, which could potentially improve speed. In this case, one
            cannot apply separate optimizer groups to different original parameters in the wrapped
            module (e.g. setting bias terms or any BatchNorm submodules to have zero weight decay) since
            all the original parameters now become a single concatenated vector.
        sync_module_states (bool): If ``True``, then each FSDP module will broadcast module parameters
            and buffers from rank 0 to ensure that they are replicated across ranks (adding communication
            overhead and more GPU memory overhead during initialization).
        use_spmd (bool): If ``True``, use SPMD based FSDP.
        shard_output_callable (callable): A callable to shard the output of the forwpass.
            The callable should have the signature (output, mesh) -> None. If None, the default
            implementation will shard the first tensor in the output. If the output is a tuple,
            only the first tensor will be sharded.
    """
    size: int = 1
    wrap_layer_cls: Set[str] = field(default_factory=set)
    flatten_parameters: bool = True
    sync_module_states: bool = False
    use_spmd: bool = False
    shard_output_callable: callable = None

    def validate(self):
        assert isinstance(self.size,
                          int), "FSDPConfig.size should be of int type"
        assert isinstance(
            self.wrap_layer_cls,
            set), "FSDPConfig.wrap_layer_cls should be of set type"
        assert isinstance(
            self.flatten_parameters,
            bool), "FSDPConfig.flatten_parameters should be of bool type"
        assert isinstance(
            self.sync_module_states,
            bool), "FSDPConfig.sync_module_states should be of bool type"
        if self.size < 1:
            raise ValueError(f"FSDPConfig.size should be >= 1")
        for cls in self.wrap_layer_cls:
            assert isinstance(
                cls,
                str), "cls in FSDPConfig.wrap_layer_cls should be of str type"


class SPConfig(BaseConfig):
    """Configuration for 
    """
    size: int = 1

    def validate(self):
        assert isinstance(self.size, int), "SPConfig.size should be of int type"


@dataclass
class DistConfig(BaseConfig):
    """Configuration for distributed parallel

    Args:
        dp (DPConfig): Configuration for data parallel.
        tp (TPConfig): Configuration for tensor parallel.
        pp (DPConfig): Configuration for pipeline parallel.
        fsdp (FSDPConfig): Configuration for fully sharded data parallel.
        topology (List[str]): The topological relationship of various parallel strategies on devices. It is
            represented as a list of 'dp', 'fsdp', 'pp' and 'tp'. The parallel strategies at the beginning
            of the list have larger intervals between ranks within the communication group, which tends to
            favor inter-node communication. On the other hand, the strategies towards the end of the list
            tend to favor intra-node communication.
    """
    dp: DPConfig = field(default_factory=DPConfig)
    tp: TPConfig = field(default_factory=TPConfig)
    pp: PPConfig = field(default_factory=PPConfig)
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    sp: SPConfig = field(default_factory=SPConfig)
    topology: List[str] = field(
        default_factory=lambda: ['dp', 'fsdp', 'pp', 'tp'])

    def validate(self):
        assert isinstance(self.dp,
                          DPConfig), "DistConfig.dp should be of DPConfig type"
        assert isinstance(self.tp,
                          TPConfig), "DistConfig.tp should be of PConfig type"
        assert isinstance(self.pp,
                          PPConfig), "DistConfig.dp should be of TDPConfig type"
        assert isinstance(
            self.fsdp,
            FSDPConfig), "DistConfig.fsdp should be of FSDPConfig type"
        assert isinstance(self.sp,
                          SPConfig), "DistConfig.sp should be of SPConfig type"
        assert isinstance(self.topology,
                          list), "DistConfig.topology should be of list type"

        if self.dp.size is None:
            used_gpus = self.pp.size * self.fsdp.size * self.tp.size
            assert ta.dist.world_size() % used_gpus == 0, "The configured parallel numbers " \
                "(pp.size * fsdp.size * tp.size) needs to be divisible by world size."
            self.dp.size = ta.dist.world_size() // used_gpus

        self.dp.validate()
        self.tp.validate()
        self.pp.validate()
        self.fsdp.validate()
        assert len(self.topology) == len(set(self.topology)), "There should not be duplicate elements in " \
            "DistConfig.topology"
        for t in self.topology:
            if t not in ['dp', 'fsdp', 'pp', 'tp', 'sp']:
                raise ValueError(
                    f"Expect 'dp', 'pp', 'tp' or 'fsdp' in DistConfig.topology, but got {t}"
                )


# TODO: support dict
@dataclass
class Config(BaseConfig):
    """Configuration for TorchAcc

    Args:
        backend (str): Backend used for acceleration. Options: 'lazy', 'eager'.
        compute (ComputeConfig): Configuration for computational optimization.
        memory (MemoryConfig): Configuration for memory optimization.
        dist (DistConfig): Configuration for distributed parallel.
        _mesh (Optional[ta.dist.Mesh]): Distributed communication component. Mesh should be obtained
            by get_mesh().
        dataloader (DataLoaderConfig): Configuration for data loader optimization.
    """
    backend: str = 'lazy'
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    dist: DistConfig = field(default_factory=DistConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)

    def validate(self):
        assert isinstance(self.backend,
                          str), "Config.backend should be of str type"
        assert isinstance(
            self.compute,
            ComputeConfig), "Config.compute should be of ComputeConfig type"
        assert isinstance(
            self.memory,
            MemoryConfig), "Config.memory should be of MemoryConfig type"
        assert isinstance(
            self.dataloader, DataLoaderConfig
        ), "Config.dataloader should be of DataLoaderConfig type"
        assert isinstance(
            self.dist, DistConfig), "Config.dist should be of DistConfig type"

        assert self.backend in ['lazy', 'eager'
                               ], "Config.backend should be 'lazy' or 'eager'"

        self.compute.validate()
        self.memory.validate()
        self.dataloader.validate()
        self.dist.validate()

    def get_mesh(self):
        """Get the distributed communication component Mesh. Mesh defines the individual communication
           groups for various distributed strategies.
        """
        if hasattr(self, "_mesh"):
            return self._mesh
        self.validate()
        if dist.is_initialized():
            assert dist.get_backend() == ta.dist.BACKEND_NAME, "The backend for initializing the distributed" \
                f" process group should be {ta.dist.BACKEND_NAME}."
        else:
            dist.init_process_group(backend=ta.dist.BACKEND_NAME)
            dist.barrier()
        if self.dist.sp.size > 1 and os.getenv('XLA_USE_SPMD', '0') == '0':
            context_parallel.initialize_context_parallel(self.dist.sp.size)
        self._mesh = ta.dist.Mesh(
            dp_num=self.dist.dp.size,
            pp_num=self.dist.pp.size,
            tp_num=self.dist.tp.size,
            fsdp_num=self.dist.fsdp.size,
            sp_num=self.dist.sp.size,
            topology=self.dist.topology)
        ta.get_global_context().mesh = self._mesh
        return self._mesh

    def is_distributed_parallel(self):
        """Whether it is distributed parallel.
        """
        if self.dist.dp.size > 1:
            return True
        if self.dist.tp.size > 1:
            return True
        if self.dist.pp.size > 1:
            return True
        if self.dist.fsdp.size > 1:
            return True
        if self.dist.sp.size > 1:
            return True
        return False

    def is_tracing_enabled(self):
        """Whether tracing of model is enabled.
           Currently only pipeline parallelism will enable tracing.
        """
        return self.dist.pp.size > 1

    def is_lazy_backend(self):
        """Whether lazy backend is enabled.
        """
        return self.backend == 'lazy'

    def is_eager_backend(self):
        """Whether eager backend is enabled.
        """
        return self.backend == 'eager'
