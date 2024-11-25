from torchacc.utils.import_utils import is_torch_xla_available
if is_torch_xla_available():
    import torch_xla.distributed.spmd as xs
    Mesh = xs.Mesh
    mark_sharding = xs.mark_sharding
