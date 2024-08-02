import importlib
import sys

import accelerate
import transformers

import torchacc

_ACCELERATE_HF_TRAINER = False
RETURN_TRUE = lambda check_device=True: True
RETURN_FALSE = lambda: False


def _func_check_and_set(module_name, func_name, func_value):
    err_msg = f'''{func_name} not found in {module_name}.
                  Replace HF {module_name.__package__} with an older version.'''
    assert hasattr(module_name, func_name), err_msg
    setattr(module_name, func_name, func_value)


def patch_accelerate():
    # return true in is_tpu_available to enable torch_xla fsdp.
    _func_check_and_set(accelerate.utils.imports, 'is_tpu_available',
                        RETURN_TRUE)
    _func_check_and_set(accelerate.utils, 'is_tpu_available', RETURN_TRUE)
    _func_check_and_set(accelerate.state, 'is_tpu_available', RETURN_TRUE)

    # reload these modules to import torch_xla
    importlib.reload(accelerate.checkpointing)
    importlib.reload(accelerate.data_loader)
    importlib.reload(accelerate.optimizer)
    importlib.reload(accelerate.utils.operations)
    importlib.reload(accelerate.utils.other)
    importlib.reload(accelerate.utils.random)

    # put accelerate.state last to make state singleton working
    importlib.reload(accelerate.state)
    _func_check_and_set(accelerate.state, 'is_tpu_available', RETURN_TRUE)


def patch_transformers():
    # return true in is_tpu_available to enable torch_xla fsdp.
    _func_check_and_set(transformers, 'is_torch_tpu_available', RETURN_TRUE)

    # tensorboard does not support nested json format which used in fsdp
    _func_check_and_set(transformers.integrations.integration_utils,
                        'is_tensorboard_available', RETURN_FALSE)
    # reload training_args because it uses accelerate.state and accelerate.utils
    importlib.reload(transformers.training_args)


def accelerate_hf_trainer(enable=True):
    '''Accelerate huggingface transformers model training with torchacc.
    This API should cooperate with huggingface trainer API.

    Put the function calling at the top of your python main entrance file.
    ```python
    import torchacc
    torchacc.accelerate_hf_trainer()
    ```

    Args:
    enable: Enable acceleration when set to true.
    '''
    global _ACCELERATE_HF_TRAINER
    _ACCELERATE_HF_TRAINER = enable

    if enable:
        # apply patches， patch accelerate first.
        patch_accelerate()
        patch_transformers()

        # get lazy device and set_replication to init pjrt client
        _ = torchacc.lazy_device()
    else:
        # set torch_xla with None to disable torch xla
        sys.modules['torch_xla'] = None
