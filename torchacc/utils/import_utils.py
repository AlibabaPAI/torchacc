import importlib

IS_TORCH_XLA_AVAILABLE = None


def is_torch_xla_available():
    global IS_TORCH_XLA_AVAILABLE
    if IS_TORCH_XLA_AVAILABLE is None:
        IS_TORCH_XLA_AVAILABLE = importlib.util.find_spec(
            "torch_xla") is not None
    return IS_TORCH_XLA_AVAILABLE
