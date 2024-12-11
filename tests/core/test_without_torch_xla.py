import sys


def test_torchacc_without_torch_xla():
    # disable the torch_xla in environment
    sys.modules["torch_xla"] = None
    from torchacc.utils.import_utils import is_torch_xla_available
    assert is_torch_xla_available() == False
    import torchacc
