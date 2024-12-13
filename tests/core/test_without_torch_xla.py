import subprocess


def test_torchacc_without_torch_xla():
    python_code = """
import sys
sys.modules["torch_xla"] = None
from torchacc.utils.import_utils import is_torch_xla_available
assert is_torch_xla_available() == False
import torchacc
    """
    command = ['python', '-c', python_code]
    # To avoid being affected by imports from other tests,
    # this testing is done here in the subprocess.
    result = subprocess.run(command, check=True, text=True)
