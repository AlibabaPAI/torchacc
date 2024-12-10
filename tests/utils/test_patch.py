import unittest
import torch
import torchacc as ta


class PatchAutocastTest(unittest.TestCase):

    def _matmul_with_autocast(self, lhs, rhs, first_dtype, second_dtype):
        with torch.autocast(device_type=first_dtype, dtype=torch.bfloat16):
            first = torch.matmul(lhs, rhs)
            with torch.autocast(device_type=second_dtype, enabled=False):
                second = torch.matmul(lhs, rhs)
        return (first, second)

    def test_patch_autocast(self):
        device = ta.lazy_device()

        t1 = torch.rand([2,2], device=device)
        t2 = torch.rand([2,2], device=device)

        first, second = self._matmul_with_autocast(t1, t2, 'cuda', 'cuda')
        assert first.dtype==torch.bfloat16
        assert second.dtype==torch.float32

        first, second = self._matmul_with_autocast(t1, t2, 'xla', 'xla')
        assert first.dtype==torch.bfloat16
        assert second.dtype==torch.float32

        first, second = self._matmul_with_autocast(t1, t2, 'cuda', 'xla')
        assert first.dtype==torch.bfloat16
        assert second.dtype==torch.float32

        first, second = self._matmul_with_autocast(t1, t2, 'xla', 'cuda')
        assert first.dtype==torch.bfloat16
        assert second.dtype==torch.float32