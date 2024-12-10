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

        lhs = torch.rand([2, 2], device=device)
        rhs = torch.rand([2, 2], device=device)

        first, second = self._matmul_with_autocast(lhs, rhs, 'cuda', 'cuda')
        self.assertEqual(first.dtype, torch.bfloat16)
        self.assertEqual(second.dtype, torch.float32)

        first, second = self._matmul_with_autocast(lhs, rhs, 'xla', 'xla')
        self.assertEqual(first.dtype, torch.bfloat16)
        self.assertEqual(second.dtype, torch.float32)

        first, second = self._matmul_with_autocast(lhs, rhs, 'cuda', 'xla')
        self.assertEqual(first.dtype, torch.bfloat16)
        self.assertEqual(second.dtype, torch.float32)

        first, second = self._matmul_with_autocast(lhs, rhs, 'xla', 'cuda')
        self.assertEqual(first.dtype, torch.bfloat16)
        self.assertEqual(second.dtype, torch.float32)
