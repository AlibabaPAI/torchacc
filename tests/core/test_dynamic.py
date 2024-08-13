import torch
import torchacc as ta


class TestDynamic:
    def test_mark_dynamic(self):
        device = ta.lazy_device()
        t = torch.randn(3, 3, 4).to(device)
        ta.mark_dynamic(t, [-1, 1], [4, 4])
        # TODO: fix this?
        assert isinstance(t.numel(), torch.SymInt)
        assert isinstance(t.shape[0], int)
        assert isinstance(t.shape[1], torch.SymInt)
        assert isinstance(t.shape[-1], torch.SymInt)
