from torch.utils.data import Dataset


def set_seed(seed=2023):
    import torch
    torch.manual_seed(seed)


class EchoDataset(object):
    """A dataset returns the complete data in each iteration.
    """

    def __init__(self, data, repeat_count):
        self._data = data
        self._repeat_count = repeat_count
        self._count = 0

    def __iter__(self):
        return EchoDataset(self._data, self._repeat_count)

    def __len__(self):
        return self._repeat_count

    def __next__(self):
        if self._count >= self._repeat_count:
            raise StopIteration
        self._count += 1
        return self._data


class RawDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
