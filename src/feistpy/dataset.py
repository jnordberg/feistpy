from typing import TYPE_CHECKING

from pygfc import Permutation

if TYPE_CHECKING:
    from torch.utils.data import Dataset


class FeistelDataset:
    """Dataset that shuffles another dataset according to a Feistel permutation."""

    def __init__(self, dataset: "Dataset", seed: int = 0, rounds: int = 8):
        assert hasattr(dataset, "__len__"), "Wrapped dataset must implement __len__"
        self.dataset = dataset
        self.num_items = len(dataset)  # type: ignore
        self.perm = Permutation(self.num_items, rounds, seed)

    def __len__(self):
        return self.num_items

    def __getitem__(self, index):
        return self.dataset[self.perm[index]]
