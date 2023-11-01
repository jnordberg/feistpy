import math
from typing import TYPE_CHECKING, Iterable, Iterator, Literal, Optional, Union

from enum import Enum

from pygfc import Permutation

if TYPE_CHECKING:
    from torch.utils.data import Dataset


class RoundingMode(Enum):
    """Enum for specifying rounding modes when the dataset is not divisible by the number of replicas.

    Attributes:
        truncate: Truncate the dataset to a length that is divisible by the number of replicas.
        pad: Pad the dataset to a length divisible by the number of replicas, repeating items.
        uneven: Retain dataset length, allowing each replica to have a different number of items.
    """

    truncate = "truncate"
    pad = "pad"
    uneven = "uneven"


class SamplingStrategy(Enum):
    """Enum for specifying the sampling strategy to use.

    Attributes:
        block: Each replica gets a contiguous block of indices. When shuffled, blocks are shuffled independently.
        shard: Each replica gets a slice of the indices. When shuffled, all indices are shuffled together.
    """

    block = "block"
    shard = "shard"


class FeistelSampler(Iterable[int]):
    """PyTorch Sampler using a Feistel permutation to sample from a dataset.

    This sampler is a drop-in replacement for `torch.utils.data.DistributedSampler` but
    it can also be used in a single-process setting.

    The only behavior that is different from `DistributedSampler` is that by default,
    this sampler truncates the dataset to the nearest length that is evenly divisible by
    the number of replicas (which is probably what you want). This can be changed by
    setting `mode="pad"` or `mode="uneven"`.

    Max items supported: 2^64
    """

    def __init__(
        self,
        dataset: Union["Dataset", int],
        num_replicas: int = 1,
        rank: int = 0,
        batch_size: int = 1,
        shuffle: bool = True,
        seed: int = 0,
        rounds: int = 8,
        mode: Union[RoundingMode, Literal["truncate", "pad", "uneven"]] = RoundingMode.truncate,
        strategy: Union[SamplingStrategy, Literal["block", "shard"]] = SamplingStrategy.block,
    ):
        """Initializes the FeistelSampler.

        Args:
            dataset: The source dataset or its length.
            num_replicas: Number of replicas (processes) to sample for. Default is 1.
            rank: Rank of the current process. Default is 0.
            batch_size: Batch size, only effective when strategy="shard". Default is 1.
            shuffle: If True, shuffles the dataset. Default is True.
            seed: Seed for the Feistel permutation (used if shuffle=True). Default is 0.
            rounds: Number of rounds for the Feistel permutation (used if shuffle=True). Default is 8.
            mode: Mode when dataset is not divisible by the number of replicas.
                Options are "truncate", "pad", or "uneven". Default is "truncate".
            strategy: Sampling strategy to adopt. Can be "block" or "shard". Default is "block".

        Notes:
            - When shuffling in "block" mode, each block is shuffled independently.
            - "uneven" mode is not recommended for distributed training as it will likely cause
                size mismatches between replicas.
        """
        if isinstance(mode, str):
            mode = RoundingMode(mode)
        if isinstance(strategy, str):
            strategy = SamplingStrategy(strategy)

        assert num_replicas > 0, "num_replicas must be positive"
        assert rank >= 0 and rank < num_replicas, "rank must be in the interval [0, num_replicas)"

        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.epoch = 0
        self.rounds = rounds
        self.mode = mode
        self.shuffle = shuffle
        self.seed = seed
        self.strategy = strategy

        self.num_items = int(dataset) if isinstance(dataset, int) else len(dataset)  # type: ignore
        bz = 1 if self.strategy == SamplingStrategy.block else self.batch_size
        if self.mode != RoundingMode.truncate:
            # Dataset is not evenly divisible and we're not truncating, so round up.
            self.block_size = math.ceil(self.num_items / self.num_replicas / bz) * bz
        else:
            # Dataset is evenly divisible or we're truncating, so round down.
            self.block_size = math.floor(self.num_items / self.num_replicas / bz) * bz

        if self.mode == RoundingMode.uneven and self.rank == self.num_replicas - 1:
            # Uneven mode, last replica gets the remainder.
            self.replica_samples = self.num_items - self.block_size * (self.num_replicas - 1)
            self.total_samples = self.num_items
        else:
            self.replica_samples = self.block_size
            self.total_samples = self.block_size * self.num_replicas

    def _block_iter(self) -> Iterator[int]:
        start = self.rank * self.block_size
        if self.shuffle:
            index_iter = Permutation(
                self.replica_samples, self.rounds, self.seed + self.epoch + self.rank
            )
        else:
            index_iter = iter(range(self.replica_samples))
        for i in index_iter:
            yield (start + i) % self.num_items

    def _shard_iter(self) -> Iterator[int]:
        num_samples = self.num_items if self.mode == RoundingMode.uneven else self.total_samples
        start = self.rank * self.batch_size
        stride = self.num_replicas * self.batch_size
        index_iter = iter(range(start, num_samples, stride))
        if self.batch_size > 1:
            index_iter = (
                i + j for i in index_iter for j in range(self.batch_size) if i + j < num_samples
            )
        if self.shuffle:
            perm = Permutation(num_samples, self.rounds, self.seed + self.epoch)
            index_iter = (perm[i] for i in index_iter)
        return (i % self.num_items for i in index_iter)

    def __iter__(self) -> Iterator[int]:
        if self.strategy == SamplingStrategy.block:
            return self._block_iter()
        elif self.strategy == SamplingStrategy.shard:
            return self._shard_iter()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def __len__(self) -> int:
        return self.replica_samples

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for the sampler.

        Effective only when shuffle is set to True.

        Args:
            epoch: The epoch number to set.
        """
        self.epoch = epoch
