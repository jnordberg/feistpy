# feistpy

Dataset sampling for machine learning using Feistel networks.

## Installation

```bash
pip install feistpy
```

## Usage

Standalone:

```python

from feistpy import FeistelSampler

sampler = FeistelSampler(
    1_000_000_000,  # total number of samples
    rank=0,  # rank of this process
    num_replicas=32,  # aka world size
)

# iterate over indices
for i in sampler:
    print(i)

```

With PyTorch:

```python
from feistpy import FeistelSampler
from torch.utils.data import DataLoader
import torch.distributed as dist

dataset = ...  # some dataset

sampler = FeistelSampler(
    dataset,
    rank=dist.get_rank(),
    num_replicas=dist.get_world_size(),
)

loader = DataLoader(
    dataset,
    batch_size=8192,
    num_workers=8,
    sampler=sampler,
)

for epoch in range(100):
    sampler.set_epoch(epoch)
    for batch in loader:
        # do something with batch

```

## Benefits

- Small memory footprint and fast sampling
- Deterministic shuffling across ranks and epochs
- Supports up to 2^64 items
- Advanced sampling strategies (see [sampler.py](./src/feistpy/sampler.py))

## Acknowledgements

This library uses the excellent [gfc](https://github.com/maxmouchet/gfc) library for the
fast generation of Feistel permutations.

## License

MIT
