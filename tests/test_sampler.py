import pytest
import math
from torch.utils.data import DataLoader, Dataset

from feistpy import FeistelSampler


class DummyDataset(Dataset):
    def __init__(self, num_items):
        self.num_items = num_items

    def __len__(self):
        return self.num_items

    def __getitem__(self, index):
        return index


def format_list(items):
    return ", ".join(f"{item:02d}" for item in items)


@pytest.mark.parametrize("mode", ["truncate", "pad", "uneven"])
@pytest.mark.parametrize("strategy", ["block", "shard", "shard-striped"])
@pytest.mark.parametrize("num_replicas", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("num_items", [97, 98, 99, 100, 101])
@pytest.mark.parametrize("batch_size", [1, 3, 8])
@pytest.mark.parametrize("shuffle", [False, True], ids=["sequential", "shuffled"])
def test_feistel_sampler(mode, strategy, num_replicas, num_items, batch_size, shuffle):
    sampler_bz = 1
    if strategy.startswith("shard-striped"):
        strategy = "shard"
        sampler_bz = batch_size

    dataset = DummyDataset(num_items)

    rank_items = []
    rank_lengths = []
    rank_batches = []
    loader_steps = []
    actual_steps = []
    for rank in range(num_replicas):
        sampler = FeistelSampler(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            mode=mode,
            batch_size=sampler_bz,
            strategy=strategy,
        )
        rank_lengths.append(len(sampler))
        print(f"Rank {rank} - Length: {len(sampler)}")
        loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        items = []
        batches = []
        loader_steps.append(len(loader))
        step = 0
        for i, batch in enumerate(loader):
            batch = batch.tolist()
            print(f"Rank {rank} - B{i:02d}: {format_list(batch)}")
            items += batch
            batches.append(batch)
            step += 1
        actual_steps.append(step)
        rank_items.append(items)
        rank_batches.append(batches)

    all_items = []
    for rank in range(num_replicas):
        all_items += rank_items[rank]

    print("-" * 80)
    for rank in range(num_replicas):
        print(f"Rank {rank}: {format_list(rank_items[rank])}")

    sorted_items = sorted(all_items)

    # in block mode, when not shuffling and padding, items should be sorted
    if not shuffle and strategy == "block" and mode != "pad":
        assert sorted_items == all_items

    # samplers should report correct lengths
    assert sum(rank_lengths) == len(all_items)

    if mode != "uneven":
        assert loader_steps == actual_steps
        # all shards should have the same number of steps
        assert len(set(rank_lengths)) == 1
        # sampler lengths should be what we get
        assert rank_lengths == [len(items) for items in rank_items]

    # if we are in uneven mode, we should see all items exactly once
    if mode == "uneven":
        assert sorted_items == list(range(num_items))

    # in truncate mode we should never see more than num_items
    if mode == "truncate":
        assert len(all_items) <= num_items

    # in pad mode we should never see less than num_items
    if mode == "pad":
        assert len(all_items) >= num_items

    # we should never see any item outside of the range [0, num_items)
    assert list(item < num_items for item in all_items) == [True] * len(all_items)

    # we should only have duplicates if we are using the "pad" mode
    duplicated_items = [item for item in all_items if all_items.count(item) > 1]
    if mode != "pad":
        assert len(duplicated_items) == 0

    if strategy == "block" and not shuffle:
        # rank items should be sequential, e.g. rank 0 should see 0, 1, 2, etc
        if mode == "uneven":
            assert all_items == list(range(num_items))
        elif mode == "truncate":
            expected_num_items = math.floor(num_items / num_replicas) * num_replicas
            assert all_items == list(range(expected_num_items))
        elif mode == "pad":
            expected_first_items = list(range(num_items))
            assert all_items[:num_items] == expected_first_items
            total = math.ceil(num_items / num_replicas) * num_replicas
            remainder = total - num_items
            assert len(all_items) == total
            assert all_items[num_items:] == expected_first_items[:remainder]

    if strategy == "shard" and not shuffle:
        if mode == "truncate":
            if num_replicas > 1:
                expected_num_items = (
                    math.floor(num_items / num_replicas / sampler_bz) * num_replicas * sampler_bz
                )
            else:
                expected_num_items = num_items
            assert len(all_items) == expected_num_items
        elif mode == "pad":
            if num_replicas > 1:
                expected_num_items = (
                    math.ceil(num_items / num_replicas / sampler_bz) * num_replicas * sampler_bz
                )
            else:
                expected_num_items = num_items
            assert len(all_items) == expected_num_items
        elif mode == "uneven":
            assert len(all_items) == num_items
