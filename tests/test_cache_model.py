
"""Correctness tests for DynaMem's block-level LRU cache."""

import pytest

from cache_model import CacheModel, CPU_CACHE_LOCATION
from config import make_default_config
from data_directory import DataDirectory
from models import DataBlock


@pytest.fixture
def system():
    """Create a fresh cache with three 3 MiB blocks."""

    config = make_default_config()
    block_size = 3 * 1024 * 1024

    blocks = [
        DataBlock("B001", "cache_test", block_size, 0),
        DataBlock("B002", "cache_test", block_size, 1),
        DataBlock("B003", "cache_test", block_size, 2),
        DataBlock("B004", "cache_test", block_size, 3),
    ]

    directory = DataDirectory(config, blocks)
    cache = CacheModel(config, directory)

    return config, directory, cache, block_size


def test_first_access_misses_and_fills_cache(system):
    """Previously unseen blocks miss and become cache-resident."""

    _, directory, cache, block_size = system

    result = cache.access_blocks(
        ["B001", "B002"],
        time_ns=10.0,
    )

    assert result.requested_blocks == 2
    assert result.cache_hits == 0
    assert result.cache_misses == 2
    assert result.input_bytes_transferred == 2 * block_size

    assert cache.contains("B001")
    assert cache.contains("B002")

    assert directory.has_copy("B001", CPU_CACHE_LOCATION)
    assert directory.has_copy("B002", CPU_CACHE_LOCATION)

    assert cache.used_bytes == 2 * block_size


def test_repeated_access_hits_without_retransferring_inputs(system):
    """Resident blocks can be accessed without another input transfer."""

    _, _, cache, block_size = system

    cache.access_blocks(["B001", "B002"], time_ns=10.0)

    result = cache.access_blocks(
        ["B001", "B002"],
        time_ns=20.0,
    )

    assert result.cache_hits == 2
    assert result.cache_misses == 0
    assert result.input_bytes_transferred == 0
    assert result.transfer_time_ns == 0.0

    assert result.hit_ratio == 1.0
    assert cache.used_bytes == 2 * block_size


def test_lru_eviction_removes_the_least_recently_used_block(system):
    """Access order must determine which block is evicted."""

    _, directory, cache, block_size = system

    cache.access_blocks(
        ["B001", "B002"],
        time_ns=10.0,
    )

    # B001 becomes the most recently used block.
    cache.access_blocks(["B001"], time_ns=20.0)

    # B002 is now the least recently used block.
    result = cache.access_blocks(
        ["B003"],
        time_ns=30.0,
    )

    assert result.evicted_block_ids == ("B002",)

    assert cache.contains("B001")
    assert not cache.contains("B002")
    assert cache.contains("B003")

    assert directory.has_copy("B001", CPU_CACHE_LOCATION)
    assert not directory.has_copy("B002", CPU_CACHE_LOCATION)
    assert directory.has_copy("B003", CPU_CACHE_LOCATION)

    assert cache.used_bytes == 2 * block_size
    assert cache.used_bytes <= cache.capacity_bytes


def test_cache_never_exceeds_its_capacity(system):
    """A cache fill must evict enough blocks to respect capacity."""

    _, _, cache, block_size = system

    result = cache.access_blocks(
        ["B001", "B002", "B003"],
        time_ns=10.0,
    )

    assert result.cache_misses == 3
    assert cache.used_bytes <= cache.capacity_bytes

    # Three blocks need 9 MiB, while the default cache holds 8 MiB.
    assert cache.used_bytes == 2 * block_size
    assert len(result.evicted_block_ids) == 1


def test_oversized_block_is_read_but_not_cached(system):
    """A block larger than the cache must not remain resident."""

    config, directory, _, _ = system

    oversized_block = DataBlock(
        "BIG",
        "oversized_test",
        config.cache.capacity_bytes + 1,
        0,
    )

    oversized_directory = DataDirectory(
        config,
        [oversized_block],
    )

    oversized_cache = CacheModel(
        config,
        oversized_directory,
    )

    result = oversized_cache.access_blocks(
        ["BIG"],
        time_ns=1.0,
    )

    assert result.cache_misses == 1
    assert result.input_bytes_transferred == (
        oversized_block.size_bytes
    )

    assert not oversized_cache.contains("BIG")
    assert not oversized_directory.has_copy(
        "BIG",
        CPU_CACHE_LOCATION,
    )

    assert oversized_cache.used_bytes == 0


def test_repeated_block_ids_in_one_batch_are_deduplicated(system):
    """A batch processes each unique block only once."""

    _, _, cache, block_size = system

    result = cache.access_blocks(
        ["B001", "B001", "B002", "B001"],
        time_ns=5.0,
    )

    assert result.requested_blocks == 2
    assert result.cache_hits + result.cache_misses == 2

    assert result.cache_misses == 2
    assert result.input_bytes_transferred == 2 * block_size


def test_cache_counters_and_hit_ratio_are_consistent(system):
    """Cumulative statistics must match individual access outcomes."""

    _, _, cache, block_size = system

    first = cache.access_blocks(
        ["B001", "B002"],
        time_ns=10.0,
    )

    second = cache.access_blocks(
        ["B001", "B003"],
        time_ns=20.0,
    )

    # B001 hits; B003 misses and triggers an eviction if needed.
    assert first.cache_hits == 0
    assert first.cache_misses == 2

    assert second.cache_hits == 1
    assert second.cache_misses == 1

    summary = cache.summary()

    assert summary["total_hits"] == 1
    assert summary["total_misses"] == 3
    assert summary["hit_ratio"] == pytest.approx(0.25)

    assert summary["total_bytes_transferred"] == 3 * block_size
    assert summary["total_evictions"] == 1


def test_flush_removes_all_directory_copies(system):
    """Flushing empties cache occupancy and removes cache metadata."""

    _, directory, cache, _ = system

    cache.access_blocks(
        ["B001", "B002"],
        time_ns=10.0,
    )

    cache.flush()

    assert cache.used_bytes == 0
    assert cache.resident_block_ids == ()

    assert not directory.has_copy("B001", CPU_CACHE_LOCATION)
    assert not directory.has_copy("B002", CPU_CACHE_LOCATION)

    summary = cache.summary()

    assert summary["total_hits"] == 0
    assert summary["total_misses"] == 0
    assert summary["total_bytes_transferred"] == 0
    assert summary["total_evictions"] == 0


def test_flush_can_preserve_historical_statistics(system):
    """A cache reset can preserve counters when requested."""

    _, _, cache, block_size = system

    cache.access_blocks(
        ["B001"],
        time_ns=10.0,
    )

    cache.flush(reset_statistics=False)

    summary = cache.summary()

    assert cache.used_bytes == 0
    assert summary["resident_blocks"] == 0

    assert summary["total_misses"] == 1
    assert summary["total_bytes_transferred"] == block_size


def test_empty_access_batch_is_rejected(system):
    """A cache access must specify at least one block."""

    _, _, cache, _ = system

    with pytest.raises(ValueError):
        cache.access_blocks([], time_ns=0.0)


def test_access_time_cannot_move_backwards(system):
    """Cache operations cannot travel backwards in simulated time."""

    _, _, cache, _ = system

    cache.access_blocks(
        ["B001"],
        time_ns=20.0,
    )

    with pytest.raises(ValueError):
        cache.access_blocks(
            ["B002"],
            time_ns=10.0,
        )


def test_unknown_block_is_rejected(system):
    """Unregistered blocks must not be inserted into the cache."""

    _, _, cache, _ = system

    with pytest.raises(KeyError):
        cache.access_blocks(
            ["UNKNOWN"],
            time_ns=10.0,
        )


def test_cache_line_rounding_is_respected(system):
    """Resident capacity is rounded up to cache-line boundaries."""

    config, directory, cache, _ = system

    small_block = DataBlock(
        "SMALL",
        "rounding_test",
        65,
        4,
    )

    rounding_directory = DataDirectory(
        config,
        [small_block],
    )

    rounding_cache = CacheModel(
        config,
        rounding_directory,
    )

    rounding_cache.access_blocks(
        ["SMALL"],
        time_ns=1.0,
    )

    # A 65-byte block occupies two 64-byte cache lines.
    assert rounding_cache.used_bytes == 128
