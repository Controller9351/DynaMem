
"""Correctness tests for DynaMem's runtime workload monitor."""

import random

import pytest

from config import make_default_config
from data_directory import DataDirectory
from models import (
    AccessPattern,
    ComputationRequest,
    DataBlock,
)
from runtime_monitor import RuntimeMonitor


@pytest.fixture
def system():
    """Create blocks, directory and monitor with independent state."""

    config = make_default_config()

    blocks = [
        DataBlock(
            block_id=f"B{index:03d}",
            dataset_id="monitor_test",
            size_bytes=1024,
            bank_id=index % config.memory.num_banks,
        )
        for index in range(128)
    ]

    directory = DataDirectory(config, blocks)
    monitor = RuntimeMonitor(config, directory)

    return config, directory, monitor, blocks


def test_pattern_is_unknown_before_minimum_history(system):
    """The monitor should avoid classifying an undersized sample."""

    _, _, monitor, blocks = system

    for index in range(15):
        monitor.observe_access(
            blocks[index].block_id,
            time_ns=float(index),
        )

    assert monitor.get_observed_pattern("monitor_test") is None


def test_sequential_access_is_classified_as_streaming(system):
    """Consecutive accesses to new blocks should indicate streaming."""

    _, _, monitor, blocks = system

    for index, block in enumerate(blocks[:32]):
        monitor.observe_access(
            block.block_id,
            time_ns=float(index),
        )

    assert (
        monitor.get_observed_pattern("monitor_test")
        == AccessPattern.STREAMING
    )

    statistics = monitor.get_statistics("monitor_test")

    assert statistics["unique_ratio"] == pytest.approx(1.0)
    assert statistics["reuse_ratio"] == pytest.approx(0.0)
    assert statistics["sequential_ratio"] == pytest.approx(1.0)


def test_repeated_access_is_classified_as_reused(system):
    """Repeated access to a small set of blocks should indicate reuse."""

    _, _, monitor, blocks = system

    for index in range(40):
        block = blocks[index % 2]

        monitor.observe_access(
            block.block_id,
            time_ns=float(index),
        )

    assert (
        monitor.get_observed_pattern("monitor_test")
        == AccessPattern.REUSED
    )

    statistics = monitor.get_statistics("monitor_test")

    assert statistics["unique_ratio"] == pytest.approx(2 / 40)
    assert statistics["reuse_ratio"] == pytest.approx(1 - 2 / 40)


def test_random_access_is_classified_as_random(system):
    """A shuffled sequence with weak adjacency should indicate random access."""

    _, _, monitor, blocks = system

    indices = list(range(64))
    random.Random(2026).shuffle(indices)

    for time_ns, index in enumerate(indices):
        monitor.observe_access(
            blocks[index].block_id,
            time_ns=float(time_ns),
        )

    statistics = monitor.get_statistics("monitor_test")

    assert statistics["unique_ratio"] == pytest.approx(1.0)
    assert (
        statistics["sequential_ratio"]
        < 0.65
    )

    assert (
        monitor.get_observed_pattern("monitor_test")
        == AccessPattern.RANDOM
    )


def test_statistics_reflect_the_sliding_window(system):
    """Only accesses in the configured window should affect statistics."""

    config, _, monitor, blocks = system

    # Generate more accesses than the configured history capacity.
    total_accesses = config.runtime.history_window + 20

    for index in range(total_accesses):
        monitor.observe_access(
            blocks[index % len(blocks)].block_id,
            time_ns=float(index),
        )

    statistics = monitor.get_statistics("monitor_test")

    assert statistics["window_size"] == config.runtime.history_window
    assert statistics["window_size"] == 64

    assert monitor.total_observed_accesses == total_accesses


def test_dataset_histories_are_independent():
    """Access history from one dataset must not classify another."""

    config = make_default_config()

    blocks = [
        DataBlock("A001", "dataset_A", 1024, 0),
        DataBlock("A002", "dataset_A", 1024, 1),
        DataBlock("B001", "dataset_B", 1024, 2),
        DataBlock("B002", "dataset_B", 1024, 3),
    ]

    directory = DataDirectory(config, blocks)
    monitor = RuntimeMonitor(config, directory)

    # Repeatedly access only dataset A.
    for index in range(20):
        block_id = "A001" if index % 2 == 0 else "A002"

        monitor.observe_access(
            block_id,
            time_ns=float(index),
        )

    assert (
        monitor.get_observed_pattern("dataset_A")
        == AccessPattern.REUSED
    )

    assert monitor.get_observed_pattern("dataset_B") is None
    assert monitor.get_statistics("dataset_B")["window_size"] == 0


def test_declared_request_pattern_does_not_control_inference(system):
    """Inference must use observed blocks, not the request's label."""

    _, directory, monitor, blocks = system

    # Label every request as RANDOM, but present consecutive block accesses.
    for index, block in enumerate(blocks[:24]):
        request = ComputationRequest(
            request_id=f"R{index:03d}",
            dataset_id="monitor_test",
            operation="BITMAP_AND",
            input_block_ids=(block.block_id,),
            output_bytes=128,
            operation_count=100,
            access_pattern=AccessPattern.RANDOM,
            arrival_time_ns=float(index),
        )

        monitor.observe_request(
            request,
            time_ns=float(index),
        )

    # The monitor observes the actual sequence of consecutive block IDs.
    assert (
        monitor.get_observed_pattern("monitor_test")
        == AccessPattern.STREAMING
    )

    assert monitor.total_observed_accesses == 24
    assert directory.total_accesses == 24


def test_monitor_adapts_after_access_pattern_changes(system):
    """Recent observations should eventually replace the previous classification."""

    _, _, monitor, blocks = system

    time_ns = 0.0

    # Phase 1: sequential access.
    for block in blocks[:64]:
        monitor.observe_access(
            block.block_id,
            time_ns=time_ns,
        )
        time_ns += 1.0

    assert (
        monitor.get_observed_pattern("monitor_test")
        == AccessPattern.STREAMING
    )

    # Phase 2: repeatedly access two blocks.
    for index in range(40):
        block = blocks[index % 2]

        monitor.observe_access(
            block.block_id,
            time_ns=time_ns,
        )
        time_ns += 1.0

    assert (
        monitor.get_observed_pattern("monitor_test")
        == AccessPattern.REUSED
    )

    # Phase 3: access a shuffled set of distinct blocks.
    indices = list(range(64))
    random.Random(77).shuffle(indices)

    for index in indices:
        monitor.observe_access(
            blocks[index].block_id,
            time_ns=time_ns,
        )
        time_ns += 1.0

    assert (
        monitor.get_observed_pattern("monitor_test")
        == AccessPattern.RANDOM
    )


def test_invalid_time_is_rejected_without_recording_access(system):
    """Negative timestamps must not update access statistics."""

    _, directory, monitor, blocks = system

    with pytest.raises(ValueError):
        monitor.observe_access(
            blocks[0].block_id,
            time_ns=-1.0,
        )

    assert monitor.total_observed_accesses == 0
    assert directory.total_accesses == 0


def test_time_cannot_move_backwards(system):
    """Observed timestamps must be globally non-decreasing."""

    _, directory, monitor, blocks = system

    monitor.observe_access(
        blocks[0].block_id,
        time_ns=10.0,
    )

    with pytest.raises(ValueError):
        monitor.observe_access(
            blocks[1].block_id,
            time_ns=9.0,
        )

    assert monitor.total_observed_accesses == 1
    assert directory.total_accesses == 1


def test_unknown_block_is_rejected(system):
    """The monitor must not invent information for unregistered blocks."""

    _, directory, monitor, _ = system

    with pytest.raises(KeyError):
        monitor.observe_access(
            "UNKNOWN_BLOCK",
            time_ns=1.0,
        )

    assert monitor.total_observed_accesses == 0
    assert directory.total_accesses == 0


def test_reuse_and_unique_ratios_sum_to_one(system):
    """The unique and reuse ratios must be complementary."""

    _, _, monitor, blocks = system

    pattern = [
        blocks[0],
        blocks[1],
        blocks[0],
        blocks[2],
        blocks[1],
        blocks[0],
        blocks[2],
        blocks[2],
        blocks[1],
        blocks[0],
        blocks[1],
        blocks[2],
        blocks[0],
        blocks[1],
        blocks[2],
        blocks[0],
    ]

    for time_ns, block in enumerate(pattern):
        monitor.observe_access(
            block.block_id,
            time_ns=float(time_ns),
        )

    statistics = monitor.get_statistics("monitor_test")

    assert (
        statistics["unique_ratio"]
        + statistics["reuse_ratio"]
    ) == pytest.approx(1.0)

    assert statistics["unique_blocks"] == 3
    assert statistics["window_size"] == 16
