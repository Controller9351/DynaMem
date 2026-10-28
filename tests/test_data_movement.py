
"""Tests for DynaMem's simulated data-movement accounting."""

import pytest

from config import make_default_config
from models import AccessPattern, ComputationRequest, DataBlock
from simulator import DynaMemSimulator
from workload_generator import WorkloadTrace


@pytest.fixture
def config():
    """Return a fresh architecture configuration for each test."""
    return make_default_config()


def make_request(
    request_id="R001",
    block_ids=("B001", "B002"),
    dataset_id="movement_test",
    output_bytes=512,
    arrival_time_ns=0.0,
):
    """Create a bitmap operation with known input and output sizes."""
    return ComputationRequest(
        request_id=request_id,
        dataset_id=dataset_id,
        operation="BITMAP_AND",
        input_block_ids=block_ids,
        output_bytes=output_bytes,
        operation_count=1000,
        access_pattern=AccessPattern.STREAMING,
        arrival_time_ns=arrival_time_ns,
    )


def make_trace(blocks, requests, name="movement_test"):
    """Build a trace with a known data layout and request sequence."""
    return WorkloadTrace(
        name=name,
        blocks=tuple(blocks),
        requests=tuple(requests),
    )


@pytest.fixture
def two_bank_trace():
    """Two equal-sized inputs stored in different memory banks."""
    blocks = [
        DataBlock("B001", "movement_test", 4096, 0),
        DataBlock("B002", "movement_test", 4096, 1),
    ]

    requests = [
        make_request(),
    ]

    return make_trace(blocks, requests)


def run_single_request(config, trace, policy_name):
    """Run one policy and return its completed execution record."""
    simulator = DynaMemSimulator(
        config=config,
        trace=trace,
        policy_name=policy_name,
    )

    result = simulator.run()

    assert len(result.records) == 1

    return result.records[0]


def test_cpu_counts_input_and_output_movement(config, two_bank_trace):
    """CPU inputs come from memory; output is committed to memory."""
    record = run_single_request(
        config,
        two_bank_trace,
        "CPU_ONLY",
    )

    assert record.input_bytes_moved == 8192
    assert record.output_bytes_moved == 512
    assert record.cross_bank_bytes_moved == 0

    assert record.total_moved_bytes == 8704

    # CPU_MEMORY: latency + bytes / bandwidth, for input and output.
    expected_transfer_time = (
        30.0 + 8192 / 32.0
        + 30.0 + 512 / 32.0
    )

    assert record.transfer_time_ns == pytest.approx(
        expected_transfer_time
    )


def test_nmp_counts_input_and_output_movement(config, two_bank_trace):
    """NMP transfers inputs to the engine and commits the result."""
    record = run_single_request(
        config,
        two_bank_trace,
        "STATIC_NMP",
    )

    assert record.input_bytes_moved == 8192
    assert record.output_bytes_moved == 512
    assert record.cross_bank_bytes_moved == 0

    assert record.total_moved_bytes == 8704

    expected_transfer_time = (
        8.0 + 8192 / 128.0
        + 8.0 + 512 / 128.0
    )

    assert record.transfer_time_ns == pytest.approx(
        expected_transfer_time
    )


def test_cim_counts_only_remote_input_movement(config, two_bank_trace):
    """CIM records cross-bank traffic while keeping its output local."""
    record = run_single_request(
        config,
        two_bank_trace,
        "STATIC_CIM_A",
    )

    # Equal-sized inputs in banks 0 and 1 lead to a tie; the current
    # deterministic policy chooses CIM_A:0.
    assert record.resource_name == "CIM_A:0"

    # B001 is local; B002 must move from bank 1 to bank 0.
    assert record.input_bytes_moved == 0
    assert record.output_bytes_moved == 0
    assert record.cross_bank_bytes_moved == 4096

    assert record.total_moved_bytes == 4096

    expected_transfer_time = 25.0 + 4096 / 16.0

    assert record.transfer_time_ns == pytest.approx(
        expected_transfer_time
    )


def test_local_cim_inputs_need_no_interbank_transfer(config):
    """Inputs already in the selected bank incur no inter-bank traffic."""
    blocks = [
        DataBlock("B001", "local_test", 4096, 0),
    ]

    requests = [
        make_request(
            dataset_id="local_test",
            block_ids=("B001",),
        ),
    ]

    trace = make_trace(
        blocks,
        requests,
        name="local_cim_test",
    )

    record = run_single_request(
        config,
        trace,
        "STATIC_CIM_A",
    )

    assert record.resource_name == "CIM_A:0"
    assert record.input_bytes_moved == 0
    assert record.output_bytes_moved == 0
    assert record.cross_bank_bytes_moved == 0
    assert record.total_moved_bytes == 0
    assert record.transfer_time_ns == 0.0


def test_cpu_cache_reuse_avoids_repeated_input_transfer(config):
    """A second CPU request reuses previously cached inputs."""
    blocks = [
        DataBlock("B001", "cache_test", 4096, 0),
        DataBlock("B002", "cache_test", 4096, 1),
    ]

    requests = [
        make_request(
            request_id="R001",
            dataset_id="cache_test",
            arrival_time_ns=0.0,
        ),
        make_request(
            request_id="R002",
            dataset_id="cache_test",
            arrival_time_ns=1000.0,
        ),
    ]

    trace = make_trace(
        blocks,
        requests,
        name="cache_reuse_test",
    )

    simulator = DynaMemSimulator(
        config,
        trace,
        policy_name="CPU_ONLY",
    )

    result = simulator.run()

    records = {
        record.request_id: record
        for record in result.records
    }

    first = records["R001"]
    second = records["R002"]

    assert first.input_bytes_moved == 8192
    assert second.input_bytes_moved == 0

    # The output is still committed to memory for both requests.
    assert first.output_bytes_moved == 512
    assert second.output_bytes_moved == 512

    assert first.total_moved_bytes == 8704
    assert second.total_moved_bytes == 512

    assert result.cache_summary["total_hits"] == 2
    assert result.cache_summary["total_misses"] == 2


def test_movement_categories_are_nonnegative(config, two_bank_trace):
    """Every movement category must be non-negative."""
    policies = (
        "CPU_ONLY",
        "STATIC_CIM_A",
        "STATIC_CIM_P",
        "STATIC_NMP",
    )

    for policy in policies:
        record = run_single_request(
            config,
            two_bank_trace,
            policy,
        )

        assert record.input_bytes_moved >= 0
        assert record.output_bytes_moved >= 0
        assert record.cross_bank_bytes_moved >= 0

        assert record.total_moved_bytes == (
            record.input_bytes_moved
            + record.output_bytes_moved
            + record.cross_bank_bytes_moved
        )


def test_simulation_summary_matches_execution_records(config, two_bank_trace):
    """Aggregate data-movement totals must match individual records."""
    simulator = DynaMemSimulator(
        config,
        two_bank_trace,
        policy_name="CPU_ONLY",
    )

    result = simulator.run()
    summary = result.summary()

    assert summary["input_bytes_moved"] == sum(
        record.input_bytes_moved for record in result.records
    )

    assert summary["output_bytes_moved"] == sum(
        record.output_bytes_moved for record in result.records
    )

    assert summary["cross_bank_bytes_moved"] == sum(
        record.cross_bank_bytes_moved for record in result.records
    )

    assert summary["total_bytes_moved"] == sum(
        record.total_moved_bytes for record in result.records
    )


def test_cim_remote_transfer_is_not_counted_as_cpu_traffic(
    config,
    two_bank_trace,
):
    """Cross-bank movement is separate from CPU/NMP input transfers."""
    record = run_single_request(
        config,
        two_bank_trace,
        "STATIC_CIM_P",
    )

    assert record.input_bytes_moved == 0
    assert record.output_bytes_moved == 0
    assert record.cross_bank_bytes_moved == 4096
