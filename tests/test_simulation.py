
"""Integration and event-timing tests for the DynaMem simulator."""

import pytest

from config import make_default_config
from models import AccessPattern, ComputationRequest, DataBlock
from simulator import DynaMemSimulator
from workload_generator import WorkloadTrace


@pytest.fixture
def config():
    """Return a fresh architecture configuration."""
    return make_default_config()


def make_trace(
    name="simulation_test",
    num_requests=3,
    arrival_interval_ns=100.0,
    block_size_bytes=1024,
    operation_count=1000,
):
    """Create a small deterministic workload for simulator tests."""

    dataset_id = "simulation_test"

    blocks = [
        DataBlock(
            block_id=f"B{index:03d}",
            dataset_id=dataset_id,
            size_bytes=block_size_bytes,
            bank_id=index % 8,
        )
        for index in range(max(2, num_requests))
    ]

    requests = []

    for index in range(num_requests):
        requests.append(
            ComputationRequest(
                request_id=f"R{index:03d}",
                dataset_id=dataset_id,
                operation="BITMAP_AND",
                input_block_ids=(
                    f"B{index:03d}",
                ),
                output_bytes=128,
                operation_count=operation_count,
                access_pattern=AccessPattern.STREAMING,
                arrival_time_ns=index * arrival_interval_ns,
            )
        )

    return WorkloadTrace(
        name=name,
        blocks=tuple(blocks),
        requests=tuple(requests),
    )


def test_all_requests_complete(config):
    """Every submitted request must have one completed record."""

    trace = make_trace(num_requests=5)

    simulator = DynaMemSimulator(
        config,
        trace,
        policy_name="CPU_ONLY",
    )

    result = simulator.run()

    assert len(result.records) == 5
    assert result.summary()["completed_requests"] == 5

    expected_ids = {
        request.request_id for request in trace.requests
    }

    actual_ids = {
        record.request_id for record in result.records
    }

    assert actual_ids == expected_ids


def test_record_timestamps_are_consistent(config):
    """Arrival, decision, computation start and finish are ordered."""

    trace = make_trace(num_requests=4)

    result = DynaMemSimulator(
        config,
        trace,
        policy_name="CPU_ONLY",
    ).run()

    arrivals = {
        request.request_id: request.arrival_time_ns
        for request in trace.requests
    }

    for record in result.records:
        assert record.arrival_time_ns == arrivals[record.request_id]

        assert (
            record.arrival_time_ns
            <= record.decision_time_ns
            <= record.start_time_ns
            <= record.finish_time_ns
        )

        assert record.total_latency_ns == pytest.approx(
            record.finish_time_ns - record.arrival_time_ns
        )

        assert record.compute_time_ns > 0


def test_makespan_matches_first_arrival_to_last_completion(config):
    """Makespan must describe elapsed system time, not summed task time."""

    trace = make_trace(
        num_requests=5,
        arrival_interval_ns=50.0,
    )

    result = DynaMemSimulator(
        config,
        trace,
        policy_name="CPU_ONLY",
    ).run()

    first_arrival = min(
        record.arrival_time_ns for record in result.records
    )

    last_completion = max(
        record.finish_time_ns for record in result.records
    )

    assert result.makespan_ns == pytest.approx(
        last_completion - first_arrival
    )

    assert result.makespan_ns > 0


def test_controller_overhead_is_recorded(config):
    """Adaptive execution must include its configured decision overhead."""

    trace = make_trace(num_requests=3)

    result = DynaMemSimulator(
        config,
        trace,
        policy_name="ADAPTIVE",
    ).run()

    expected_overhead = config.runtime.decision_overhead_ns

    assert len(result.records) == 3

    for record in result.records:
        assert record.decision_overhead_ns == pytest.approx(
            expected_overhead
        )

    # CPU_ONLY has no adaptive-controller overhead in the current baseline.
    baseline_result = DynaMemSimulator(
        config,
        trace,
        policy_name="CPU_ONLY",
    ).run()

    assert all(
        record.decision_overhead_ns == 0
        for record in baseline_result.records
    )


def test_resource_contention_produces_queue_wait(config):
    """Concurrent requests should wait when competing for a shared resource."""

    trace = make_trace(
        name="contention_test",
        num_requests=6,
        arrival_interval_ns=0.0,
        block_size_bytes=1024,
        operation_count=100_000,
    )

    result = DynaMemSimulator(
        config,
        trace,
        policy_name="STATIC_NMP",
    ).run()

    assert len(result.records) == 6

    # NMP is a single shared engine. Under this deliberately heavy
    # workload, at least one request should wait for a resource.
    assert any(
        record.queue_wait_ns > 0
        for record in result.records
    )

    # Requests must still finish successfully despite contention.
    assert all(
        record.finish_time_ns >= record.arrival_time_ns
        for record in result.records
    )


def test_resource_utilization_stays_within_modeled_capacity(config):
    """Busy time divided by capacity and makespan must remain bounded."""

    trace = make_trace(
        num_requests=8,
        arrival_interval_ns=0.0,
        operation_count=10_000,
    )

    result = DynaMemSimulator(
        config,
        trace,
        policy_name="CPU_ONLY",
    ).run()

    utilization = result.summary()["resource_utilization"]

    for resource_name, value in utilization.items():
        assert -1e-9 <= value <= 1.0 + 1e-9, (
            f"{resource_name} utilization was {value}"
        )


def test_resource_busy_time_is_nonnegative(config):
    """No resource may have negative simulated busy time."""

    trace = make_trace(num_requests=5)

    result = DynaMemSimulator(
        config,
        trace,
        policy_name="STATIC_CIM_P",
    ).run()

    assert result.resource_busy_ns

    for resource_name, busy_time in result.resource_busy_ns.items():
        assert busy_time >= 0, resource_name


def test_repeated_runs_are_deterministic(config):
    """Identical configurations and traces should reproduce the same records."""

    trace = make_trace(
        num_requests=5,
        arrival_interval_ns=30.0,
    )

    first = DynaMemSimulator(
        config,
        trace,
        policy_name="STATIC_NMP",
    ).run()

    second = DynaMemSimulator(
        config,
        trace,
        policy_name="STATIC_NMP",
    ).run()

    assert first.records == second.records
    assert first.makespan_ns == second.makespan_ns
    assert first.resource_busy_ns == second.resource_busy_ns


def test_simulator_cannot_run_twice(config):
    """Independent experiments must construct fresh simulator instances."""

    trace = make_trace(num_requests=2)

    simulator = DynaMemSimulator(
        config,
        trace,
        policy_name="CPU_ONLY",
    )

    simulator.run()

    with pytest.raises(RuntimeError):
        simulator.run()


def test_summary_totals_match_execution_records(config):
    """Aggregate latency, energy and movement must match the records."""

    trace = make_trace(num_requests=5)

    result = DynaMemSimulator(
        config,
        trace,
        policy_name="STATIC_NMP",
    ).run()

    summary = result.summary()

    assert summary["completed_requests"] == len(result.records)

    assert summary["total_request_latency_ns"] == pytest.approx(
        sum(record.total_latency_ns for record in result.records)
    )

    assert summary["total_energy_pj"] == pytest.approx(
        sum(record.energy_pj for record in result.records)
    )

    assert summary["total_bytes_moved"] == sum(
        record.total_moved_bytes for record in result.records
    )

    assert summary["total_compute_time_ns"] == pytest.approx(
        sum(record.compute_time_ns for record in result.records)
    )
