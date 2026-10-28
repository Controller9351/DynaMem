
"""Correctness tests for DynaMem's cost estimator."""

import pytest

from config import make_default_config
from cost_model import CostModel
from data_directory import DataDirectory
from models import AccessPattern, ComputationRequest, DataBlock


@pytest.fixture
def system():
    """Create a fresh, independent configuration and data directory."""

    config = make_default_config()

    blocks = [
        DataBlock("B001", "demo", 4096, 0),
        DataBlock("B002", "demo", 8192, 1),
        DataBlock("B003", "demo", 16384, 2),
    ]

    directory = DataDirectory(config, blocks)
    cost_model = CostModel(config, directory)

    return config, directory, cost_model


def make_request(
    operation="BITMAP_AND",
    block_ids=("B001", "B002"),
    output_bytes=512,
    operation_count=1000,
    dataset_id="demo",
):
    """Create a valid request for testing cost estimates."""

    return ComputationRequest(
        request_id="TEST_R001",
        dataset_id=dataset_id,
        operation=operation,
        input_block_ids=block_ids,
        output_bytes=output_bytes,
        operation_count=operation_count,
        access_pattern=AccessPattern.REUSED,
    )


def test_candidates_respect_operation_support(system):
    """Only resources supporting an operation should be candidates."""

    _, _, model = system

    bitmap_candidates = set(
        model.candidate_resources(make_request())
    )

    expected_bitmap_candidates = {
        "CPU",
        "NMP",
        *(f"CIM_A:{bank}" for bank in range(8)),
        *(f"CIM_P:{bank}" for bank in range(8)),
    }

    assert bitmap_candidates == expected_bitmap_candidates

    matmul_candidates = set(
        model.candidate_resources(
            make_request(operation="MATMUL")
        )
    )

    # The current configuration assigns MATMUL to CPU and NMP.
    assert matmul_candidates == {"CPU", "NMP"}


def test_cpu_estimate_accounts_for_cache_hit_and_miss(system):
    """Cached input avoids a main-memory input transfer."""

    config, directory, model = system

    directory.add_copy("B001", "CPU_CACHE")

    request = make_request()
    estimate = model.estimate(request, "CPU")

    # B001 is cached; B002 is not.
    assert estimate.input_bytes_moved == 8192
    assert estimate.output_bytes_moved == 512

    # The corrected model charges a lookup for every input block.
    assert estimate.cache_access_time_ns == 2 * 4.0

    expected_transfer_time = (
        30.0 + 8192 / 32.0
        + 30.0 + 512 / 32.0
    )

    assert estimate.transfer_time_ns == pytest.approx(
        expected_transfer_time
    )

    assert estimate.memory_access_time_ns == pytest.approx(120.0)
    assert estimate.predicted_latency_ns > 0
    assert estimate.predicted_energy_pj > 0


def test_fully_cached_inputs_still_require_output_commit(system):
    """Cached inputs save their input transfer, not the result write."""

    _, directory, model = system

    directory.add_copy("B001", "CPU_CACHE")
    directory.add_copy("B002", "CPU_CACHE")

    estimate = model.estimate(
        make_request(),
        "CPU",
    )

    assert estimate.input_bytes_moved == 0
    assert estimate.output_bytes_moved == 512

    assert estimate.transfer_time_ns == pytest.approx(
        30.0 + 512 / 32.0
    )

    assert estimate.memory_access_time_ns == pytest.approx(60.0)


def test_nmp_estimate_accounts_for_input_and_output_transfers(system):
    """NMP execution transports inputs and returns the output."""

    _, _, model = system

    estimate = model.estimate(
        make_request(),
        "NMP",
    )

    assert estimate.input_bytes_moved == 4096 + 8192
    assert estimate.output_bytes_moved == 512
    assert estimate.cross_bank_bytes_moved == 0

    expected_transfer_time = (
        8.0 + (4096 + 8192) / 128.0
        + 8.0 + 512 / 128.0
    )

    assert estimate.transfer_time_ns == pytest.approx(
        expected_transfer_time
    )

    assert estimate.memory_access_time_ns == pytest.approx(120.0)


def test_cim_estimate_accounts_for_remote_bank_data(system):
    """CIM execution in bank 0 must transfer B002 from bank 1."""

    _, _, model = system

    estimate = model.estimate(
        make_request(),
        "CIM_A:0",
    )

    assert estimate.input_bytes_moved == 0
    assert estimate.output_bytes_moved == 0
    assert estimate.cross_bank_bytes_moved == 8192

    expected_transfer_time = 25.0 + 8192 / 16.0

    assert estimate.transfer_time_ns == pytest.approx(
        expected_transfer_time
    )

    assert estimate.total_moved_bytes == 8192


def test_cim_local_input_has_no_cross_bank_transfer(system):
    """Inputs already in the selected bank require no inter-bank transfer."""

    _, _, model = system

    request = make_request(
        block_ids=("B001",),
        output_bytes=256,
    )

    estimate = model.estimate(request, "CIM_P:0")

    assert estimate.cross_bank_bytes_moved == 0
    assert estimate.transfer_time_ns == 0.0
    assert estimate.input_bytes_moved == 0
    assert estimate.output_bytes_moved == 0


def test_queue_wait_changes_latency_but_not_energy(system):
    """Queue waiting should affect predicted latency, not energy."""

    _, _, model = system
    request = make_request()

    baseline = model.estimate(
        request,
        "CPU",
        queue_wait_ns=0.0,
    )

    delayed = model.estimate(
        request,
        "CPU",
        queue_wait_ns=25.0,
    )

    assert delayed.predicted_latency_ns - baseline.predicted_latency_ns == pytest.approx(
        25.0
    )

    assert delayed.predicted_energy_pj == pytest.approx(
        baseline.predicted_energy_pj
    )


def test_decision_overhead_can_be_excluded(system):
    """Removing decision overhead should remove exactly that latency cost."""

    config, _, model = system
    request = make_request()

    with_overhead = model.estimate(
        request,
        "CPU",
        include_decision_overhead=True,
    )

    without_overhead = model.estimate(
        request,
        "CPU",
        include_decision_overhead=False,
    )

    assert (
        with_overhead.predicted_latency_ns
        - without_overhead.predicted_latency_ns
    ) == pytest.approx(config.runtime.decision_overhead_ns)

    assert with_overhead.predicted_energy_pj == pytest.approx(
        without_overhead.predicted_energy_pj
    )


def test_unknown_resource_is_rejected(system):
    """The cost model must reject a non-existent resource."""

    _, _, model = system

    with pytest.raises(ValueError):
        model.estimate(
            make_request(),
            "CIM_A:99",
        )


def test_unsupported_operation_resource_pair_is_rejected(system):
    """An operation cannot execute on an incompatible CIM engine."""

    _, _, model = system

    request = make_request(operation="MATMUL")

    with pytest.raises(ValueError):
        model.estimate(
            request,
            "CIM_A:0",
        )


def test_negative_queue_wait_is_rejected(system):
    """Queue waiting time cannot be negative."""

    _, _, model = system

    with pytest.raises(ValueError):
        model.estimate(
            make_request(),
            "CPU",
            queue_wait_ns=-1.0,
        )


def test_request_cannot_reference_another_dataset(system):
    """The request and input blocks must refer to the same dataset."""

    _, _, model = system

    request = make_request(dataset_id="different_dataset")

    with pytest.raises(ValueError):
        model.candidate_resources(request)
