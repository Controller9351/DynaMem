
"""Correctness tests for DynaMem's adaptive placement controller."""

import pytest

from config import make_default_config
from cost_model import CostModel
from data_directory import DataDirectory
from models import AccessPattern, ComputationRequest, DataBlock
from placement_controller import PlacementController
from runtime_monitor import RuntimeMonitor


@pytest.fixture
def system():
    """Create an independent controller, monitor and cost model."""

    config = make_default_config()

    blocks = [
        DataBlock("B001", "placement_test", 4096, 0),
        DataBlock("B002", "placement_test", 4096, 1),
        DataBlock("B003", "placement_test", 8192, 2),
    ]

    directory = DataDirectory(config, blocks)
    monitor = RuntimeMonitor(config, directory)
    cost_model = CostModel(config, directory)

    controller = PlacementController(
        cost_model=cost_model,
        monitor=monitor,
    )

    return config, directory, monitor, cost_model, controller


def make_request(
    request_id="R001",
    operation="BITMAP_AND",
    block_ids=("B001", "B002"),
    output_bytes=512,
    operation_count=1000,
):
    """Build a valid test request."""

    return ComputationRequest(
        request_id=request_id,
        dataset_id="placement_test",
        operation=operation,
        input_block_ids=block_ids,
        output_bytes=output_bytes,
        operation_count=operation_count,
        access_pattern=AccessPattern.STREAMING,
    )


def test_decision_selects_a_legal_candidate(system):
    """The selected execution resource must be one of the legal candidates."""

    _, _, _, cost_model, controller = system

    request = make_request()

    decision = controller.decide(request)

    assert decision.selected_resource in (
        cost_model.candidate_resources(request)
    )

    assert decision.request_id == request.request_id
    assert decision.predicted_latency_ns > 0
    assert decision.predicted_energy_pj > 0


def test_initial_decision_selects_lowest_scoring_candidate(system):
    """Without previous placement history, choose the lowest-scoring option."""

    _, _, _, _, controller = system

    request = make_request()

    decision = controller.decide(request)

    expected_resource = min(
        decision.candidate_scores,
        key=lambda name: (
            decision.candidate_scores[name],
            name,
        ),
    )

    assert decision.selected_resource == expected_resource
    assert decision.score == pytest.approx(
        decision.candidate_scores[expected_resource]
    )


def test_matmul_never_selects_cim_resources(system):
    """The controller must obey the configured operation capabilities."""

    _, _, _, cost_model, controller = system

    request = make_request(operation="MATMUL")

    candidates = set(cost_model.candidate_resources(request))
    decision = controller.decide(request)

    assert candidates == {"CPU", "NMP"}
    assert decision.selected_resource in {"CPU", "NMP"}

    assert all(
        not candidate.startswith(("CIM_A:", "CIM_P:"))
        for candidate in decision.candidate_scores
    )


def test_repeated_identical_decision_is_stable(system):
    """Unchanged conditions should not cause unnecessary switching."""

    _, _, _, _, controller = system

    request = make_request()

    first = controller.decide(request)
    second = controller.decide(request)

    assert first.selected_resource == second.selected_resource
    assert first.candidate_scores == second.candidate_scores


def test_queue_pressure_can_change_the_selected_resource(system):
    """A large delay on the previous resource should trigger reconsideration."""

    _, _, _, _, controller = system

    request = make_request(
        block_ids=("B001",),
        output_bytes=512,
    )

    first = controller.decide(request)

    queue_waits = {
        first.selected_resource: 1_000_000.0,
    }

    second = controller.decide(
        request,
        queue_waits_ns=queue_waits,
    )

    assert second.selected_resource != first.selected_resource

    assert (
        second.candidate_scores[second.selected_resource]
        < second.candidate_scores[first.selected_resource]
    )


def test_decision_overhead_is_added_only_when_requested(system):
    """The output latency includes the configured controller overhead."""

    config, _, _, _, controller = system

    request = make_request()

    with_overhead = controller.decide(
        request,
        include_decision_overhead=True,
    )

    without_overhead = controller.decide(
        request,
        include_decision_overhead=False,
    )

    assert with_overhead.selected_resource == (
        without_overhead.selected_resource
    )

    assert (
        with_overhead.predicted_latency_ns
        - without_overhead.predicted_latency_ns
    ) == pytest.approx(
        config.runtime.decision_overhead_ns
    )

    # Decision overhead changes latency, not estimated execution energy.
    assert with_overhead.predicted_energy_pj == pytest.approx(
        without_overhead.predicted_energy_pj
    )


def test_controller_does_not_trust_declared_access_pattern(system):
    """Without observations, the request's label must not become runtime knowledge."""

    _, _, monitor, _, controller = system

    request = make_request()

    assert monitor.get_observed_pattern(
        request.dataset_id
    ) is None

    decision = controller.decide(request)

    assert "insufficient observation history" in decision.reason.lower()
    assert "streaming" not in decision.reason.lower()


def test_invalid_queue_wait_is_rejected(system):
    """Negative queue-delay estimates are invalid input."""

    _, _, _, _, controller = system

    request = make_request()

    with pytest.raises(ValueError):
        controller.decide(
            request,
            queue_waits_ns={
                "CPU": -1.0,
            },
        )


def test_reset_history_clears_previous_decisions(system):
    """Resetting between experiments clears switching history and count."""

    _, _, _, _, controller = system

    request = make_request()

    controller.decide(request)
    assert controller.decision_count == 1

    controller.reset_history()

    assert controller.decision_count == 0

    # A fresh decision must still produce a legal candidate.
    decision = controller.decide(request)

    assert decision.selected_resource in decision.candidate_scores
    assert controller.decision_count == 1


def test_recent_access_history_is_visible_in_decision_reason(system):
    """Decision diagnostics should reflect observed runtime history."""

    _, _, monitor, _, controller = system

    for index in range(20):
        block_id = "B001" if index % 2 == 0 else "B002"
        monitor.observe_access(block_id, float(index))

    request = make_request()
    decision = controller.decide(request)

    assert "Observed workload behaviour: reused" in decision.reason
    assert "Recent unique-block ratio: 0.10" in decision.reason
    assert "Recent unique-block ratio" in decision.reason



def test_reuse_aware_feature_changes_cpu_candidate_score(system):
    """Observed reuse history should influence the adaptive CPU score."""

    _, _, monitor, cost_model, pattern_aware_controller = system

    # Observe repeated accesses to the same two blocks.
    for index in range(40):
        block_id = "B001" if index % 2 == 0 else "B002"

        monitor.observe_access(
            block_id,
            time_ns=float(index),
        )

    assert (
        monitor.get_observed_pattern("placement_test")
        == AccessPattern.REUSED
    )

    request = make_request(
        block_ids=("B001", "B002"),
    )

    # Default controller: reuse-aware scoring is enabled.
    pattern_aware_decision = pattern_aware_controller.decide(
        request,
    )

    # Ablation controller: same model and observations, but no reuse credit.
    pattern_blind_controller = PlacementController(
        cost_model,
        monitor,
        use_access_pattern_features=False,
    )

    pattern_blind_decision = pattern_blind_controller.decide(
        request,
    )

    assert (
        pattern_aware_decision.candidate_scores["CPU"]
        < pattern_blind_decision.candidate_scores["CPU"]
    )

    assert (
        "Reuse-aware scoring applied a discounted future CPU-cache benefit"
        in pattern_aware_decision.reason
    )

    assert (
        "Reuse-aware scoring applied a discounted future CPU-cache benefit"
        not in pattern_blind_decision.reason
    )


