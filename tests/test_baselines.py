
"""Correctness tests for DynaMem's fixed and baseline placement policies."""

import pytest

from baselines import (
    CPUOnlyPolicy,
    FixedHybridPolicy,
    StaticCIMAPolicy,
    StaticCIMPPolicy,
    StaticNMPPolicy,
    build_baseline_policies,
)
from config import make_default_config
from cost_model import CostModel
from data_directory import DataDirectory
from models import AccessPattern, ComputationRequest, DataBlock
from runtime_monitor import RuntimeMonitor


@pytest.fixture
def system():
    """Create fresh components shared by the policies under test."""

    config = make_default_config()

    blocks = [
        DataBlock("B_SMALL", "baseline_test", 512, 0),
        DataBlock("B_BOUNDARY", "baseline_test", 1024, 1),
        DataBlock("B001", "baseline_test", 4096, 0),
        DataBlock("B002", "baseline_test", 8192, 1),
    ]

    directory = DataDirectory(config, blocks)
    monitor = RuntimeMonitor(config, directory)
    cost_model = CostModel(config, directory)

    policies = build_baseline_policies(
        cost_model,
        monitor,
        size_threshold_bytes=1024,
    )

    return config, directory, monitor, cost_model, policies


def make_request(
    request_id="R001",
    operation="BITMAP_AND",
    block_ids=("B001",),
    output_bytes=512,
    operation_count=1000,
):
    """Create a valid workload request."""

    return ComputationRequest(
        request_id=request_id,
        dataset_id="baseline_test",
        operation=operation,
        input_block_ids=block_ids,
        output_bytes=output_bytes,
        operation_count=operation_count,
        access_pattern=AccessPattern.RANDOM,
    )


def test_policy_factory_builds_all_expected_policies(system):
    """The factory must expose all five baseline strategies."""

    _, _, _, _, policies = system

    assert set(policies) == {
        "CPU_ONLY",
        "STATIC_CIM_A",
        "STATIC_CIM_P",
        "STATIC_NMP",
        "FIXED_HYBRID",
    }


def test_cpu_only_policy_always_selects_cpu(system):
    """CPU-only is the conventional-computing reference policy."""

    _, _, _, _, policies = system

    for operation in (
        "BITMAP_AND",
        "FILTER",
        "VECTOR_MAP",
        "MATMUL",
        "RANDOM_LOOKUP",
    ):
        decision = policies["CPU_ONLY"].decide(
            make_request(operation=operation)
        )

        assert decision.selected_resource == "CPU"


def test_static_cim_a_prefers_cim_a_when_supported(system):
    """Static CIM-A selects a legal CIM-A instance for bitmap operations."""

    _, _, _, _, policies = system

    decision = policies["STATIC_CIM_A"].decide(
        make_request(operation="BITMAP_AND")
    )

    assert decision.selected_resource.startswith("CIM_A:")


def test_static_cim_a_falls_back_to_cpu_when_unsupported(system):
    """Unsupported operations must use CPU under the CIM-A policy."""

    _, _, _, _, policies = system

    decision = policies["STATIC_CIM_A"].decide(
        make_request(operation="MATMUL")
    )

    assert decision.selected_resource == "CPU"
    assert "falling back to CPU" in decision.reason


def test_static_cim_p_prefers_cim_p_when_supported(system):
    """Static CIM-P selects a legal CIM-P instance for filters."""

    _, _, _, _, policies = system

    decision = policies["STATIC_CIM_P"].decide(
        make_request(operation="FILTER")
    )

    assert decision.selected_resource.startswith("CIM_P:")


def test_static_cim_p_falls_back_to_cpu_when_unsupported(system):
    """CIM-P does not execute matrix multiplication in this configuration."""

    _, _, _, _, policies = system

    decision = policies["STATIC_CIM_P"].decide(
        make_request(operation="MATMUL")
    )

    assert decision.selected_resource == "CPU"
    assert "falling back to CPU" in decision.reason


def test_static_nmp_always_selects_nmp(system):
    """The NMP baseline selects its fixed processor for supported requests."""

    _, _, _, _, policies = system

    for operation in (
        "BITMAP_AND",
        "FILTER",
        "VECTOR_MAP",
        "MATMUL",
        "RANDOM_LOOKUP",
    ):
        decision = policies["STATIC_NMP"].decide(
            make_request(operation=operation)
        )

        assert decision.selected_resource == "NMP"


def test_fixed_hybrid_uses_cpu_below_threshold(system):
    """Inputs smaller than the threshold use CPU under the fixed rule."""

    _, _, _, _, policies = system

    request = make_request(
        block_ids=("B_SMALL",),
        operation="BITMAP_AND",
    )

    decision = policies["FIXED_HYBRID"].decide(request)

    assert decision.selected_resource == "CPU"
    assert "below the fixed threshold" in decision.reason


def test_fixed_hybrid_specializes_at_threshold_boundary(system):
    """Input size equal to the threshold counts as large."""

    _, _, _, _, policies = system

    request = make_request(
        block_ids=("B_BOUNDARY",),
        operation="BITMAP_OR",
    )

    decision = policies["FIXED_HYBRID"].decide(request)

    assert decision.selected_resource.startswith("CIM_P:")
    assert "Large simple operation" in decision.reason


def test_fixed_hybrid_uses_cim_p_for_large_simple_operations(system):
    """Large bitmap, filter, and reduction requests prefer CIM-P."""

    _, _, _, _, policies = system

    for operation in ("BITMAP_AND", "BITMAP_OR", "FILTER", "COUNT", "SUM"):
        request = make_request(
            operation=operation,
            block_ids=("B001", "B002"),
        )

        decision = policies["FIXED_HYBRID"].decide(request)

        assert decision.selected_resource.startswith("CIM_P:")


def test_fixed_hybrid_uses_nmp_for_large_vector_and_matrix_operations(system):
    """Large vector and matrix requests follow the fixed NMP rule."""

    _, _, _, _, policies = system

    for operation in ("VECTOR_MAP", "MATMUL"):
        request = make_request(
            operation=operation,
            block_ids=("B001", "B002"),
        )

        decision = policies["FIXED_HYBRID"].decide(request)

        assert decision.selected_resource == "NMP"


def test_fixed_hybrid_falls_back_to_cpu_without_specialized_rule(system):
    """Large random lookups use CPU because no specialized hybrid rule exists."""

    _, _, _, _, policies = system

    request = make_request(
        operation="RANDOM_LOOKUP",
        block_ids=("B001", "B002"),
    )

    decision = policies["FIXED_HYBRID"].decide(request)

    assert decision.selected_resource == "CPU"
    assert "No specialized fixed rule" in decision.reason


def test_decision_contains_scores_for_legal_candidates(system):
    """A policy must select one of the candidates for which it has scores."""

    _, _, _, cost_model, policies = system

    request = make_request(operation="BITMAP_OR")

    for policy in policies.values():
        decision = policy.decide(request)

        legal_candidates = set(
            cost_model.candidate_resources(request)
        )

        assert decision.selected_resource in legal_candidates
        assert decision.selected_resource in decision.candidate_scores
        assert all(
            score >= 0
            for score in decision.candidate_scores.values()
        )


def test_fixed_hybrid_decision_overhead_is_configurable(system):
    """Hybrid's decision overhead is included in its prediction."""

    config, _, monitor, cost_model, _ = system

    policy = FixedHybridPolicy(
        cost_model,
        monitor,
        size_threshold_bytes=1024,
        decision_overhead_ns=3.0,
    )

    request = make_request(
        block_ids=("B_SMALL",),
    )

    decision = policy.decide(request)

    estimate_without_overhead = cost_model.estimate(
        request,
        "CPU",
        include_decision_overhead=False,
    )

    assert decision.predicted_latency_ns == pytest.approx(
        estimate_without_overhead.predicted_latency_ns + 3.0
    )


def test_reset_history_resets_decision_count(system):
    """Policies can reset decision statistics between experiments."""

    _, _, _, _, policies = system

    for policy in policies.values():
        policy.decide(make_request())

        assert policy.decision_count == 1

        policy.reset_history()

        assert policy.decision_count == 0


def test_baseline_configuration_rejects_invalid_threshold(system):
    """A fixed hybrid requires a positive integer threshold."""

    _, _, monitor, cost_model, _ = system

    for invalid_threshold in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            FixedHybridPolicy(
                cost_model,
                monitor,
                size_threshold_bytes=invalid_threshold,
            )


def test_invalid_decision_overhead_is_rejected(system):
    """Baseline decision overhead cannot be negative or infinite."""

    _, _, monitor, cost_model, _ = system

    with pytest.raises(ValueError):
        CPUOnlyPolicy(
            cost_model,
            monitor,
            decision_overhead_ns=-1.0,
        )

    with pytest.raises(ValueError):
        StaticNMPPolicy(
            cost_model,
            monitor,
            decision_overhead_ns=float("inf"),
        )
