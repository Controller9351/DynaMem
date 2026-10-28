
"""
DynaMem: Baseline computation-placement policies.

Policies:
- CPUOnlyPolicy
- StaticCIMAPolicy
- StaticCIMPPolicy
- StaticNMPPolicy
- FixedHybridPolicy

All policies implement a common decide() interface, allowing the
simulator to compare them using the same workload and cost model.

Decision overheads are modelling assumptions and are configurable.
"""

from math import isfinite
from typing import Mapping

from cost_model import CostEstimate, CostModel
from models import ComputationRequest, PlacementDecision
from runtime_monitor import RuntimeMonitor


SIMPLE_DATA_PARALLEL_OPERATIONS = frozenset({
    "BITMAP_AND",
    "BITMAP_OR",
    "FILTER",
    "COUNT",
    "SUM",
})

DEFAULT_SIZE_THRESHOLD_BYTES = 256 * 1024


class BasePlacementPolicy:
    """Shared decision and scoring functionality for placement policies."""

    def __init__(
        self,
        cost_model: CostModel,
        monitor: RuntimeMonitor,
        name: str,
        decision_overhead_ns: float = 0.0,
    ):
        if monitor.directory is not cost_model.directory:
            raise ValueError(
                "Policy monitor and cost model must share a DataDirectory."
            )

        if monitor.config != cost_model.config:
            raise ValueError(
                "Policy monitor and cost model must use the same config."
            )

        if not name.strip():
            raise ValueError("Policy name cannot be empty.")

        if (
            not isfinite(decision_overhead_ns)
            or decision_overhead_ns < 0
        ):
            raise ValueError(
                "Decision overhead must be finite and non-negative."
            )

        self.cost_model = cost_model
        self.monitor = monitor
        self.config = cost_model.config
        self.name = name
        self.decision_overhead_ns = decision_overhead_ns

        self._decision_count = 0

    @property
    def decision_count(self) -> int:
        """Number of decisions made by this policy."""

        return self._decision_count

    @staticmethod
    def validate_queue_waits(
        queue_waits_ns: Mapping[str, float] | None,
    ) -> dict[str, float]:
        """Validate supplied queue-wait estimates."""

        if queue_waits_ns is None:
            return {}

        validated = {}

        for resource_name, wait_ns in queue_waits_ns.items():
            if (
                not isinstance(resource_name, str)
                or not resource_name.strip()
            ):
                raise ValueError(
                    "Queue-wait resource names cannot be empty."
                )

            if (
                not isinstance(wait_ns, (int, float))
                or isinstance(wait_ns, bool)
                or not isfinite(wait_ns)
                or wait_ns < 0
            ):
                raise ValueError(
                    f"Invalid queue wait for {resource_name}."
                )

            validated[resource_name] = float(wait_ns)

        return validated

    @staticmethod
    def score_estimates(
        estimates: Mapping[str, CostEstimate],
        latency_weight: float,
    ) -> dict[str, float]:
        """Calculate normalized latency-energy scores."""

        if not estimates:
            raise ValueError("No legal execution candidates available.")

        if not 0.0 <= latency_weight <= 1.0:
            raise ValueError(
                "Latency weight must be between 0 and 1."
            )

        min_latency = min(
            estimate.predicted_latency_ns
            for estimate in estimates.values()
        )

        min_energy = min(
            estimate.predicted_energy_pj
            for estimate in estimates.values()
        )

        if min_latency <= 0 or min_energy <= 0:
            raise ValueError(
                "Cost estimates must have positive latency and energy."
            )

        energy_weight = 1.0 - latency_weight

        return {
            name: (
                latency_weight
                * estimate.predicted_latency_ns
                / min_latency
                + energy_weight
                * estimate.predicted_energy_pj
                / min_energy
            )
            for name, estimate in estimates.items()
        }

    def _evaluate_candidates(
        self,
        request: ComputationRequest,
        queue_waits_ns: Mapping[str, float] | None,
    ) -> tuple[dict[str, CostEstimate], dict[str, float]]:
        """Estimate all legal candidates without charging adaptive overhead."""

        waits = self.validate_queue_waits(queue_waits_ns)

        estimates = self.cost_model.estimate_all(
            request,
            queue_waits_ns=waits,
            include_decision_overhead=False,
        )

        scores = self.score_estimates(
            estimates,
            self.config.runtime.latency_weight,
        )

        return estimates, scores

    def _make_decision(
        self,
        request: ComputationRequest,
        selected_resource: str,
        estimates: Mapping[str, CostEstimate],
        scores: Mapping[str, float],
        reason: str,
    ) -> PlacementDecision:
        """Create a common decision record for any policy."""

        if selected_resource not in estimates:
            raise ValueError(
                f"Selected resource {selected_resource} "
                "is not a legal candidate."
            )

        estimate = estimates[selected_resource]

        observed_pattern = self.monitor.get_observed_pattern(
            request.dataset_id
        )

        pattern_name = (
            observed_pattern.value
            if observed_pattern is not None
            else "unknown"
        )

        self._decision_count += 1

        final_reason = (
            f"Policy={self.name}; {reason}; "
            f"observed_pattern={pattern_name}."
        )

        return PlacementDecision(
            request_id=request.request_id,
            selected_resource=selected_resource,
            predicted_latency_ns=(
                estimate.predicted_latency_ns
                + self.decision_overhead_ns
            ),
            predicted_energy_pj=estimate.predicted_energy_pj,
            score=scores[selected_resource],
            reason=final_reason,
            candidate_scores=dict(scores),
        )

    def decide(
        self,
        request: ComputationRequest,
        queue_waits_ns: Mapping[str, float] | None = None,
        include_decision_overhead: bool = True,
    ) -> PlacementDecision:
        """Select a resource; implemented by each concrete policy."""

        raise NotImplementedError(
            "Concrete placement policies must implement decide()."
        )

    def reset_history(self) -> None:
        """Reset per-experiment decision statistics."""

        self._decision_count = 0


class CPUOnlyPolicy(BasePlacementPolicy):
    """Always execute requests on the conventional CPU."""

    def __init__(
        self,
        cost_model: CostModel,
        monitor: RuntimeMonitor,
        decision_overhead_ns: float = 0.0,
    ):
        super().__init__(
            cost_model,
            monitor,
            name="CPU_ONLY",
            decision_overhead_ns=decision_overhead_ns,
        )

    def decide(
        self,
        request: ComputationRequest,
        queue_waits_ns: Mapping[str, float] | None = None,
        include_decision_overhead: bool = True,
    ) -> PlacementDecision:
        estimates, scores = self._evaluate_candidates(
            request,
            queue_waits_ns,
        )

        return self._make_decision(
            request,
            selected_resource="CPU",
            estimates=estimates,
            scores=scores,
            reason="Conventional-computing baseline; always use CPU.",
        )


class StaticMemoryPolicy(BasePlacementPolicy):
    """Prefer a fixed memory-centric resource family."""

    def __init__(
        self,
        cost_model: CostModel,
        monitor: RuntimeMonitor,
        resource_family: str,
        decision_overhead_ns: float = 0.0,
    ):
        family = resource_family.upper()

        if family not in {"CIM_A", "CIM_P"}:
            raise ValueError(
                "Memory resource family must be CIM_A or CIM_P."
            )

        super().__init__(
            cost_model,
            monitor,
            name=f"STATIC_{family}",
            decision_overhead_ns=decision_overhead_ns,
        )

        self.resource_family = family

    def decide(
        self,
        request: ComputationRequest,
        queue_waits_ns: Mapping[str, float] | None = None,
        include_decision_overhead: bool = True,
    ) -> PlacementDecision:
        estimates, scores = self._evaluate_candidates(
            request,
            queue_waits_ns,
        )

        preferred = [
            name
            for name in estimates
            if name.startswith(f"{self.resource_family}:")
        ]

        if preferred:
            # The policy stays within its preferred resource family,
            # but chooses a bank using the configured cost estimates.
            selected = min(
                preferred,
                key=lambda name: (scores[name], name),
            )

            reason = (
                f"Static family preference={self.resource_family}; "
                f"selected the lowest-scoring legal bank instance."
            )
        else:
            # Unsupported operations fall back to the CPU.
            selected = "CPU"

            reason = (
                f"{self.resource_family} does not support "
                f"{request.operation}; falling back to CPU."
            )

        return self._make_decision(
            request,
            selected,
            estimates,
            scores,
            reason,
        )


class StaticCIMAPolicy(StaticMemoryPolicy):
    """Prefer computation inside a memory array."""

    def __init__(
        self,
        cost_model: CostModel,
        monitor: RuntimeMonitor,
        decision_overhead_ns: float = 0.0,
    ):
        super().__init__(
            cost_model,
            monitor,
            resource_family="CIM_A",
            decision_overhead_ns=decision_overhead_ns,
        )


class StaticCIMPPolicy(StaticMemoryPolicy):
    """Prefer computation in memory periphery."""

    def __init__(
        self,
        cost_model: CostModel,
        monitor: RuntimeMonitor,
        decision_overhead_ns: float = 0.0,
    ):
        super().__init__(
            cost_model,
            monitor,
            resource_family="CIM_P",
            decision_overhead_ns=decision_overhead_ns,
        )


class StaticNMPPolicy(BasePlacementPolicy):
    """Always prefer the near-memory processor."""

    def __init__(
        self,
        cost_model: CostModel,
        monitor: RuntimeMonitor,
        decision_overhead_ns: float = 0.0,
    ):
        super().__init__(
            cost_model,
            monitor,
            name="STATIC_NMP",
            decision_overhead_ns=decision_overhead_ns,
        )

    def decide(
        self,
        request: ComputationRequest,
        queue_waits_ns: Mapping[str, float] | None = None,
        include_decision_overhead: bool = True,
    ) -> PlacementDecision:
        estimates, scores = self._evaluate_candidates(
            request,
            queue_waits_ns,
        )

        # NMP must be available for this baseline to execute.
        if "NMP" not in estimates:
            raise RuntimeError(
                "NMP is not a legal candidate for this request."
            )

        return self._make_decision(
            request,
            selected_resource="NMP",
            estimates=estimates,
            scores=scores,
            reason="Static near-memory preference; always use NMP.",
        )


class FixedHybridPolicy(BasePlacementPolicy):
    """
    Choose a resource family using fixed operation and size thresholds.

    Small tasks default to CPU. Larger simple data-parallel tasks
    prefer CIM-P; larger vector or matrix tasks prefer NMP.
    This baseline does not adapt its thresholds from workload history.
    """

    def __init__(
        self,
        cost_model: CostModel,
        monitor: RuntimeMonitor,
        size_threshold_bytes: int = DEFAULT_SIZE_THRESHOLD_BYTES,
        decision_overhead_ns: float = 1.0,
    ):
        if (
            not isinstance(size_threshold_bytes, int)
            or isinstance(size_threshold_bytes, bool)
            or size_threshold_bytes <= 0
        ):
            raise ValueError(
                "Size threshold must be a positive integer."
            )

        super().__init__(
            cost_model,
            monitor,
            name="FIXED_HYBRID",
            decision_overhead_ns=decision_overhead_ns,
        )

        self.size_threshold_bytes = size_threshold_bytes

    def decide(
        self,
        request: ComputationRequest,
        queue_waits_ns: Mapping[str, float] | None = None,
        include_decision_overhead: bool = True,
    ) -> PlacementDecision:
        estimates, scores = self._evaluate_candidates(
            request,
            queue_waits_ns,
        )

        input_bytes = self.cost_model.directory.input_size_bytes(
            request.input_block_ids
        )

        is_large = input_bytes >= self.size_threshold_bytes

        if not is_large:
            selected = "CPU"
            reason = (
                f"Input size {input_bytes} bytes is below the fixed "
                f"threshold {self.size_threshold_bytes}; use CPU."
            )

        elif request.operation in SIMPLE_DATA_PARALLEL_OPERATIONS:
            cim_candidates = [
                name
                for name in estimates
                if name.startswith("CIM_P:")
            ]

            if cim_candidates:
                selected = min(
                    cim_candidates,
                    key=lambda name: (scores[name], name),
                )

                reason = (
                    f"Large simple operation; prefer CIM-P using "
                    f"the fixed {self.size_threshold_bytes}-byte threshold."
                )
            else:
                selected = "CPU"
                reason = (
                    "Large simple operation, but CIM-P is unavailable; "
                    "fall back to CPU."
                )

        elif request.operation in {"VECTOR_MAP", "MATMUL"}:
            if "NMP" in estimates:
                selected = "NMP"
                reason = (
                    "Large vector or matrix workload; fixed rule "
                    "prefers near-memory compute."
                )
            else:
                selected = "CPU"
                reason = (
                    "NMP unavailable; fall back to CPU."
                )

        else:
            selected = "CPU"
            reason = (
                f"No specialized fixed rule for {request.operation}; "
                "use CPU."
            )

        return self._make_decision(
            request,
            selected,
            estimates,
            scores,
            reason,
        )


def build_baseline_policies(
    cost_model: CostModel,
    monitor: RuntimeMonitor,
    size_threshold_bytes: int = DEFAULT_SIZE_THRESHOLD_BYTES,
) -> dict[str, BasePlacementPolicy]:
    """Create the baseline policy suite using shared model components."""

    return {
        "CPU_ONLY": CPUOnlyPolicy(
            cost_model,
            monitor,
        ),
        "STATIC_CIM_A": StaticCIMAPolicy(
            cost_model,
            monitor,
        ),
        "STATIC_CIM_P": StaticCIMPPolicy(
            cost_model,
            monitor,
        ),
        "STATIC_NMP": StaticNMPPolicy(
            cost_model,
            monitor,
        ),
        "FIXED_HYBRID": FixedHybridPolicy(
            cost_model,
            monitor,
            size_threshold_bytes=size_threshold_bytes,
        ),
    }
