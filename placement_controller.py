
"""
DynaMem: Adaptive computation-placement controller.

Responsibilities:
- Evaluate legal execution resources for the current request.
- Compare normalized latency and energy estimates.
- Account for current queue-wait estimates.
- Apply a switching threshold to avoid unnecessary resource changes.
- Record the selected resource and an explanation of the decision.

The controller uses only current request information, current model state,
and previously observed workload history. It does not inspect future requests.
"""

from math import isfinite
from typing import Mapping

from cache_model import CPU_CACHE_LOCATION
from cost_model import CostEstimate, CostModel
from models import AccessPattern, ComputationRequest, PlacementDecision
from runtime_monitor import RuntimeMonitor


class PlacementController:
    """Choose an execution resource using estimated runtime costs."""

    def __init__(
            self,
            cost_model: CostModel,
            monitor: RuntimeMonitor,
            use_access_pattern_features: bool = True,
            reuse_prediction_weight: float = 0.25,
    ):
        if monitor.directory is not cost_model.directory:
            raise ValueError(
                "The monitor and cost model must share the same "
                "DataDirectory instance."
            )

        if monitor.config != cost_model.config:
            raise ValueError(
                "The monitor and cost model must use the same configuration."
            )

        if not isinstance(use_access_pattern_features, bool):
            raise TypeError(
                "use_access_pattern_features must be a boolean."
            )

        if (
                not isinstance(reuse_prediction_weight, (int, float))
                or isinstance(reuse_prediction_weight, bool)
                or not isfinite(reuse_prediction_weight)
                or not 0.0 <= reuse_prediction_weight <= 1.0
        ):
            raise ValueError(
                "reuse_prediction_weight must be between 0 and 1."
            )

        self.cost_model = cost_model
        self.monitor = monitor
        self.config = cost_model.config

        self.use_access_pattern_features = use_access_pattern_features
        self.reuse_prediction_weight = float(reuse_prediction_weight)

        # Remember the previous resource for each dataset/operation pair.
        self._last_selected: dict[tuple[str, str], str] = {}
        self._decision_count = 0

    @property
    def decision_count(self) -> int:
        """Number of placement decisions made by this controller."""

        return self._decision_count

    @staticmethod
    def _validate_queue_waits(
        queue_waits_ns: Mapping[str, float] | None,
    ) -> dict[str, float]:
        """Validate and copy the supplied queue-wait estimates."""

        if queue_waits_ns is None:
            return {}

        validated = {}

        for resource_name, wait_ns in queue_waits_ns.items():
            if not isinstance(resource_name, str) or not resource_name.strip():
                raise ValueError("Queue-wait resource names cannot be empty.")

            if (
                not isinstance(wait_ns, (int, float))
                or isinstance(wait_ns, bool)
                or not isfinite(wait_ns)
                or wait_ns < 0
            ):
                raise ValueError(
                    f"Invalid queue-wait estimate for {resource_name}."
                )

            validated[resource_name] = float(wait_ns)

        return validated

    def _estimate_reuse_credit(
            self,
            request: ComputationRequest,
    ) -> dict[str, float]:
        """
        Estimate a discounted future cache benefit for CPU placement.

        Credit is applied only when:
        - Access-pattern features are enabled.
        - The monitor has classified the dataset as reused.
        - Input blocks have previous reuse history.
        - The corresponding blocks are not already in the CPU cache.

        This is a heuristic for candidate ranking, not a measured saving.
        """

        credit = {
            "latency_ns": 0.0,
            "energy_pj": 0.0,
            "expected_saved_bytes": 0.0,
        }

        if not self.use_access_pattern_features:
            return credit

        observed_pattern = self.monitor.get_observed_pattern(
            request.dataset_id
        )

        if observed_pattern != AccessPattern.REUSED:
            return credit

        directory = self.cost_model.directory

        # Count each input block only once.
        blocks = [
            directory.get_block(block_id)
            for block_id in dict.fromkeys(request.input_block_ids)
        ]

        uncached_blocks = [
            block
            for block in blocks
            if not directory.has_copy(
                block.block_id,
                CPU_CACHE_LOCATION,
            )
        ]

        if not uncached_blocks:
            return credit

        cache = self.config.cache
        line_size = cache.line_size_bytes

        # If the request's input working set exceeds cache capacity,
        # discount its estimated ability to remain resident.
        working_set_bytes = sum(
            ((block.size_bytes + line_size - 1) // line_size)
            * line_size
            for block in blocks
        )

        cache_fit_factor = min(
            1.0,
            cache.capacity_bytes / max(working_set_bytes, 1),
        )

        expected_saved_bytes = 0.0
        probability_all_reused = 1.0

        for block in uncached_blocks:
            resident_size = (
                                    (block.size_bytes + line_size - 1) // line_size
                            ) * line_size

            access_count = directory.get_access_count(
                block.block_id
            )

            # A block observed once has not yet demonstrated repeated reuse.
            # The ratio increases as repeated accesses are observed.
            if resident_size > cache.capacity_bytes or access_count <= 0:
                reuse_probability = 0.0
            else:
                reuse_probability = (
                                            block.reuse_count / access_count
                                    ) * cache_fit_factor

            reuse_probability = min(
                1.0,
                max(0.0, reuse_probability),
            )

            expected_saved_bytes += (
                    block.size_bytes * reuse_probability
            )

            # Approximate the probability that all currently uncached
            # inputs are reused together, assuming independent recurrence.
            probability_all_reused *= reuse_probability

        if expected_saved_bytes <= 0.0:
            return credit

        link = self.config.links["CPU_MEMORY"]
        memory = self.config.memory
        weight = self.reuse_prediction_weight

        # Expected input-bandwidth savings plus a discounted probability
        # of avoiding the next access's memory/link startup costs.
        latency_credit = weight * (
                expected_saved_bytes / link.bandwidth_bytes_per_ns
                + probability_all_reused
                * (link.latency_ns + memory.access_latency_ns)
        )

        # Credit only for future transfer and main-memory energy avoided.
        energy_credit = weight * expected_saved_bytes * (
                link.energy_per_byte_pj
                + memory.energy_per_byte_pj
        )

        credit["latency_ns"] = latency_credit
        credit["energy_pj"] = energy_credit
        credit["expected_saved_bytes"] = expected_saved_bytes

        return credit

    def _score_candidates(
            self,
            estimates: Mapping[str, CostEstimate],
            request: ComputationRequest,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """
        Score legal resources using normalized latency and energy.

        For reused workloads, CPU's scoring cost may include a discounted
        expected cache-reuse benefit. Reported immediate execution
        estimates remain unchanged.
        """

        if not estimates:
            raise ValueError(
                "No legal execution candidates are available."
            )

        effective_costs = {
            name: (
                estimate.predicted_latency_ns,
                estimate.predicted_energy_pj,
            )
            for name, estimate in estimates.items()
        }

        reuse_credit = self._estimate_reuse_credit(request)

        if "CPU" in effective_costs:
            cpu_latency, cpu_energy = effective_costs["CPU"]

            effective_costs["CPU"] = (
                max(
                    1e-9,
                    cpu_latency - reuse_credit["latency_ns"],
                ),
                max(
                    1e-9,
                    cpu_energy - reuse_credit["energy_pj"],
                ),
            )

        best_latency = min(
            latency for latency, _ in effective_costs.values()
        )

        best_energy = min(
            energy for _, energy in effective_costs.values()
        )

        if best_latency <= 0 or best_energy <= 0:
            raise ValueError(
                "Latency and energy reference values must be positive."
            )

        latency_weight = self.config.runtime.latency_weight
        energy_weight = 1.0 - latency_weight

        scores = {}

        for resource_name, (latency, energy) in effective_costs.items():
            scores[resource_name] = (
                    latency_weight * latency / best_latency
                    + energy_weight * energy / best_energy
            )

        return scores, reuse_credit

    def decide(
        self,
        request: ComputationRequest,
        queue_waits_ns: Mapping[str, float] | None = None,
        include_decision_overhead: bool = True,
    ) -> PlacementDecision:
        """
        Select a legal execution resource for the current request.

        queue_waits_ns supplies estimates from the simulator. If omitted,
        the estimator assumes no waiting time for any resource.
        """

        validated_waits = self._validate_queue_waits(
            queue_waits_ns
        )

        # Compare candidate execution costs without adding the same
        # controller-decision overhead to every candidate's score.
        estimates = self.cost_model.estimate_all(
            request,
            queue_waits_ns=validated_waits,
            include_decision_overhead=False,
        )

        scores, reuse_credit = self._score_candidates(
            estimates,
            request,
        )

        # Deterministic tie-breaking makes experiments reproducible.
        best_resource = min(
            scores,
            key=lambda resource_name: (
                scores[resource_name],
                resource_name,
            ),
        )

        chosen_resource = best_resource
        switching_applied = False
        switching_note = ""

        # Avoid switching for an improvement smaller than the configured
        # relative threshold, but only if the previous choice remains legal.
        key = (request.dataset_id, request.operation)
        previous_resource = self._last_selected.get(key)

        if (
            previous_resource is not None
            and previous_resource in scores
            and previous_resource != best_resource
        ):
            previous_score = scores[previous_resource]
            best_score = scores[best_resource]

            relative_improvement = (
                (previous_score - best_score)
                / previous_score
            )

            threshold = self.config.runtime.switching_threshold

            if relative_improvement < threshold:
                chosen_resource = previous_resource
                switching_applied = True
                switching_note = (
                    f"Retained {previous_resource}: the alternative's "
                    f"predicted score improvement "
                    f"({relative_improvement:.2%}) was below the "
                    f"{threshold:.2%} switching threshold."
                )

        chosen_estimate = estimates[chosen_resource]
        chosen_score = scores[chosen_resource]

        # Use already observed history. The monitor never reads the
        # request's declared access_pattern as classification evidence.
        observed_pattern = self.monitor.get_observed_pattern(
            request.dataset_id
        )

        statistics = self.monitor.get_statistics(request.dataset_id)

        if observed_pattern is None:
            pattern_description = "insufficient observation history"
        else:
            pattern_description = observed_pattern.value

        movement_bytes = chosen_estimate.total_moved_bytes

        if switching_applied:
            reason = (
                f"{switching_note} Observed workload behaviour: "
                f"{pattern_description}. Operation: {request.operation}. "
                f"Estimated latency: "
                f"{chosen_estimate.predicted_latency_ns:.2f} ns; "
                f"estimated energy: "
                f"{chosen_estimate.predicted_energy_pj:.2f} pJ; "
                f"modelled moved bytes: {movement_bytes}."
            )
        else:
            reason = (
                f"Selected the lowest-scoring legal candidate. "
                f"Observed workload behaviour: {pattern_description}. "
                f"Operation: {request.operation}. Estimated latency: "
                f"{chosen_estimate.predicted_latency_ns:.2f} ns; "
                f"estimated energy: "
                f"{chosen_estimate.predicted_energy_pj:.2f} pJ; "
                f"modelled moved bytes: {movement_bytes}."
            )

        # Keep a short diagnostic note available in the explanation.
        if statistics["window_size"] > 0:
            reason += (
                f" Recent unique-block ratio: "
                f"{statistics['unique_ratio']:.2f}; sequential ratio: "
                f"{statistics['sequential_ratio']:.2f}."
            )

        if reuse_credit["expected_saved_bytes"] > 0:
            reason += (
                " Reuse-aware scoring applied a discounted future "
                "CPU-cache benefit: expected reusable bytes="
                f"{reuse_credit['expected_saved_bytes']:.1f}, "
                f"latency credit={reuse_credit['latency_ns']:.2f} ns, "
                f"energy credit={reuse_credit['energy_pj']:.2f} pJ."
            )

        self._last_selected[key] = chosen_resource
        self._decision_count += 1

        return PlacementDecision(
            request_id=request.request_id,
            selected_resource=chosen_resource,
            predicted_latency_ns=(
    chosen_estimate.predicted_latency_ns
    + (
        self.config.runtime.decision_overhead_ns
        if include_decision_overhead
        else 0.0
    )
),
            predicted_energy_pj=chosen_estimate.predicted_energy_pj,
            score=chosen_score,
            reason=reason,
            candidate_scores=dict(scores),
        )

    def reset_history(self) -> None:
        """
        Clear placement history between independent experiments.

        This prevents one experiment's previous selections from influencing
        another experiment's switching-threshold behaviour.
        """

        self._last_selected.clear()
        self._decision_count = 0
