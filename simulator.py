
"""
DynaMem: Event-driven simulation engine.

Models:
- Request arrivals and runtime placement decisions.
- CPU, CIM-A, CIM-P, and near-memory compute resources.
- Main-memory bank contention.
- CPU-cache accesses and LRU evictions.
- CPU-memory, memory-NMP, and inter-bank data transfers.
- Queue waiting, execution time, data movement, and energy.

Time is measured in nanoseconds, data in bytes, and energy in picojoules.

The model is event-driven, but it is not a cycle-accurate hardware simulator.
"""
from baselines import build_baseline_policies
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import ceil, isfinite
from typing import Mapping

import simpy

from cache_model import CacheModel, CPU_CACHE_LOCATION
from config import SimulationConfig
from cost_model import CostModel
from data_directory import DataDirectory
from models import (
    ComputationRequest,
    ExecutionRecord,
    PlacementDecision,
)
from placement_controller import PlacementController
from runtime_monitor import RuntimeMonitor
from workload_generator import WorkloadTrace


@dataclass(frozen=True)
class SimulationResult:
    """Results and aggregate statistics from a completed simulation."""

    trace_name: str
    policy_name:str
    records: tuple[ExecutionRecord, ...]
    makespan_ns: float

    resource_busy_ns: Mapping[str, float]
    resource_capacities: Mapping[str, int]

    cache_summary: Mapping
    directory_summary: Mapping

    runtime_decision_count: int

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        """Calculate a nearest-rank percentile."""

        if not values:
            return 0.0

        ordered = sorted(values)
        rank = max(1, ceil(percentile * len(ordered)))

        return ordered[min(rank - 1, len(ordered) - 1)]

    def summary(self) -> dict:
        """Calculate high-level metrics from completed request records."""

        records = self.records
        count = len(records)

        latencies = [
            record.total_latency_ns for record in records
        ]

        total_energy = sum(
            record.energy_pj for record in records
        )

        input_bytes = sum(
            record.input_bytes_moved for record in records
        )

        output_bytes = sum(
            record.output_bytes_moved for record in records
        )

        cross_bank_bytes = sum(
            record.cross_bank_bytes_moved for record in records
        )

        total_bytes = input_bytes + output_bytes + cross_bank_bytes

        makespan_seconds = self.makespan_ns * 1e-9

        placements = Counter(
            record.resource_name for record in records
        )

        resource_utilization = {}

        for resource_name, capacity in self.resource_capacities.items():
            if self.makespan_ns <= 0:
                utilization = 0.0
            else:
                utilization = (
                    self.resource_busy_ns.get(resource_name, 0.0)
                    / (self.makespan_ns * capacity)
                )

            resource_utilization[resource_name] = utilization

        return {
            "trace_name": self.trace_name,
            "policy_name": self.policy_name,
            "request_count": count,
            "completed_requests": count,
            "makespan_ns": self.makespan_ns,
            "total_request_latency_ns": sum(latencies),
            "average_request_latency_ns": (
                sum(latencies) / count if count else 0.0
            ),
            "p50_latency_ns": self._percentile(latencies, 0.50),
            "p95_latency_ns": self._percentile(latencies, 0.95),
            "max_latency_ns": max(latencies, default=0.0),
            "throughput_requests_per_second": (
                count / makespan_seconds
                if makespan_seconds > 0
                else 0.0
            ),
            "total_energy_pj": total_energy,
            "input_bytes_moved": input_bytes,
            "output_bytes_moved": output_bytes,
            "cross_bank_bytes_moved": cross_bank_bytes,
            "total_bytes_moved": total_bytes,
            "placements": dict(placements),
            "resource_utilization": resource_utilization,
            "total_queue_wait_ns": sum(
                record.queue_wait_ns for record in records
            ),
            "total_compute_time_ns": sum(
                record.compute_time_ns for record in records
            ),
            "total_transfer_time_ns": sum(
                record.transfer_time_ns for record in records
            ),
            "total_memory_access_time_ns": sum(
                record.memory_access_time_ns for record in records
            ),
            "total_cache_access_time_ns": sum(
                record.cache_access_time_ns for record in records
            ),
            "runtime_decisions": self.runtime_decision_count,
            "cache": dict(self.cache_summary),
            "directory": dict(self.directory_summary),

        }


class DynaMemSimulator:
    """Run one workload on the adaptive DynaMem architecture."""

    def __init__(
            self,
            config: SimulationConfig,
            trace: WorkloadTrace,
            policy_name: str = "ADAPTIVE",
            hybrid_size_threshold_bytes: int = 256 * 1024,
            adaptive_pattern_aware: bool = True,
            adaptive_queue_aware: bool = True,
    ):
        self.config = config
        self.trace = trace

        if not isinstance(adaptive_pattern_aware, bool):
            raise TypeError("adaptive_pattern_aware must be a boolean.")

        if not isinstance(adaptive_queue_aware, bool):
            raise TypeError("adaptive_queue_aware must be a boolean.")

        self.adaptive_pattern_aware = adaptive_pattern_aware
        self.adaptive_queue_aware = adaptive_queue_aware


        # Every simulator instance starts with fresh runtime state.
        self.env = simpy.Environment()

        self.directory = DataDirectory(
            config,
            trace.blocks,
        )

        self.cache = CacheModel(
            config,
            self.directory,
        )

        self.monitor = RuntimeMonitor(
            config,
            self.directory,
        )

        self.cost_model = CostModel(
            config,
            self.directory,
        )

        # Adaptive policy.
        self.adaptive_controller = PlacementController(
            self.cost_model,
            self.monitor,
            use_access_pattern_features=self.adaptive_pattern_aware,
        )

        # Fixed baselines, sharing the same runtime model components.
        self.baseline_policies = build_baseline_policies(
            self.cost_model,
            self.monitor,
            size_threshold_bytes=hybrid_size_threshold_bytes,
        )

        available_policies = {
            "ADAPTIVE": self.adaptive_controller,
            **self.baseline_policies,
        }

        selected_policy_name = policy_name.upper()

        if selected_policy_name not in available_policies:
            raise ValueError(
                f"Unknown policy: {policy_name}. "
                f"Available policies: {sorted(available_policies)}"
            )

        self.policy_name = selected_policy_name
        self.policy = available_policies[selected_policy_name]

        # Compatibility alias for existing code that references controller.
        self.controller = self.policy

        # All SimPy resources are registered by name so the controller
        # and event engine refer to the same physical-modelled instances.
        self._resource_objects: dict[str, simpy.Resource] = {}
        self._resource_capacities: dict[str, int] = {}

        self._resource_busy_ns: dict[str, float] = defaultdict(float)

        self._records: list[ExecutionRecord] = []
        self._decisions: list[PlacementDecision] = []

        self._has_run = False

        self._initialize_resources()

    def _register_resource(
        self,
        name: str,
        capacity: int,
    ) -> None:
        """Create and register one simulated contention resource."""

        if capacity <= 0:
            raise ValueError("Resource capacity must be positive.")

        self._resource_objects[name] = simpy.Resource(
            self.env,
            capacity=capacity,
        )

        self._resource_capacities[name] = capacity
        self._resource_busy_ns[name] = 0.0

    def _initialize_resources(self) -> None:
        """Create compute resources, links, memory banks, and cache port."""

        # CPU and NMP are shared engines with multiple or single slots.
        for resource_type in ("CPU", "NMP"):
            capacity = self.config.resources[resource_type].instances

            self._register_resource(
                resource_type,
                capacity,
            )

        # Each CIM compute engine is associated with one memory bank.
        for resource_type in ("CIM_A", "CIM_P"):
            engine_config = self.config.resources[resource_type]

            for bank_id in range(engine_config.instances):
                self._register_resource(
                    f"{resource_type}:{bank_id}",
                    1,
                )

        # Data-transfer links have limited concurrent-transfer capacity.
        for link_name, link_config in self.config.links.items():
            self._register_resource(
                f"LINK:{link_name}",
                link_config.max_concurrent_transfers,
            )

        # Each main-memory bank has its own contention resource.
        for bank_id in range(self.config.memory.num_banks):
            self._register_resource(
                f"BANK:{bank_id}",
                self.config.memory.max_concurrent_requests_per_bank,
            )

        # One cache port serializes cache lookup/fill batches.
        self._register_resource("CPU_CACHE_PORT", 1)

    def _run_stage(self, resource_names, duration_ns: float):
        """
        Reserve the named resources for a simulated service stage.

        Returns the time spent waiting for all requested resources.

        Resource names are acquired in a deterministic order, helping
        prevent cyclic resource-acquisition deadlocks.
        """

        if not isfinite(duration_ns) or duration_ns < 0:
            raise ValueError(
                "Stage duration must be finite and non-negative."
            )

        unique_names = sorted(set(resource_names))
        pending_requests = []

        wait_started_ns = self.env.now

        try:
            for name in unique_names:
                if name not in self._resource_objects:
                    raise KeyError(
                        f"Unknown simulated resource: {name}"
                    )

                resource = self._resource_objects[name]
                event = resource.request()

                pending_requests.append(
                    (name, resource, event)
                )

            if pending_requests:
                yield self.env.all_of(
                    [event for _, _, event in pending_requests]
                )

            queue_wait_ns = self.env.now - wait_started_ns

            if duration_ns > 0:
                yield self.env.timeout(duration_ns)

                for name, _, _ in pending_requests:
                    self._resource_busy_ns[name] += duration_ns

            return queue_wait_ns

        finally:
            # Release anything acquired and cancel requests that remain
            # queued if the stage exits abnormally.
            for _, resource, event in reversed(pending_requests):
                if event.triggered:
                    resource.release(event)
                else:
                    event.cancel()

    def _estimate_queue_waits(
        self,
        request: ComputationRequest,
    ) -> dict[str, float]:
        """
        Produce approximate wait estimates for placement decisions.

        The estimate uses current queue lengths and each candidate's
        predicted service duration. It is not a perfect prediction of
        future completion times; the event engine measures actual waits.
        """

        estimates = self.cost_model.estimate_all(
            request,
            include_decision_overhead=False,
        )

        waits = {}

        for resource_name, estimate in estimates.items():
            resource = self._resource_objects[resource_name]

            jobs_ahead = max(
                0,
                resource.count
                + len(resource.queue)
                - resource.capacity
                + 1,
            )

            estimated_service_ns = max(
                1.0,
                estimate.compute_time_ns
                + estimate.transfer_time_ns
                + estimate.memory_access_time_ns
                + estimate.cache_access_time_ns,
            )

            waits[resource_name] = (
                ceil(jobs_ahead / resource.capacity)
                * estimated_service_ns
            )

        return waits

    def _bank_resource_names(
        self,
        bank_ids,
    ) -> list[str]:
        """Convert memory-bank IDs to registered resource names."""

        return sorted({
            f"BANK:{bank_id}"
            for bank_id in bank_ids
        })

    def _process_cpu_inputs(
        self,
        request: ComputationRequest,
        stats: dict,
    ):
        """Fetch CPU inputs through the simulated cache hierarchy."""

        block_ids = tuple(dict.fromkeys(request.input_block_ids))

        cache_resource = self._resource_objects["CPU_CACHE_PORT"]

        wait_started_ns = self.env.now
        cache_request = cache_resource.request()

        yield cache_request

        stats["queue_wait_ns"] += (
            self.env.now - wait_started_ns
        )

        cache_port_started_ns = self.env.now

        try:
            # Take a snapshot before the cache model changes residency.
            cached_before = {
                block_id: self.cache.contains(block_id)
                for block_id in block_ids
            }

            result = self.cache.access_blocks(
                block_ids,
                time_ns=self.env.now,
            )

            newly_missed = [
                block_id
                for block_id in block_ids
                if not cached_before[block_id]
            ]

            # Keep new copies invisible to placement estimates until the
            # modelled memory read and transfer have completed.
            for block_id in newly_missed:
                if self.directory.has_copy(
                    block_id,
                    CPU_CACHE_LOCATION,
                ):
                    self.directory.remove_copy(
                        block_id,
                        CPU_CACHE_LOCATION,
                    )

            stats["cache_access_time_ns"] += (
                result.cache_access_time_ns
            )

            stats["input_bytes_moved"] += (
                result.input_bytes_transferred
            )

            stats["energy_pj"] += result.energy_pj

            # Cache lookup service.
            if result.cache_access_time_ns > 0:
                yield self.env.timeout(
                    result.cache_access_time_ns
                )

            # Service the input read through the relevant memory banks.
            if result.input_bytes_transferred > 0:
                missed_bank_ids = [
                    self.directory.get_home_bank(block_id)
                    for block_id in newly_missed
                ]

                if result.memory_access_time_ns > 0:
                    wait_ns = yield self.env.process(
                        self._run_stage(
                            self._bank_resource_names(missed_bank_ids),
                            result.memory_access_time_ns,
                        )
                    )

                    stats["queue_wait_ns"] += wait_ns

                    stats["memory_access_time_ns"] += (
                        result.memory_access_time_ns
                    )

                # Transfer the missed input bytes over the shared link.
                if result.transfer_time_ns > 0:
                    wait_ns = yield self.env.process(
                        self._run_stage(
                            ["LINK:CPU_MEMORY"],
                            result.transfer_time_ns,
                        )
                    )

                    stats["queue_wait_ns"] += wait_ns

                    stats["transfer_time_ns"] += (
                        result.transfer_time_ns
                    )

            # Make successfully resident blocks visible again only after
            # their simulated fill operations have completed.
            for block_id in newly_missed:
                if (
                    self.cache.contains(block_id)
                    and not self.directory.has_copy(
                        block_id,
                        CPU_CACHE_LOCATION,
                    )
                ):
                    self.directory.add_copy(
                        block_id,
                        CPU_CACHE_LOCATION,
                    )

        finally:
            # The model holds the cache port during this batch's
            # lookup and fill activity.
            self._resource_busy_ns["CPU_CACHE_PORT"] += (
                self.env.now - cache_port_started_ns
            )

            cache_resource.release(cache_request)

    def _process_nmp_inputs(
        self,
        request: ComputationRequest,
        stats: dict,
    ):
        """Read inputs from memory and transfer them to near-memory compute."""

        blocks = [
            self.directory.get_block(block_id)
            for block_id in dict.fromkeys(request.input_block_ids)
        ]

        input_bytes = sum(
            block.size_bytes for block in blocks
        )

        bank_ids = [block.bank_id for block in blocks]

        # Access the input blocks at their home memory banks.
        memory_latency = self.config.memory.access_latency_ns

        wait_ns = yield self.env.process(
            self._run_stage(
                self._bank_resource_names(bank_ids),
                memory_latency,
            )
        )

        stats["queue_wait_ns"] += wait_ns
        stats["memory_access_time_ns"] += memory_latency

        stats["energy_pj"] += (
            input_bytes * self.config.memory.energy_per_byte_pj
        )

        # Move the input data into the NMP's local working area.
        link_name = "MEMORY_NMP"
        link = self.config.links[link_name]

        transfer_time_ns = (
            link.latency_ns
            + input_bytes / link.bandwidth_bytes_per_ns
        )

        wait_ns = yield self.env.process(
            self._run_stage(
                [f"LINK:{link_name}"],
                transfer_time_ns,
            )
        )

        stats["queue_wait_ns"] += wait_ns
        stats["transfer_time_ns"] += transfer_time_ns
        stats["input_bytes_moved"] += input_bytes

        stats["energy_pj"] += (
            input_bytes * link.energy_per_byte_pj
        )

    def _process_cim_inputs(
        self,
        request: ComputationRequest,
        target_bank: int,
        stats: dict,
    ):
        """Read input blocks and move remote data to a CIM bank if needed."""

        blocks = [
            self.directory.get_block(block_id)
            for block_id in dict.fromkeys(request.input_block_ids)
        ]

        input_bytes = sum(
            block.size_bytes for block in blocks
        )

        # All source banks must service this input-read stage.
        source_bank_ids = [
            block.bank_id for block in blocks
        ]

        memory_latency = self.config.memory.access_latency_ns

        wait_ns = yield self.env.process(
            self._run_stage(
                self._bank_resource_names(source_bank_ids),
                memory_latency,
            )
        )

        stats["queue_wait_ns"] += wait_ns
        stats["memory_access_time_ns"] += memory_latency

        stats["energy_pj"] += (
            input_bytes * self.config.memory.energy_per_byte_pj
        )

        # Inputs already in the target bank require no inter-bank transfer.
        cross_bank_bytes = sum(
            block.size_bytes
            for block in blocks
            if block.bank_id != target_bank
        )

        stats["cross_bank_bytes_moved"] += cross_bank_bytes

        if cross_bank_bytes == 0:
            return

        link_name = "BANK_TO_BANK"
        link = self.config.links[link_name]

        transfer_time_ns = (
            link.latency_ns
            + cross_bank_bytes / link.bandwidth_bytes_per_ns
        )

        # The target bank is also reserved while receiving the transferred
        # data. This introduces contention on both the link and destination.
        wait_ns = yield self.env.process(
            self._run_stage(
                [
                    f"LINK:{link_name}",
                    f"BANK:{target_bank}",
                ],
                transfer_time_ns,
            )
        )

        stats["queue_wait_ns"] += wait_ns
        stats["transfer_time_ns"] += transfer_time_ns

        stats["energy_pj"] += (
            cross_bank_bytes * link.energy_per_byte_pj
        )

        # Copying data into the target bank incurs an additional write cost.
        stats["energy_pj"] += (
            cross_bank_bytes * self.config.memory.energy_per_byte_pj
        )

    def _execute_compute(
        self,
        request: ComputationRequest,
        resource_name: str,
        stats: dict,
    ):
        """Acquire the selected compute engine and execute the request."""

        resource = self._resource_objects[resource_name]
        resource_config = self._resource_config(resource_name)

        wait_started_ns = self.env.now
        compute_request = resource.request()

        yield compute_request

        stats["queue_wait_ns"] += (
            self.env.now - wait_started_ns
        )

        try:
            start_time_ns = self.env.now

            compute_time_ns = (
                resource_config.launch_overhead_ns
                + request.operation_count / resource_config.ops_per_ns
            )

            yield self.env.timeout(compute_time_ns)

            self._resource_busy_ns[resource_name] += compute_time_ns

            stats["compute_time_ns"] += compute_time_ns

            stats["energy_pj"] += (
                resource_config.fixed_energy_pj
                + request.operation_count
                * resource_config.energy_per_op_pj
            )

            return start_time_ns

        finally:
            resource.release(compute_request)

    def _resource_config(self, resource_name: str):
        """Retrieve a compute-resource configuration by instance name."""

        if resource_name in ("CPU", "NMP"):
            return self.config.resources[resource_name]

        resource_type, _ = resource_name.split(":", 1)

        return self.config.resources[resource_type]

    def _process_output(
        self,
        request: ComputationRequest,
        resource_type: str,
        target_bank: int | None,
        stats: dict,
    ):
        """Commit a computation result to its modelled destination."""

        output_bytes = request.output_bytes

        if output_bytes == 0:
            return

        memory = self.config.memory

        if resource_type in ("CPU", "NMP"):
            link_name = (
                "CPU_MEMORY"
                if resource_type == "CPU"
                else "MEMORY_NMP"
            )

            link = self.config.links[link_name]

            transfer_time_ns = (
                link.latency_ns
                + output_bytes / link.bandwidth_bytes_per_ns
            )

            # Transfer the result before the destination memory write.
            wait_ns = yield self.env.process(
                self._run_stage(
                    [f"LINK:{link_name}"],
                    transfer_time_ns,
                )
            )

            stats["queue_wait_ns"] += wait_ns
            stats["transfer_time_ns"] += transfer_time_ns
            stats["output_bytes_moved"] += output_bytes

            stats["energy_pj"] += (
                output_bytes * link.energy_per_byte_pj
            )

            # This model writes the result to the first input block's
            # home bank because the request does not specify an output bank.
            output_bank_id = self.directory.get_home_bank(
                request.input_block_ids[0]
            )

            memory_latency = memory.access_latency_ns

            wait_ns = yield self.env.process(
                self._run_stage(
                    [f"BANK:{output_bank_id}"],
                    memory_latency,
                )
            )

            stats["queue_wait_ns"] += wait_ns
            stats["memory_access_time_ns"] += memory_latency

            stats["energy_pj"] += (
                output_bytes * memory.energy_per_byte_pj
            )

        else:
            # CIM leaves the result in its selected memory bank.
            if target_bank is None:
                raise RuntimeError(
                    "CIM output handling requires a target bank."
                )

            memory_latency = memory.access_latency_ns

            wait_ns = yield self.env.process(
                self._run_stage(
                    [f"BANK:{target_bank}"],
                    memory_latency,
                )
            )

            stats["queue_wait_ns"] += wait_ns
            stats["memory_access_time_ns"] += memory_latency

            stats["energy_pj"] += (
                output_bytes * memory.energy_per_byte_pj
            )

            # The result stays in the selected bank: no inter-resource
            # output transfer is counted in this initial model.

    def _process_request(self, request: ComputationRequest):
        """Simulate one arriving request from admission to completion."""

        if request.arrival_time_ns > self.env.now:
            yield self.env.timeout(
                request.arrival_time_ns - self.env.now
            )

        arrival_time_ns = self.env.now

        # Estimate current compute-resource congestion before placement.
        if (
                self.policy_name == "ADAPTIVE"
                and not self.adaptive_queue_aware
        ):
            # Ablation mode: the adaptive controller receives no queue estimates.
            queue_waits_ns = {}
        else:
            queue_waits_ns = self._estimate_queue_waits(request)

        decision = self.policy.decide(
            request,
            queue_waits_ns=queue_waits_ns,
            include_decision_overhead=True,
        )

        self._decisions.append(decision)

        decision_time_ns = self.env.now

        stats = {
            "queue_wait_ns": 0.0,
            "compute_time_ns": 0.0,
            "transfer_time_ns": 0.0,
            "memory_access_time_ns": 0.0,
            "cache_access_time_ns": 0.0,
            "decision_overhead_ns": 0.0,
            "input_bytes_moved": 0,
            "output_bytes_moved": 0,
            "cross_bank_bytes_moved": 0,
            "energy_pj": 0.0,
        }

        # Runtime computation-placement overhead is part of total latency.
        # Adaptive placement pays the configured runtime-controller overhead.
        # Fixed policies use their own explicitly configured overhead.
        decision_overhead_ns = getattr(
            self.policy,
            "decision_overhead_ns",
            self.config.runtime.decision_overhead_ns,
        )

        if decision_overhead_ns > 0:
            yield self.env.timeout(decision_overhead_ns)

        stats["decision_overhead_ns"] = decision_overhead_ns

        # Record accesses only when this request reaches its simulated
        # access phase. The monitor does not inspect future trace requests.
        self.monitor.observe_request(
            request,
            time_ns=self.env.now,
        )

        resource_name = decision.selected_resource

        if resource_name in ("CPU", "NMP"):
            resource_type = resource_name
            target_bank = None
        else:
            resource_type, bank_text = resource_name.split(":", 1)
            target_bank = int(bank_text)

        # Inputs are accessed before the selected compute engine executes.
        if resource_type == "CPU":
            yield self.env.process(
                self._process_cpu_inputs(request, stats)
            )

        elif resource_type == "NMP":
            yield self.env.process(
                self._process_nmp_inputs(request, stats)
            )

        else:
            yield self.env.process(
                self._process_cim_inputs(
                    request,
                    target_bank,
                    stats,
                )
            )

        # The start timestamp is when the selected compute engine begins
        # executing, after input movement and resource queueing.
        start_time_ns = yield self.env.process(
            self._execute_compute(
                request,
                resource_name,
                stats,
            )
        )

        # Process the output after the computation completes.
        yield self.env.process(
            self._process_output(
                request,
                resource_type,
                target_bank,
                stats,
            )
        )

        finish_time_ns = self.env.now

        record = ExecutionRecord(
            request_id=request.request_id,
            resource_name=resource_name,
            arrival_time_ns=arrival_time_ns,
            decision_time_ns=decision_time_ns,
            start_time_ns=start_time_ns,
            finish_time_ns=finish_time_ns,
            input_bytes_moved=stats["input_bytes_moved"],
            output_bytes_moved=stats["output_bytes_moved"],
            cross_bank_bytes_moved=stats["cross_bank_bytes_moved"],
            queue_wait_ns=stats["queue_wait_ns"],
            compute_time_ns=stats["compute_time_ns"],
            transfer_time_ns=stats["transfer_time_ns"],
            decision_overhead_ns=stats["decision_overhead_ns"],
            energy_pj=stats["energy_pj"],
            memory_access_time_ns=stats["memory_access_time_ns"],
            cache_access_time_ns=stats["cache_access_time_ns"],
        )

        self._records.append(record)

    def run(self) -> SimulationResult:
        """Run the complete workload and return aggregate results."""

        if self._has_run:
            raise RuntimeError(
                "This simulator has already run. Create a new instance "
                "for an independent experiment."
            )

        self._has_run = True

        for request in self.trace.requests:
            self.env.process(
                self._process_request(request)
            )

        self.env.run()

        records = tuple(
            sorted(
                self._records,
                key=lambda record: (
                    record.arrival_time_ns,
                    record.request_id,
                ),
            )
        )

        first_arrival_ns = min(
            (record.arrival_time_ns for record in records),
            default=0.0,
        )

        last_completion_ns = max(
            (record.finish_time_ns for record in records),
            default=first_arrival_ns,
        )

        makespan_ns = max(
            0.0,
            last_completion_ns - first_arrival_ns,
        )

        return SimulationResult(
            trace_name=self.trace.name,
            policy_name=self.policy_name,
            records=records,
            makespan_ns=makespan_ns,
            resource_busy_ns=dict(self._resource_busy_ns),
            resource_capacities=dict(self._resource_capacities),
            cache_summary=self.cache.summary(),
            directory_summary=self.directory.summary(),
            runtime_decision_count=self.policy.decision_count,
        )
