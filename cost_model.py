
"""
DynaMem: Runtime latency, energy and data-movement cost estimation.

Responsibilities:
- Identify legal execution resources for a request.
- Estimate data transfers based on block locations.
- Account for CPU-cache copies and memory-bank placement.
- Estimate computation time, transfer time and memory-access time.
- Estimate energy from computation, memory, cache and links.
- Include dynamic placement overhead when requested.

All estimates use the configured simulation parameters.
They are model predictions, not physical hardware measurements.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

from config import SimulationConfig
from data_directory import DataDirectory
from models import ComputationRequest


@dataclass(frozen=True)
class CostEstimate:
    """Estimated cost of executing a request on one resource."""

    resource_name: str

    predicted_latency_ns: float
    predicted_energy_pj: float

    queue_wait_ns: float
    compute_time_ns: float
    transfer_time_ns: float
    memory_access_time_ns: float
    cache_access_time_ns: float
    decision_overhead_ns: float

    input_bytes_moved: int
    output_bytes_moved: int
    cross_bank_bytes_moved: int

    def __post_init__(self):
        if not self.resource_name.strip():
            raise ValueError("Resource name cannot be empty.")

        time_values = {
            "predicted latency": self.predicted_latency_ns,
            "queue wait": self.queue_wait_ns,
            "compute time": self.compute_time_ns,
            "transfer time": self.transfer_time_ns,
            "memory-access time": self.memory_access_time_ns,
            "cache-access time": self.cache_access_time_ns,
            "decision overhead": self.decision_overhead_ns,
        }

        for name, value in time_values.items():
            if not isfinite(value) or value < 0:
                raise ValueError(
                    f"{name.capitalize()} must be finite and non-negative."
                )

        if (
            not isfinite(self.predicted_energy_pj)
            or self.predicted_energy_pj < 0
        ):
            raise ValueError(
                "Predicted energy must be finite and non-negative."
            )

        byte_values = {
            "input bytes moved": self.input_bytes_moved,
            "output bytes moved": self.output_bytes_moved,
            "cross-bank bytes moved": self.cross_bank_bytes_moved,
        }

        for name, value in byte_values.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name.capitalize()} must be an integer.")

            if value < 0:
                raise ValueError(f"{name.capitalize()} cannot be negative.")

    @property
    def total_moved_bytes(self) -> int:
        """Total bytes transported between the modelled locations."""

        return (
            self.input_bytes_moved
            + self.output_bytes_moved
            + self.cross_bank_bytes_moved
        )


class CostModel:
    """Estimate the cost of executing requests on DynaMem resources."""

    def __init__(
        self,
        config: SimulationConfig,
        directory: DataDirectory,
    ):
        self.config = config
        self.directory = directory

        if directory.config != config:
            raise ValueError(
                "CostModel and DataDirectory must use the same configuration."
            )

    def candidate_resources(
        self,
        request: ComputationRequest,
    ) -> list[str]:
        """
        Return all legal execution choices for the request.

        CIM-A and CIM-P are represented by individual bank instances:
        CIM_A:0, CIM_A:1, CIM_P:0, CIM_P:1, and so on.
        """

        self._validate_request_blocks(request)

        candidates = []

        for resource_type in ("CPU", "CIM_A", "CIM_P", "NMP"):
            resource = self.config.resources[resource_type]

            if not resource.supports(request.operation):
                continue

            if resource_type in ("CIM_A", "CIM_P"):
                for bank_id in range(resource.instances):
                    candidates.append(
                        f"{resource_type}:{bank_id}"
                    )
            else:
                candidates.append(resource_type)

        return candidates

    def _validate_request_blocks(
        self,
        request: ComputationRequest,
    ) -> list:
        """Validate and return the blocks referenced by a request."""

        blocks = []

        for block_id in request.input_block_ids:
            block = self.directory.get_block(block_id)

            if block.dataset_id != request.dataset_id:
                raise ValueError(
                    f"Block {block_id} belongs to dataset "
                    f"{block.dataset_id}, not {request.dataset_id}."
                )

            blocks.append(block)

        return blocks

    def _resolve_resource(
        self,
        resource_name: str,
    ) -> tuple[str, int | None, object]:
        """Resolve a resource name into its type and optional bank."""

        if resource_name in ("CPU", "NMP"):
            return (
                resource_name,
                None,
                self.config.resources[resource_name],
            )

        if ":" not in resource_name:
            raise ValueError(
                "CIM resources must specify an instance, "
                "for example 'CIM_P:2'."
            )

        resource_type, instance_text = resource_name.split(":", 1)

        if resource_type not in ("CIM_A", "CIM_P"):
            raise ValueError(
                f"Unknown execution resource: {resource_name}"
            )

        try:
            bank_id = int(instance_text)
        except ValueError as exc:
            raise ValueError(
                f"Invalid resource instance: {resource_name}"
            ) from exc

        resource = self.config.resources[resource_type]

        if not 0 <= bank_id < resource.instances:
            raise ValueError(
                f"Instance {bank_id} does not exist for {resource_type}."
            )

        return resource_type, bank_id, resource

    def _transfer_time_ns(
        self,
        link_name: str,
        bytes_moved: int,
    ) -> float:
        """
        Estimate one bulk transfer.

        Model:
            transfer time = link latency + bytes / bandwidth
        """

        if bytes_moved <= 0:
            return 0.0

        link = self.config.links[link_name]

        return (
            link.latency_ns
            + bytes_moved / link.bandwidth_bytes_per_ns
        )

    def _transfer_energy_pj(
        self,
        link_name: str,
        bytes_moved: int,
    ) -> float:
        """Estimate the energy consumed by a data-link transfer."""

        link = self.config.links[link_name]

        return bytes_moved * link.energy_per_byte_pj

    def estimate(
        self,
        request: ComputationRequest,
        resource_name: str,
        queue_wait_ns: float = 0.0,
        include_decision_overhead: bool = True,
    ) -> CostEstimate:
        """
        Estimate end-to-end latency and energy for one execution choice.

        The queue wait is supplied by the caller. The cost model itself
        does not schedule requests or reserve execution resources.
        """

        if (
            not isfinite(queue_wait_ns)
            or queue_wait_ns < 0
        ):
            raise ValueError(
                "Queue wait must be finite and non-negative."
            )

        if not isinstance(include_decision_overhead, bool):
            raise TypeError(
                "include_decision_overhead must be a boolean."
            )

        blocks = self._validate_request_blocks(request)

        (
            resource_type,
            target_bank,
            resource,
        ) = self._resolve_resource(resource_name)

        if not resource.supports(request.operation):
            raise ValueError(
                f"{resource_name} does not support "
                f"{request.operation}."
            )

        input_bytes = sum(block.size_bytes for block in blocks)
        output_bytes = request.output_bytes

        memory = self.config.memory

        # Movement and access accounting.
        input_bytes_moved = 0
        output_bytes_moved = 0
        cross_bank_bytes_moved = 0

        transfer_time_ns = 0.0
        memory_access_time_ns = 0.0
        cache_access_time_ns = 0.0

        link_energy_pj = 0.0
        memory_energy_pj = 0.0
        cache_energy_pj = 0.0

        if resource_type == "CPU":
            # A recorded CPU-cache copy avoids the main-memory input
            # transfer in this simplified cache model.
            cached_blocks = [
                block
                for block in blocks
                if self.directory.has_copy(
                    block.block_id,
                    "CPU_CACHE",
                )
            ]

            uncached_blocks = [
                block
                for block in blocks
                if not self.directory.has_copy(
                    block.block_id,
                    "CPU_CACHE",
                )
            ]

            input_bytes_moved = sum(
                block.size_bytes for block in uncached_blocks
            )

            cache_access_time_ns = (
                    len(blocks) * self.config.cache.hit_latency_ns
            )

            cache_energy_pj = (
                    len(blocks) * self.config.cache.energy_per_access_pj
            )

            # Main-memory input transfer for cache misses.
            input_transfer_time = self._transfer_time_ns(
                "CPU_MEMORY",
                input_bytes_moved,
            )

            # Assume the result is committed to main memory.
            output_bytes_moved = output_bytes

            output_transfer_time = self._transfer_time_ns(
                "CPU_MEMORY",
                output_bytes_moved,
            )

            transfer_time_ns = (
                input_transfer_time + output_transfer_time
            )

            if input_bytes_moved > 0:
                memory_access_time_ns += memory.access_latency_ns

            if output_bytes_moved > 0:
                memory_access_time_ns += memory.access_latency_ns

            link_energy_pj += self._transfer_energy_pj(
                "CPU_MEMORY",
                input_bytes_moved + output_bytes_moved,
            )

            memory_energy_pj += (
                input_bytes_moved + output_bytes_moved
            ) * memory.energy_per_byte_pj

        elif resource_type == "NMP":
            # Inputs travel from main memory to the near-memory
            # processor, and the output is committed to main memory.
            input_bytes_moved = input_bytes
            output_bytes_moved = output_bytes

            transfer_time_ns = (
                self._transfer_time_ns(
                    "MEMORY_NMP",
                    input_bytes_moved,
                )
                + self._transfer_time_ns(
                    "MEMORY_NMP",
                    output_bytes_moved,
                )
            )

            if input_bytes_moved > 0:
                memory_access_time_ns += memory.access_latency_ns

            if output_bytes_moved > 0:
                memory_access_time_ns += memory.access_latency_ns

            link_energy_pj += self._transfer_energy_pj(
                "MEMORY_NMP",
                input_bytes_moved + output_bytes_moved,
            )

            memory_energy_pj += (
                input_bytes_moved + output_bytes_moved
            ) * memory.energy_per_byte_pj

        else:
            # CIM execution is attached to one specific bank. Blocks
            # in other banks must be moved to the selected bank.
            assert target_bank is not None

            cross_bank_bytes_moved = sum(
                block.size_bytes
                for block in blocks
                if block.bank_id != target_bank
            )

            transfer_time_ns = self._transfer_time_ns(
                "BANK_TO_BANK",
                cross_bank_bytes_moved,
            )

            # Inputs are accessed locally or transferred to the target
            # bank. The result remains in that target bank.
            if input_bytes > 0:
                memory_access_time_ns += memory.access_latency_ns

            if output_bytes > 0:
                memory_access_time_ns += memory.access_latency_ns

            # Reading original inputs and writing the result.
            memory_energy_pj += (
                input_bytes + output_bytes
            ) * memory.energy_per_byte_pj

            # Cross-bank inputs require an additional destination write.
            memory_energy_pj += (
                cross_bank_bytes_moved
                * memory.energy_per_byte_pj
            )

            link_energy_pj += self._transfer_energy_pj(
                "BANK_TO_BANK",
                cross_bank_bytes_moved,
            )

        # Compute time includes the engine's launch/setup overhead.
        compute_time_ns = (
            resource.launch_overhead_ns
            + request.operation_count / resource.ops_per_ns
        )

        compute_energy_pj = (
            resource.fixed_energy_pj
            + request.operation_count * resource.energy_per_op_pj
        )

        decision_overhead_ns = (
            self.config.runtime.decision_overhead_ns
            if include_decision_overhead
            else 0.0
        )

        # This is a serial estimate: overlapping transfers and compute
        # are not assumed in this initial cost model.
        predicted_latency_ns = (
            queue_wait_ns
            + decision_overhead_ns
            + cache_access_time_ns
            + memory_access_time_ns
            + transfer_time_ns
            + compute_time_ns
        )

        predicted_energy_pj = (
            compute_energy_pj
            + memory_energy_pj
            + link_energy_pj
            + cache_energy_pj
        )

        return CostEstimate(
            resource_name=resource_name,
            predicted_latency_ns=predicted_latency_ns,
            predicted_energy_pj=predicted_energy_pj,
            queue_wait_ns=queue_wait_ns,
            compute_time_ns=compute_time_ns,
            transfer_time_ns=transfer_time_ns,
            memory_access_time_ns=memory_access_time_ns,
            cache_access_time_ns=cache_access_time_ns,
            decision_overhead_ns=decision_overhead_ns,
            input_bytes_moved=input_bytes_moved,
            output_bytes_moved=output_bytes_moved,
            cross_bank_bytes_moved=cross_bank_bytes_moved,
        )

    def estimate_all(
        self,
        request: ComputationRequest,
        queue_waits_ns: Mapping[str, float] | None = None,
        include_decision_overhead: bool = True,
    ) -> dict[str, CostEstimate]:
        """Estimate all legal candidates for one request."""

        queue_waits_ns = queue_waits_ns or {}

        return {
            resource_name: self.estimate(
                request=request,
                resource_name=resource_name,
                queue_wait_ns=queue_waits_ns.get(
                    resource_name,
                    0.0,
                ),
                include_decision_overhead=include_decision_overhead,
            )
            for resource_name in self.candidate_resources(request)
        }
