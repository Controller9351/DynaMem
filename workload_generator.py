
"""
DynaMem: Workload generation.

Generates deterministic synthetic traces for:
1. Highly reusable data
2. Sequential streaming
3. Random access
4. Workloads with changing access patterns
5. Mixed-size requests within a single trace

All generated workloads are synthetic simulation inputs.

Backward compatibility:
    If min_blocks_per_request and max_blocks_per_request are omitted,
    every request uses the original blocks_per_request value.
"""

import random
from dataclasses import dataclass
from math import isfinite

from config import SimulationConfig, SUPPORTED_OPERATIONS
from models import AccessPattern, ComputationRequest, DataBlock


DEFAULT_OPERATIONS = tuple(sorted(SUPPORTED_OPERATIONS))

DEFAULT_HOT_FRACTION = 0.10
DEFAULT_REUSE_PROBABILITY = 0.90


@dataclass(frozen=True)
class WorkloadTrace:
    """A workload containing registered memory blocks and requests."""

    name: str
    blocks: tuple[DataBlock, ...]
    requests: tuple[ComputationRequest, ...]

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Workload name cannot be empty.")

        block_ids = [block.block_id for block in self.blocks]

        if len(block_ids) != len(set(block_ids)):
            raise ValueError("Data-block IDs must be unique.")

        request_ids = [
            request.request_id for request in self.requests
        ]

        if len(request_ids) != len(set(request_ids)):
            raise ValueError("Request IDs must be unique.")

        known_blocks = {
            block.block_id: block
            for block in self.blocks
        }

        previous_arrival = -1.0

        for request in self.requests:
            if request.arrival_time_ns < previous_arrival:
                raise ValueError(
                    "Requests must be ordered by arrival time."
                )

            previous_arrival = request.arrival_time_ns

            for block_id in request.input_block_ids:
                if block_id not in known_blocks:
                    raise ValueError(
                        f"Request {request.request_id} references "
                        f"unknown block {block_id}."
                    )

                if (
                    known_blocks[block_id].dataset_id
                    != request.dataset_id
                ):
                    raise ValueError(
                        f"Block {block_id} belongs to a different dataset."
                    )

    @property
    def blocks_by_id(self) -> dict[str, DataBlock]:
        """Return a lookup table from block ID to block metadata."""

        return {
            block.block_id: block
            for block in self.blocks
        }

    @property
    def total_data_bytes(self) -> int:
        """Return total bytes stored in unique blocks in the trace."""

        return sum(
            block.size_bytes for block in self.blocks
        )

    @property
    def request_sizes_bytes(self) -> tuple[int, ...]:
        """Return each request's total input size in bytes."""

        blocks_by_id = self.blocks_by_id

        return tuple(
            sum(
                blocks_by_id[block_id].size_bytes
                for block_id in request.input_block_ids
            )
            for request in self.requests
        )


class WorkloadGenerator:
    """Generate repeatable workloads for the DynaMem simulator."""

    def __init__(
        self,
        config: SimulationConfig,
        seed: int | None = None,
    ):
        self.config = config
        self.seed = (
            config.random_seed
            if seed is None
            else seed
        )
        self.rng = random.Random(self.seed)

    @staticmethod
    def _validate_common(
        num_requests: int,
        num_blocks: int,
        block_size_bytes: int,
        blocks_per_request: int,
        arrival_interval_ns: float,
    ) -> None:
        """Validate parameters shared by all workload types."""

        integer_parameters = {
            "num_requests": num_requests,
            "num_blocks": num_blocks,
            "block_size_bytes": block_size_bytes,
            "blocks_per_request": blocks_per_request,
        }

        for name, value in integer_parameters.items():
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(
                    f"{name} must be a positive integer."
                )

        if (
            not isinstance(arrival_interval_ns, (int, float))
            or isinstance(arrival_interval_ns, bool)
            or not isfinite(arrival_interval_ns)
            or arrival_interval_ns < 0
        ):
            raise ValueError(
                "Arrival interval must be finite and non-negative."
            )

    @staticmethod
    def _resolve_block_count_range(
        blocks_per_request: int,
        num_blocks: int,
        min_blocks_per_request: int | None,
        max_blocks_per_request: int | None,
    ) -> tuple[int, int]:
        """
        Resolve the request-size range.

        Examples:
            Existing behaviour:
                blocks_per_request=4

            Mixed-size requests:
                blocks_per_request=4,
                min_blocks_per_request=1,
                max_blocks_per_request=12

            If only a minimum is supplied, blocks_per_request is
            used as the maximum. If only a maximum is supplied,
            blocks_per_request is used as the minimum.
        """

        min_count = (
            blocks_per_request
            if min_blocks_per_request is None
            else min_blocks_per_request
        )

        max_count = (
            blocks_per_request
            if max_blocks_per_request is None
            else max_blocks_per_request
        )

        for name, value in (
            ("min_blocks_per_request", min_count),
            ("max_blocks_per_request", max_count),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(
                    f"{name} must be a positive integer."
                )

        if min_count > max_count:
            raise ValueError(
                "Minimum blocks per request cannot exceed maximum."
            )

        if max_count > num_blocks:
            raise ValueError(
                "Maximum blocks per request cannot exceed "
                "the total number of data blocks."
            )

        return min_count, max_count

    def _sample_request_block_counts(
        self,
        num_requests: int,
        min_count: int,
        max_count: int,
    ) -> list[int]:
        """Choose the input-block count for each request."""

        # Preserve the old random-number sequence when every request
        # has a fixed input size.
        if min_count == max_count:
            return [min_count] * num_requests

        return [
            self.rng.randint(min_count, max_count)
            for _ in range(num_requests)
        ]

    @staticmethod
    def _validate_operations(
        operations: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Validate and normalize supported operation names."""

        normalized = []

        for operation in operations:
            if not isinstance(operation, str):
                raise ValueError(
                    "Every operation name must be a string."
                )

            normalized.append(operation.upper())

        normalized = tuple(normalized)

        if not normalized:
            raise ValueError("At least one operation is required.")

        unknown = set(normalized) - SUPPORTED_OPERATIONS

        if unknown:
            raise ValueError(
                f"Unsupported operations: {sorted(unknown)}"
            )

        return normalized

    @staticmethod
    def _validate_reuse_parameters(
        hot_fraction: float,
        reuse_probability: float,
    ) -> None:
        """Validate hot-set and reuse probabilities."""

        if (
            not isinstance(hot_fraction, (int, float))
            or isinstance(hot_fraction, bool)
            or not isfinite(hot_fraction)
            or not 0 < hot_fraction <= 1
        ):
            raise ValueError(
                "Hot fraction must be finite and in (0, 1]."
            )

        if (
            not isinstance(reuse_probability, (int, float))
            or isinstance(reuse_probability, bool)
            or not isfinite(reuse_probability)
            or not 0 <= reuse_probability <= 1
        ):
            raise ValueError(
                "Reuse probability must be finite and between 0 and 1."
            )

    def _create_blocks(
        self,
        dataset_id: str,
        num_blocks: int,
        block_size_bytes: int,
    ) -> tuple[DataBlock, ...]:
        """Create blocks distributed round-robin over memory banks."""

        if not isinstance(dataset_id, str) or not dataset_id.strip():
            raise ValueError("Dataset ID cannot be empty.")

        num_banks = self.config.memory.num_banks

        blocks_per_bank = [
            0 for _ in range(num_banks)
        ]

        for index in range(num_blocks):
            blocks_per_bank[index % num_banks] += 1

        max_bank_bytes = (
            max(blocks_per_bank) * block_size_bytes
        )

        if (
            max_bank_bytes
            > self.config.memory.capacity_per_bank_bytes
        ):
            raise ValueError(
                "Generated workload exceeds the configured "
                "capacity of at least one memory bank."
            )

        blocks = []

        for index in range(num_blocks):
            blocks.append(
                DataBlock(
                    block_id=f"{dataset_id}_B{index:06d}",
                    dataset_id=dataset_id,
                    size_bytes=block_size_bytes,
                    bank_id=index % num_banks,
                )
            )

        return tuple(blocks)

    @staticmethod
    def _output_size(
        operation: str,
        input_bytes: int,
    ) -> int:
        """Estimate output size using the current synthetic model."""

        if operation in {"COUNT", "SUM"}:
            return 8

        if operation == "FILTER":
            return max(1, input_bytes // 4)

        if operation == "RANDOM_LOOKUP":
            return min(input_bytes, 64)

        if operation == "MATMUL":
            return max(8, input_bytes // 4)

        if operation in {"BITMAP_AND", "BITMAP_OR", "VECTOR_MAP"}:
            return input_bytes

        raise ValueError(f"Unknown operation: {operation}")

    @staticmethod
    def _operation_count(
        operation: str,
        input_bytes: int,
        blocks_per_request: int,
    ) -> int:
        """
        Estimate primitive-operation count.

        These counts are abstract workload parameters, not executed
        Python instructions or detailed hardware instruction counts.
        """

        if operation == "RANDOM_LOOKUP":
            return blocks_per_request

        base_count = max(1, input_bytes // 8)

        if operation == "MATMUL":
            return base_count * 16

        return base_count

    def _make_request(
        self,
        request_index: int,
        dataset_id: str,
        block_ids: tuple[str, ...],
        blocks_by_id: dict[str, DataBlock],
        access_pattern: AccessPattern,
        operations: tuple[str, ...],
        arrival_interval_ns: float,
    ) -> ComputationRequest:
        """Create one request from the chosen input data blocks."""

        operation = self.rng.choice(operations)

        input_bytes = sum(
            blocks_by_id[block_id].size_bytes
            for block_id in block_ids
        )

        return ComputationRequest(
            request_id=f"R{request_index:07d}",
            dataset_id=dataset_id,
            operation=operation,
            input_block_ids=block_ids,
            output_bytes=self._output_size(
                operation,
                input_bytes,
            ),
            operation_count=self._operation_count(
                operation,
                input_bytes,
                len(block_ids),
            ),
            access_pattern=access_pattern,
            arrival_time_ns=(
                request_index * arrival_interval_ns
            ),
        )

    def _select_reused_blocks(
        self,
        all_block_ids: tuple[str, ...],
        hot_block_ids: tuple[str, ...],
        blocks_per_request: int,
        reuse_probability: float,
    ) -> tuple[str, ...]:
        """Prefer the hot set while avoiding duplicate inputs in a request."""

        selected = []

        for _ in range(blocks_per_request):
            if self.rng.random() < reuse_probability:
                preferred_pool = hot_block_ids
            else:
                preferred_pool = all_block_ids

            available = [
                block_id
                for block_id in preferred_pool
                if block_id not in selected
            ]

            if not available:
                available = [
                    block_id
                    for block_id in all_block_ids
                    if block_id not in selected
                ]

            selected.append(self.rng.choice(available))

        return tuple(selected)

    def generate_reused(
        self,
        num_requests: int = 256,
        num_blocks: int = 1024,
        block_size_bytes: int = 4096,
        blocks_per_request: int = 4,
        hot_fraction: float = DEFAULT_HOT_FRACTION,
        reuse_probability: float = DEFAULT_REUSE_PROBABILITY,
        arrival_interval_ns: float = 10.0,
        operations: tuple[str, ...] = DEFAULT_OPERATIONS,
        dataset_id: str = "reused_dataset",
        min_blocks_per_request: int | None = None,
        max_blocks_per_request: int | None = None,
    ) -> WorkloadTrace:
        """Generate requests that repeatedly access a hot data subset."""

        self._validate_common(
            num_requests,
            num_blocks,
            block_size_bytes,
            blocks_per_request,
            arrival_interval_ns,
        )

        min_count, max_count = self._resolve_block_count_range(
            blocks_per_request,
            num_blocks,
            min_blocks_per_request,
            max_blocks_per_request,
        )

        self._validate_reuse_parameters(
            hot_fraction,
            reuse_probability,
        )

        operations = self._validate_operations(operations)

        blocks = self._create_blocks(
            dataset_id,
            num_blocks,
            block_size_bytes,
        )

        blocks_by_id = {
            block.block_id: block
            for block in blocks
        }

        all_ids = tuple(blocks_by_id)

        hot_count = max(
            1,
            round(num_blocks * hot_fraction),
        )

        hot_ids = all_ids[:hot_count]

        request_block_counts = self._sample_request_block_counts(
            num_requests,
            min_count,
            max_count,
        )

        requests = []

        for index, selected_count in enumerate(request_block_counts):
            selected = self._select_reused_blocks(
                all_ids,
                hot_ids,
                selected_count,
                reuse_probability,
            )

            requests.append(
                self._make_request(
                    index,
                    dataset_id,
                    selected,
                    blocks_by_id,
                    AccessPattern.REUSED,
                    operations,
                    arrival_interval_ns,
                )
            )

        return WorkloadTrace(
            "high_reuse",
            blocks,
            tuple(requests),
        )

    def generate_streaming(
        self,
        num_requests: int = 256,
        num_blocks: int = 1024,
        block_size_bytes: int = 4096,
        blocks_per_request: int = 4,
        arrival_interval_ns: float = 10.0,
        operations: tuple[str, ...] = DEFAULT_OPERATIONS,
        dataset_id: str = "streaming_dataset",
        min_blocks_per_request: int | None = None,
        max_blocks_per_request: int | None = None,
    ) -> WorkloadTrace:
        """Generate sequential requests over fresh input blocks."""

        self._validate_common(
            num_requests,
            num_blocks,
            block_size_bytes,
            blocks_per_request,
            arrival_interval_ns,
        )

        min_count, max_count = self._resolve_block_count_range(
            blocks_per_request,
            num_blocks,
            min_blocks_per_request,
            max_blocks_per_request,
        )

        operations = self._validate_operations(operations)

        request_block_counts = self._sample_request_block_counts(
            num_requests,
            min_count,
            max_count,
        )

        required_blocks = sum(request_block_counts)

        if required_blocks > num_blocks:
            raise ValueError(
                "Streaming needs at least the sum of the generated "
                "request block counts as unique blocks. Increase "
                "num_blocks or reduce the number/size of requests."
            )

        blocks = self._create_blocks(
            dataset_id,
            num_blocks,
            block_size_bytes,
        )

        blocks_by_id = {
            block.block_id: block
            for block in blocks
        }

        all_ids = tuple(blocks_by_id)

        requests = []
        cursor = 0

        for index, selected_count in enumerate(request_block_counts):
            selected = all_ids[
                cursor:cursor + selected_count
            ]

            cursor += selected_count

            requests.append(
                self._make_request(
                    index,
                    dataset_id,
                    selected,
                    blocks_by_id,
                    AccessPattern.STREAMING,
                    operations,
                    arrival_interval_ns,
                )
            )

        return WorkloadTrace(
            "streaming",
            blocks,
            tuple(requests),
        )

    def generate_random(
        self,
        num_requests: int = 256,
        num_blocks: int = 1024,
        block_size_bytes: int = 4096,
        blocks_per_request: int = 4,
        arrival_interval_ns: float = 10.0,
        operations: tuple[str, ...] = DEFAULT_OPERATIONS,
        dataset_id: str = "random_dataset",
        min_blocks_per_request: int | None = None,
        max_blocks_per_request: int | None = None,
    ) -> WorkloadTrace:
        """Generate requests with random input-block selections."""

        self._validate_common(
            num_requests,
            num_blocks,
            block_size_bytes,
            blocks_per_request,
            arrival_interval_ns,
        )

        min_count, max_count = self._resolve_block_count_range(
            blocks_per_request,
            num_blocks,
            min_blocks_per_request,
            max_blocks_per_request,
        )

        operations = self._validate_operations(operations)

        blocks = self._create_blocks(
            dataset_id,
            num_blocks,
            block_size_bytes,
        )

        blocks_by_id = {
            block.block_id: block
            for block in blocks
        }

        all_ids = tuple(blocks_by_id)

        request_block_counts = self._sample_request_block_counts(
            num_requests,
            min_count,
            max_count,
        )

        requests = []

        for index, selected_count in enumerate(request_block_counts):
            selected = tuple(
                self.rng.sample(
                    all_ids,
                    selected_count,
                )
            )

            requests.append(
                self._make_request(
                    index,
                    dataset_id,
                    selected,
                    blocks_by_id,
                    AccessPattern.RANDOM,
                    operations,
                    arrival_interval_ns,
                )
            )

        return WorkloadTrace(
            "random_access",
            blocks,
            tuple(requests),
        )

    def generate_phase_changing(
        self,
        num_requests: int = 300,
        num_blocks: int = 1024,
        block_size_bytes: int = 4096,
        blocks_per_request: int = 4,
        hot_fraction: float = DEFAULT_HOT_FRACTION,
        reuse_probability: float = DEFAULT_REUSE_PROBABILITY,
        arrival_interval_ns: float = 10.0,
        operations: tuple[str, ...] = DEFAULT_OPERATIONS,
        dataset_id: str = "changing_dataset",
        min_blocks_per_request: int | None = None,
        max_blocks_per_request: int | None = None,
    ) -> WorkloadTrace:
        """
        Generate a trace with streaming, reuse and random phases.

        Phase boundaries are fixed by request count, but the runtime
        controller should infer the observed behaviour rather than
        receive future phase information.
        """

        self._validate_common(
            num_requests,
            num_blocks,
            block_size_bytes,
            blocks_per_request,
            arrival_interval_ns,
        )

        min_count, max_count = self._resolve_block_count_range(
            blocks_per_request,
            num_blocks,
            min_blocks_per_request,
            max_blocks_per_request,
        )

        self._validate_reuse_parameters(
            hot_fraction,
            reuse_probability,
        )

        operations = self._validate_operations(operations)

        streaming_count = num_requests // 3
        reused_count = num_requests // 3
        random_count = (
            num_requests
            - streaming_count
            - reused_count
        )

        request_block_counts = self._sample_request_block_counts(
            num_requests,
            min_count,
            max_count,
        )

        streaming_block_counts = request_block_counts[
            :streaming_count
        ]

        required_streaming_blocks = sum(
            streaming_block_counts
        )

        if required_streaming_blocks > num_blocks:
            raise ValueError(
                "Insufficient unique blocks for the streaming phase. "
                "Increase num_blocks or reduce streaming request sizes."
            )

        blocks = self._create_blocks(
            dataset_id,
            num_blocks,
            block_size_bytes,
        )

        blocks_by_id = {
            block.block_id: block
            for block in blocks
        }

        all_ids = tuple(blocks_by_id)

        hot_count = max(
            1,
            round(num_blocks * hot_fraction),
        )

        hot_ids = all_ids[:hot_count]

        requests = []
        request_index = 0
        cursor = 0

        # Phase 1: sequentially scan fresh, consecutive blocks.
        for index in range(streaming_count):
            selected_count = request_block_counts[index]

            selected = all_ids[
                cursor:cursor + selected_count
            ]

            cursor += selected_count

            requests.append(
                self._make_request(
                    request_index,
                    dataset_id,
                    selected,
                    blocks_by_id,
                    AccessPattern.STREAMING,
                    operations,
                    arrival_interval_ns,
                )
            )

            request_index += 1

        # Phase 2: repeatedly access the hot subset when selected.
        for local_index in range(reused_count):
            selected_count = request_block_counts[
                streaming_count + local_index
            ]

            selected = self._select_reused_blocks(
                all_ids,
                hot_ids,
                selected_count,
                reuse_probability,
            )

            requests.append(
                self._make_request(
                    request_index,
                    dataset_id,
                    selected,
                    blocks_by_id,
                    AccessPattern.REUSED,
                    operations,
                    arrival_interval_ns,
                )
            )

            request_index += 1

        # Phase 3: scatter accesses across the full dataset.
        for local_index in range(random_count):
            count_index = (
                streaming_count
                + reused_count
                + local_index
            )

            selected_count = request_block_counts[count_index]

            selected = tuple(
                self.rng.sample(
                    all_ids,
                    selected_count,
                )
            )

            requests.append(
                self._make_request(
                    request_index,
                    dataset_id,
                    selected,
                    blocks_by_id,
                    AccessPattern.RANDOM,
                    operations,
                    arrival_interval_ns,
                )
            )

            request_index += 1

        return WorkloadTrace(
            "phase_changing",
            blocks,
            tuple(requests),
        )
