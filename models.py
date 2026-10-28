
"""
DynaMem: Data models for the memory-centric architecture simulator.

This module defines the objects exchanged between workload generation,
memory management, runtime placement, and simulation components.
"""

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Mapping

from config import SUPPORTED_OPERATIONS


class AccessPattern(str, Enum):
    """Describes how a workload accesses its input data."""

    REUSED = "reused"
    STREAMING = "streaming"
    RANDOM = "random"


@dataclass
class DataBlock:
    """
    Represents a block of data stored in a main-memory bank.

    The block's home bank is its canonical location. A CPU cache or
    another component may hold a copy, which will be tracked separately.
    """

    block_id: str
    dataset_id: str
    size_bytes: int
    bank_id: int

    access_count: int = 0
    last_access_time_ns: float | None = None

    def __post_init__(self):
        if not self.block_id.strip():
            raise ValueError("Block ID cannot be empty.")

        if not self.dataset_id.strip():
            raise ValueError("Dataset ID cannot be empty.")

        if self.size_bytes <= 0:
            raise ValueError("Block size must be positive.")

        if self.bank_id < 0:
            raise ValueError("Bank ID cannot be negative.")

        if self.access_count < 0:
            raise ValueError("Access count cannot be negative.")

        if self.last_access_time_ns is not None:
            if (
                not isfinite(self.last_access_time_ns)
                or self.last_access_time_ns < 0
            ):
                raise ValueError(
                    "Last access time must be finite and non-negative."
                )

    @property
    def reuse_count(self) -> int:
        """Number of accesses after the first access."""

        return max(0, self.access_count - 1)

    def record_access(self, time_ns: float) -> None:
        """Record an access to this block at simulated time time_ns."""

        if not isfinite(time_ns) or time_ns < 0:
            raise ValueError(
                "Access time must be finite and non-negative."
            )

        if (
            self.last_access_time_ns is not None
            and time_ns < self.last_access_time_ns
        ):
            raise ValueError(
                "Access time cannot move backwards for a data block."
            )

        self.access_count += 1
        self.last_access_time_ns = time_ns


@dataclass(frozen=True)
class ComputationRequest:
    """
    Represents one computation submitted to DynaMem.

    Input size is derived from the referenced DataBlock objects,
    rather than duplicated here.
    """

    request_id: str
    dataset_id: str
    operation: str
    input_block_ids: tuple[str, ...]

    output_bytes: int
    operation_count: int

    access_pattern: AccessPattern
    arrival_time_ns: float = 0.0

    def __post_init__(self):
        if not self.request_id.strip():
            raise ValueError("Request ID cannot be empty.")

        if not self.dataset_id.strip():
            raise ValueError("Dataset ID cannot be empty.")

        # Allow users to provide either a string or an enum value.
        normalized_operation = self.operation.upper()
        object.__setattr__(self, "operation", normalized_operation)

        if normalized_operation not in SUPPORTED_OPERATIONS:
            raise ValueError(
                f"Unsupported operation: {normalized_operation}"
            )

        # Convert a supplied list of block IDs into an immutable tuple.
        object.__setattr__(
            self,
            "input_block_ids",
            tuple(self.input_block_ids),
        )

        if not self.input_block_ids:
            raise ValueError(
                "A computation request must reference input blocks."
            )

        if any(
            not isinstance(block_id, str) or not block_id.strip()
            for block_id in self.input_block_ids
        ):
            raise ValueError(
                "Every input block ID must be a non-empty string."
            )

        if self.output_bytes < 0:
            raise ValueError("Output size cannot be negative.")

        if self.operation_count <= 0:
            raise ValueError("Operation count must be positive.")

        if not isinstance(self.access_pattern, AccessPattern):
            object.__setattr__(
                self,
                "access_pattern",
                AccessPattern(self.access_pattern),
            )

        if (
            not isfinite(self.arrival_time_ns)
            or self.arrival_time_ns < 0
        ):
            raise ValueError(
                "Arrival time must be finite and non-negative."
            )


@dataclass(frozen=True)
class PlacementDecision:
    """
    Stores the runtime controller's decision for one request.

    Candidate scores allow us to inspect why the chosen resource
    was preferred over the alternatives.
    """

    request_id: str
    selected_resource: str

    predicted_latency_ns: float
    predicted_energy_pj: float
    score: float

    reason: str
    candidate_scores: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not self.request_id.strip():
            raise ValueError("Decision request ID cannot be empty.")

        if not self.selected_resource.strip():
            raise ValueError("Selected resource cannot be empty.")

        if not self.reason.strip():
            raise ValueError("Decision reason cannot be empty.")

        numeric_values = {
            "predicted latency": self.predicted_latency_ns,
            "predicted energy": self.predicted_energy_pj,
            "score": self.score,
        }

        for name, value in numeric_values.items():
            if not isfinite(value) or value < 0:
                raise ValueError(
                    f"{name.capitalize()} must be finite and non-negative."
                )

        for resource_name, candidate_score in self.candidate_scores.items():
            if not resource_name.strip():
                raise ValueError(
                    "Candidate resource names cannot be empty."
                )

            if (
                not isfinite(candidate_score)
                or candidate_score < 0
            ):
                raise ValueError(
                    "Candidate scores must be finite and non-negative."
                )


@dataclass(frozen=True)
class ExecutionRecord:
    """
    Records the outcome of one simulated computation.

    All time values use nanoseconds; energy uses picojoules;
    data movement is measured in bytes.
    """

    request_id: str
    resource_name: str

    arrival_time_ns: float
    decision_time_ns: float
    start_time_ns: float
    finish_time_ns: float

    input_bytes_moved: int
    output_bytes_moved: int
    cross_bank_bytes_moved: int

    queue_wait_ns: float
    compute_time_ns: float
    transfer_time_ns: float
    decision_overhead_ns: float

    energy_pj: float
    memory_access_time_ns: float = 0.0
    cache_access_time_ns: float = 0.0

    def __post_init__(self):
        if not self.request_id.strip():
            raise ValueError("Execution-record request ID cannot be empty.")

        if not self.resource_name.strip():
            raise ValueError("Execution-record resource cannot be empty.")

        time_values = {
            "arrival time": self.arrival_time_ns,
            "decision time": self.decision_time_ns,
            "start time": self.start_time_ns,
            "finish time": self.finish_time_ns,
            "queue wait": self.queue_wait_ns,
            "compute time": self.compute_time_ns,
            "transfer time": self.transfer_time_ns,
            "decision overhead": self.decision_overhead_ns,
            "memory access time": self.memory_access_time_ns,
            "cache access time": self.cache_access_time_ns,
        }

        for name, value in time_values.items():
            if not isfinite(value) or value < 0:
                raise ValueError(
                    f"{name.capitalize()} must be finite and non-negative."
                )

        if not (
            self.arrival_time_ns
            <= self.decision_time_ns
            <= self.start_time_ns
            <= self.finish_time_ns
        ):
            raise ValueError(
                "Expected arrival <= decision <= start <= finish."
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

        if not isfinite(self.energy_pj) or self.energy_pj < 0:
            raise ValueError(
                "Recorded energy must be finite and non-negative."
            )

    @property
    def total_latency_ns(self) -> float:
        """Elapsed simulated time from arrival to completion."""

        return self.finish_time_ns - self.arrival_time_ns

    @property
    def total_moved_bytes(self) -> int:
        """Sum the movement categories recorded for this request."""

        return (
            self.input_bytes_moved
            + self.output_bytes_moved
            + self.cross_bank_bytes_moved
        )
