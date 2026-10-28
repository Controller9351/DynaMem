
"""
DynaMem: Configuration for the memory-centric architecture simulator.

Units:
    Time       : nanoseconds (ns)
    Data size  : bytes
    Bandwidth
    : bytes per nanosecond (bytes/ns)
    Compute    : operations per nanosecond (ops/ns)
    Energy     : picojoules (pJ)

All default hardware values are illustrative simulation parameters.
They are not measurements from real hardware.
"""

from dataclasses import dataclass
from typing import Mapping


SUPPORTED_OPERATIONS = frozenset({
    "BITMAP_AND",
    "BITMAP_OR",
    "FILTER",
    "COUNT",
    "SUM",
    "VECTOR_MAP",
    "MATMUL",
    "RANDOM_LOOKUP",
})

REQUIRED_RESOURCES = frozenset({
    "CPU",
    "CIM_A",
    "CIM_P",
    "NMP",
})

REQUIRED_LINKS = frozenset({
    "CPU_MEMORY",
    "MEMORY_NMP",
    "BANK_TO_BANK",
})


@dataclass(frozen=True)
class CacheConfig:
    """Configuration for the simulated CPU cache."""

    capacity_bytes: int = 8 * 1024 * 1024
    line_size_bytes: int = 64
    hit_latency_ns: float = 4.0
    energy_per_access_pj: float = 2.0

    def __post_init__(self):
        if self.capacity_bytes <= 0:
            raise ValueError("Cache capacity must be positive.")

        if self.line_size_bytes <= 0:
            raise ValueError("Cache line size must be positive.")

        if self.capacity_bytes % self.line_size_bytes != 0:
            raise ValueError(
                "Cache capacity must be divisible by its line size."
            )

        if self.hit_latency_ns < 0:
            raise ValueError("Cache hit latency cannot be negative.")

        if self.energy_per_access_pj < 0:
            raise ValueError("Cache access energy cannot be negative.")


@dataclass(frozen=True)
class MemoryConfig:
    """Configuration for main memory and its banks."""

    num_banks: int = 8
    capacity_per_bank_bytes: int = 512 * 1024 * 1024

    access_latency_ns: float = 60.0
    bandwidth_per_bank_bytes_per_ns: float = 16.0
    energy_per_byte_pj: float = 0.08

    max_concurrent_requests_per_bank: int = 1

    def __post_init__(self):
        if self.num_banks <= 0:
            raise ValueError("Number of memory banks must be positive.")

        if self.capacity_per_bank_bytes <= 0:
            raise ValueError("Memory-bank capacity must be positive.")

        if self.access_latency_ns < 0:
            raise ValueError("Memory access latency cannot be negative.")

        if self.bandwidth_per_bank_bytes_per_ns <= 0:
            raise ValueError("Memory-bank bandwidth must be positive.")

        if self.energy_per_byte_pj < 0:
            raise ValueError("Memory energy per byte cannot be negative.")

        if self.max_concurrent_requests_per_bank <= 0:
            raise ValueError(
                "Each bank must support at least one concurrent request."
            )


@dataclass(frozen=True)
class LinkConfig:
    """Configuration for a simulated data-movement link."""

    bandwidth_bytes_per_ns: float
    latency_ns: float
    energy_per_byte_pj: float

    max_concurrent_transfers: int = 1

    def __post_init__(self):
        if self.bandwidth_bytes_per_ns <= 0:
            raise ValueError("Link bandwidth must be positive.")

        if self.latency_ns < 0:
            raise ValueError("Link latency cannot be negative.")

        if self.energy_per_byte_pj < 0:
            raise ValueError("Link energy per byte cannot be negative.")

        if self.max_concurrent_transfers <= 0:
            raise ValueError(
                "A link must support at least one concurrent transfer."
            )


@dataclass(frozen=True)
class ComputeResourceConfig:
    """Configuration for one type of compute resource."""

    instances: int
    ops_per_ns: float

    launch_overhead_ns: float
    energy_per_op_pj: float
    fixed_energy_pj: float

    supported_operations: frozenset[str]
    supports_all: bool = False

    def __post_init__(self):
        if self.instances <= 0:
            raise ValueError("Compute-resource instances must be positive.")

        if self.ops_per_ns <= 0:
            raise ValueError("Compute throughput must be positive.")

        if self.launch_overhead_ns < 0:
            raise ValueError("Launch overhead cannot be negative.")

        if self.energy_per_op_pj < 0 or self.fixed_energy_pj < 0:
            raise ValueError("Compute energy values cannot be negative.")

        unknown_operations = (
            self.supported_operations - SUPPORTED_OPERATIONS
        )

        if unknown_operations:
            raise ValueError(
                f"Unknown operations: {sorted(unknown_operations)}"
            )

    def supports(self, operation: str) -> bool:
        """Return whether this resource supports an operation."""

        return (
            self.supports_all
            or operation in self.supported_operations
        )


@dataclass(frozen=True)
class RuntimeConfig:
    """Configuration for adaptive runtime placement."""

    decision_overhead_ns: float = 8.0

    # 1.0 prioritizes latency; 0.0 prioritizes energy.
    latency_weight: float = 0.7

    # Weight used when updating estimates from observed outcomes.
    ema_alpha: float = 0.2

    # Require a meaningful predicted improvement before switching.
    switching_threshold: float = 0.05

    history_window: int = 64

    def __post_init__(self):
        if self.decision_overhead_ns < 0:
            raise ValueError(
                "Runtime decision overhead cannot be negative."
            )

        if not 0.0 <= self.latency_weight <= 1.0:
            raise ValueError("Latency weight must be between 0 and 1.")

        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("EMA alpha must be in (0, 1].")

        if not 0.0 <= self.switching_threshold < 1.0:
            raise ValueError(
                "Switching threshold must be in [0, 1)."
            )

        if self.history_window <= 0:
            raise ValueError("History window must be positive.")


@dataclass(frozen=True)
class SimulationConfig:
    """Complete configuration for one DynaMem simulation."""

    memory: MemoryConfig
    cache: CacheConfig

    links: Mapping[str, LinkConfig]
    resources: Mapping[str, ComputeResourceConfig]

    runtime: RuntimeConfig
    random_seed: int = 42

    def __post_init__(self):
        missing_resources = REQUIRED_RESOURCES - self.resources.keys()

        if missing_resources:
            raise ValueError(
                f"Missing compute resources: {sorted(missing_resources)}"
            )

        missing_links = REQUIRED_LINKS - self.links.keys()

        if missing_links:
            raise ValueError(
                f"Missing data links: {sorted(missing_links)}"
            )

        # Each bank has its own simulated CIM-A and CIM-P instance.
        for resource_name in ("CIM_A", "CIM_P"):
            instances = self.resources[resource_name].instances

            if instances != self.memory.num_banks:
                raise ValueError(
                    f"{resource_name} has {instances} instances, "
                    f"but memory has {self.memory.num_banks} banks."
                )


def make_default_config() -> SimulationConfig:
    """Create the default, configurable DynaMem architecture."""

    memory = MemoryConfig()

    resources = {
        "CPU": ComputeResourceConfig(
            instances=4,
            ops_per_ns=16.0,
            launch_overhead_ns=3.0,
            energy_per_op_pj=0.50,
            fixed_energy_pj=10.0,
            supported_operations=SUPPORTED_OPERATIONS,
            supports_all=True,
        ),

        "CIM_A": ComputeResourceConfig(
            instances=memory.num_banks,
            ops_per_ns=128.0,
            launch_overhead_ns=5.0,
            energy_per_op_pj=0.08,
            fixed_energy_pj=5.0,
            supported_operations=frozenset({
                "BITMAP_AND",
                "BITMAP_OR",
            }),
        ),

        "CIM_P": ComputeResourceConfig(
            instances=memory.num_banks,
            ops_per_ns=256.0,
            launch_overhead_ns=7.0,
            energy_per_op_pj=0.12,
            fixed_energy_pj=7.0,
            supported_operations=frozenset({
                "BITMAP_AND",
                "BITMAP_OR",
                "FILTER",
                "COUNT",
                "SUM",
            }),
        ),

        "NMP": ComputeResourceConfig(
            instances=1,
            ops_per_ns=64.0,
            launch_overhead_ns=8.0,
            energy_per_op_pj=0.20,
            fixed_energy_pj=10.0,
            supported_operations=SUPPORTED_OPERATIONS,
        ),
    }

    links = {
        "CPU_MEMORY": LinkConfig(
            bandwidth_bytes_per_ns=32.0,
            latency_ns=30.0,
            energy_per_byte_pj=1.20,
        ),

        "MEMORY_NMP": LinkConfig(
            bandwidth_bytes_per_ns=128.0,
            latency_ns=8.0,
            energy_per_byte_pj=0.25,
        ),

        "BANK_TO_BANK": LinkConfig(
            bandwidth_bytes_per_ns=16.0,
            latency_ns=25.0,
            energy_per_byte_pj=0.50,
        ),
    }

    return SimulationConfig(
        memory=memory,
        cache=CacheConfig(),
        links=links,
        resources=resources,
        runtime=RuntimeConfig(),
        random_seed=42,
    )
