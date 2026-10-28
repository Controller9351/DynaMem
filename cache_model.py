
"""
DynaMem: CPU-cache simulation using block-level LRU replacement.

Responsibilities:
- Track cached data blocks and their resident sizes.
- Simulate cache hits and misses.
- Transfer missed blocks from main memory.
- Evict least-recently-used blocks when capacity is exceeded.
- Synchronize CPU-cache copies with the DataDirectory.
- Estimate cache, memory and transfer time and energy.

This is a block-level simulation, not a cycle-accurate cache model.
All time values use nanoseconds; energy values use picojoules.
"""

from collections import OrderedDict
from dataclasses import dataclass
from math import isfinite
from typing import Iterable

from config import SimulationConfig
from data_directory import DataDirectory


CPU_CACHE_LOCATION = "CPU_CACHE"


@dataclass(frozen=True)
class CacheAccessResult:
    """Outcome and estimated cost of one batch of cache accesses."""

    requested_blocks: int
    cache_hits: int
    cache_misses: int

    input_bytes_transferred: int
    evicted_block_ids: tuple[str, ...]

    cache_access_time_ns: float
    memory_access_time_ns: float
    transfer_time_ns: float

    energy_pj: float

    def __post_init__(self):
        for name in (
            "requested_blocks",
            "cache_hits",
            "cache_misses",
            "input_bytes_transferred",
        ):
            value = getattr(self, name)

            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError(
                    f"{name} must be a non-negative integer."
                )

        if self.cache_hits + self.cache_misses != self.requested_blocks:
            raise ValueError(
                "Cache hits plus misses must equal requested blocks."
            )

        for name in (
            "cache_access_time_ns",
            "memory_access_time_ns",
            "transfer_time_ns",
            "energy_pj",
        ):
            value = getattr(self, name)

            if not isfinite(value) or value < 0:
                raise ValueError(
                    f"{name} must be finite and non-negative."
                )

    @property
    def hit_ratio(self) -> float:
        """Fraction of requested blocks that were cache hits."""

        if self.requested_blocks == 0:
            return 0.0

        return self.cache_hits / self.requested_blocks

    @property
    def total_access_time_ns(self) -> float:
        """Total modelled time for this cache-access batch."""

        return (
            self.cache_access_time_ns
            + self.memory_access_time_ns
            + self.transfer_time_ns
        )


class CacheModel:
    """Model a limited-capacity CPU cache with LRU replacement."""

    def __init__(
        self,
        config: SimulationConfig,
        directory: DataDirectory,
    ):
        if directory.config != config:
            raise ValueError(
                "CacheModel and DataDirectory must use the same configuration."
            )

        self.config = config
        self.directory = directory

        # OrderedDict maintains least-recently-used entries first.
        # Mapping: block ID -> cache-resident size in bytes.
        self._entries: OrderedDict[str, int] = OrderedDict()

        self._used_bytes = 0
        self._total_hits = 0
        self._total_misses = 0
        self._total_bytes_transferred = 0
        self._total_evictions = 0

        self._last_access_time_ns = 0.0

    @property
    def used_bytes(self) -> int:
        """Current cache capacity occupied by resident blocks."""

        return self._used_bytes

    @property
    def capacity_bytes(self) -> int:
        """Configured CPU-cache capacity."""

        return self.config.cache.capacity_bytes

    @property
    def resident_block_ids(self) -> tuple[str, ...]:
        """Resident block IDs, from least recently used to most recent."""

        return tuple(self._entries.keys())

    @property
    def hit_ratio(self) -> float:
        """Hit ratio accumulated over all recorded cache accesses."""

        total = self._total_hits + self._total_misses

        if total == 0:
            return 0.0

        return self._total_hits / total

    def contains(self, block_id: str) -> bool:
        """Check whether a block currently resides in the CPU cache."""

        self.directory.get_block(block_id)
        return block_id in self._entries

    def _resident_size(self, block_size_bytes: int) -> int:
        """Round a block's resident footprint up to cache-line boundaries."""

        line_size = self.config.cache.line_size_bytes

        return (
            (block_size_bytes + line_size - 1) // line_size
        ) * line_size

    def _evict_one(self) -> str:
        """Evict the least-recently-used block and return its ID."""

        if not self._entries:
            raise RuntimeError("Cannot evict from an empty cache.")

        block_id, resident_size = self._entries.popitem(last=False)

        self._used_bytes -= resident_size

        self.directory.remove_copy(
            block_id,
            CPU_CACHE_LOCATION,
        )

        self._total_evictions += 1

        return block_id

    def access_blocks(
        self,
        block_ids: Iterable[str],
        time_ns: float,
    ) -> CacheAccessResult:
        """
        Access a batch of blocks and update cache residency.

        A unique missed block is fetched from main memory once in this
        batch. The aggregate transfer is modelled as one bulk transfer.

        Blocks larger than the entire cache can be read but are not cached.
        """

        if not isfinite(time_ns) or time_ns < 0:
            raise ValueError(
                "Cache-access time must be finite and non-negative."
            )

        if time_ns < self._last_access_time_ns:
            raise ValueError(
                "Cache-access times cannot move backwards."
            )

        # Remove repeated IDs while preserving request order.
        unique_ids = tuple(dict.fromkeys(block_ids))

        if not unique_ids:
            raise ValueError(
                "At least one data block must be accessed."
            )

        # Validate all IDs before changing cache state.
        blocks = {
            block_id: self.directory.get_block(block_id)
            for block_id in unique_ids
        }

        hits = 0
        misses = 0
        bytes_to_fetch = 0
        evicted_ids = []

        for block_id in unique_ids:
            block = blocks[block_id]

            if block_id in self._entries:
                hits += 1

                # A hit makes this block the most recently used.
                self._entries.move_to_end(block_id)

                continue

            misses += 1
            bytes_to_fetch += block.size_bytes

            resident_size = self._resident_size(block.size_bytes)

            # The block can be read from memory, but cannot be retained
            # if its rounded footprint exceeds the entire cache.
            if resident_size > self.capacity_bytes:
                continue

            # Evict old blocks until the new block fits.
            while (
                self._used_bytes + resident_size
                > self.capacity_bytes
            ):
                evicted_ids.append(self._evict_one())

            self._entries[block_id] = resident_size
            self._used_bytes += resident_size

            self.directory.add_copy(
                block_id,
                CPU_CACHE_LOCATION,
            )

        # Each referenced unique block incurs a cache lookup cost.
        # Misses additionally incur memory and bulk-transfer costs.
        cache_access_time_ns = (
            len(unique_ids)
            * self.config.cache.hit_latency_ns
        )

        if bytes_to_fetch > 0:
            link = self.config.links["CPU_MEMORY"]

            transfer_time_ns = (
                link.latency_ns
                + bytes_to_fetch / link.bandwidth_bytes_per_ns
            )

            memory_access_time_ns = (
                self.config.memory.access_latency_ns
            )

            energy_pj = (
                len(unique_ids)
                * self.config.cache.energy_per_access_pj
                + bytes_to_fetch * link.energy_per_byte_pj
                + bytes_to_fetch
                * self.config.memory.energy_per_byte_pj
            )
        else:
            transfer_time_ns = 0.0
            memory_access_time_ns = 0.0

            energy_pj = (
                len(unique_ids)
                * self.config.cache.energy_per_access_pj
            )

        self._total_hits += hits
        self._total_misses += misses
        self._total_bytes_transferred += bytes_to_fetch
        self._last_access_time_ns = time_ns

        return CacheAccessResult(
            requested_blocks=len(unique_ids),
            cache_hits=hits,
            cache_misses=misses,
            input_bytes_transferred=bytes_to_fetch,
            evicted_block_ids=tuple(evicted_ids),
            cache_access_time_ns=cache_access_time_ns,
            memory_access_time_ns=memory_access_time_ns,
            transfer_time_ns=transfer_time_ns,
            energy_pj=energy_pj,
        )

    def flush(self, reset_statistics: bool = True) -> None:
        """
        Evict every cached block.

        Use this between independent experiments when a cold cache is
        required. Runtime statistics can optionally be retained.
        """

        for block_id in tuple(self._entries.keys()):
            self.directory.remove_copy(
                block_id,
                CPU_CACHE_LOCATION,
            )

        self._entries.clear()
        self._used_bytes = 0

        if reset_statistics:
            self._total_hits = 0
            self._total_misses = 0
            self._total_bytes_transferred = 0
            self._total_evictions = 0

    def summary(self) -> dict:
        """Return a snapshot of cache occupancy and cumulative statistics."""

        total_accesses = self._total_hits + self._total_misses

        return {
            "capacity_bytes": self.capacity_bytes,
            "used_bytes": self.used_bytes,
            "free_bytes": self.capacity_bytes - self.used_bytes,
            "resident_blocks": len(self._entries),
            "total_hits": self._total_hits,
            "total_misses": self._total_misses,
            "hit_ratio": (
                self._total_hits / total_accesses
                if total_accesses
                else 0.0
            ),
            "total_bytes_transferred": self._total_bytes_transferred,
            "total_evictions": self._total_evictions,
        }
