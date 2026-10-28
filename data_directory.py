
"""
DynaMem: Data directory and memory-location tracking.

Responsibilities:
- Register data blocks and validate their home memory banks.
- Track block access counts and access times.
- Track copies held by the CPU cache or other components.
- Track memory-bank occupancy and access statistics.
- Provide input-size and data-location queries.

A new DataDirectory starts with fresh access statistics for each simulation.
All sizes are in bytes and times are in nanoseconds.
"""

from collections import defaultdict
from typing import Iterable

from config import SimulationConfig
from models import DataBlock


class DataDirectory:
    """Maintain the simulated system's view of data locations."""

    def __init__(
        self,
        config: SimulationConfig,
        blocks: Iterable[DataBlock] = (),
    ):
        self.config = config

        # Registered canonical data blocks, indexed by unique block ID.
        self._blocks: dict[str, DataBlock] = {}

        # Number of bytes stored in each block's home memory bank.
        self._bank_occupancy = [
            0 for _ in range(config.memory.num_banks)
        ]

        # Locations that currently hold a copy of each block.
        self._locations: dict[str, set[str]] = {}

        # Runtime access statistics.
        self._bank_accesses = [
            0 for _ in range(config.memory.num_banks)
        ]
        self._dataset_accesses: dict[str, int] = defaultdict(int)
        self._total_accesses = 0

        # Register blocks in the order supplied by the workload.
        for block in blocks:
            self.add_block(block)

    @staticmethod
    def bank_location(bank_id: int) -> str:
        """Return the standard location name for a memory bank."""

        if not isinstance(bank_id, int) or isinstance(bank_id, bool):
            raise ValueError("Bank ID must be an integer.")

        if bank_id < 0:
            raise ValueError("Bank ID cannot be negative.")

        return f"BANK:{bank_id}"

    def add_block(self, block: DataBlock) -> None:
        """
        Register a new block with fresh runtime access statistics.

        The source block is copied so that simulating one policy does
        not mutate the original workload trace or another simulation.
        """

        if not isinstance(block, DataBlock):
            raise TypeError("Only DataBlock objects can be registered.")

        if block.block_id in self._blocks:
            raise ValueError(
                f"Duplicate data-block ID: {block.block_id}"
            )

        num_banks = self.config.memory.num_banks

        if block.bank_id >= num_banks:
            raise ValueError(
                f"Block {block.block_id} uses invalid bank "
                f"{block.bank_id}; configured banks are "
                f"0 to {num_banks - 1}."
            )

        projected_occupancy = (
            self._bank_occupancy[block.bank_id] + block.size_bytes
        )

        bank_capacity = self.config.memory.capacity_per_bank_bytes

        if projected_occupancy > bank_capacity:
            raise ValueError(
                f"Adding block {block.block_id} exceeds the capacity "
                f"of memory bank {block.bank_id}."
            )

        # Preserve the block's static identity and placement, but start
        # a clean runtime history for this simulation.
        stored_block = DataBlock(
            block_id=block.block_id,
            dataset_id=block.dataset_id,
            size_bytes=block.size_bytes,
            bank_id=block.bank_id,
        )

        self._blocks[stored_block.block_id] = stored_block

        self._bank_occupancy[stored_block.bank_id] += (
            stored_block.size_bytes
        )

        # The home bank always contains the canonical copy.
        self._locations[stored_block.block_id] = {
            self.bank_location(stored_block.bank_id)
        }

    def get_block(self, block_id: str) -> DataBlock:
        """Return registered block metadata by block ID."""

        try:
            return self._blocks[block_id]
        except KeyError as exc:
            raise KeyError(
                f"Unknown data block: {block_id}"
            ) from exc

    @property
    def blocks(self) -> tuple[DataBlock, ...]:
        """Return all registered blocks as a tuple."""

        return tuple(self._blocks.values())

    @property
    def block_ids(self) -> tuple[str, ...]:
        """Return all registered block IDs."""

        return tuple(self._blocks.keys())

    @property
    def total_data_bytes(self) -> int:
        """Return the total size of unique blocks in main memory."""

        return sum(self._bank_occupancy)

    @property
    def total_accesses(self) -> int:
        """Return the total number of recorded block accesses."""

        return self._total_accesses

    @property
    def dataset_accesses(self) -> dict[str, int]:
        """Return a copy of the per-dataset access statistics."""

        return dict(self._dataset_accesses)

    @property
    def accesses_by_bank(self) -> dict[int, int]:
        """Return the recorded number of accesses to each home bank."""

        return {
            bank_id: count
            for bank_id, count in enumerate(self._bank_accesses)
        }

    def get_home_bank(self, block_id: str) -> int:
        """Return the canonical main-memory bank for a block."""

        return self.get_block(block_id).bank_id

    def record_access(
        self,
        block_id: str,
        time_ns: float,
    ) -> None:
        """Record one observed access to a block."""

        block = self.get_block(block_id)

        block.record_access(time_ns)

        self._total_accesses += 1
        self._bank_accesses[block.bank_id] += 1
        self._dataset_accesses[block.dataset_id] += 1

    def get_access_count(self, block_id: str) -> int:
        """Return the observed access count for one block."""

        return self.get_block(block_id).access_count

    def get_reuse_count(self, block_id: str) -> int:
        """Return the number of accesses after the first access."""

        return self.get_block(block_id).reuse_count

    def get_locations(self, block_id: str) -> tuple[str, ...]:
        """Return the currently recorded locations holding a block."""

        self.get_block(block_id)

        return tuple(sorted(self._locations[block_id]))

    def has_copy(
        self,
        block_id: str,
        location: str,
    ) -> bool:
        """Check whether a location is recorded as holding a copy."""

        self.get_block(block_id)

        return location in self._locations[block_id]

    def add_copy(
        self,
        block_id: str,
        location: str,
    ) -> bool:
        """
        Record a copy after a transfer or cache-fill event.

        Returns True if a new copy was recorded, or False if that
        location already held a copy.

        This method updates metadata only; it does not simulate a
        transfer or charge its latency and energy.
        """

        self.get_block(block_id)

        if not isinstance(location, str) or not location.strip():
            raise ValueError("Copy location must be a non-empty string.")

        if location in self._locations[block_id]:
            return False

        self._locations[block_id].add(location)
        return True

    def remove_copy(
        self,
        block_id: str,
        location: str,
    ) -> bool:
        """
        Remove a recorded copy, for example after a cache eviction.

        The canonical home-memory copy cannot be removed.
        Returns False if the requested non-canonical copy is absent.
        """

        block = self.get_block(block_id)
        home_location = self.bank_location(block.bank_id)

        if location == home_location:
            raise ValueError(
                "Cannot remove a block's canonical home-memory copy."
            )

        if location not in self._locations[block_id]:
            return False

        self._locations[block_id].remove(location)
        return True

    def input_size_bytes(
        self,
        block_ids: Iterable[str],
    ) -> int:
        """Calculate the total size of the referenced input blocks."""

        return sum(
            self.get_block(block_id).size_bytes
            for block_id in block_ids
        )

    def get_bank_occupancy_bytes(self, bank_id: int) -> int:
        """Return the number of canonical data bytes in one bank."""

        if (
            not isinstance(bank_id, int)
            or isinstance(bank_id, bool)
            or not 0 <= bank_id < self.config.memory.num_banks
        ):
            raise ValueError("Invalid memory-bank ID.")

        return self._bank_occupancy[bank_id]

    def get_bank_free_bytes(self, bank_id: int) -> int:
        """Return the remaining main-memory capacity of one bank."""

        return (
            self.config.memory.capacity_per_bank_bytes
            - self.get_bank_occupancy_bytes(bank_id)
        )

    def get_dataset_blocks(
        self,
        dataset_id: str,
    ) -> tuple[DataBlock, ...]:
        """Return blocks that belong to a particular dataset."""

        return tuple(
            block
            for block in self._blocks.values()
            if block.dataset_id == dataset_id
        )

    def summary(self) -> dict:
        """Return a snapshot of the directory's key statistics."""

        return {
            "block_count": len(self._blocks),
            "total_data_bytes": self.total_data_bytes,
            "total_accesses": self.total_accesses,
            "dataset_accesses": self.dataset_accesses,
            "accesses_by_bank": self.accesses_by_bank,
            "bank_occupancy_bytes": {
                bank_id: occupancy
                for bank_id, occupancy in enumerate(
                    self._bank_occupancy
                )
            },
        }
