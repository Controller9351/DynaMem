
"""
DynaMem: Runtime workload monitoring.

Responsibilities:
- Observe input-block accesses as requests enter execution.
- Track a sliding window of recent accesses for each dataset.
- Estimate reuse and sequential-access characteristics.
- Infer whether the observed behaviour is reused, streaming or random.
- Provide runtime statistics to the placement controller.

Important:
The monitor does not inspect ComputationRequest.access_pattern.
It infers behaviour only from observed accesses and known block order.
"""

from collections import defaultdict, deque
from math import isfinite

from config import SimulationConfig
from data_directory import DataDirectory
from models import AccessPattern, ComputationRequest


# Classification thresholds. These are tunable model parameters.
MIN_OBSERVATIONS = 16
STREAMING_MIN_UNIQUE_RATIO = 0.75
STREAMING_MIN_SEQUENTIAL_RATIO = 0.65
REUSED_MAX_UNIQUE_RATIO = 0.85


class RuntimeMonitor:
    """Observe runtime memory access behaviour."""

    def __init__(
        self,
        config: SimulationConfig,
        directory: DataDirectory,
    ):
        self.config = config
        self.directory = directory

        window_size = config.runtime.history_window

        # Each dataset maintains an independent sliding access window.
        self._windows = defaultdict(
            lambda: deque(maxlen=window_size)
        )

        # Record block positions using registered memory layout.
        # This is static metadata, not future access information.
        self._block_rank = {}
        next_rank_by_dataset = defaultdict(int)

        for block in directory.blocks:
            dataset_id = block.dataset_id

            self._block_rank[block.block_id] = (
                next_rank_by_dataset[dataset_id]
            )

            next_rank_by_dataset[dataset_id] += 1

        self._last_observation_time_ns = 0.0
        self._total_observed_accesses = 0

    @property
    def total_observed_accesses(self) -> int:
        """Number of block accesses observed by this monitor."""

        return self._total_observed_accesses

    def observe_access(
        self,
        block_id: str,
        time_ns: float,
    ) -> AccessPattern | None:
        """
        Record one observed block access.

        Returns the inferred pattern when sufficient observations
        are available; otherwise returns None.
        """

        if not isfinite(time_ns) or time_ns < 0:
            raise ValueError(
                "Observation time must be finite and non-negative."
            )

        if time_ns < self._last_observation_time_ns:
            raise ValueError(
                "Observation times cannot move backwards."
            )

        block = self.directory.get_block(block_id)

        # Update the authoritative access history.
        self.directory.record_access(block_id, time_ns)

        # Store only information actually observed at runtime.
        event = (
            block_id,
            self._block_rank[block_id],
            time_ns,
        )

        self._windows[block.dataset_id].append(event)

        self._last_observation_time_ns = time_ns
        self._total_observed_accesses += 1

        return self.get_observed_pattern(block.dataset_id)

    def observe_request(
        self,
        request: ComputationRequest,
        time_ns: float,
    ) -> AccessPattern | None:
        """
        Observe the input blocks of a request.

        Call this when the request's input-access event is modelled
        to occur, not merely because a future request exists in a trace.
        """

        if not isinstance(request, ComputationRequest):
            raise TypeError(
                "Expected a ComputationRequest object."
            )

        # Validate the current request against registered data.
        for block_id in request.input_block_ids:
            block = self.directory.get_block(block_id)

            if block.dataset_id != request.dataset_id:
                raise ValueError(
                    f"Block {block_id} belongs to dataset "
                    f"{block.dataset_id}, not {request.dataset_id}."
                )

        inferred_pattern = None

        for block_id in request.input_block_ids:
            inferred_pattern = self.observe_access(
                block_id,
                time_ns,
            )

        return inferred_pattern

    def get_observed_pattern(
        self,
        dataset_id: str,
    ) -> AccessPattern | None:
        """
        Infer the recent access pattern for a dataset.

        Returns None if there are not enough observations.
        """

        statistics = self.get_statistics(dataset_id)

        if statistics["window_size"] < MIN_OBSERVATIONS:
            return None

        unique_ratio = statistics["unique_ratio"]
        sequential_ratio = statistics["sequential_ratio"]

        # Sequential scanning takes priority when there is strong
        # evidence of consecutive access to previously unseen blocks.
        if (
            sequential_ratio >= STREAMING_MIN_SEQUENTIAL_RATIO
            and unique_ratio >= STREAMING_MIN_UNIQUE_RATIO
        ):
            return AccessPattern.STREAMING

        # A smaller fraction of unique blocks indicates repeated access.
        if unique_ratio <= REUSED_MAX_UNIQUE_RATIO:
            return AccessPattern.REUSED

        # Sufficient observations, high uniqueness and weak sequential
        # locality suggest irregular/random access.
        return AccessPattern.RANDOM

    def get_statistics(self, dataset_id: str) -> dict:
        """
        Return statistics calculated from recent observed accesses.

        The values are descriptive features, not probabilities
        or calibrated confidence scores.
        """

        if not isinstance(dataset_id, str) or not dataset_id.strip():
            raise ValueError("Dataset ID cannot be empty.")

        events = list(self._windows[dataset_id])
        window_size = len(events)

        if window_size == 0:
            return {
                "dataset_id": dataset_id,
                "window_size": 0,
                "unique_blocks": 0,
                "unique_ratio": 0.0,
                "reuse_ratio": 0.0,
                "sequential_ratio": 0.0,
                "observed_pattern": None,
            }

        unique_blocks = len({
            block_id
            for block_id, _, _ in events
        })

        unique_ratio = unique_blocks / window_size
        reuse_ratio = 1.0 - unique_ratio

        # Count consecutive accesses to adjacent logical block ranks.
        consecutive_pairs = 0

        for index in range(1, window_size):
            previous_rank = events[index - 1][1]
            current_rank = events[index][1]

            if current_rank == previous_rank + 1:
                consecutive_pairs += 1

        if window_size >= 2:
            sequential_ratio = (
                consecutive_pairs / (window_size - 1)
            )
        else:
            sequential_ratio = 0.0

        observed_pattern = self.get_observed_pattern_without_stats(
            window_size,
            unique_ratio,
            sequential_ratio,
        )

        return {
            "dataset_id": dataset_id,
            "window_size": window_size,
            "unique_blocks": unique_blocks,
            "unique_ratio": unique_ratio,
            "reuse_ratio": reuse_ratio,
            "sequential_ratio": sequential_ratio,
            "observed_pattern": (
                observed_pattern.value
                if observed_pattern is not None
                else None
            ),
        }

    @staticmethod
    def get_observed_pattern_without_stats(
        window_size: int,
        unique_ratio: float,
        sequential_ratio: float,
    ) -> AccessPattern | None:
        """Classify already-computed features without recursive calls."""

        if window_size < MIN_OBSERVATIONS:
            return None

        if (
            sequential_ratio >= STREAMING_MIN_SEQUENTIAL_RATIO
            and unique_ratio >= STREAMING_MIN_UNIQUE_RATIO
        ):
            return AccessPattern.STREAMING

        if unique_ratio <= REUSED_MAX_UNIQUE_RATIO:
            return AccessPattern.REUSED

        return AccessPattern.RANDOM
