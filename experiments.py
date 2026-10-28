"""Experiment orchestration for DynaMem.

Every policy receives the same deterministic workload trace for a given
workload and seed. Each policy runs with a fresh simulator instance.
"""

from __future__ import annotations

import ast
import csv
import inspect
import json
import math
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Mapping, Sequence

from config import make_default_config
from metrics import summarize_result
from simulator import DynaMemSimulator
from workload_generator import DEFAULT_OPERATIONS, WorkloadGenerator


DEFAULT_POLICIES = (
    "ADAPTIVE",
    "CPU_ONLY",
    "STATIC_CIM_A",
    "STATIC_CIM_P",
    "STATIC_NMP",
    "FIXED_HYBRID",
)

DEFAULT_WORKLOADS = (
    "reused",
    "streaming",
    "random",
    "phase_changing",
)

DEFAULT_SEEDS = (42, 43, 44)


class ExperimentRunner:
    """Run policy comparisons, dataset-size sweeps, and weight sweeps."""

    def __init__(
        self,
        config=None,
        output_dir: str | Path = "results",
        seeds: Sequence[int] = DEFAULT_SEEDS,
        num_requests: int = 90,
        num_blocks: int = 2048,
        block_size_bytes: int = 4096,
        blocks_per_request: int = 4,
        operations: Sequence[str] | None = None,
        policies: Sequence[str] = DEFAULT_POLICIES,
        workloads: Sequence[str] = DEFAULT_WORKLOADS,
        hybrid_size_threshold_bytes: int = 256 * 1024,
        min_blocks_per_request: int | None = None,
        max_blocks_per_request: int | None = None,
        workload_types: Sequence[str] | None = None,
    ) -> None:
        self.config = config or make_default_config()
        self.output_dir = Path(output_dir)

        self.seeds = tuple(int(seed) for seed in seeds)
        self.num_requests = int(num_requests)
        self.num_blocks = int(num_blocks)
        self.block_size_bytes = int(block_size_bytes)
        self.blocks_per_request = int(blocks_per_request)

        self.operations = (
            tuple(operations)
            if operations is not None
            else tuple(DEFAULT_OPERATIONS)
        )

        self.policies = tuple(
            str(policy).strip().upper()
            for policy in policies
        )

        # Backward-compatible alias used by the command-line interface.
        selected_workloads = (
            workloads if workload_types is None else workload_types
        )

        self.workloads = tuple(
            str(workload).strip().lower().replace("-", "_")
            for workload in selected_workloads
        )

        self.hybrid_size_threshold_bytes = int(
            hybrid_size_threshold_bytes
        )

        self.min_blocks_per_request = (
            None
            if min_blocks_per_request is None
            else int(min_blocks_per_request)
        )

        self.max_blocks_per_request = (
            None
            if max_blocks_per_request is None
            else int(max_blocks_per_request)
        )

        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """Validate runner settings before experiments start."""

        if not self.seeds:
            raise ValueError("At least one random seed is required.")

        if not self.policies:
            raise ValueError("At least one policy is required.")

        if not self.workloads:
            raise ValueError("At least one workload is required.")

        if not self.operations:
            raise ValueError("operations cannot be empty.")

        if self.num_requests < 1:
            raise ValueError("num_requests must be at least 1.")

        if self.num_blocks < 1:
            raise ValueError("num_blocks must be at least 1.")

        if self.block_size_bytes < 1:
            raise ValueError("block_size_bytes must be at least 1.")

        if self.blocks_per_request < 1:
            raise ValueError("blocks_per_request must be at least 1.")

        if self.hybrid_size_threshold_bytes < 0:
            raise ValueError(
                "hybrid_size_threshold_bytes cannot be negative."
            )

        minimum, maximum = self._resolved_request_block_range()

        if minimum < 1 or maximum < 1:
            raise ValueError(
                "Request block-count bounds must be at least 1."
            )

        if minimum > maximum:
            raise ValueError(
                "Minimum request block count cannot exceed maximum."
            )

        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Duplicate seeds are not allowed.")

        if len(set(self.policies)) != len(self.policies):
            raise ValueError("Duplicate policies are not allowed.")

        if len(set(self.workloads)) != len(self.workloads):
            raise ValueError("Duplicate workloads are not allowed.")

    def _resolved_request_block_range(self) -> tuple[int, int]:
        """Resolve optional bounds against the legacy fixed-size setting."""

        minimum = (
            self.blocks_per_request
            if self.min_blocks_per_request is None
            else self.min_blocks_per_request
        )

        maximum = (
            self.blocks_per_request
            if self.max_blocks_per_request is None
            else self.max_blocks_per_request
        )

        return minimum, maximum

    @staticmethod
    def _call_with_supported_kwargs(
        callable_obj,
        kwargs: Mapping[str, Any],
    ):
        """Pass only keyword arguments accepted by a callable."""

        try:
            parameters = inspect.signature(callable_obj).parameters
        except (TypeError, ValueError):
            return callable_obj(**dict(kwargs))

        accepts_arbitrary = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )

        if accepts_arbitrary:
            return callable_obj(**dict(kwargs))

        accepted = {
            key: value
            for key, value in kwargs.items()
            if key in parameters
        }

        return callable_obj(**accepted)

    def _validate_capacity(
        self,
        workload_name: str,
        num_blocks: int,
        num_requests: int,
    ) -> None:
        """Validate block capacity needed by streaming portions of a trace."""

        _, maximum = self._resolved_request_block_range()

        canonical = (
            workload_name.strip()
            .lower()
            .replace("-", "_")
        )

        if canonical == "random_access":
            canonical = "random"

        if canonical == "streaming":
            required = num_requests * maximum

            if num_blocks < required:
                raise ValueError(
                    f"Streaming workload needs at least {required} "
                    f"blocks for {num_requests} requests with up to "
                    f"{maximum} blocks per request; received {num_blocks}."
                )

        elif canonical == "phase_changing":
            streaming_requests = math.ceil(num_requests / 3)
            required = streaming_requests * maximum

            if num_blocks < required:
                raise ValueError(
                    f"Phase-changing workload needs at least {required} "
                    f"blocks for its initial streaming phase; "
                    f"received {num_blocks}."
                )

    def _create_trace(
        self,
        workload_name: str,
        seed: int,
        num_blocks: int | None = None,
        num_requests: int | None = None,
        block_size_bytes: int | None = None,
        config=None,
    ):
        """Create one deterministic trace with optional mixed-size requests."""

        active_config = config or self.config

        active_blocks = (
            self.num_blocks
            if num_blocks is None
            else int(num_blocks)
        )

        active_requests = (
            self.num_requests
            if num_requests is None
            else int(num_requests)
        )

        active_block_size = (
            self.block_size_bytes
            if block_size_bytes is None
            else int(block_size_bytes)
        )

        self._validate_capacity(
            workload_name,
            active_blocks,
            active_requests,
        )

        # The WorkloadGenerator in this project requires config.
        generator_kwargs = {
            "config": active_config,
            "seed": int(seed),
            "num_blocks": active_blocks,
            "block_size_bytes": active_block_size,
            "num_banks": active_config.memory.num_banks,
            "operations": self.operations,
        }

        generator = self._call_with_supported_kwargs(
            WorkloadGenerator,
            generator_kwargs,
        )

        canonical_name = (
            workload_name.strip()
            .lower()
            .replace("-", "_")
        )

        aliases = {
            "random_access": "random",
            "phasechanging": "phase_changing",
        }

        canonical_name = aliases.get(
            canonical_name,
            canonical_name,
        )

        method_name = f"generate_{canonical_name}"
        generate = getattr(generator, method_name, None)

        if generate is None:
            supported = ", ".join(DEFAULT_WORKLOADS)

            raise ValueError(
                f"Unknown workload '{workload_name}'. "
                f"Supported workloads: {supported}."
            )

        method_kwargs = {
            "num_requests": active_requests,
            "num_blocks": active_blocks,
            "block_size_bytes": active_block_size,
            "blocks_per_request": self.blocks_per_request,
            "min_blocks_per_request": self.min_blocks_per_request,
            "max_blocks_per_request": self.max_blocks_per_request,
            "operations": self.operations,
        }

        return self._call_with_supported_kwargs(
            generate,
            method_kwargs,
        )

    @staticmethod
    def _summary_dict(result) -> dict[str, Any]:
        """Convert simulation metrics to a dictionary."""

        summary = summarize_result(result)

        if isinstance(summary, Mapping):
            row = dict(summary)

        elif hasattr(result, "summary"):
            result_summary = result.summary()

            row = (
                dict(result_summary)
                if isinstance(result_summary, Mapping)
                else {}
            )

        else:
            row = {}

        records = getattr(result, "records", None)

        if records is None:
            records = getattr(result, "execution_records", None)

        records = list(records or [])

        def first_present(names: Sequence[str]):
            for name in names:
                if name in row and row[name] is not None:
                    return row[name]
            return None

        # Makespan.
        if first_present(
            ("makespan_ns", "simulation_makespan_ns")
        ) is None:
            direct = getattr(result, "makespan_ns", None)

            if direct is not None:
                row["makespan_ns"] = direct

            elif records:
                end_times = [
                    getattr(record, "end_time_ns", 0.0)
                    for record in records
                ]

                row["makespan_ns"] = max(end_times, default=0.0)

        # Per-request latency statistics.
        latencies = [
            float(record.total_latency_ns)
            for record in records
            if getattr(record, "total_latency_ns", None) is not None
        ]

        if latencies:
            if first_present(
                (
                    "mean_latency_ns",
                    "average_latency_ns",
                    "avg_latency_ns",
                    "latency_mean_ns",
                )
            ) is None:
                row["mean_latency_ns"] = mean(latencies)

            ordered = sorted(latencies)
            p95_index = max(
                0,
                math.ceil(0.95 * len(ordered)) - 1,
            )

            if first_present(
                ("p95_latency_ns", "latency_p95_ns", "p95_ns")
            ) is None:
                row["p95_latency_ns"] = ordered[p95_index]

        # Total estimated energy.
        if first_present(
            (
                "total_energy_pj",
                "estimated_energy_pj",
                "energy_pj",
            )
        ) is None and records:
            row["total_energy_pj"] = sum(
                float(
                    getattr(record, "energy_pj", 0.0) or 0.0
                )
                for record in records
            )

        # Total movement across the paths counted by the simulator.
        if first_present(
            (
                "total_moved_bytes",
                "total_bytes_moved",
                "bytes_moved",
            )
        ) is None and records:
            row["total_moved_bytes"] = sum(
                float(
                    getattr(record, "total_moved_bytes", 0.0) or 0.0
                )
                for record in records
            )

        return row

    @staticmethod
    def _number(
        row: Mapping[str, Any],
        names: Sequence[str],
    ) -> float | None:
        """Return the first available numeric metric."""

        for name in names:
            value = row.get(name)

            if value is None or isinstance(value, bool):
                continue

            try:
                return float(value)
            except (TypeError, ValueError):
                continue

        return None

    def _metadata(
        self,
        workload: str,
        policy: str,
        seed: int,
        num_blocks: int,
        num_requests: int,
        config,
    ) -> dict[str, Any]:
        """Build reproducibility metadata for an experiment row."""

        return {
            "workload": workload,
            "policy": policy,
            "seed": seed,
            "num_requests": num_requests,
            "num_blocks": num_blocks,
            "block_size_bytes": self.block_size_bytes,
            "blocks_per_request": self.blocks_per_request,
            "min_blocks_per_request": self.min_blocks_per_request,
            "max_blocks_per_request": self.max_blocks_per_request,
            "operations": ";".join(self.operations),
            "latency_weight": config.runtime.latency_weight,
            "hybrid_size_threshold_bytes": (
                self.hybrid_size_threshold_bytes
            ),
        }

    def _make_comparisons(
        self,
        workload: str,
        seed: int,
        num_blocks: int,
        num_requests: int,
        config,
        policy_summaries: Mapping[str, Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Compare each policy with CPU_ONLY when available."""

        if not policy_summaries:
            return []

        reference_policy = (
            "CPU_ONLY"
            if "CPU_ONLY" in policy_summaries
            else next(iter(policy_summaries))
        )

        reference = policy_summaries[reference_policy]

        reference_makespan = self._number(
            reference,
            ("makespan_ns", "simulation_makespan_ns"),
        )

        reference_latency = self._number(
            reference,
            (
                "mean_latency_ns",
                "average_latency_ns",
                "avg_latency_ns",
                "latency_mean_ns",
            ),
        )

        reference_p95 = self._number(
            reference,
            ("p95_latency_ns", "latency_p95_ns", "p95_ns"),
        )

        reference_energy = self._number(
            reference,
            (
                "total_energy_pj",
                "estimated_energy_pj",
                "energy_pj",
            ),
        )

        reference_movement = self._number(
            reference,
            (
                "total_moved_bytes",
                "total_bytes_moved",
                "bytes_moved",
            ),
        )

        comparisons = []

        for policy, summary in policy_summaries.items():
            makespan = self._number(
                summary,
                ("makespan_ns", "simulation_makespan_ns"),
            )

            latency = self._number(
                summary,
                (
                    "mean_latency_ns",
                    "average_latency_ns",
                    "avg_latency_ns",
                    "latency_mean_ns",
                ),
            )

            p95 = self._number(
                summary,
                ("p95_latency_ns", "latency_p95_ns", "p95_ns"),
            )

            energy = self._number(
                summary,
                (
                    "total_energy_pj",
                    "estimated_energy_pj",
                    "energy_pj",
                ),
            )

            movement = self._number(
                summary,
                (
                    "total_moved_bytes",
                    "total_bytes_moved",
                    "bytes_moved",
                ),
            )

            def reduction(
                reference_value: float | None,
                current_value: float | None,
            ):
                if (
                    reference_value is None
                    or current_value is None
                    or reference_value == 0
                ):
                    return None

                return (
                    (reference_value - current_value)
                    / reference_value
                    * 100.0
                )

            speedup = None

            if (
                reference_makespan is not None
                and makespan not in (None, 0)
            ):
                speedup = reference_makespan / makespan

            row = self._metadata(
                workload,
                policy,
                seed,
                num_blocks,
                num_requests,
                config,
            )

            row.update(
                {
                    "reference_policy": reference_policy,
                    "speedup": speedup,
                    "latency_reduction_pct": reduction(
                        reference_latency,
                        latency,
                    ),
                    "energy_reduction_pct": reduction(
                        reference_energy,
                        energy,
                    ),
                    "data_movement_reduction_pct": reduction(
                        reference_movement,
                        movement,
                    ),
                    "movement_reduction_pct": reduction(
                        reference_movement,
                        movement,
                    ),
                    "p95_latency_ns": p95,
                    "reference_p95_latency_ns": reference_p95,
                }
            )

            comparisons.append(row)

        return comparisons

    def _execute(
        self,
        output_dir: str | Path,
        num_blocks_values: Iterable[int],
        config=None,
    ) -> dict[str, Any]:
        """Run the requested experiments and write three CSV files."""

        active_config = config or self.config
        raw_rows: list[dict[str, Any]] = []
        comparison_rows: list[dict[str, Any]] = []

        for active_blocks in num_blocks_values:
            active_blocks = int(active_blocks)

            for workload in self.workloads:
                for seed in self.seeds:
                    # All policies receive the exact same trace for this
                    # workload, seed and dataset size.
                    trace = self._create_trace(
                        workload_name=workload,
                        seed=seed,
                        num_blocks=active_blocks,
                        num_requests=self.num_requests,
                        block_size_bytes=self.block_size_bytes,
                        config=active_config,
                    )

                    policy_summaries: dict[str, dict[str, Any]] = {}

                    for policy in self.policies:
                        simulator = DynaMemSimulator(
                            active_config,
                            trace,
                            policy_name=policy,
                            hybrid_size_threshold_bytes=(
                                self.hybrid_size_threshold_bytes
                            ),
                        )

                        result = simulator.run()
                        summary = self._summary_dict(result)

                        row = self._metadata(
                            workload,
                            policy,
                            seed,
                            active_blocks,
                            self.num_requests,
                            active_config,
                        )

                        row.update(summary)

                        # Keep run metadata consistent and explicit.
                        row.update(
                            {
                                "workload": workload,
                                "policy": policy,
                                "seed": seed,
                                "num_requests": self.num_requests,
                                "num_blocks": active_blocks,
                                "block_size_bytes": self.block_size_bytes,
                                "blocks_per_request": (
                                    self.blocks_per_request
                                ),
                                "min_blocks_per_request": (
                                    self.min_blocks_per_request
                                ),
                                "max_blocks_per_request": (
                                    self.max_blocks_per_request
                                ),
                                "operations": ";".join(self.operations),
                                "latency_weight": (
                                    active_config.runtime.latency_weight
                                ),
                                "hybrid_size_threshold_bytes": (
                                    self.hybrid_size_threshold_bytes
                                ),
                            }
                        )

                        raw_rows.append(row)
                        policy_summaries[policy] = summary

                    comparison_rows.extend(
                        self._make_comparisons(
                            workload,
                            seed,
                            active_blocks,
                            self.num_requests,
                            active_config,
                            policy_summaries,
                        )
                    )

        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)

        raw_path = destination / "raw_run_results.csv"
        per_seed_path = destination / "per_seed_comparisons.csv"
        aggregate_path = destination / "aggregate_comparisons.csv"

        aggregate_rows = self._aggregate_comparisons(
            comparison_rows
        )

        # Normalize any nested placement data to valid JSON while writing.
        self._write_csv(raw_path, raw_rows)
        self._write_csv(per_seed_path, comparison_rows)
        self._write_csv(aggregate_path, aggregate_rows)

        paths = {
            "raw_results": raw_path,
            "per_seed_comparisons": per_seed_path,
            "aggregate_comparisons": aggregate_path,
        }

        # Keep both the row data and nested path interface expected by
        # check_experiments.py and main.py.
        return {
            "raw_rows": raw_rows,
            "comparison_rows": comparison_rows,
            "per_seed_rows": comparison_rows,
            "aggregate_rows": aggregate_rows,
            "paths": paths,
            **paths,
        }

    @staticmethod
    def _write_csv(
        path: Path,
        rows: Sequence[Mapping[str, Any]],
    ) -> None:
        """Write CSV rows while preserving nested data as valid JSON.

        Dictionaries and lists are serialized using json.dumps. If a
        placement-related field contains a string representation of a
        Python dictionary, ast.literal_eval is used to parse it safely
        before converting it to valid JSON.
        """

        if not rows:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
            return

        fieldnames: list[str] = []
        seen: set[str] = set()

        for row in rows:
            for field in row:
                if field not in seen:
                    seen.add(field)
                    fieldnames.append(field)

        def serialize_value(field: str, value: Any) -> Any:
            """Convert nested or Python-style placement data to JSON."""

            if isinstance(value, (dict, list, tuple)):
                return json.dumps(
                    value,
                    ensure_ascii=False,
                    default=str,
                )

            field_name = field.lower()

            should_normalize_string = (
                "placement" in field_name
                or "distribution" in field_name
                or field_name.endswith("_json")
            )

            if not isinstance(value, str) or not should_normalize_string:
                return value

            stripped = value.strip()

            if not stripped:
                return value

            # First accept already-valid JSON.
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None

                # Then try a Python literal such as "{'CPU': 10}".
                # literal_eval does not execute arbitrary Python code.
                try:
                    parsed = ast.literal_eval(stripped)
                except (ValueError, SyntaxError, TypeError):
                    pass

            if isinstance(parsed, (dict, list, tuple)):
                return json.dumps(
                    parsed,
                    ensure_ascii=False,
                    default=str,
                )

            # Do not silently change an unrecognised value.
            return value

        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )

            writer.writeheader()

            for row in rows:
                serialized_row = {
                    field: serialize_value(field, value)
                    for field, value in row.items()
                }

                writer.writerow(serialized_row)

    @staticmethod
    def _aggregate_comparisons(
        rows: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Calculate means and standard deviations across random seeds."""

        if not rows:
            return []

        group_fields = (
            "workload",
            "policy",
            "num_requests",
            "num_blocks",
            "block_size_bytes",
            "blocks_per_request",
            "min_blocks_per_request",
            "max_blocks_per_request",
            "operations",
            "latency_weight",
            "hybrid_size_threshold_bytes",
            "reference_policy",
        )

        groups: dict[
            tuple[Any, ...],
            list[Mapping[str, Any]],
        ] = defaultdict(list)

        for row in rows:
            key = tuple(row.get(field) for field in group_fields)
            groups[key].append(row)

        output: list[dict[str, Any]] = []
        excluded = set(group_fields) | {"seed"}

        for key, group in groups.items():
            aggregate = dict(zip(group_fields, key))

            numeric_fields = sorted(
                {
                    field
                    for row in group
                    for field, value in row.items()
                    if (
                        field not in excluded
                        and isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and value is not None
                    )
                }
            )

            for field in numeric_fields:
                values = [
                    float(row[field])
                    for row in group
                    if (
                        isinstance(row.get(field), (int, float))
                        and not isinstance(row.get(field), bool)
                    )
                ]

                if values:
                    aggregate[f"{field}_mean"] = mean(values)
                    aggregate[f"{field}_std"] = (
                        stdev(values) if len(values) > 1 else 0.0
                    )

            aggregate["seed_count"] = len(group)
            output.append(aggregate)

        return output

    def run_suite(
        self,
        seeds: Sequence[int] | None = None,
        num_requests: int | None = None,
        num_blocks: int | None = None,
        block_size_bytes: int | None = None,
        blocks_per_request: int | None = None,
        operations: Sequence[str] | None = None,
        policies: Sequence[str] | None = None,
        workloads: Sequence[str] | None = None,
        output_dir: str | Path | None = None,
        min_blocks_per_request: int | None = None,
        max_blocks_per_request: int | None = None,
        hybrid_size_threshold_bytes: int | None = None,
        workload_types: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Run the standard suite with optional per-call overrides.

        workload_types is retained as a backward-compatible alias for
        workloads.
        """

        destination = (
            self.output_dir
            if output_dir is None
            else Path(output_dir)
        )

        if (
            workloads is not None
            and workload_types is not None
            and tuple(workloads) != tuple(workload_types)
        ):
            raise ValueError(
                "workloads and workload_types were both supplied "
                "with conflicting values."
            )

        selected_workloads = (
            workloads
            if workloads is not None
            else workload_types
        )

        overrides_requested = any(
            value is not None
            for value in (
                seeds,
                num_requests,
                num_blocks,
                block_size_bytes,
                blocks_per_request,
                operations,
                policies,
                workloads,
                workload_types,
                min_blocks_per_request,
                max_blocks_per_request,
                hybrid_size_threshold_bytes,
            )
        )

        if not overrides_requested:
            return self._execute(
                destination,
                [self.num_blocks],
                config=self.config,
            )

        active_runner = ExperimentRunner(
            config=self.config,
            output_dir=destination,
            seeds=self.seeds if seeds is None else seeds,
            num_requests=(
                self.num_requests
                if num_requests is None
                else num_requests
            ),
            num_blocks=(
                self.num_blocks
                if num_blocks is None
                else num_blocks
            ),
            block_size_bytes=(
                self.block_size_bytes
                if block_size_bytes is None
                else block_size_bytes
            ),
            blocks_per_request=(
                self.blocks_per_request
                if blocks_per_request is None
                else blocks_per_request
            ),
            operations=(
                self.operations
                if operations is None
                else operations
            ),
            policies=(
                self.policies
                if policies is None
                else policies
            ),
            workloads=(
                self.workloads
                if selected_workloads is None
                else selected_workloads
            ),
            hybrid_size_threshold_bytes=(
                self.hybrid_size_threshold_bytes
                if hybrid_size_threshold_bytes is None
                else hybrid_size_threshold_bytes
            ),
            min_blocks_per_request=(
                self.min_blocks_per_request
                if min_blocks_per_request is None
                else min_blocks_per_request
            ),
            max_blocks_per_request=(
                self.max_blocks_per_request
                if max_blocks_per_request is None
                else max_blocks_per_request
            ),
        )

        return active_runner._execute(
            destination,
            [active_runner.num_blocks],
            config=active_runner.config,
        )

    def run_size_sweep(
        self,
        block_counts: Sequence[int],
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Compare policies across different dataset sizes."""

        values = tuple(int(value) for value in block_counts)

        if not values:
            raise ValueError(
                "block_counts must contain at least one value."
            )

        if any(value < 1 for value in values):
            raise ValueError(
                "Every block count in a size sweep must be positive."
            )

        destination = (
            self.output_dir / "size_sweep"
            if output_dir is None
            else Path(output_dir)
        )

        return self._execute(
            destination,
            values,
            config=self.config,
        )

    def run_weight_sweep(
        self,
        latency_weights: Sequence[float],
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Compare execution policies across latency-energy weights."""

        weights = tuple(
            float(weight)
            for weight in latency_weights
        )

        if not weights:
            raise ValueError(
                "latency_weights must contain at least one value."
            )

        if any(not 0.0 <= weight <= 1.0 for weight in weights):
            raise ValueError(
                "Every latency weight must be between 0 and 1."
            )

        destination = (
            self.output_dir / "weight_sweep"
            if output_dir is None
            else Path(output_dir)
        )

        raw_rows = []
        comparison_rows = []

        for weight in weights:
            weighted_runtime = replace(
                self.config.runtime,
                latency_weight=weight,
            )

            weighted_config = replace(
                self.config,
                runtime=weighted_runtime,
            )

            run_dir = destination / f"weight_{weight:g}"

            result = self._execute(
                run_dir,
                [self.num_blocks],
                config=weighted_config,
            )

            raw_rows.extend(
                self._read_csv_rows(
                    result["paths"]["raw_results"]
                )
            )

            comparison_rows.extend(
                self._read_csv_rows(
                    result["paths"]["per_seed_comparisons"]
                )
            )

        destination.mkdir(parents=True, exist_ok=True)

        raw_path = destination / "raw_run_results.csv"
        per_seed_path = destination / "per_seed_comparisons.csv"
        aggregate_path = destination / "aggregate_comparisons.csv"

        aggregate_rows = self._aggregate_comparisons(
            comparison_rows
        )

        self._write_csv(raw_path, raw_rows)
        self._write_csv(per_seed_path, comparison_rows)
        self._write_csv(aggregate_path, aggregate_rows)

        paths = {
            "raw_results": raw_path,
            "per_seed_comparisons": per_seed_path,
            "aggregate_comparisons": aggregate_path,
        }

        return {
            "raw_rows": raw_rows,
            "comparison_rows": comparison_rows,
            "per_seed_rows": comparison_rows,
            "aggregate_rows": aggregate_rows,
            "paths": paths,
            **paths,
        }

    @staticmethod
    def _read_csv_rows(
        path: Path,
    ) -> list[dict[str, Any]]:
        """Read CSV output and convert numeric values back to numbers."""

        def parse_value(value: str) -> Any:
            if value == "":
                return None

            try:
                return int(value)
            except (TypeError, ValueError):
                pass

            try:
                return float(value)
            except (TypeError, ValueError):
                return value

        with path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as file:
            return [
                {
                    key: parse_value(value)
                    for key, value in row.items()
                }
                for row in csv.DictReader(file)
            ]