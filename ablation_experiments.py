
"""
DynaMem: Ablation experiments for adaptive placement features.

Adaptive variants:
- ADAPTIVE_FULL
- ADAPTIVE_NO_PATTERN
- ADAPTIVE_NO_QUEUE
- ADAPTIVE_NEITHER

The runner also includes CPU-only and fixed-placement baselines.

Outputs:
- ablation_raw_results.csv
- ablation_per_seed.csv
- ablation_aggregate.csv

All measurements are produced by the configured simulation model.
"""

from dataclasses import replace
from pathlib import Path

from config import SimulationConfig, make_default_config
from experiments import (
    DEFAULT_OPERATIONS,
    DEFAULT_POLICIES,
    DEFAULT_WORKLOADS,
    aggregate_comparisons,
    write_csv,
)
from metrics import compare_results, summarize_result
from simulator import DynaMemSimulator
from workload_generator import WorkloadGenerator


ADAPTIVE_VARIANTS = {
    "ADAPTIVE_FULL": {
        "adaptive_pattern_aware": True,
        "adaptive_queue_aware": True,
    },
    "ADAPTIVE_NO_PATTERN": {
        "adaptive_pattern_aware": False,
        "adaptive_queue_aware": True,
    },
    "ADAPTIVE_NO_QUEUE": {
        "adaptive_pattern_aware": True,
        "adaptive_queue_aware": False,
    },
    "ADAPTIVE_NEITHER": {
        "adaptive_pattern_aware": False,
        "adaptive_queue_aware": False,
    },
}

FIXED_BASELINES = (
    "CPU_ONLY",
    "STATIC_CIM_A",
    "STATIC_CIM_P",
    "STATIC_NMP",
    "FIXED_HYBRID",
)

DEFAULT_ABLATION_VARIANTS = (
    "ADAPTIVE_FULL",
    "ADAPTIVE_NO_PATTERN",
    "ADAPTIVE_NO_QUEUE",
    "ADAPTIVE_NEITHER",
    *FIXED_BASELINES,
)


class AblationExperimentRunner:
    """Run controlled comparisons of DynaMem's adaptive features."""

    def __init__(
        self,
        config: SimulationConfig | None = None,
        output_dir: str | Path = "results/ablation",
        variants: tuple[str, ...] = DEFAULT_ABLATION_VARIANTS,
    ):
        self.config = (
            make_default_config()
            if config is None
            else config
        )

        self.output_dir = Path(output_dir)

        self.variants = tuple(
            variant.upper() for variant in variants
        )

        if not self.variants:
            raise ValueError("At least one variant must be supplied.")

        if len(set(self.variants)) != len(self.variants):
            raise ValueError("Ablation variants must be unique.")

        allowed = (
            set(ADAPTIVE_VARIANTS)
            | set(FIXED_BASELINES)
        )

        unknown = set(self.variants) - allowed

        if unknown:
            raise ValueError(
                f"Unknown ablation variants: {sorted(unknown)}"
            )

        if "CPU_ONLY" not in self.variants:
            raise ValueError(
                "CPU_ONLY is required as the comparison reference."
            )

    def _create_trace(
        self,
        workload_type: str,
        seed: int,
        num_requests: int,
        num_blocks: int,
        block_size_bytes: int,
        blocks_per_request: int,
        operations: tuple[str, ...],
        hot_fraction: float,
        reuse_probability: float,
        arrival_interval_ns: float,
    ):
        """Generate one trace that every variant will share."""

        generator = WorkloadGenerator(
            self.config,
            seed=seed,
        )

        common = {
            "num_requests": num_requests,
            "num_blocks": num_blocks,
            "block_size_bytes": block_size_bytes,
            "blocks_per_request": blocks_per_request,
            "arrival_interval_ns": arrival_interval_ns,
            "operations": operations,
            "dataset_id": f"{workload_type}_seed_{seed}",
        }

        if workload_type == "reused":
            return generator.generate_reused(
                **common,
                hot_fraction=hot_fraction,
                reuse_probability=reuse_probability,
            )

        if workload_type == "streaming":
            return generator.generate_streaming(**common)

        if workload_type == "random":
            return generator.generate_random(**common)

        if workload_type == "phase_changing":
            return generator.generate_phase_changing(
                **common,
                hot_fraction=hot_fraction,
                reuse_probability=reuse_probability,
            )

        raise ValueError(
            f"Unknown workload type: {workload_type}"
        )

    def _run_variant(
        self,
        trace,
        variant_name: str,
        hybrid_size_threshold_bytes: int,
    ):
        """Run one policy or adaptive ablation in a fresh simulator."""

        if variant_name in ADAPTIVE_VARIANTS:
            settings = ADAPTIVE_VARIANTS[variant_name]

            simulator = DynaMemSimulator(
                config=self.config,
                trace=trace,
                policy_name="ADAPTIVE",
                hybrid_size_threshold_bytes=(
                    hybrid_size_threshold_bytes
                ),
                adaptive_pattern_aware=(
                    settings["adaptive_pattern_aware"]
                ),
                adaptive_queue_aware=(
                    settings["adaptive_queue_aware"]
                ),
            )

            result = simulator.run()

            # The simulator executes the ADAPTIVE policy in every
            # variant. Rename the result for fair variant comparison.
            return replace(
                result,
                policy_name=variant_name,
            )

        # Fixed baseline runs have their own policy names and overheads.
        simulator = DynaMemSimulator(
            config=self.config,
            trace=trace,
            policy_name=variant_name,
            hybrid_size_threshold_bytes=(
                hybrid_size_threshold_bytes
            ),
        )

        return simulator.run()

    @staticmethod
    def _variant_metadata(variant_name: str) -> dict:
        """Describe which features are enabled for a variant."""

        if variant_name in ADAPTIVE_VARIANTS:
            settings = ADAPTIVE_VARIANTS[variant_name]

            return {
                "adaptive_pattern_aware": (
                    settings["adaptive_pattern_aware"]
                ),
                "adaptive_queue_aware": (
                    settings["adaptive_queue_aware"]
                ),
            }

        return {
            "adaptive_pattern_aware": None,
            "adaptive_queue_aware": None,
        }

    def run(
        self,
        seeds: tuple[int, ...] = (42, 43, 44),
        workload_types: tuple[str, ...] = DEFAULT_WORKLOADS,
        num_requests: int = 90,
        num_blocks: int = 2048,
        block_size_bytes: int = 4096,
        blocks_per_request: int = 4,
        operations: tuple[str, ...] = DEFAULT_OPERATIONS,
        hot_fraction: float = 0.10,
        reuse_probability: float = 0.90,
        arrival_interval_ns: float = 10.0,
        hybrid_size_threshold_bytes: int = 256 * 1024,
        reference_policy: str = "CPU_ONLY",
    ) -> dict:
        """
        Run ablation variants on identical workload traces.

        Returns raw runs, per-seed comparisons, aggregate statistics,
        and the locations of the generated CSV files.
        """

        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError(
                "Seeds must be non-empty and unique."
            )

        if any(
            not isinstance(seed, int) or isinstance(seed, bool)
            for seed in seeds
        ):
            raise ValueError("Every seed must be an integer.")

        if not workload_types:
            raise ValueError("At least one workload is required.")

        if len(set(workload_types)) != len(workload_types):
            raise ValueError("Workload types must be unique.")

        unknown_workloads = (
            set(workload_types) - set(DEFAULT_WORKLOADS)
        )

        if unknown_workloads:
            raise ValueError(
                f"Unknown workloads: {sorted(unknown_workloads)}"
            )

        if reference_policy not in self.variants:
            raise ValueError(
                "The comparison reference must be included "
                "in the ablation variants."
            )

        if (
            num_requests <= 0
            or num_blocks <= 0
            or block_size_bytes <= 0
            or blocks_per_request <= 0
        ):
            raise ValueError(
                "Request count, block count, block size and "
                "blocks per request must be positive."
            )

        if blocks_per_request > num_blocks:
            raise ValueError(
                "Blocks per request cannot exceed block count."
            )

        if "streaming" in workload_types:
            if num_requests * blocks_per_request > num_blocks:
                raise ValueError(
                    "Streaming requires at least "
                    "num_requests * blocks_per_request blocks."
                )

        if "phase_changing" in workload_types:
            required_streaming_blocks = (
                (num_requests // 3) * blocks_per_request
            )

            if required_streaming_blocks > num_blocks:
                raise ValueError(
                    "Insufficient blocks for the streaming phase."
                )

        if not operations:
            raise ValueError("At least one operation is required.")

        if not 0 < hot_fraction <= 1:
            raise ValueError("Hot fraction must be in (0, 1].")

        if not 0 <= reuse_probability <= 1:
            raise ValueError(
                "Reuse probability must be between 0 and 1."
            )

        if arrival_interval_ns < 0:
            raise ValueError(
                "Arrival interval cannot be negative."
            )

        raw_rows = []
        comparison_rows = []

        for seed in seeds:
            for workload_type in workload_types:
                print(
                    f"\nWorkload={workload_type}, seed={seed}"
                )

                trace = self._create_trace(
                    workload_type=workload_type,
                    seed=seed,
                    num_requests=num_requests,
                    num_blocks=num_blocks,
                    block_size_bytes=block_size_bytes,
                    blocks_per_request=blocks_per_request,
                    operations=operations,
                    hot_fraction=hot_fraction,
                    reuse_probability=reuse_probability,
                    arrival_interval_ns=arrival_interval_ns,
                )

                results_by_variant = {}

                # Every variant receives the same trace but a new
                # simulator instance with fresh cache and runtime state.
                for variant_name in self.variants:
                    print(f"  Running {variant_name}")

                    result = self._run_variant(
                        trace,
                        variant_name,
                        hybrid_size_threshold_bytes,
                    )

                    results_by_variant[variant_name] = result

                    row = summarize_result(result)

                    row.update({
                        "workload_type": workload_type,
                        "seed": seed,
                        "num_requests": num_requests,
                        "num_blocks": num_blocks,
                        "block_size_bytes": block_size_bytes,
                        "blocks_per_request": blocks_per_request,
                        "latency_weight": (
                            self.config.runtime.latency_weight
                        ),
                        **self._variant_metadata(variant_name),
                    })

                    raw_rows.append(row)

                # Compare all results against the same CPU-only reference.
                current_comparisons = compare_results(
                    results_by_variant,
                    reference_policy=reference_policy,
                )

                for row in current_comparisons:
                    variant_name = row["policy_name"]

                    row.update({
                        "workload_type": workload_type,
                        "seed": seed,
                        "num_requests": num_requests,
                        "num_blocks": num_blocks,
                        "block_size_bytes": block_size_bytes,
                        "blocks_per_request": blocks_per_request,
                        "latency_weight": (
                            self.config.runtime.latency_weight
                        ),
                        **self._variant_metadata(variant_name),
                    })

                    comparison_rows.append(row)

        aggregate_rows = aggregate_comparisons(
            comparison_rows
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        paths = {
            "raw_results": write_csv(
                raw_rows,
                self.output_dir / "ablation_raw_results.csv",
            ),
            "per_seed_comparisons": write_csv(
                comparison_rows,
                self.output_dir / "ablation_per_seed.csv",
            ),
            "aggregate_comparisons": write_csv(
                aggregate_rows,
                self.output_dir / "ablation_aggregate.csv",
            ),
        }

        print("\nAblation experiment complete.")

        for label, path in paths.items():
            print(f"{label}: {path.resolve()}")

        return {
            "raw_rows": raw_rows,
            "comparison_rows": comparison_rows,
            "aggregate_rows": aggregate_rows,
            "paths": paths,
        }


if __name__ == "__main__":
    runner = AblationExperimentRunner()

    results = runner.run(
        seeds=(42, 43, 44),
        workload_types=DEFAULT_WORKLOADS,
        num_requests=90,
        num_blocks=2048,
        block_size_bytes=4096,
        blocks_per_request=4,
    )

    print("\nAblation summary")
    print("=" * 90)

    for row in results["aggregate_rows"]:
        print(
            f"{row['workload_type']:16} "
            f"{row['policy_name']:22} "
            f"seeds={row['seed_count']:2} "
            f"speedup={row['mean_speedup_vs_reference']} "
            f"energy_reduction="
            f"{row['mean_energy_reduction_pct']}% "
            f"movement_reduction="
            f"{row['mean_data_movement_reduction_pct']}%"
        )
