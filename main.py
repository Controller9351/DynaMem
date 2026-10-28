"""
DynaMem: Main command-line entry point.

Modes:
    suite         Run standard workload and policy comparisons.
    size-sweep    Evaluate multiple dataset sizes.
    weight-sweep  Evaluate latency-energy objective weights.

All performance and energy values are produced by the simulation model.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

from config import make_default_config
from experiments import (
    DEFAULT_OPERATIONS,
    DEFAULT_POLICIES,
    DEFAULT_WORKLOADS,
    ExperimentRunner,
)
from plots import (
    generate_standard_plots,
    plot_latency_weight_sweep,
    plot_size_sweep,
)


def positive_integer(value: str) -> int:
    """Parse an integer that must be greater than zero."""

    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a valid integer."
        ) from exc

    if number <= 0:
        raise argparse.ArgumentTypeError(
            "Value must be greater than zero."
        )

    return number


def parse_integer_list(value: str) -> tuple[int, ...]:
    """Parse comma-separated positive integers."""

    try:
        numbers = tuple(
            int(item.strip())
            for item in value.split(",")
            if item.strip()
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Provide comma-separated integers, e.g. 42,43,44."
        ) from exc

    if not numbers:
        raise argparse.ArgumentTypeError(
            "At least one integer is required."
        )

    if any(number <= 0 for number in numbers):
        raise argparse.ArgumentTypeError(
            "All integers must be positive."
        )

    if len(numbers) != len(set(numbers)):
        raise argparse.ArgumentTypeError(
            "Duplicate values are not allowed."
        )

    return numbers


def parse_float_list(value: str) -> tuple[float, ...]:
    """Parse comma-separated latency weights from 0 to 1."""

    try:
        numbers = tuple(
            float(item.strip())
            for item in value.split(",")
            if item.strip()
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Provide comma-separated numbers, e.g. 0,0.5,1."
        ) from exc

    if not numbers:
        raise argparse.ArgumentTypeError(
            "At least one weight is required."
        )

    if any(not 0.0 <= number <= 1.0 for number in numbers):
        raise argparse.ArgumentTypeError(
            "Each latency weight must be between 0 and 1."
        )

    if len(numbers) != len(set(numbers)):
        raise argparse.ArgumentTypeError(
            "Duplicate weights are not allowed."
        )

    return numbers


def parse_string_list(value: str) -> tuple[str, ...]:
    """Parse comma-separated strings."""

    values = tuple(
        item.strip()
        for item in value.split(",")
        if item.strip()
    )

    if not values:
        raise argparse.ArgumentTypeError(
            "At least one value is required."
        )

    return values


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""

    parser = argparse.ArgumentParser(
        description="Run DynaMem memory-centric computing simulations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        choices=("suite", "size-sweep", "weight-sweep"),
        default="suite",
        help="Experiment type to execute.",
    )

    parser.add_argument(
        "--seeds",
        type=parse_integer_list,
        default=(42, 43, 44),
        help="Independent random seeds, comma-separated.",
    )

    parser.add_argument(
        "--workloads",
        type=parse_string_list,
        default=DEFAULT_WORKLOADS,
        help=(
            "Workloads: reused, streaming, random, phase_changing."
        ),
    )

    parser.add_argument(
        "--policies",
        type=parse_string_list,
        default=DEFAULT_POLICIES,
        help=(
            "Policies: ADAPTIVE, CPU_ONLY, STATIC_CIM_A, "
            "STATIC_CIM_P, STATIC_NMP, FIXED_HYBRID."
        ),
    )

    parser.add_argument(
        "--requests",
        type=positive_integer,
        default=90,
        help="Number of requests in each workload.",
    )

    parser.add_argument(
        "--blocks",
        type=positive_integer,
        default=2048,
        help="Number of memory data blocks for suite/weight-sweep mode.",
    )

    parser.add_argument(
        "--block-size",
        type=positive_integer,
        default=4096,
        help="Size of each data block in bytes.",
    )

    parser.add_argument(
        "--blocks-per-request",
        type=positive_integer,
        default=4,
        help=(
            "Legacy fixed input-block count and fallback for omitted bounds."
        ),
    )

    parser.add_argument(
        "--min-blocks-per-request",
        type=positive_integer,
        default=None,
        help=(
            "Minimum input blocks per request for mixed-size workloads. "
            "Defaults to --blocks-per-request when omitted."
        ),
    )

    parser.add_argument(
        "--max-blocks-per-request",
        type=positive_integer,
        default=None,
        help=(
            "Maximum input blocks per request for mixed-size workloads. "
            "Defaults to --blocks-per-request when omitted."
        ),
    )

    parser.add_argument(
        "--operations",
        type=parse_string_list,
        default=DEFAULT_OPERATIONS,
        help="Supported operation types, comma-separated.",
    )

    parser.add_argument(
        "--sizes",
        type=parse_integer_list,
        default=(512, 1024, 2048, 4096),
        help="Dataset sizes in blocks for size-sweep mode.",
    )

    parser.add_argument(
        "--weights",
        type=parse_float_list,
        default=(0.0, 0.25, 0.5, 0.75, 1.0),
        help="Latency weights for weight-sweep mode.",
    )

    parser.add_argument(
        "--latency-weight",
        type=float,
        default=None,
        help=(
            "Override the latency weight for suite and starting configuration."
        ),
    )

    parser.add_argument(
        "--hybrid-threshold",
        type=positive_integer,
        default=256 * 1024,
        help="Fixed-hybrid input-size threshold in bytes.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for CSV files and graphs.",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip graph generation.",
    )

    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command-line arguments and workload capacity."""

    workloads = tuple(
        item.strip().lower().replace("-", "_")
        for item in args.workloads
    )

    policies = tuple(
        item.strip().upper()
        for item in args.policies
    )

    operations = tuple(
        item.strip().upper()
        for item in args.operations
    )

    if not workloads:
        raise ValueError("At least one workload is required.")

    if not policies:
        raise ValueError("At least one policy is required.")

    if len(policies) < 2:
        raise ValueError(
            "At least two policies are required for comparison."
        )

    if len(set(policies)) != len(policies):
        raise ValueError("Duplicate policies are not allowed.")

    if len(set(workloads)) != len(workloads):
        raise ValueError("Duplicate workloads are not allowed.")

    if len(set(operations)) != len(operations):
        raise ValueError("Duplicate operations are not allowed.")

    known_workloads = set(DEFAULT_WORKLOADS)
    unknown_workloads = set(workloads) - known_workloads

    if unknown_workloads:
        raise ValueError(
            f"Unknown workloads: {sorted(unknown_workloads)}"
        )

    known_policies = set(DEFAULT_POLICIES)
    unknown_policies = set(policies) - known_policies

    if unknown_policies:
        raise ValueError(
            f"Unknown policies: {sorted(unknown_policies)}"
        )

    if "CPU_ONLY" not in policies:
        raise ValueError(
            "CPU_ONLY must be included as a reference policy."
        )

    known_operations = set(DEFAULT_OPERATIONS)
    unknown_operations = set(operations) - known_operations

    if unknown_operations:
        raise ValueError(
            f"Unknown operations: {sorted(unknown_operations)}"
        )

    if args.latency_weight is not None:
        if not 0.0 <= args.latency_weight <= 1.0:
            raise ValueError(
                "--latency-weight must be between 0 and 1."
            )

    # Preserve the legacy behavior when mixed-size bounds are omitted.
    minimum = (
        args.blocks_per_request
        if args.min_blocks_per_request is None
        else args.min_blocks_per_request
    )

    maximum = (
        args.blocks_per_request
        if args.max_blocks_per_request is None
        else args.max_blocks_per_request
    )

    if minimum > maximum:
        raise ValueError(
            "--min-blocks-per-request cannot exceed "
            "--max-blocks-per-request."
        )

    # A dataset must at least be large enough to support one request.
    required_blocks = maximum

    # Streaming requests use fresh blocks.
    if "streaming" in workloads:
        required_blocks = max(
            required_blocks,
            args.requests * maximum,
        )

    # The first phase of phase_changing is streaming.
    if "phase_changing" in workloads:
        streaming_requests = math.ceil(args.requests / 3)

        required_blocks = max(
            required_blocks,
            streaming_requests * maximum,
        )

    if args.mode == "size-sweep":
        too_small = [
            size
            for size in args.sizes
            if size < required_blocks
        ]

        if too_small:
            raise ValueError(
                "Dataset sizes are too small for the selected workloads "
                "and request-size range. "
                f"Need at least {required_blocks} blocks; "
                f"invalid sizes: {too_small}."
            )

    elif args.blocks < required_blocks:
        raise ValueError(
            f"The selected workloads and request-size range require at "
            f"least {required_blocks} blocks; --blocks is {args.blocks}."
        )


def print_aggregate_summary(rows: list[dict[str, Any]]) -> None:
    """Display aggregate metrics using current or legacy field names."""

    print("\nAggregate comparison")
    print("=" * 108)

    print(
        f"{'Workload':20} "
        f"{'Policy':16} "
        f"{'Seeds':>6} "
        f"{'Mean speedup':>14} "
        f"{'Energy red. %':>15} "
        f"{'Movement red. %':>17}"
    )

    print("-" * 108)

    def display(value: Any, digits: int = 3) -> str:
        if value is None or value == "":
            return "N/A"

        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return "N/A"

    for row in rows:
        workload = row.get(
            "workload_type",
            row.get("workload", "N/A"),
        )

        policy = row.get(
            "policy_name",
            row.get("policy", "N/A"),
        )

        speedup = row.get(
            "mean_speedup_vs_reference",
            row.get("speedup_mean"),
        )

        energy = row.get(
            "mean_energy_reduction_pct",
            row.get("energy_reduction_pct_mean"),
        )

        movement = row.get(
            "mean_data_movement_reduction_pct",
            row.get(
                "data_movement_reduction_pct_mean",
                row.get("movement_reduction_pct_mean"),
            ),
        )

        seed_count = row.get("seed_count", "N/A")

        print(
            f"{str(workload):20} "
            f"{str(policy):16} "
            f"{str(seed_count):>6} "
            f"{display(speedup):>14} "
            f"{display(energy):>15} "
            f"{display(movement):>17}"
        )


def prepare_plot_csv(
    source_path: str | Path,
    destination_path: str | Path,
    aggregate: bool,
) -> Path:
    """Add legacy column aliases to a copy of a CSV for plotting compatibility.

    The original experimental CSV is not modified.
    """

    source = Path(source_path)
    destination = Path(destination_path)

    destination.parent.mkdir(parents=True, exist_ok=True)

    with source.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    aliases = {
        "workload_type": "workload",
        "policy_name": "policy",
    }

    if aggregate:
        aliases.update(
            {
                "mean_speedup_vs_reference": "speedup_mean",
                "mean_latency_reduction_pct": (
                    "latency_reduction_pct_mean"
                ),
                "mean_energy_reduction_pct": (
                    "energy_reduction_pct_mean"
                ),
                "mean_data_movement_reduction_pct": (
                    "data_movement_reduction_pct_mean"
                ),
                "mean_movement_reduction_pct": (
                    "movement_reduction_pct_mean"
                ),
                "mean_latency_ns": "mean_latency_ns_mean",
                "mean_p95_latency_ns": "p95_latency_ns_mean",
                "dataset_size": "num_blocks",
                "dataset_size_blocks": "num_blocks",
            }
        )
    else:
        aliases.update(
            {
                "dataset_size": "num_blocks",
                "dataset_size_blocks": "num_blocks",
            }
        )

    for alias in aliases:
        if alias not in fieldnames:
            fieldnames.append(alias)

    for row in rows:
        for alias, original in aliases.items():
            if not row.get(alias) and row.get(original) is not None:
                row[alias] = row[original]

        # Support the alternate movement-metric name if necessary.
        if aggregate and not row.get(
            "mean_data_movement_reduction_pct"
        ):
            value = row.get("movement_reduction_pct_mean")

            if value is not None:
                row["mean_data_movement_reduction_pct"] = value

    with destination.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)

    return destination


def main() -> int:
    """Parse arguments, run experiments, save results and optionally plot."""

    parser = build_parser()
    args = parser.parse_args()

    try:
        validate_arguments(args)

        config = make_default_config()

        if args.latency_weight is not None:
            config = replace(
                config,
                runtime=replace(
                    config.runtime,
                    latency_weight=args.latency_weight,
                ),
            )

        # Configure the runner through the current ExperimentRunner interface.
        runner = ExperimentRunner(
            config=config,
            output_dir=args.output_dir,
            seeds=args.seeds,
            num_requests=args.requests,
            num_blocks=args.blocks,
            block_size_bytes=args.block_size,
            blocks_per_request=args.blocks_per_request,
            operations=tuple(
                operation.upper()
                for operation in args.operations
            ),
            policies=tuple(
                policy.upper()
                for policy in args.policies
            ),
            workloads=tuple(
                workload.strip().lower().replace("-", "_")
                for workload in args.workloads
            ),
            hybrid_size_threshold_bytes=args.hybrid_threshold,
            min_blocks_per_request=args.min_blocks_per_request,
            max_blocks_per_request=args.max_blocks_per_request,
        )

        if args.mode == "suite":
            results = runner.run_suite()

            print_aggregate_summary(
                results["aggregate_rows"]
            )

            if not args.no_plots:
                plot_dir = args.output_dir / "plot_compatibility"

                aggregate_csv = prepare_plot_csv(
                    results["paths"]["aggregate_comparisons"],
                    plot_dir / "suite_aggregate.csv",
                    aggregate=True,
                )

                raw_csv = prepare_plot_csv(
                    results["paths"]["raw_results"],
                    plot_dir / "suite_raw.csv",
                    aggregate=False,
                )

                plot_paths = generate_standard_plots(
                    aggregate_csv=aggregate_csv,
                    raw_csv=raw_csv,
                    output_dir=args.output_dir / "figures",
                )

                print("\nGenerated figures:")

                for path in plot_paths:
                    print(Path(path).resolve())

        elif args.mode == "size-sweep":
            results = runner.run_size_sweep(
                block_counts=args.sizes,
                output_dir=args.output_dir / "size_sweep",
            )

            print_aggregate_summary(
                results["aggregate_rows"]
            )

            if not args.no_plots:
                aggregate_csv = prepare_plot_csv(
                    results["paths"]["aggregate_comparisons"],
                    args.output_dir
                    / "plot_compatibility"
                    / "size_aggregate.csv",
                    aggregate=True,
                )

                plot_paths = plot_size_sweep(
                    aggregate_csv=aggregate_csv,
                    output_dir=(
                        args.output_dir
                        / "figures"
                        / "size_sweep"
                    ),
                )

                print("\nGenerated figures:")

                for path in plot_paths:
                    print(Path(path).resolve())

        else:
            results = runner.run_weight_sweep(
                latency_weights=args.weights,
                output_dir=args.output_dir / "weight_sweep",
            )

            print_aggregate_summary(
                results["aggregate_rows"]
            )

            if not args.no_plots:
                aggregate_csv = prepare_plot_csv(
                    results["paths"]["aggregate_comparisons"],
                    args.output_dir
                    / "plot_compatibility"
                    / "weight_aggregate.csv",
                    aggregate=True,
                )

                plot_paths = plot_latency_weight_sweep(
                    aggregate_csv=aggregate_csv,
                    output_dir=(
                        args.output_dir
                        / "figures"
                        / "weight_sweep"
                    ),
                )

                print("\nGenerated figures:")

                for path in plot_paths:
                    print(Path(path).resolve())

        print("\nResult files:")

        for label, path in results["paths"].items():
            print(f"  {label}: {Path(path).resolve()}")

        print("\nDynaMem experiment completed successfully.")
        return 0

    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    return 1


if __name__ == "__main__":
    raise SystemExit(main())