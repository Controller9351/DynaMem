
"""
DynaMem: Plotting and visualization utilities.

Reads the CSV files produced by experiments.py and creates:
- Speedup comparisons
- Energy-reduction comparisons
- Data-movement-reduction comparisons
- P95 request-latency comparisons
- Resource-placement distributions
- Dataset-size sweep plots
- Latency-energy weight sweep plots

Uses Matplotlib only; no Seaborn dependency is required.
"""

import argparse
import csv
import json
import math

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


POLICY_ORDER = [
    "CPU_ONLY",
    "ADAPTIVE",
    "STATIC_CIM_A",
    "STATIC_CIM_P",
    "STATIC_NMP",
    "FIXED_HYBRID",
]

WORKLOAD_ORDER = [
    "reused",
    "streaming",
    "random",
    "phase_changing",
]

RESOURCE_FAMILIES = [
    "CPU",
    "CIM_A",
    "CIM_P",
    "NMP",
]


def read_csv_rows(path: str | Path) -> list[dict]:
    """Read a CSV file and return its rows as dictionaries."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Results file not found: {path.resolve()}"
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(
            f"The results file is empty: {path.resolve()}"
        )

    return rows


def _number(value) -> float | None:
    """Convert a CSV value into a finite float, or None if unavailable."""

    if value is None or value == "":
        return None

    try:
        number = float(value)
    except (ValueError, TypeError):
        return None

    if not math.isfinite(number):
        return None

    return number


def _ordered_values(values, preferred_order):
    """Put known values in a stable order and append unknown values."""

    values = set(values)

    ordered = [
        item for item in preferred_order
        if item in values
    ]

    ordered.extend(
        sorted(values - set(ordered))
    )

    return ordered


def _validate_single_configuration(rows: list[dict]) -> None:
    """
    Prevent standard comparison plots from silently mixing configurations.

    Size sweeps and objective-weight sweeps should use their dedicated
    plotting functions.
    """

    for column in (
        "num_blocks",
        "block_size_bytes",
        "blocks_per_request",
        "latency_weight",
    ):
        values = {
            row.get(column, "")
            for row in rows
            if row.get(column, "") != ""
        }

        if len(values) > 1:
            raise ValueError(
                f"Standard plots require a single value for {column}. "
                "Use the appropriate sweep-plot function instead."
            )


def _plot_grouped_metric(
    rows: list[dict],
    output_path: Path,
    metric_column: str,
    std_column: str,
    title: str,
    y_label: str,
    reference_line: float | None = None,
) -> Path:
    """Generate a grouped bar chart with optional standard-deviation bars."""

    workloads = _ordered_values(
        (row["workload_type"] for row in rows),
        WORKLOAD_ORDER,
    )

    policies = _ordered_values(
        (row["policy_name"] for row in rows),
        POLICY_ORDER,
    )

    values = {}

    for row in rows:
        key = (row["workload_type"], row["policy_name"])

        if key in values:
            raise ValueError(
                f"Duplicate aggregate row for {key}. "
                "Check the experiment configuration."
            )

        values[key] = (
            _number(row.get(metric_column)),
            _number(row.get(std_column)),
        )

    figure, axis = plt.subplots(figsize=(12, 6))

    group_positions = list(range(len(workloads)))
    total_group_width = 0.80
    bar_width = total_group_width / max(1, len(policies))

    for policy_index, policy in enumerate(policies):
        offsets = [
            position
            - total_group_width / 2
            + (policy_index + 0.5) * bar_width
            for position in group_positions
        ]

        heights = []
        error_bars = []

        for workload in workloads:
            value, std = values.get(
                (workload, policy),
                (None, None),
            )

            heights.append(
                value if value is not None else float("nan")
            )

            error_bars.append(
                std if std is not None else 0.0
            )

        axis.bar(
            offsets,
            heights,
            width=bar_width,
            yerr=error_bars,
            capsize=2,
            label=policy,
        )

    if reference_line is not None:
        axis.axhline(
            reference_line,
            linestyle="--",
            linewidth=1,
        )

    axis.set_title(title)
    axis.set_ylabel(y_label)
    axis.set_xlabel("Workload type")
    axis.set_xticks(group_positions)
    axis.set_xticklabels(workloads)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(
        loc="best",
        fontsize="small",
    )

    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def plot_standard_comparisons(
    aggregate_csv: str | Path,
    output_dir: str | Path = "results/figures",
) -> list[Path]:
    """
    Plot the aggregate comparisons produced by ExperimentRunner.run_suite().
    """

    rows = read_csv_rows(aggregate_csv)
    _validate_single_configuration(rows)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    specifications = [
        (
            "mean_speedup_vs_reference",
            "std_speedup_vs_reference",
            "System speedup relative to CPU-only",
            "Speedup (higher is better)",
            1.0,
            "speedup_by_workload.png",
        ),
        (
            "mean_energy_reduction_pct",
            "std_energy_reduction_pct",
            "Estimated energy reduction relative to CPU-only",
            "Energy reduction (%)",
            0.0,
            "energy_reduction_by_workload.png",
        ),
        (
            "mean_data_movement_reduction_pct",
            "std_data_movement_reduction_pct",
            "Modelled data-movement reduction relative to CPU-only",
            "Data-movement reduction (%)",
            0.0,
            "movement_reduction_by_workload.png",
        ),
        (
            "mean_p95_latency_ns",
            "std_p95_latency_ns",
            "P95 request latency across workloads",
            "P95 latency (ns; lower is better)",
            None,
            "p95_latency_by_workload.png",
        ),
    ]

    created_paths = []

    for (
        metric_column,
        std_column,
        title,
        y_label,
        reference_line,
        filename,
    ) in specifications:
        output_path = output_dir / filename

        created_paths.append(
            _plot_grouped_metric(
                rows=rows,
                output_path=output_path,
                metric_column=metric_column,
                std_column=std_column,
                title=title,
                y_label=y_label,
                reference_line=reference_line,
            )
        )

    return created_paths


def _resource_family(resource_name: str) -> str:
    """Map a specific resource instance to its resource family."""

    if resource_name in {"CPU", "NMP"}:
        return resource_name

    if ":" in resource_name:
        return resource_name.split(":", 1)[0]

    return resource_name


def plot_placement_distributions(
    raw_csv: str | Path,
    output_dir: str | Path = "results/figures",
) -> list[Path]:
    """
    Plot the fraction of requests placed on each resource family.

    Multiple random seeds are combined so that each policy's distribution
    reflects the complete set of executed requests in the raw results file.
    """

    rows = read_csv_rows(raw_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate placement counts by workload and policy across seeds.
    counts = defaultdict(CounterLike)

    for row in rows:
        workload = row.get("workload_type", "")
        policy = row.get("policy_name", "")
        raw_placements = row.get("placements", "")

        if not raw_placements:
            continue

        try:
            placements = json.loads(raw_placements)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid placement JSON for {workload}/{policy}."
            ) from exc

        for resource_name, count in placements.items():
            family = _resource_family(resource_name)

            if family not in RESOURCE_FAMILIES:
                raise ValueError(
                    f"Unknown compute-resource family: {family}"
                )

            counts[(workload, policy)][family] += int(count)

    if not counts:
        raise ValueError(
            "No placement information was found in the raw results CSV."
        )

    workloads = _ordered_values(
        (key[0] for key in counts),
        WORKLOAD_ORDER,
    )

    policies = _ordered_values(
        (key[1] for key in counts),
        POLICY_ORDER,
    )

    created_paths = []

    for workload in workloads:
        figure, axis = plt.subplots(figsize=(11, 6))

        positions = list(range(len(policies)))
        bottom = [0.0] * len(policies)

        for family in RESOURCE_FAMILIES:
            heights = []

            for policy in policies:
                family_count = counts[(workload, policy)][family]
                total_count = sum(
                    counts[(workload, policy)].values()
                )

                proportion = (
                    family_count / total_count
                    if total_count
                    else 0.0
                )

                heights.append(proportion)

            axis.bar(
                positions,
                heights,
                bottom=bottom,
                label=family,
            )

            bottom = [
                previous + height
                for previous, height in zip(bottom, heights)
            ]

        axis.set_title(
            f"Execution-resource placement distribution: {workload}"
        )
        axis.set_ylabel("Fraction of requests")
        axis.set_xlabel("Placement policy")
        axis.set_ylim(0, 1)
        axis.set_xticks(positions)
        axis.set_xticklabels(
            policies,
            rotation=20,
            ha="right",
        )
        axis.grid(axis="y", alpha=0.25)
        axis.legend(title="Resource family")

        figure.tight_layout()

        output_path = output_dir / (
            f"placement_distribution_{workload}.png"
        )

        figure.savefig(
            output_path,
            dpi=180,
            bbox_inches="tight",
        )

        plt.close(figure)
        created_paths.append(output_path)

    return created_paths


class CounterLike(dict):
    """Dictionary whose missing resource-family counters default to zero."""

    def __missing__(self, key):
        value = 0
        self[key] = value
        return value


def plot_size_sweep(
    aggregate_csv: str | Path,
    output_dir: str | Path = "results/figures/size_sweep",
) -> list[Path]:
    """
    Plot mean speedup versus dataset size from size_sweep_aggregate.csv.

    Creates one figure per workload.
    """

    rows = read_csv_rows(aggregate_csv)

    required_columns = {
        "workload_type",
        "policy_name",
        "num_blocks",
        "mean_speedup_vs_reference",
        "std_speedup_vs_reference",
    }

    if not required_columns.issubset(rows[0]):
        raise ValueError(
            "The supplied file is not a dataset-size sweep aggregate CSV."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    workloads = _ordered_values(
        (row["workload_type"] for row in rows),
        WORKLOAD_ORDER,
    )

    policies = _ordered_values(
        (row["policy_name"] for row in rows),
        POLICY_ORDER,
    )

    created_paths = []

    for workload in workloads:
        figure, axis = plt.subplots(figsize=(10, 6))

        for policy in policies:
            subset = [
                row for row in rows
                if row["workload_type"] == workload
                and row["policy_name"] == policy
            ]

            subset.sort(
                key=lambda row: float(row["num_blocks"])
            )

            x_values = [
                float(row["num_blocks"])
                for row in subset
            ]

            y_values = [
                _number(row.get("mean_speedup_vs_reference"))
                for row in subset
            ]

            error_values = [
                _number(row.get("std_speedup_vs_reference")) or 0.0
                for row in subset
            ]

            if not any(value is not None for value in y_values):
                continue

            axis.errorbar(
                x_values,
                [
                    value if value is not None else float("nan")
                    for value in y_values
                ],
                yerr=error_values,
                marker="o",
                capsize=3,
                label=policy,
            )

        axis.axhline(1.0, linestyle="--", linewidth=1)
        axis.set_title(f"Dataset-size sensitivity: {workload}")
        axis.set_xlabel("Number of data blocks")
        axis.set_ylabel("Mean speedup relative to CPU-only")
        axis.grid(alpha=0.25)
        axis.legend(fontsize="small")

        figure.tight_layout()

        output_path = output_dir / (
            f"speedup_size_sweep_{workload}.png"
        )

        figure.savefig(
            output_path,
            dpi=180,
            bbox_inches="tight",
        )

        plt.close(figure)
        created_paths.append(output_path)

    return created_paths


def plot_latency_weight_sweep(
    aggregate_csv: str | Path,
    output_dir: str | Path = "results/figures/weight_sweep",
) -> list[Path]:
    """
    Plot speedup and energy reduction against latency_weight.

    Creates separate speedup and energy-reduction figures for each workload.
    """

    rows = read_csv_rows(aggregate_csv)

    required_columns = {
        "workload_type",
        "policy_name",
        "latency_weight",
        "mean_speedup_vs_reference",
        "mean_energy_reduction_pct",
    }

    if not required_columns.issubset(rows[0]):
        raise ValueError(
            "The supplied file is not a latency-weight sweep aggregate CSV."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    workloads = _ordered_values(
        (row["workload_type"] for row in rows),
        WORKLOAD_ORDER,
    )

    policies = _ordered_values(
        (row["policy_name"] for row in rows),
        POLICY_ORDER,
    )

    created_paths = []

    for workload in workloads:
        for metric, title, y_label, filename_prefix in (
            (
                "mean_speedup_vs_reference",
                "Speedup versus latency weight",
                "Mean speedup relative to CPU-only",
                "speedup",
            ),
            (
                "mean_energy_reduction_pct",
                "Energy reduction versus latency weight",
                "Mean estimated energy reduction (%)",
                "energy_reduction",
            ),
        ):
            figure, axis = plt.subplots(figsize=(10, 6))

            for policy in policies:
                subset = [
                    row for row in rows
                    if row["workload_type"] == workload
                    and row["policy_name"] == policy
                ]

                subset.sort(
                    key=lambda row: float(row["latency_weight"])
                )

                x_values = [
                    float(row["latency_weight"])
                    for row in subset
                ]

                y_values = [
                    _number(row.get(metric))
                    for row in subset
                ]

                if not any(value is not None for value in y_values):
                    continue

                axis.plot(
                    x_values,
                    [
                        value if value is not None else float("nan")
                        for value in y_values
                    ],
                    marker="o",
                    label=policy,
                )

            axis.set_title(f"{title}: {workload}")
            axis.set_xlabel("Latency weight (0 = energy-focused, 1 = latency-focused)")
            axis.set_ylabel(y_label)
            axis.grid(alpha=0.25)
            axis.legend(fontsize="small")

            figure.tight_layout()

            output_path = output_dir / (
                f"{filename_prefix}_weight_sweep_{workload}.png"
            )

            figure.savefig(
                output_path,
                dpi=180,
                bbox_inches="tight",
            )

            plt.close(figure)
            created_paths.append(output_path)

    return created_paths


def generate_standard_plots(
    aggregate_csv: str | Path,
    raw_csv: str | Path,
    output_dir: str | Path = "results/figures",
) -> list[Path]:
    """Generate the standard comparison and placement figures."""

    output_dir = Path(output_dir)

    paths = plot_standard_comparisons(
        aggregate_csv,
        output_dir,
    )

    paths.extend(
        plot_placement_distributions(
            raw_csv,
            output_dir,
        )
    )

    return paths


def main():
    """Command-line entry point."""

    parser = argparse.ArgumentParser(
        description="Plot DynaMem experiment results."
    )

    parser.add_argument(
        "--aggregate",
        default="results/aggregate_comparisons.csv",
        help="Aggregate-comparison CSV from experiments.py.",
    )

    parser.add_argument(
        "--raw",
        default="results/raw_run_results.csv",
        help="Raw-run CSV from experiments.py.",
    )

    parser.add_argument(
        "--output-dir",
        default="results/figures",
        help="Directory in which to save figures.",
    )

    parser.add_argument(
        "--size-sweep",
        default=None,
        help="Optional size_sweep_aggregate.csv path.",
    )

    parser.add_argument(
        "--weight-sweep",
        default=None,
        help="Optional latency_weight_sweep_aggregate.csv path.",
    )

    args = parser.parse_args()

    paths = generate_standard_plots(
        aggregate_csv=args.aggregate,
        raw_csv=args.raw,
        output_dir=args.output_dir,
    )

    if args.size_sweep:
        paths.extend(
            plot_size_sweep(
                args.size_sweep,
                Path(args.output_dir) / "size_sweep",
            )
        )

    if args.weight_sweep:
        paths.extend(
            plot_latency_weight_sweep(
                args.weight_sweep,
                Path(args.output_dir) / "weight_sweep",
            )
        )

    print("\nGenerated figures:")

    for path in paths:
        print(path.resolve())


if __name__ == "__main__":
    main()
