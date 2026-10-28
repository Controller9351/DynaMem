
"""
DynaMem: Metrics and policy-comparison utilities.

Calculates:
- Makespan speedup
- Average, median, and P95 request latency
- Throughput
- Estimated energy and energy per request
- Data-movement reduction
- Latency and throughput changes
- Resource-placement distributions

Comparisons require equivalent workload traces.

All energy and performance values come from the simulator's model,
not measurements from physical hardware.
"""

import csv
import json
from math import isfinite
from pathlib import Path
from typing import Mapping

from simulator import SimulationResult


def safe_ratio(
    numerator: float,
    denominator: float,
) -> float | None:
    """Return numerator / denominator, or None if undefined."""

    if denominator == 0:
        return None

    return numerator / denominator


def percentage_reduction(
    reference_value: float,
    measured_value: float,
) -> float | None:
    """
    Return percentage reduction relative to a reference.

    Positive values indicate a reduction; negative values indicate
    an increase compared with the reference.
    """

    if reference_value == 0:
        return None

    return 100.0 * (
        1.0 - measured_value / reference_value
    )


def percentage_increase(
    reference_value: float,
    measured_value: float,
) -> float | None:
    """
    Return percentage increase relative to a reference.

    Negative values indicate a decrease compared with the reference.
    """

    if reference_value == 0:
        return None

    return 100.0 * (
        measured_value / reference_value - 1.0
    )


def summarize_result(result: SimulationResult) -> dict:
    """Return the simulator's summary with additional per-request metrics."""

    summary = result.summary()

    completed = summary["completed_requests"]

    summary["energy_per_request_pj"] = (
        summary["total_energy_pj"] / completed
        if completed
        else None
    )

    summary["bytes_moved_per_request"] = (
        summary["total_bytes_moved"] / completed
        if completed
        else None
    )

    summary["energy_per_byte_moved_pj"] = safe_ratio(
        summary["total_energy_pj"],
        summary["total_bytes_moved"],
    )

    return summary


def _validate_comparable_results(
    results: Mapping[str, SimulationResult],
    reference_policy: str,
) -> None:
    """Ensure results describe the same workload and valid policy labels."""

    if not results:
        raise ValueError("At least one simulation result is required.")

    if reference_policy not in results:
        raise ValueError(
            f"Reference policy {reference_policy!r} was not found. "
            f"Available policies: {sorted(results)}"
        )

    if len(results) < 2:
        raise ValueError(
            "At least two policies are required for comparison."
        )

    reference = results[reference_policy]

    reference_ids = {
        record.request_id for record in reference.records
    }

    for policy_name, result in results.items():
        if not isinstance(policy_name, str) or not policy_name.strip():
            raise ValueError("Policy names must be non-empty strings.")

        if result.policy_name != policy_name:
            raise ValueError(
                f"Result key {policy_name!r} does not match "
                f"its recorded policy name {result.policy_name!r}."
            )

        if result.trace_name != reference.trace_name:
            raise ValueError(
                "Cannot compare results from different workload traces: "
                f"{reference.trace_name!r} versus {result.trace_name!r}."
            )

        request_ids = {
            record.request_id for record in result.records
        }

        if request_ids != reference_ids:
            raise ValueError(
                f"Policy {policy_name!r} completed a different set "
                "of request IDs than the reference policy."
            )

        if len(request_ids) != len(result.records):
            raise ValueError(
                f"Policy {policy_name!r} contains duplicate "
                "request IDs in its execution records."
            )


def compare_results(
    results: Mapping[str, SimulationResult],
    reference_policy: str = "CPU_ONLY",
) -> list[dict]:
    """
    Compare simulation outcomes with a reference policy.

    Speedup uses system makespan:
        reference makespan / measured makespan

    Latency, energy, and bytes-moved reductions are reported separately.
    """

    _validate_comparable_results(
        results,
        reference_policy,
    )

    reference_result = results[reference_policy]
    reference = summarize_result(reference_result)

    rows = []

    for policy_name, result in results.items():
        current = summarize_result(result)

        speedup = safe_ratio(
            reference["makespan_ns"],
            current["makespan_ns"],
        )

        throughput_ratio = safe_ratio(
            current["throughput_requests_per_second"],
            reference["throughput_requests_per_second"],
        )

        row = {
            "policy_name": policy_name,
            "trace_name": current["trace_name"],
            "reference_policy": reference_policy,
            "request_count": current["completed_requests"],
            "makespan_ns": current["makespan_ns"],
            "speedup_vs_reference": speedup,
            "average_latency_ns": (
                current["average_request_latency_ns"]
            ),
            "average_latency_reduction_pct": percentage_reduction(
                reference["average_request_latency_ns"],
                current["average_request_latency_ns"],
            ),
            "p50_latency_ns": current["p50_latency_ns"],
            "p95_latency_ns": current["p95_latency_ns"],
            "p95_latency_reduction_pct": percentage_reduction(
                reference["p95_latency_ns"],
                current["p95_latency_ns"],
            ),
            "throughput_requests_per_second": (
                current["throughput_requests_per_second"]
            ),
            "throughput_change_pct": (
                percentage_increase(
                    reference["throughput_requests_per_second"],
                    current["throughput_requests_per_second"],
                )
            ),
            "throughput_ratio_vs_reference": throughput_ratio,
            "total_energy_pj": current["total_energy_pj"],
            "energy_per_request_pj": (
                current["energy_per_request_pj"]
            ),
            "energy_reduction_pct": percentage_reduction(
                reference["total_energy_pj"],
                current["total_energy_pj"],
            ),
            "total_bytes_moved": current["total_bytes_moved"],
            "bytes_moved_per_request": (
                current["bytes_moved_per_request"]
            ),
            "data_movement_reduction_pct": percentage_reduction(
                reference["total_bytes_moved"],
                current["total_bytes_moved"],
            ),
            "input_bytes_moved": current["input_bytes_moved"],
            "output_bytes_moved": current["output_bytes_moved"],
            "cross_bank_bytes_moved": (
                current["cross_bank_bytes_moved"]
            ),
            "total_queue_wait_ns": current["total_queue_wait_ns"],
            "total_compute_time_ns": current["total_compute_time_ns"],
            "total_transfer_time_ns": current["total_transfer_time_ns"],
            "runtime_decisions": current["runtime_decisions"],
            "placement_distribution": json.dumps(
                current["placements"],
                sort_keys=True,
            ),
        }

        rows.append(row)

    # Display the reference first, followed by policies with higher
    # speedup. Policies with undefined speedup appear at the end.
    rows.sort(
        key=lambda row: (
            row["policy_name"] != reference_policy,
            row["speedup_vs_reference"] is None,
            -(row["speedup_vs_reference"] or 0.0),
            row["policy_name"],
        )
    )

    return rows


def save_comparison_csv(
    rows: list[dict],
    output_path: str | Path,
) -> Path:
    """Save policy-comparison rows to a CSV file."""

    if not rows:
        raise ValueError("Cannot save an empty comparison.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    for row in rows:
        if set(row.keys()) != set(fieldnames):
            raise ValueError(
                "All comparison rows must have the same columns."
            )

        for value in row.values():
            if isinstance(value, float) and not isfinite(value):
                raise ValueError(
                    "CSV metrics cannot contain NaN or infinity."
                )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    return path
