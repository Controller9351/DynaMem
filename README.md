# DynaMem — Adaptive Memory-Centric Computing Simulator

DynaMem is a Python-based simulator that explores how choosing where a computation runs can affect execution time, estimated energy consumption, and data movement. It compares a conventional CPU with computation-in-memory resources and a near-memory processor, using an adaptive placement controller to select an execution resource for each request.

The project focuses on data locality, resource contention, runtime access monitoring, and the trade-offs between moving data and processing it closer to memory.

> **Note:** DynaMem is a software simulation, not a physical hardware implementation. Its performance and energy results are estimates produced by the configured model.

## 1. Project Overview

In conventional systems, data often has to move from memory to a processor before computation can occur. For workloads that process large datasets, these transfers can contribute significantly to total execution cost.

DynaMem explores whether this cost can be reduced by selecting an execution location according to the current request, data location, estimated transfer costs, and resource availability.

The architecture includes:

- A conventional CPU.
- A simplified CPU cache using LRU replacement.
- Main memory divided into banks.
- CIM-A resources representing computation within a memory array.
- CIM-P resources representing computation in memory periphery.
- A near-memory processor (NMP).
- Links connecting the CPU, memory, NMP, and memory banks.
- A data directory that tracks block locations and copies.
- A runtime monitor for observing access behaviour.
- A cost model and adaptive placement controller.
- Fixed execution policies for comparison.
- An event-driven simulator and experimental analysis tools.

## 2. Main Features

### Adaptive execution placement

The controller estimates the latency and energy of legal execution resources and selects an option according to a configurable objective. The decision can account for estimated queue waiting, and the controller includes a reuse-aware scoring feature.

### Multiple workload patterns

DynaMem supports four workload types:

- `reused` — repeatedly accesses a subset of blocks.
- `streaming` — accesses fresh, consecutive groups of blocks.
- `random` — selects blocks in random order.
- `phase_changing` — moves through streaming, reused, and random phases.

### Mixed-size requests

Requests can use different numbers of input blocks. This allows the same trace to contain requests with different nominal input sizes.

The options `--min-blocks-per-request` and `--max-blocks-per-request` configure the request-size range. When omitted, the legacy fixed-size setting is retained.

For example, with 4,096-byte blocks and a range of one to eight blocks, request input sizes can range from 4 KiB to 32 KiB.

### Baseline comparison

The adaptive controller is compared with five fixed policies:

- `CPU_ONLY`
- `STATIC_CIM_A`
- `STATIC_CIM_P`
- `STATIC_NMP`
- `FIXED_HYBRID`

### Experiment analysis

The experiment runner supports standard comparisons, dataset-size sweeps, and latency-weight sweeps. The ablation runner separately examines the effect of queue-aware and reuse-aware features.

The project records simulated latency, makespan, throughput, estimated energy, modelled data movement, and resource utilization. CSV outputs and graphs support further analysis.

## 3. Project Structure

The main project files are organized as follows.

| File | Purpose |
|---|---|
| `config.py` | Defines architecture parameters and default configuration |
| `models.py` | Defines data blocks, computation requests, decisions, and execution records |
| `workload_generator.py` | Generates synthetic workload traces |
| `data_directory.py` | Tracks canonical block locations, copies, and access history |
| `runtime_monitor.py` | Classifies observed access patterns |
| `cost_model.py` | Estimates candidate execution costs |
| `placement_controller.py` | Selects execution resources |
| `cache_model.py` | Simulates cache access and LRU replacement |
| `simulator.py` | Runs the event-driven simulation |
| `baselines.py` | Implements the fixed comparison policies |
| `metrics.py` | Calculates metrics and policy comparisons |
| `experiments.py` | Runs experiments and saves result CSV files |
| `ablation_experiments.py` | Evaluates adaptive-controller feature ablations |
| `plots.py` | Generates experiment visualizations |
| `main.py` | Provides the command-line interface |
| `requirements.txt` | Lists required Python packages |
| `tests/` | Contains automated tests |

`README.md` provides the operating instructions, while `REPORT.md` describes the architecture, methodology, results, and limitations in greater detail.

## 4. Requirements and Installation

Use Python 3.11 or another version compatible with the dependencies and project code.

Open a terminal in the project directory and install the requirements:

```bash
python -m pip install -r requirements.txt
```

If a dependency is missing, install the relevant package and rerun the command.

## 5. Run the Project

### Display available options

```bash
python main.py --help
```

### Run the standard experiment suite

```bash
python main.py --mode suite --seeds 42,43,44
```

This runs the configured workloads and policies using three random seeds.

### Run a smaller smoke test

```bash
python main.py --mode suite --seeds 42 --requests 12 --blocks 256 --workloads reused,random --policies ADAPTIVE,CPU_ONLY --operations BITMAP_AND --no-plots
```

This is useful for checking that the command-line interface, workload generation, simulation, and result export work together.

### Run a mixed-size experiment

```bash
python main.py --mode suite --seeds 42,43,44 --min-blocks-per-request 1 --max-blocks-per-request 8 --output-dir results_mixed_size
```

With the default 4,096-byte block size, this permits nominal request input sizes between 4 KiB and 32 KiB.

The exact sizes present in a trace depend on the seed and number of requests. The configured range does not guarantee that every possible size will occur in every run.

### Skip plots

```bash
python main.py --mode suite --seeds 42,43,44 --min-blocks-per-request 1 --max-blocks-per-request 8 --output-dir results_mixed_size --no-plots
```

### Change the fixed-hybrid threshold

```bash
python main.py --mode suite --seeds 42,43,44 --min-blocks-per-request 1 --max-blocks-per-request 8 --hybrid-threshold 8192 --output-dir results_hybrid_check
```

This changes the configured fixed-hybrid threshold to 8,192 bytes. It is a threshold experiment, not a claim that this value is optimal.

### Run the dataset-size sweep

```bash
python main.py --mode size-sweep --sizes 512,1024,2048,4096
```

### Run the latency-weight sweep

```bash
python main.py --mode weight-sweep --weights 0,0.25,0.5,0.75,1
```

### Run the automated test suite

```bash
python -m pytest tests/ -v
```

The current project verification reported **83 passing tests**. Rerun the test suite after making further changes.

## 6. Understanding the Results

The experiment runner saves raw metrics, per-seed comparisons, and aggregate comparisons.

The standard suite uses CPU-only as the main reference for calculating speedup and percentage reductions.

### Makespan and speedup

Makespan is the elapsed simulated time from the first request's arrival to the last request's completion.

Speedup is calculated as:

**Speedup = CPU-only makespan / policy makespan**

A speedup above 1 means the policy completed the workload faster in the simulation.

### Estimated energy reduction

The energy model combines configured computation, memory, cache, and transfer costs.

A positive energy-reduction percentage indicates lower estimated energy than CPU-only. A negative value indicates higher estimated energy.

### Modelled data movement

Movement reduction compares the movement bytes recorded by the simulator with those recorded for CPU-only.

The counter covers the movement paths represented in the current model. It does not necessarily include every local memory read or write, so it should not be interpreted as total physical memory traffic.

### Important interpretation

A policy may improve latency and estimated energy without reducing movement in every workload. The metrics should be considered together rather than treating any single metric as proof that a policy is universally better.

All performance and energy values are simulated estimates, not measurements from physical hardware.

## 7. Latest Recorded Mixed-Size Results

The following results were reported from the mixed-size experiment suite using three random seeds: 42, 43, and 44.

| Workload | Policy | Mean speedup | Estimated energy reduction (%) | Modelled movement reduction (%) |
|---|---|---:|---:|---:|
| Reused | ADAPTIVE | 3.457 | 63.761 | -30.259 |
| Streaming | ADAPTIVE | 4.402 | 73.687 | 7.598 |
| Random | ADAPTIVE | 3.979 | 72.325 | 15.143 |
| Phase-changing | ADAPTIVE | 3.901 | 69.013 | -9.124 |

The adaptive controller achieved the largest mean speedup for the streaming workload in this experiment. It also reduced estimated energy across all four workloads.

Movement results varied. Reused and phase-changing workloads showed increased tracked movement compared with CPU-only. These outcomes are part of the results and should be reported rather than omitted.

The fixed-hybrid policy matched CPU-only in this configuration because the default threshold was 256 KiB, while the maximum nominal request input was 32 KiB. Requests therefore remained below its threshold.

These figures describe the recorded simulation configuration. They should not be interpreted as guarantees of performance or energy improvements in real hardware.

## 8. Generated Output

The standard suite creates the following CSV files:

| File | Contents |
|---|---|
| `raw_run_results.csv` | Metrics from individual policy runs |
| `per_seed_comparisons.csv` | Comparisons for individual seeds |
| `aggregate_comparisons.csv` | Aggregate means and standard deviations |

When plotting is enabled, the `figures/` directory contains charts for speedup, energy reduction, movement reduction, P95 latency, and placement distribution by workload.

Size sweeps and weight sweeps save their outputs in separate directories. Use `--output-dir` to keep experiments organized and avoid overwriting previous results.

The files in the output directory should be retained alongside the seed list and command used for each experiment so the reported results remain traceable.

## 9. Testing Status

The project owner reported that the updated `experiments.py` and `main.py` executed successfully, the mixed-size workload generation check passed, the plots were generated, and all 83 automated tests passed.

The tests cover the implemented behaviour of the cost model, data movement, placement, simulator, cache, runtime monitor, and baseline policies.

Passing tests confirms the assertions covered by those tests. It does not independently prove the accuracy of the assumed physical parameters or guarantee that adaptive placement wins for every workload.

## 10. Limitations

- Default latency, bandwidth, compute-rate, and energy parameters are illustrative model assumptions.
- Computation is represented through abstract operation counts and processing rates.
- The cache model is a simplified block-level LRU model.
- The memory and transfer model is not cycle-accurate.
- Queue-wait estimates are approximate; actual waits are determined during simulation.
- Access-pattern classification uses heuristic thresholds.
- Reuse-aware scoring estimates a potential benefit rather than guaranteeing future cache hits.
- Modelled movement does not necessarily count every local memory operation.
- No physical hardware implementation or measurement is included.

These limitations define the scope of the conclusions that can be drawn from the simulator.

## 11. Future Work

Possible extensions include calibrating parameters against published hardware measurements, improving queue-wait prediction, adding more detailed memory timing, refining the cache and output-placement models, and evaluating the adaptive controller over more seeds and parameter settings.

Additional ablation and sensitivity experiments would help determine how stable the reported results are when the workload or architecture parameters change.

## 12. Conclusion

DynaMem provides a configurable framework for examining how computation placement affects simulated performance, estimated energy consumption, and data movement.

The project combines workload generation, a model of memory and compute resources, runtime monitoring, adaptive decision-making, fixed baselines, automated tests, and experiment reporting. Its mixed-size workload support allows it to evaluate requests with different input requirements within the same trace.

The latest reported experiments show that adaptive placement performed favourably against CPU-only in simulated makespan and estimated energy across the four tested workloads, while movement outcomes varied by workload. These results illustrate why placement decisions should be evaluated using multiple metrics and clearly stated assumptions.

The project is intended as a system-level exploration of memory-centric computing. Its findings are valid within the assumptions and limitations of the simulation model; further calibration and independent validation would be needed before applying them to physical hardware.