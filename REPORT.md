# DynaMem: Development of a Memory-Centric Computing Architecture for Data-Local Execution and Intelligent Computation Placement

## Abstract

In conventional computer systems, data is often moved from memory to a processor before computation can take place. For applications that process large datasets, these transfers can contribute significantly to execution time and energy consumption. Memory-centric computing explores whether some operations can be performed closer to the data, reducing the need for unnecessary movement.

DynaMem is a Python-based simulation project developed to study this idea. It models a conventional CPU, a simplified CPU cache, multiple memory banks, computation-in-memory resources, a near-memory processor, and the links connecting these components. An adaptive placement controller estimates the cost of executing each request on the available resources and selects an execution location according to the configured latency-energy objective.

The project evaluates reusable data, sequential streaming, random access, and phase-changing workloads. It compares adaptive placement with fixed CPU, CIM-A, CIM-P, near-memory, and hybrid policies. The simulator records simulated latency, makespan, throughput, estimated energy consumption, modelled data movement, queue waiting, resource utilization, and placement decisions. The workload generator also supports mixed-size requests, allowing requests in the same trace to have different numbers of input blocks.

The latest reported experiment suite used three random seeds and covered all four workload types. Adaptive placement achieved mean speedups ranging from 3.457× to 4.402× against CPU-only across the tested workloads, with estimated energy reductions between 63.761% and 73.687%. However, movement outcomes varied, and two workloads showed increased movement under the simulator's tracked counters. These results are estimates under the configured model assumptions, not physical hardware measurements.

## 1. Introduction

### 1.1 Background

Computer performance depends not only on how quickly a processor performs an operation, but also on how efficiently it obtains the data needed by that operation.

A conventional processor may spend time waiting for memory accesses or data transfers even when the computation itself is relatively simple. This can become a significant issue when processing large collections of data or repeatedly accessing the same dataset.

Memory-centric computing addresses this challenge by moving selected computation closer to the location of the data. Depending on the architecture, a suitable operation may execute within a memory array, in the memory periphery, or on a processor located near memory.

These alternatives have different advantages and constraints. A resource within memory may reduce transfer requirements but support only a restricted set of operations. A near-memory processor may support more general computation but still require data to travel over a link. A conventional CPU may be preferable when its cache already holds the relevant input or when other processing resources are busy.

DynaMem was developed to examine these trade-offs using a configurable software model.

### 1.2 Problem statement

A single execution policy is not necessarily suitable for all workloads.

Repeated access to a small set of blocks differs from sequential streaming. Random access creates different locality conditions, while a changing workload may require the execution strategy to adapt over time.

The execution decision should therefore consider more than the nominal speed of a compute resource. It should also account for the amount and location of input data, operation support, transfer latency, resource queues, cache state, and the relative importance of latency and energy.

The problem addressed by DynaMem is how to select an execution location for each request using these modelled characteristics, then evaluate whether adaptive selection performs better than fixed alternatives under the same workload conditions.

### 1.3 Aim

The aim of this project is to design and simulate an adaptive memory-centric computing architecture that selects between conventional, in-memory, and near-memory execution based on estimated cost and observed system behaviour.

### 1.4 Objectives

The project has the following objectives:

1. Model a memory hierarchy containing multiple banks and different compute resources.
2. Represent data movement between the CPU, memory, near-memory processor, and memory banks.
3. Implement a simplified CPU cache with finite capacity and LRU replacement.
4. Generate multiple memory-access patterns, including mixed-size requests.
5. Observe access behaviour and infer the current pattern during simulation.
6. Estimate the cost of executing a request on each legal resource.
7. Select execution resources using an adaptive latency-energy objective.
8. Simulate arrivals, transfers, queues, computation, and request completion.
9. Compare adaptive placement with fixed execution policies.
10. Examine latency, energy, movement, and resource-utilization metrics.
11. Validate the software using automated tests.
12. Document the limitations of the model and the assumptions behind the results.

### 1.5 Scope

DynaMem is implemented in Python and uses discrete-event simulation. It does not implement physical memory circuitry, processor RTL, or a hardware accelerator.

Its configurable parameters represent assumptions about latency, bandwidth, processing rate, energy, and resource availability. The model is intended to support system-level comparisons and explore how placement decisions change under different conditions.

The output should be interpreted as the behaviour of the simulated architecture rather than a prediction guaranteed to match a commercial processor or a future hardware design.

## 2. Architecture Overview

### 2.1 System components

The simulator contains the following major components:

- Conventional CPU
- CPU cache
- Main memory divided into banks
- CIM-A resources
- CIM-P resources
- Near-memory processor
- CPU-memory link
- Memory-NMP link
- Bank-to-bank link
- Data directory
- Runtime monitor
- Cost model
- Placement controller
- Event-driven simulation engine
- Fixed baseline policies
- Metrics, experiments, and plotting modules

Each component has a separate responsibility, allowing it to be changed and tested without combining all of the logic in a single module.

### 2.2 Execution flow

When a trace is generated, it contains data blocks and computation requests with arrival times. At runtime, the simulator schedules each request, obtains information about the current state, and asks the selected policy to choose an execution resource.

The adaptive controller compares the cost of legal candidates using the configured objective. Once a resource is selected, the simulator models the necessary input accesses, transfers, waiting time, compute service, and output handling. A completion record is created for each request.

The high-level process is:

**Workload generation → request arrival → access monitoring → candidate cost estimation → placement decision → data access and transfer → computation → output handling → execution record → metrics and experiment comparison.**

### 2.3 Conventional CPU

The CPU represents general-purpose processing. It can support the operations allowed by the configuration and acts as the fallback for some baseline policies when a specialized resource cannot execute an operation.

CPU execution may require cache lookup, main-memory access, and data transfers. The model includes the configured latency, compute rate, and energy coefficients for these stages.

### 2.4 CPU cache

The CPU cache is a finite-capacity store for recently used blocks. The implementation uses a simplified block-level LRU replacement policy.

A cache hit occurs when the requested block is already resident. On a miss, the model obtains the block from its backing memory path. If there is insufficient cache capacity, the least recently used resident blocks are evicted until the new block can fit, unless the block is too large to be retained at all.

The cache model records hits, misses, evictions, access costs, energy estimates, and cache-copy metadata. It does not implement detailed cache associativity, coherence protocols, or a complete physical cache-line transaction model.

### 2.5 Main memory and banks

Main memory consists of multiple banks. Each block has an identity, dataset identifier, size, and canonical home bank.

The bank count, bank capacity, memory access latency, bandwidth, and energy coefficients are configurable. Bank-level resources allow the simulator to model contention and some degree of concurrency between requests.

### 2.6 CIM-A and CIM-P

CIM-A represents computation within a memory array. In the current default configuration, it supports a restricted operation set, particularly bitmap operations. Each bank has a CIM-A instance, and input data located in another bank may need a transfer before the operation can proceed.

CIM-P represents computation associated with the memory periphery. It supports a broader configured set of operations, including bitmap operations, filtering, counting, and summation.

The two resource types have separate configured compute rates, launch overheads, and energy costs. These values are illustrative parameters supplied to the simulator; they are not measured characteristics of actual CIM devices.

### 2.7 Near-memory processor

The near-memory processor represents a more general compute resource close to main memory. It supports the operations allowed by the configuration and is connected to memory through a dedicated link.

The model accounts for input and output transfer over this link, as well as waiting and execution on the NMP resource.

### 2.8 Data-transfer links

DynaMem models the following primary transfer paths:

1. CPU and main memory
2. Main memory and near-memory processing
3. Transfers between memory banks

Each link has a configured bandwidth, startup latency, energy-per-byte cost, and concurrency limit. The transfer model estimates service time using the link latency and the amount of data transferred.

The model does not represent every possible transfer overlap or low-level memory protocol detail. Transfer costs should therefore be interpreted according to the model's defined behaviour.

## 3. Software Design and Modules

The implementation is divided into separate modules.

### 3.1 Configuration — `config.py`

The configuration module defines the parameters for the CPU cache, memory, links, compute resources, and runtime controller.

It includes values such as bank count, memory capacity, cache size, access latency, link bandwidth, energy coefficients, processing rates, launch overheads, and the latency-energy weight.

The configuration is validated so that required resources, links, and operation definitions are present. The default configuration provides a consistent starting point for experiments, while allowing the parameters to be changed for sensitivity analysis.

### 3.2 Shared data models — `models.py`

This module defines the shared data structures used throughout the simulator.

`DataBlock` represents the memory block and its metadata. `ComputationRequest` describes a computation, its input block identifiers, operation type, output size, operation count, access-pattern label, and arrival time. `PlacementDecision` stores the selected resource and its decision information. `ExecutionRecord` stores timing, movement, queueing and energy information for a completed request.

The data structures validate important values such as request sizes, identifiers, supported operations, and timestamps.

### 3.3 Workload generation — `workload_generator.py`

The workload generator produces data blocks and computation requests using a seeded pseudo-random source. It distributes blocks over the configured banks and generates operations and input selections for each workload.

It supports reusable, streaming, random-access, and phase-changing workloads. The number of input blocks can remain fixed or vary between configured minimum and maximum bounds.

When mixed-size generation is enabled, the input size of a request is the number of selected input blocks multiplied by the configured block size. The same trace can therefore contain different nominal request sizes.

The generated workload is synthetic. Operation counts, output sizes, and arrival schedules are abstractions used by the model rather than measurements collected from real application execution.

### 3.4 Data directory — `data_directory.py`

The data directory stores metadata about block identity, canonical home location, copies, and access history.

It supports queries needed by the cost model and placement controller, including determining where a block is located and whether an additional copy is recorded.

The directory updates metadata when copies are added or removed. It does not independently simulate the time or energy required for a transfer; those costs are handled by the cache and simulator components.

### 3.5 Runtime monitor — `runtime_monitor.py`

The runtime monitor maintains a bounded window of recent observed block accesses for each dataset.

It derives statistics such as unique-block ratio, reuse ratio, and sequential ratio. A minimum observation count is required before classification.

The current heuristic recognises streaming behaviour when sequential locality and unique-block ratio are high, reused behaviour when the unique-block ratio is sufficiently low, and random behaviour otherwise.

This classifier uses observed block accesses rather than trusting the declared pattern label of the generated request. However, the classification is still heuristic, and a fixed threshold may not correctly identify every mixed or transitional access pattern.

### 3.6 Cost model — `cost_model.py`

The cost model identifies which resources support the requested operation and estimates the cost of each legal candidate.

The latency model combines relevant terms such as queue-wait estimate, decision overhead, cache access, memory access, transfer time, and compute time. Compute time is based on the configured launch overhead and abstract operation count divided by the resource's processing rate.

Estimated energy is calculated from configured compute, memory, cache, and transfer coefficients.

The model gives the controller a consistent way to rank candidates. Its estimates are not identical to final simulated timing because the event engine later models actual contention and resource waits.

### 3.7 Placement controller — `placement_controller.py`

The adaptive controller evaluates the legal execution candidates and calculates a normalised latency-energy score.

The general objective is:

**Score = α × normalised latency + (1 − α) × normalised energy**

The parameter α ranges from zero to one. A higher value gives more importance to latency, while a lower value gives more importance to estimated energy.

The controller also supports resource-switching hysteresis. It remembers previous choices and can retain the previous choice if an alternative offers an improvement smaller than the configured threshold, provided the previous resource remains legal.

The reuse-aware feature can assign CPU a discounted potential future benefit when observed access behaviour indicates reuse. This credit changes the ranking score rather than directly reducing the reported cost of the current request. It is a heuristic and should be evaluated through ablation and sensitivity experiments.

Queue-aware scoring incorporates the simulator's approximate queue-wait estimates. The feature can be disabled for ablation studies, while actual resource contention continues to be modelled during execution.

### 3.8 Cache model — `cache_model.py`

The cache model provides block-level lookup, insertion, eviction, and residency tracking using LRU replacement.

It records cache hits and misses, access costs, movement associated with fills, and estimated cache energy. Cache copies are synchronised with the data directory.

The implementation intentionally avoids the complexity of a complete hardware cache model. Its results should be interpreted in the context of block-level LRU behaviour.

### 3.9 Simulator — `simulator.py`

The simulator uses SimPy to schedule request arrivals and model resource contention.

It creates resources for CPU and NMP compute, bank-specific CIM engines, memory banks, transfer links, and the cache port. Requests that need busy resources wait until the required resources become available.

For each request, the simulator runs the policy decision, input access and transfer stages, computation, and output handling. It records completion timestamps and accumulated modelled costs.

The simulator is event-driven rather than cycle-accurate. It represents computation through aggregate operation counts and processing rates and does not reproduce every hardware instruction or memory command.

### 3.10 Baseline policies — `baselines.py`

The project includes CPU-only, static CIM-A, static CIM-P, static NMP, and fixed-hybrid strategies.

These policies provide reference points against which the adaptive controller can be evaluated. The hybrid policy uses a fixed threshold and operation categories instead of reacting to a runtime cost estimate for every decision.

The baseline decision rules and their configured overheads should be documented when presenting comparisons.

### 3.11 Metrics — `metrics.py`

The metrics module calculates summaries and compares policy results. It checks that compared runs refer to compatible workloads and request identifiers.

It calculates makespan-based speedup, latency comparisons, estimated energy reductions, modelled movement reductions, throughput, and resource-utilization metrics where available.

The movement metric follows the byte counters implemented in the simulator. It does not necessarily include all local memory activity, so it must not be described as total physical memory traffic.

### 3.12 Experiment runner — `experiments.py`

The experiment runner generates one trace for each workload and seed and gives that same trace to every policy. Each policy receives a fresh simulator instance.

It saves the individual results, per-seed comparisons, and aggregate means and standard deviations to CSV files. It also supports dataset-size and latency-weight sweeps.

The runner supports mixed-size request bounds and retains the `workload_types` argument as a compatibility alias for `workloads`. It returns result rows and the paths to the generated CSV files.

### 3.13 Plotting — `plots.py`

The plotting module creates graphs from the experiment outputs. The standard suite generates comparisons for speedup, estimated energy reduction, movement reduction, P95 latency, and resource placement.

Separate plots support dataset-size and latency-weight sweeps. The figures provide a visual summary of the CSV data rather than replacing the raw results.

### 3.14 Main entry point — `main.py`

The command-line interface exposes the standard suite, dataset-size sweep, and latency-weight sweep. Users can select seeds, workloads, policies, request counts, block counts, block size, operation types, mixed-size bounds, hybrid threshold, output directory, and whether plots should be generated.

The CLI validates the selected settings and passes them to the experiment runner. This allows the same code to be used for small integration checks and larger experiments.

### 3.15 Ablation runner — `ablation_experiments.py`

The ablation runner compares the full adaptive controller with versions that disable reuse-aware scoring, queue-wait estimates, or both.

This helps investigate whether individual runtime features contribute to the observed results. The variants should use the same workload traces and fresh simulator state for a fair comparison.

## 4. Workload Design

### 4.1 Reused data

The reused workload repeatedly selects blocks from a hot subset of the dataset, with some probability of selecting blocks from the wider dataset.

It is designed to examine reuse, cache behaviour, and the effect of repeated accesses on resource placement.

### 4.2 Streaming data

The streaming workload assigns fresh consecutive blocks to requests. It represents a sequence of requests that progress through a dataset without deliberately revisiting the same input blocks.

The generator checks that sufficient blocks are available for the unique input groups required by the streaming phase.

### 4.3 Random access

The random-access workload selects blocks in random order across the dataset. It does not intentionally create a hot set or sequential access order, though repeats can still occur by chance.

### 4.4 Phase-changing access

The phase-changing workload combines streaming, reused, and random-access phases in one trace. It is intended to examine whether the runtime monitor and placement controller adapt as the observed access behaviour changes.

### 4.5 Mixed-size requests

The workload generator supports a minimum and maximum number of blocks per request. For example, if the block size is 4,096 bytes and requests can contain between one and eight blocks, their nominal input sizes range from 4,096 bytes to 32,768 bytes.

The seed determines the randomly sampled block counts. Consequently, not every possible size is guaranteed to appear in every trace.

When mixed-size bounds are omitted, the legacy `blocks_per_request` setting is used to preserve fixed-size behaviour.

## 5. Execution Policies

Six policies are included in the standard experiment suite.

| Policy | Behaviour |
|---|---|
| `ADAPTIVE` | Selects among legal resources using the cost model and enabled runtime features |
| `CPU_ONLY` | Executes requests on the conventional CPU |
| `STATIC_CIM_A` | Prefers CIM-A for supported operations and uses the defined fallback when needed |
| `STATIC_CIM_P` | Prefers CIM-P for supported operations and uses the defined fallback when needed |
| `STATIC_NMP` | Prefers near-memory execution |
| `FIXED_HYBRID` | Uses a fixed request-size threshold and operation categories to select a resource |

CPU-only is used as the primary reference for speedup and percentage-reduction calculations.

The static policies test whether adaptive placement provides a benefit over choosing one resource family. Fixed hybrid provides a simpler size-based and operation-based comparison.

The outcome depends on the configured architecture, so no baseline should be assumed to be best for every workload.

## 6. Experimental Methodology

### 6.1 Standard suite

For each workload and seed, the experiment runner generates one trace and evaluates the selected policies on that trace. Each policy begins with fresh cache, directory, queue, and runtime-monitor state.

This avoids giving one policy an advantage simply because another policy ran first and changed the state of the system.

### 6.2 Dataset-size sweep

The size sweep varies the number of data blocks while retaining the selected experiment settings. It can reveal how the working-set size affects relative policy performance.

Each tested size must be large enough for the chosen workload and request-size range, especially when streaming requires fresh blocks.

### 6.3 Latency-weight sweep

The weight sweep varies the latency-energy parameter from zero to one. It helps show how the controller's objective affects placement decisions and resulting metrics.

The preferred weight is workload-dependent and should be determined from the experiments rather than selected in advance.

### 6.4 Ablation experiments

The ablation study compares these adaptive variants:

- `ADAPTIVE_FULL`
- `ADAPTIVE_NO_PATTERN`
- `ADAPTIVE_NO_QUEUE`
- `ADAPTIVE_NEITHER`

It also includes fixed policies for reference.

The aim is to measure whether the reuse-aware and queue-aware features contribute to results beyond the base adaptive mechanism.

### 6.5 Seeds and reproducibility

Random seeds make the workload traces repeatable. The same trace is used for all policies within a workload-seed comparison.

The reported final suite used seeds 42, 43, and 44. Three seeds are enough to perform a basic repeatability check, but they provide limited statistical evidence. More seeds and variability analysis are recommended for stronger conclusions.

## 7. Metrics

### 7.1 Makespan

Makespan is the simulated time from the first request's arrival to the last request's completion. It captures the overall elapsed time of a workload executed with concurrent resources.

### 7.2 Speedup

Speedup is calculated as:

**Speedup = CPU-only makespan / policy makespan**

A value above one indicates a lower simulated makespan than CPU-only.

### 7.3 Latency

Request latency is the time between request arrival and request completion. Average latency describes the overall distribution, while P95 latency helps identify the slower requests.

### 7.4 Estimated energy reduction

Estimated energy is calculated from the configured resource and transfer coefficients.

Energy reduction compares a policy's estimated energy with the CPU-only estimate. A positive percentage indicates lower estimated energy; a negative value indicates higher estimated energy.

### 7.5 Modelled movement reduction

The movement metric compares the bytes counted by the simulator's movement counters against CPU-only.

The metric covers tracked transfer categories and may not include all local memory activity. It should be interpreted as modelled movement rather than a complete count of physical memory traffic.

### 7.6 Resource utilization

Utilization represents the fraction of simulated resource capacity that was busy during the run. It can help identify possible bottlenecks or underused resources.

### 7.7 Placement distribution

The placement distribution shows how frequently each policy selects CPU, CIM-A, CIM-P, or NMP. It can explain the controller's behaviour but does not establish effectiveness on its own.

## 8. Latest Mixed-Size Experiment Results

The reported final experiment suite used three seeds (42, 43, and 44), four workloads, and six execution policies. It allowed requests to contain between one and eight input blocks, with a default block size of 4,096 bytes.

The following table summarizes the reported adaptive results against CPU-only.

| Workload | Mean speedup | Estimated energy reduction | Modelled movement reduction |
|---|---:|---:|---:|
| Reused | 3.457× | 63.761% | -30.259% |
| Streaming | 4.402× | 73.687% | 7.598% |
| Random | 3.979× | 72.325% | 15.143% |
| Phase-changing | 3.901× | 69.013% | -9.124% |

### 8.1 Streaming

The streaming workload achieved the highest reported adaptive speedup, at 4.402×. Estimated energy reduction was 73.687%, while tracked movement reduction was 7.598%.

Under the configured model, this result suggests that the adaptive resource choices were effective for the streaming trace. It does not establish that the same speedup would occur on physical hardware.

### 8.2 Random access

The random-access workload achieved 3.979× mean speedup and 72.325% estimated energy reduction. Tracked movement fell by 15.143% relative to CPU-only.

This was a favourable outcome across all three reported metrics, but it remains conditional on the workload generator and configured cost parameters.

### 8.3 Reused data

The reused workload achieved 3.457× mean speedup and 63.761% estimated energy reduction.

However, tracked movement increased by 30.259% compared with CPU-only. This shows that reduced simulated makespan and estimated energy did not imply reduced movement for this trace.

### 8.4 Phase-changing access

The phase-changing workload achieved 3.901× mean speedup and 69.013% estimated energy reduction.

Tracked movement increased by 9.124%. This is relevant because changing locality can cause different resources to become preferable at different points in the run.

### 8.5 Comparison with static policies

The static NMP policy was the strongest fixed comparison in terms of simulated speedup across these four reported workloads.

| Workload | Static NMP speedup |
|---|---:|
| Reused | 2.953× |
| Streaming | 4.177× |
| Random | 3.958× |
| Phase-changing | 3.444× |

Adaptive placement had higher reported mean speedup than static NMP for all four workloads. The difference for random access was small, so further seeds and sensitivity experiments would help determine whether that advantage is robust.

### 8.6 Fixed-hybrid result

The fixed-hybrid policy matched CPU-only in this experiment.

The configured hybrid threshold was 256 KiB, while the maximum nominal input size for the mixed-size requests was 32 KiB. Therefore, the specialized branches were not triggered for these request sizes.

This outcome reflects the selected threshold and request-size range. It does not show that fixed-hybrid placement can never be useful. A separate threshold sweep would be needed to evaluate it fairly.

### 8.7 Overall interpretation

Across the four tested workloads, adaptive placement showed lower simulated makespan and lower estimated energy than CPU-only. The movement results differed across workloads.

The findings therefore support a workload-dependent conclusion: adaptive placement can be beneficial within the configured model, but its advantages are not identical across latency, energy, and data movement.

All reported values are simulation results. They should not be described as physical measurements or guarantees of performance improvement on real hardware.

## 9. Testing and Verification

The project includes automated tests covering the cost model, data movement, placement controller, event-driven simulation, cache model, runtime monitor, and baseline policies.

The tests check conditions such as:

- Valid resource selection and operation support.
- Expected cost and movement calculations.
- Cache hits, misses, and eviction.
- Runtime access-pattern classification.
- Queue behaviour and request completion.
- Correctness of baseline policy rules.
- Result generation and experiment integration.

The mixed-size workload generator was tested with 100 random-access requests. The check reported a minimum input size of 4,096 bytes, a maximum of 32,768 bytes, and all eight possible sizes in the configured range.

As reported by the project owner on 28 October 2026, the integration commands executed successfully, all eight standard figures were generated, and the test suite reported **83 passed**.

These checks establish that the tested software behaviour was correct for the cases covered. They do not independently validate the physical accuracy of the model parameters or prove that the adaptive policy is universally superior.

## 10. Limitations

DynaMem has several limitations that affect how its results should be interpreted.

1. **Illustrative parameters:** the default latency, bandwidth, compute rate, and energy values are assumptions rather than measurements from a specific device.
2. **Abstract computation:** operation counts and processing rates represent computation in aggregate rather than detailed instruction execution.
3. **Simplified cache:** the cache uses block-level LRU behaviour and does not model all hardware cache features.
4. **Simplified memory timing:** memory banks and links use aggregate service costs rather than cycle-level command sequences.
5. **Approximate queue prediction:** predicted waiting can differ from actual waits in the event simulation.
6. **Heuristic access classification:** fixed thresholds may not correctly represent every mixed or changing access pattern.
7. **Heuristic reuse credit:** the reuse-aware feature estimates a potential future benefit rather than guaranteeing future CPU-cache hits.
8. **Limited movement accounting:** local memory activity is not necessarily included in the tracked movement total.
9. **No physical validation:** the project does not implement hardware or measure physical energy.

These limitations define the scope of the current conclusions. They also provide clear directions for improving the model.

## 11. Future Work

Several additions could make the simulator more realistic and the findings more robust.

- Calibrate model parameters using published hardware measurements.
- Improve queue prediction by considering the dependencies between banks, links, and compute resources.
- Model more detailed memory timing and possible overlap between transfers and computation.
- Extend the cache model with more realistic associativity and write behaviour.
- Make workload computation and output-size models more explicit.
- Evaluate runtime classification on a wider range of mixed patterns.
- Calibrate reuse-aware scoring from observed simulation results.
- Run more random seeds and report confidence intervals where appropriate.
- Perform sensitivity experiments on compute rates, transfer bandwidth, cache size, and controller parameters.
- Compare results with a more detailed simulator or published results when suitable workloads and metrics are available.

## 12. Conclusion

DynaMem is a configurable simulator for exploring adaptive computation placement in a memory-centric architecture. It combines a model of memory banks, CPU caching, computation-in-memory and near-memory resources, data-transfer links, runtime monitoring, cost estimation, and event-driven execution.

The project compares adaptive placement with conventional and fixed execution strategies across multiple workload types. Its mixed-size workload support also allows request sizes to vary within a single trace.

The reported experiments showed favourable simulated makespan and estimated-energy outcomes for adaptive placement relative to CPU-only across four workloads. However, data movement increased for reused and phase-changing access in the current experiment, and the fixed-hybrid result was constrained by a threshold larger than every request's nominal input size.

The results illustrate why memory-centric placement should be evaluated with multiple metrics and clearly stated assumptions. DynaMem provides a framework for conducting these comparisons in a repeatable way, but further calibration and independent validation would be needed before applying its conclusions to real hardware.

## Appendix A. Key Terms

| Term | Meaning |
|---|---|
| CIM | Computation-in-Memory |
| CIM-A | Simulated computation within a memory array |
| CIM-P | Simulated computation in memory periphery |
| NMP | Near-memory processor |
| Data locality | How close data is to the resource processing it |
| LRU | Least Recently Used replacement policy |
| Makespan | Elapsed simulated time from first arrival to final completion |
| P95 latency | The 95th-percentile request latency |
| Placement policy | The strategy used to select an execution resource |
| Reuse ratio | One minus the unique-block ratio in the monitor's recent window |
| Seed | Value used to make pseudo-random trace generation reproducible |
| Ablation study | An experiment that disables a feature to examine its contribution |

## Appendix B. Commands for Reproducing Experiments

Run the standard experiment suite:

```bash
python main.py --mode suite --seeds 42,43,44
```

Run a mixed-size suite:

```bash
python main.py --mode suite --seeds 42,43,44 --min-blocks-per-request 1 --max-blocks-per-request 8 --output-dir results_mixed_size
```

Run without plots:

```bash
python main.py --mode suite --seeds 42,43,44 --min-blocks-per-request 1 --max-blocks-per-request 8 --output-dir results_mixed_size --no-plots
```

Run a dataset-size sweep:

```bash
python main.py --mode size-sweep --sizes 512,1024,2048,4096
```

Run a latency-weight sweep:

```bash
python main.py --mode weight-sweep --weights 0,0.25,0.5,0.75,1
```

Run automated tests:

```bash
python -m pytest tests/ -v
```

Before reporting any new results, verify the output CSVs and preserve the command, seeds, workload settings, and configuration used to generate them.