# SkyPilot Spot Simulator

This simulator models the execution of tasks on cloud instances, considering spot instance availability and preemption based on trace data. It allows comparing different strategies for managing spot and on-demand instances to complete tasks within a deadline.

## Running Simulations

There are two main ways to run simulations:

1.  **Command Line Interface (CLI):** Specify all parameters directly via command-line arguments. Use `python main.py --help` to see available options.
2.  **YAML Scenarios:** Define one or more simulation scenarios in a YAML file (e.g., `scenarios.yaml`) and run specific scenarios using the `--scenarios-config` and `--run-scenarios` arguments. This is useful for organizing and reproducing complex setups.

   ```bash
   # Example: Run a specific scenario from the YAML file
   python main.py --scenarios-config scenarios.yaml --run-scenarios ChainedTask_With_SubEnvs --output-dir exp-yaml/ChainedTask_SubEnvs
   ```

## Task Types

Tasks define the computational work to be done.

### `SingleTask`

Represents a simple task with a single, continuous duration requirement.

**YAML Configuration:**

```yaml
MySingleTaskScenario:
  task:
    type: SingleTask
    config:
      duration: 47.0 # Total duration in hours
  # ... other scenario settings (env, deadline, etc.)
```

### `ChainedTask`

Represents a sequence of sub-tasks that must be completed in order. The total task is done only when all sub-tasks are finished. Currently, sub-tasks are assumed to be `SingleTask`.

**YAML Configuration:**

```yaml
MyChainedTaskScenario:
  task:
    type: ChainedTask
    config:
      sub_tasks:
        - duration: 25.0 # Duration of first sub-task in hours
        - duration: 22.0 # Duration of second sub-task in hours
        # Add more sub-tasks as needed
  # ... other scenario settings (env, deadline, etc.)
```

## Environment Types

Environments simulate the underlying cloud infrastructure and spot instance availability based on trace files.

### `TraceEnv`

Simulates a single environment (e.g., a specific instance type in one availability zone) based on a single trace file. Assumes trace files are JSON containing availability data over time.

**YAML Configuration:**

```yaml
MySingleRegionScenario:
  env:
    type: trace
    trace_files:
      # Path to the single trace file
      - data/real/ping_based/random_start_time/us-west-2a_v100_1/0.json
    # env_start_hours: 0 # Optional start offset in hours
  # ... other scenario settings
```

### `MultiTraceEnv`

Simulates multiple environments concurrently (e.g., different regions or availability zones) using multiple trace files. The strategy can potentially switch between these simulated regions.

**YAML Configuration:**

```yaml
MyMultiRegionScenario:
  env:
    type: multi_trace
    trace_files:
      # List of paths to trace files, one per simulated region/env
      - data/real/ping_based/random_start_time/us-west-2a_v100_1/0.json
      - data/real/ping_based/random_start_time/us-west-2b_v100_1/0.json
    # env_start_hours: 0 # Optional start offset common to all traces
  # ... other scenario settings
```

### `SubtaskMultiEnvSwitcher`

This environment is specifically designed to work with `ChainedTask`. It holds a list of underlying environments (typically `MultiTraceEnv` or `TraceEnv`), one for each sub-task defined in the `ChainedTask`. As the simulation progresses through the sub-tasks of the `ChainedTask`, the `SubtaskMultiEnvSwitcher` automatically switches to using the corresponding underlying environment from its list.

**Key Requirement: Indexed Trace Files**
When using `SubtaskMultiEnvSwitcher`, the simulation often needs to run multiple independent trials using different trace segments. This is achieved by using *indexed* trace files. The system expects trace files within the specified directories to be named like `0.json`, `1.json`, `2.json`, etc. The simulator will find directories containing corresponding indices across *all* sub-task environment definitions and create a separate simulation run (using a distinct `SubtaskMultiEnvSwitcher` instance) for each common index found.

**YAML Configuration:**

```yaml
ChainedTask_With_SubEnvs:
  task:
    type: ChainedTask
    config:
      sub_tasks:
        - duration: 25.0 # Sub-task 0
        - duration: 22.0 # Sub-task 1
  env:
    type: subtask_multi_env_switcher
    # List of environment configurations, matching the order of sub_tasks
    sub_task_envs:
      - # Environment config for Sub-task 0
        trace_files: # This sub-task uses a MultiTraceEnv internally
          # Directories containing indexed traces (e.g., 0.json, 1.json...)
          - data/real/ping_based/random_start_time/us-west-2a_v100_1
          - data/real/ping_based/random_start_time/us-west-2b_v100_1
        # env_start_hours: 0 # Optional for this sub-env
      - # Environment config for Sub-task 1
        trace_files: # This sub-task also uses a MultiTraceEnv
          # Directories containing indexed traces (e.g., 0.json, 1.json...)
          - data/real/ping_based/random_start_time/us-west-2a_k80_1
          - data/real/ping_based/random_start_time/us-west-2b_k80_1
        env_start_hours: 10 # Optional: Specific start time for this sub-env's traces
  # Global settings for the overall scenario
  deadline_hours: 52.0
  restart_overhead_hours: [0.2] # Can be a single value or list matching sub-tasks
```

In the example above:
*   When the simulation is working on sub-task 0, the switcher uses a `MultiTraceEnv` based on the `v100` traces.
*   When the simulation progresses to sub-task 1, the switcher automatically starts using a `MultiTraceEnv` based on the `k80` traces (starting at hour 10 within those traces).
*   The simulator finds common indices (e.g., `0.json`, `1.json`) in *all four* specified directories and runs separate simulations for each index.

## Scenario Configuration (`scenarios.yaml`)

The YAML file allows defining multiple named scenarios. Each scenario typically specifies:

*   `task`: Defines the `type` (`SingleTask` or `ChainedTask`) and its specific `config`.
*   `env`: Defines the `type` (`trace`, `multi_trace`, or `subtask_multi_env_switcher`) and its configuration (e.g., `trace_files`, `sub_task_envs`).
*   `deadline_hours`: The overall deadline for completing the task.
*   `restart_overhead_hours`: The time penalty incurred when restarting a task after preemption or switching instance types. Can be a single float or a list (useful for `ChainedTask` if overhead differs per sub-task).
*   Other parameters can often be overridden here or set globally via CLI/config file.

Refer to `scenarios.yaml` for more examples.

```
sky launch ./cluster.yaml
```

## Artifact

Data and exp can be found in `gs://spot-vs-demand-exp`.

## Spot / On-demand

### Strategies
#### Only Spot
#### Only On-demand


#### Strawman (Greedy)

### Time Sliced
`TRACE_PATH` can be a folder containing multiple traces.
```
python ./main.py --strategy=time_sliced \
            --slice-interval-hours=$i \
            --env trace \
            --trace-file $TRACE_PATH \
            --restart-overhead-hours=0.1 \
            --deadline-hours=52 \
            --task-duration-hours=48 | tee ./$basename-slice-$i.out 2>&1 &
```

### Trace generation
#### Generate from distribution

```
python ./sky_spot/traces/generate.py --trace-folder data/two_exp \
--generator two_exp \
--gap-seconds 600 \
--length 4320 \
--num-traces 1 \
--alive-scale 24.4920 \
--wait-scale 6.4126984
```

```
python ./sky_spot/traces/generate.py --trace-folder data/two_gamma \
--generator two_gamma \
--gap-seconds 600 \
--length 4320 \
--num-traces 5 \
--alive-a 5 \
--alive-scale 20 \
--wait-a 5 \
--wait-scale 20 \
--seed 0
```

#### Get Random Start
```
python ./sky_spot/traces/random_start.py```
