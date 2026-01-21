
import configargparse
import logging
import os
import sys
import yaml
try:
    import wandb as _wandb_module
except ModuleNotFoundError:  # pragma: no cover
    class _WandbStub:
        run = None

        @staticmethod
        def init(*args, **kwargs):
            return None

    wandb = _WandbStub()
else:
    wandb = _wandb_module
if not hasattr(wandb, "run"):
    wandb.run = None
import re
from collections import defaultdict
from typing import Sequence, Type
import importlib.util
from colorama import init, Fore, Style

# Initialize colorama for cross-platform color support
init(autoreset=True)

from sky_spot import env as env_lib
from sky_spot.env import MultiTraceEnv, TraceEnv 
from sky_spot import simulate
from sky_spot.strategies import strategy as strategy_lib
from sky_spot.task import SingleTask, ChainedTask, Task

if os.environ.get('ENABLE_WANDB', '0') == '1':
    wandb.init(project='sky-spot')
logger = logging.getLogger(__name__)
    
def load_strategy_from_file(file_path: str) -> Type[strategy_lib.Strategy]:
    """Dynamically loads a strategy class from a Python file."""
    try:
        module_name = os.path.splitext(os.path.basename(file_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load spec for module from {file_path}")
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find all strategy classes, but exclude base classes
        found_classes = []
        for attr in dir(module):
            obj = getattr(module, attr)
            if isinstance(obj, type) and issubclass(obj, strategy_lib.Strategy) and obj is not strategy_lib.Strategy:
                # Also exclude MultiRegionStrategy base class
                if obj.__name__ not in ['Strategy', 'MultiRegionStrategy']:
                    found_classes.append(obj)
                    logger.info(f"Found strategy class '{obj.__name__}' in {file_path}")
        
        if found_classes:
            # Return the first non-base strategy class found
            return found_classes[0]

        raise AttributeError(f"No concrete strategy class found in {file_path}")
    except Exception as e:
        logger.error(f"Failed to load strategy from {file_path}: {e}")
        raise


def find_indexed_traces(dir_path):
    indexed_files = {}
    if not os.path.isdir(dir_path):
        logger.warning(f"Path is not a directory: {dir_path}")
        return indexed_files
    try:
        for filename in os.listdir(dir_path):
            if filename.endswith('.json'):
                match = re.match(r"(\d+)\.json", filename)
                if match:
                    index = int(match.group(1))
                    indexed_files[index] = os.path.join(dir_path, filename)
    except OSError as e:
        logger.error(f"Error reading directory {dir_path}: {e}")
    return indexed_files


if __name__ == '__main__':
    root_logger = logging.getLogger('sky_spot')

    def setup_logger():
        logging_level = os.environ.get('LOG_LEVEL', 'DEBUG')
        handler = logging.StreamHandler(sys.stdout)
        
        class SimpleColorFormatter(logging.Formatter):
            def format(self, record):
                # Get the last part of the module name
                name_parts = record.name.split('.')
                short_name = name_parts[-1] if name_parts else record.name
                
                # Apply colors based on level
                if record.levelname == 'ERROR':
                    color = Fore.RED
                elif record.levelname == 'WARNING':
                    color = Fore.YELLOW
                elif record.levelname == 'DEBUG':
                    color = Fore.LIGHTBLACK_EX  # Gray
                else:
                    color = ''  # No color for INFO
                
                return f"{color}[{short_name}] {record.getMessage()}{Style.RESET_ALL}"
        
        handler.setFormatter(SimpleColorFormatter())
        handler.setLevel(logging_level)
        root_logger.setLevel(logging_level)
        root_logger.addHandler(handler)

    setup_logger()

    parser = configargparse.ArgumentParser('Skypilot spot simulator')

    parser.add_argument('--scenarios-config',
                        type=str,
                        help='Optional YAML file defining simulation scenarios.')
    parser.add_argument('--run-scenarios',
                        type=str,
                        nargs='+',
                        help='Names of scenarios to run from the YAML config file.')

    parser.add_argument('--config',
                        type=str,
                        default=None,
                        is_config_file=True,
                        required=False)
    group = parser.add_argument_group('Global options')
    group.add_argument('--deadline-hours',
                       type=float,
                       default=10,
                       help='Deadline of the task in hours')
    group.add_argument(
        '--task-duration-hours',
        type=float,
        nargs='+',
        default=[10],
        help=
        'Duration(s) of task(s) in hours. For chained tasks, provide multiple values.'
    )
    group.add_argument(
        '--restart-overhead-hours',
        type=float,
        nargs='+',
        default=[0.2],
        help=
        'Overhead(s) of restarting tasks in hours. Provide multiple values for different tasks.'
    )
    group.add_argument(
        '--checkpoint-size-gb',
        type=float,
        default=50.0,
        help='Size of the checkpoint in GB for calculating migration times (default: 50GB)'
    )
    group.add_argument('--output-dir',
                       type=str,
                       default='exp/',
                       help='Output directory')
    
    # --- MODIFIED PART: Define strategy args at the top level ---
    strategy_group = parser.add_argument_group('Strategy Selection')
    strategy_group.add_argument('--strategy-file',
                       type=str,
                       default=None,
                       help='Path to a Python file defining the strategy to use.')
    strategy_group.add_argument('--strategy',
                            type=str,
                            default='strawman',
                            choices=strategy_lib.Strategy.SUBCLASSES.keys(),
                            help='Name of the built-in strategy to use.')
    # --- END MODIFIED PART ---

    args, _ = parser.parse_known_args()

    # This logic block is for when you run with a YAML scenario config.
    if args.scenarios_config:
        # (This entire 'if' block remains unchanged)
        logger.info(f"Loading scenarios from YAML: {args.scenarios_config}")
        with open(args.scenarios_config, 'r', encoding='utf-8') as f:
            all_scenarios = yaml.safe_load(f)

        scenarios_to_run_names = (
            args.run_scenarios
            if args.run_scenarios
            else list(all_scenarios.keys())
        )
        assert len(scenarios_to_run_names) > 0, "No scenarios to run"
        logger.info(f"Planning to run scenarios: {scenarios_to_run_names}")

        scenario_name = scenarios_to_run_names[0]
        logger.info(f"Running scenario: {scenario_name}")
        scenario_spec = all_scenarios[scenario_name]

        wandb.init(project='sky-spot', name=scenario_name, reinit=True)

        task_spec = scenario_spec.get('task', {})
        task_type_name = task_spec.get('type')
        task_config = task_spec.get('config', {})

        if task_type_name == 'SingleTask':
            current_task = SingleTask(config=task_config)
        elif task_type_name == 'ChainedTask':
            current_task = ChainedTask(config=task_config)
        else:
            raise ValueError(f"Unknown task type: {task_type_name}")
        logger.info(f"Task: {current_task}")

        env_spec = scenario_spec.get('env', {})
        env_type_name = env_spec.get('type')
        default_env_start_hours = getattr(args, 'env_start_hours', 0.0)

        envs: Sequence[env_lib.Env] = []
        trace_param: str

        if env_type_name == 'subtask_multi_env_switcher':
            # This logic remains unchanged
            logger.info("Processing 'subtask_multi_env_switcher' environment type.")
            sub_task_env_configs = env_spec.get('sub_task_envs')
            assert isinstance(sub_task_env_configs, list), (
                f"Scenario '{scenario_name}' must have a list in 'sub_task_envs'."
            )
            num_sub_tasks = len(sub_task_env_configs)
            assert num_sub_tasks > 0, "'sub_task_envs' list cannot be empty."
            
            subtask_indexed_files: list[dict[int, list[str]]] = []
            common_indices = None
            logger.info(
                f"Processing scenario '{scenario_name}': Finding corresponding trace files for {num_sub_tasks} sub-tasks..."
            )
            for i, sub_cfg in enumerate(sub_task_env_configs):
                trace_paths = sub_cfg.get('trace_files', [])
                if not trace_paths:
                     raise ValueError(f"Sub-task env config {i} must contain 'trace_files'.")
                
                indexed_files_per_region = [] 
                for path in trace_paths:
                     indexed_files_in_path = find_indexed_traces(path)
                     indexed_files_per_region.append(indexed_files_in_path)
                
                current_subtask_indices = set(indexed_files_per_region[0].keys()) if indexed_files_per_region else set()
                for region_files in indexed_files_per_region[1:]:
                    current_subtask_indices.intersection_update(region_files.keys())
                    
                if not current_subtask_indices:
                     raise ValueError(f"No common trace indices found across regions for sub-task {i}. Check paths: {trace_paths}")
                
                aligned_files_for_subtask = defaultdict(list)
                for idx in sorted(list(current_subtask_indices)):
                    for region_files in indexed_files_per_region:
                        if idx in region_files:
                             aligned_files_for_subtask[idx].append(region_files[idx])
                        else: 
                             raise RuntimeError(f"Logic error: Index {idx} not found in region files after intersection.")
                subtask_indexed_files.append(aligned_files_for_subtask)
                
                if common_indices is None:
                    common_indices = current_subtask_indices
                else:
                    common_indices.intersection_update(current_subtask_indices)

            if common_indices is None or not common_indices:
                raise ValueError("No common trace indices found across *all* sub-tasks.")
                
            num_runs = len(common_indices)
            print(f"--> Found {num_runs} matching trace file sets across all sub-tasks.")
            print(f"--> Creating {num_runs} simulation environment instance(s)...")
            
            from sky_spot.env import SubtaskMultiEnvSwitcher # Import here
            final_common_indices = sorted(list(common_indices))
            
            for i in range(num_runs):
                run_idx = final_common_indices[i]
                sub_envs_for_this_run = []
                for k in range(num_sub_tasks):
                     files_for_subtask_run = subtask_indexed_files[k][run_idx]
                     sub_cfg = sub_task_env_configs[k]
                     env_start_hours = sub_cfg.get('env_start_hours', default_env_start_hours)
                     sub_env = MultiTraceEnv(trace_files=files_for_subtask_run, env_start_hours=env_start_hours)
                     sub_envs_for_this_run.append(sub_env)
                     
                switcher = SubtaskMultiEnvSwitcher(sub_environments=sub_envs_for_this_run)
                envs.append(switcher) 
            
            print(f"--> Finished creating {len(envs)} environment instance(s).")
            trace_param = f"subtask_switcher_{num_sub_tasks}subs_{num_runs}runs"
        elif env_type_name == 'multi_trace':
            env_trace_files = env_spec.get('trace_files', [])
            env_start_hours = env_spec.get('env_start_hours', default_env_start_hours)
            envs = MultiTraceEnv.create_env(trace_files_or_dirs=env_trace_files, env_start_hours=env_start_hours)
            trace_param = f"multi_region_{','.join(os.path.basename(f) for f in env_trace_files)}"
        elif env_type_name == 'trace':
            env_trace_files = env_spec.get('trace_files', [])
            env_start_hours = env_spec.get('env_start_hours', default_env_start_hours)
            if len(env_trace_files) != 1:
                 raise ValueError(f"Scenario '{scenario_name}' with env type 'trace' requires exactly one path.")
            envs = TraceEnv.create_env(trace_file_or_dir=env_trace_files[0], env_start_hours=env_start_hours)
            trace_param = os.path.basename(env_trace_files[0])
        else:
            raise ValueError(f"Unknown env type in scenario: {env_type_name}")
        
        assert envs, f"Failed to create environment(s) for scenario '{scenario_name}'."
        logger.info(f"Environment(s) created: {envs}")

        if env_type_name != 'subtask_multi_env_switcher':
            first_env_config = envs[0].config
            if 'trace_files' in first_env_config:
                trace_param = (
                    f"multi_region_{','.join(os.path.basename(f) for f in first_env_config['trace_files'])}"
                )
            elif 'trace_file' in first_env_config:
                trace_param = os.path.basename(first_env_config['trace_file'])
            else:
                trace_param = f"{env_type_name}_unknown_trace"

        deadline = scenario_spec.get('deadline_hours', args.deadline_hours)
        restart_overhead = scenario_spec.get('restart_overhead_hours', args.restart_overhead_hours)
        scenario_output_dir = os.path.join(args.output_dir, scenario_name)

        args.deadline_hours = deadline
        args.restart_overhead_hours = restart_overhead
        args.env_start_hours = (
            envs[0].config.get('default_env_start_hours', default_env_start_hours)
            if env_type_name == 'subtask_multi_env_switcher'
            else env_start_hours
        )
        args.output_dir = scenario_output_dir

        parser.set_defaults(deadline_hours=deadline,
                            restart_overhead_hours=restart_overhead)

        if args.strategy_file:
            logger.info(f"Dynamically loading strategy from: {args.strategy_file}")
            StrategyClass = load_strategy_from_file(args.strategy_file)
        else:
            logger.info(f"Loading built-in strategy: {args.strategy}")
            StrategyClass = strategy_lib.Strategy.get(args.strategy)
        
        strategy = StrategyClass._from_args(parser)
        args = strategy.args
        logger.info(f"Strategy: {strategy.name}")

    else:
        # This logic block is for when you run with CLI arguments directly.
        envs = env_lib.Env.from_args(parser)
        
        if args.strategy_file:
            logger.info(f"Dynamically loading strategy from: {args.strategy_file}")
            StrategyClass = load_strategy_from_file(args.strategy_file)
        else:
            logger.info(f"Loading built-in strategy: {args.strategy}")
            StrategyClass = strategy_lib.Strategy.get(args.strategy)
        
        strategy = StrategyClass._from_args(parser)
        args = strategy.args

        assert envs, "Env.from_args did not return any environments based on CLI arguments."
        if len(args.task_duration_hours) == 1:
            current_task = SingleTask(config={
                'duration': args.task_duration_hours[0],
                'checkpoint_size_gb': args.checkpoint_size_gb
            })
        else:
            sub_tasks_config = [{'duration': duration} for duration in args.task_duration_hours]
            current_task = ChainedTask(config={
                'sub_tasks': sub_tasks_config,
                'checkpoint_size_gb': args.checkpoint_size_gb
            })
        logger.info(f"Task: {current_task}")

        trace_param: str
        if hasattr(args, 'trace_file') and args.trace_file:
            trace_param = args.trace_file
            logger.info(f"Using trace_file: {trace_param}")
        elif hasattr(args, 'trace_files') and args.trace_files:
            trace_files_str = ",".join([os.path.basename(f) for f in args.trace_files])
            trace_param = f"multi_region_{trace_files_str}"
            logger.info(f"Using multiple trace files: {args.trace_files}")
            logger.info(f"Combined name: {trace_param}")
        else:
            trace_param = "unknown_cli_trace"


    final_env_start_hours = getattr(args, 'env_start_hours', 0.0)
    assert 'current_task' in locals(), "current_task was not defined before simulate call."

    print(f"Starting simulation with {len(envs)} environment instance(s)...")
    simulate.simulate(envs=envs,
                      strategy=strategy,
                      task=current_task,
                      trace_file=trace_param,
                      deadline_hours=strategy.args.deadline_hours,
                      restart_overhead_hours=strategy.args.restart_overhead_hours,
                      env_start_hours=final_env_start_hours,
                      output_dir=args.output_dir,
                      kwargs=vars(strategy.args))
