"""SkyPilot-based parallel execution module for benchmarks using Python API."""

import os
import time
import json
import logging
import tempfile
import shutil
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import sky

logger = logging.getLogger(__name__)


def check_skypilot_api_compatibility():
    """Check SkyPilot API compatibility before launching tasks."""
    try:
        sky.status()
        return True
    except Exception as e:
        if "version mismatch" in str(e).lower():
            logger.error("🚨 SkyPilot API version mismatch detected!")
            logger.error(str(e))
            logger.info("💡 Fix: sky api stop; sky api start")
            return False
        logger.warning(f"⚠️  SkyPilot API issue (proceeding anyway): {e}")
        return True


class SkyPilotExecutor:
    """Manages parallel execution of simulation tasks using SkyPilot Python API."""
    
    def __init__(self, max_parallel_clusters: int = 10, cloud: Optional[str] = 'gcp', instance_type: Optional[str] = None, auto_down: bool = True):
        self.max_parallel_clusters = max_parallel_clusters
        self.cloud = cloud  # None means auto-select
        self.instance_type = instance_type  # None means auto-select
        self.auto_down = auto_down  # Whether to auto-terminate clusters
        self.active_clusters = []
        self.temp_dirs = []  # Track temp directories for cleanup
        
    def create_batch_workdir(self, tasks: List[Dict], params: Dict, batch_id: str) -> str:
        """Create a working directory with necessary task and parameter files."""
        
        # Create temporary directory
        workdir = tempfile.mkdtemp(prefix=f"skypilot_batch_{batch_id}_")
        self.temp_dirs.append(workdir)
        
        # Add task indices to preserve order
        for i, task in enumerate(tasks):
            task['_task_index'] = task.get('_task_index', i)  # Preserve global index if exists
        
        # Save tasks file
        tasks_file = os.path.join(workdir, "tasks.json")
        with open(tasks_file, 'w') as f:
            json.dump(tasks, f)
        
        # Save params file
        params_file = os.path.join(workdir, "params.json")
        with open(params_file, 'w') as f:
            json.dump(params, f)
        
        return workdir
    
    def execute_batch(self, tasks: List[Dict], cache_dir: Path, params: Dict) -> List[Dict]:
        """Execute a batch of tasks in parallel using SkyPilot Python API."""
        
        # Pre-check: Verify SkyPilot API compatibility
        if not check_skypilot_api_compatibility():
            logger.error("❌ SkyPilot API compatibility check failed. Please fix the version mismatch and try again.")
            # Return failed results for all tasks
            return [{**task, "cost": float('nan'), "error": "SkyPilot API version mismatch"} for task in tasks]
        
        # Add global indices to all tasks to preserve order
        for i, task in enumerate(tasks):
            task['_global_index'] = i
            
        results = []
        
        # Calculate optimal batch size based on tasks per cluster
        tasks_per_cluster = 50  # Increased for better instance utilization
        num_clusters = min(
            self.max_parallel_clusters,
            (len(tasks) + tasks_per_cluster - 1) // tasks_per_cluster
        )
        
        # Distribute tasks across clusters
        task_batches = []
        for i in range(num_clusters):
            start_idx = i * len(tasks) // num_clusters
            end_idx = (i + 1) * len(tasks) // num_clusters
            if start_idx < end_idx:
                task_batches.append(tasks[start_idx:end_idx])
        
        logger.info(f"Distributing {len(tasks)} tasks across {len(task_batches)} clusters")
        
        # Phase 1: Launch all clusters in parallel (non-blocking)
        logger.info("📤 Phase 1: Launching all clusters in parallel...")
        cluster_info = []  # Store (batch_idx, batch_tasks, cluster_name, job_id)
        
        with ThreadPoolExecutor(max_workers=min(len(task_batches), 10)) as executor:
            launch_futures = []
            for batch_idx, batch_tasks in enumerate(task_batches):
                future = executor.submit(
                    self._launch_cluster,  # Only launch, don't wait
                    batch_idx,
                    batch_tasks,
                    cache_dir,
                    params
                )
                launch_futures.append((batch_idx, batch_tasks, future))
            
            # Collect launch results
            for batch_idx, batch_tasks, future in launch_futures:
                try:
                    cluster_name, job_id, run_id = future.result()
                    cluster_info.append((batch_idx, batch_tasks, cluster_name, job_id, run_id))
                    logger.info(f"✅ Launched cluster {cluster_name} with job {job_id}")
                except Exception as e:
                    logger.error(f"❌ Failed to launch batch {batch_idx}: {e}")
                    results.extend([{**task, "cost": float('nan'), "error": str(e)} for task in batch_tasks])
        
        # Phase 2: Wait for all jobs to complete in parallel
        logger.info("⏳ Phase 2: Waiting for all jobs to complete...")
        wait_futures = []
        with ThreadPoolExecutor(max_workers=min(len(cluster_info), 10)) as executor:
            for batch_idx, batch_tasks, cluster_name, job_id, run_id in cluster_info:
                future = executor.submit(
                    self._wait_and_download,  # Wait and download results
                    batch_idx,
                    batch_tasks,
                    cluster_name,
                    job_id,
                    run_id,
                    cache_dir,
                    params
                )
                wait_futures.append(future)
            
            # Collect results from all clusters
            for future in as_completed(wait_futures):
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                except Exception as e:
                    logger.error(f"Cluster batch failed: {e}")
        
        # Sort results by global index to maintain order
        results.sort(key=lambda x: x.get('_global_index', 0))
        
        # Remove the temporary index field
        for result in results:
            result.pop('_global_index', None)
            result.pop('_task_index', None)
        
        # Cost values should now be scalar floats from the corrected batch_worker
        
        # Summarize failures
        failed_tasks = [r for r in results if pd.isna(r.get('cost', float('nan'))) or 'error' in r]
        if failed_tasks:
            logger.warning(f"⚠️  {len(failed_tasks)}/{len(results)} tasks failed:")
            for task in failed_tasks[:10]:  # Show first 10 failures
                error_msg = task.get('error', 'Unknown error')
                strategy = task.get('strategy', 'unknown')
                trace = task.get('trace_index', 'unknown')
                logger.warning(f"   - Strategy: {strategy}, Trace: {trace}, Error: {error_msg}")
            if len(failed_tasks) > 10:
                logger.warning(f"   ... and {len(failed_tasks) - 10} more failures")
        else:
            logger.info(f"✅ All {len(results)} tasks completed successfully")
        
        return results
    
    def _launch_cluster(self, batch_idx: int, tasks: List[Dict], 
                       cache_dir: Path, params: Dict) -> tuple[str, int, str]:
        """Launch a cluster for a batch of tasks and return immediately."""
        cluster_name = f"benchmark-batch-{batch_idx}"
        workdir = None
        
        try:
            # Create working directory with all files
            workdir = self.create_batch_workdir(tasks, params, str(batch_idx))
            
            # Use a single persistent bucket with unique paths
            base_bucket_name = 'skypilot-benchmark-results'  # Single bucket for caching
            
            # Generate unique run_id for this execution
            run_id = str(int(time.time()))
            
            # Create run script - no need for git clone since we'll sync the code
            output_mount_path = f"/tmp/results_{cluster_name}"
            # Add run_id to the output path for unique results
            results_subdir = f"{run_id}/batch_{batch_idx}"
            run_script = f"""#!/bin/bash
set -e

# Activate skypilot runtime environment
source ~/skypilot-runtime/bin/activate

# Install dependencies using the mounted requirements.txt
pip install -r /tmp/requirements.txt

# Create output directory (for legacy backup, though we'll write to S3 mount)
mkdir -p /tmp/results

# Debug: Check if files are synced
echo "=== Checking synced files ===" 2>&1 | tee -a /tmp/debug.log
echo "Current directory: $(pwd)" 2>&1 | tee -a /tmp/debug.log
echo "Mounted files:" 2>&1 | tee -a /tmp/debug.log
ls -la /tmp/ | grep -E "(tasks|params|scripts_multi|sky_spot|requirements|main)" 2>&1 | tee -a /tmp/debug.log
echo "Scripts multi contents:" 2>&1 | tee -a /tmp/debug.log
ls -la /tmp/scripts_multi/ 2>&1 | tee -a /tmp/debug.log

# Debug: Check GCS mount
echo "=== Checking GCS mount ===" 2>&1 | tee -a /tmp/debug.log
ls -la {output_mount_path}/ 2>&1 | tee -a /tmp/debug.log

# Set PYTHONPATH to include the mounted directories
export PYTHONPATH=/tmp:$PYTHONPATH

# Disable wandb to avoid API key issues
export WANDB_MODE=offline

# Create subdirectory for this run and batch
mkdir -p {output_mount_path}/{results_subdir}

# Run batch processor using the mounted files - write directly to S3 mount
cd /tmp
python scripts_multi/benchmark_components/batch_worker.py \\
  --tasks-file /tmp/tasks.json \\
  --params-json /tmp/params.json \\
  --output-dir {output_mount_path}/{results_subdir} 2>&1 | tee /tmp/batch_worker.log
  
# Debug: Check if results were written
echo "=== Checking results after batch processing ===" 2>&1 | tee -a /tmp/debug.log
ls -la {output_mount_path}/ 2>&1 | tee -a /tmp/debug.log
"""
            
            # Create SkyPilot task without workdir (will use file_mounts instead)
            task = sky.Task(
                name=f'benchmark-batch-{batch_idx}',
                run=run_script
            )
            
            # Set file mounts
            # Create results directory in workdir  
            results_dir = os.path.join(workdir, 'results')
            os.makedirs(results_dir, exist_ok=True)
            
            # Set file mounts for task and param files
            project_root = Path(__file__).parent.parent.parent
            
            # Use the same persistent bucket for all runs (for caching)
            s3_bucket_name = base_bucket_name
            
            file_mounts: dict[str, Any] = {
                '/tmp/tasks.json': os.path.join(workdir, 'tasks.json'),
                '/tmp/params.json': os.path.join(workdir, 'params.json'),
                # Mount specific directories needed for the task
                '/tmp/scripts_multi': str(project_root / 'scripts_multi'),
                '/tmp/sky_spot': str(project_root / 'sky_spot'),
                '/tmp/requirements.txt': str(project_root / 'requirements.txt'),
                '/tmp/main.py': str(project_root / 'main.py')
            }
            
            # Mount the entire data directory for multi-region tasks
            data_dir = project_root / 'data' / 'converted_multi_region_aligned'
            if data_dir.exists():
                file_mounts['/tmp/data/converted_multi_region_aligned'] = str(data_dir)
            
            task.set_file_mounts(file_mounts)
            
            # Set up S3 storage using storage_mounts
            storage = sky.Storage(
                name=s3_bucket_name,
                source=None,  # Empty bucket
                mode=sky.StorageMode.MOUNT
            )
            task.set_storage_mounts({output_mount_path: storage})
            
            # Set resources - force GCP
            resources_kwargs = {
                'cpus': '16+', 
                'memory': '32+',
                'cloud': sky.GCP()  # Always use GCP
            }
                    
            if self.instance_type:
                resources_kwargs['instance_type'] = self.instance_type
                
            task.set_resources(sky.Resources(**resources_kwargs))  # type: ignore
            
            # Launch cluster
            logger.info(f"🚀 Launching cluster {cluster_name} with {len(tasks)} tasks")
            logger.info(f"📍 Cloud: GCP")
            cpu_info = resources_kwargs.get('cpus', '16+')
            mem_info = resources_kwargs.get('memory', '32+')
            logger.info(f"📊 Resources: CPUs {cpu_info}, Memory {mem_info}GB")
            
            # Launch and get job info
            logger.info(f"⏳ Starting cluster provisioning...")
            launch_start = time.time()
            request_id = sky.launch(task, cluster_name=cluster_name)
            self.active_clusters.append(cluster_name)
            logger.info(f"✅ Launch command sent, request_id: {request_id}")
            
            # Stream and get the job_id 
            job_id, handle = sky.stream_and_get(request_id)
            logger.info(f"✅ Job launched with job_id: {job_id}")
            
            return cluster_name, job_id, run_id
            
        except Exception as e:
            logger.error(f"Failed to launch batch {batch_idx}: {e}")
            raise e
            
        finally:
            # Clean up workdir only
            if workdir and os.path.exists(workdir):
                shutil.rmtree(workdir)
    
    def _wait_and_download(self, batch_idx: int, tasks: List[Dict], 
                          cluster_name: str, job_id: int, run_id: str,
                          cache_dir: Path, params: Dict) -> List[Dict]:
        """Wait for job completion and download results."""
        try:
            # Wait for job completion by tailing logs synchronously
            logger.info(f"⏳ Waiting for job {job_id} completion on {cluster_name}...")
            
            try:
                # Tail logs synchronously - this blocks until job completes
                sky.tail_logs(cluster_name, job_id, follow=True)
                logger.info(f"✅ Job {job_id} completed successfully")
            except Exception as e:
                logger.warning(f"⚠️  Error tailing logs for job {job_id}: {e}, but job may have completed")
            
            # Add buffer time for GCS file sync after job completion
            logger.info(f"📝 Adding buffer time for GCS results sync...")
            time.sleep(60)  # Reduced to 60 seconds since job is actually complete
            logger.info(f"✅ Buffer time completed for job {job_id}")
            
            # Use the same persistent bucket for all runs
            s3_bucket_name = 'skypilot-benchmark-results'
            
            # Download results from GCS bucket after job completes
            results = self._download_results_from_s3(s3_bucket_name, tasks, run_id, batch_idx)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to wait/download batch {batch_idx}: {e}")
            return [{**task, "cost": float('nan'), "error": str(e)} for task in tasks]
            
        finally:
            if self.auto_down:
                self._terminate_cluster(cluster_name)
            else:
                logger.info(f"⚠️  Cluster {cluster_name} is still running for debugging. Run 'sky down {cluster_name}' to terminate it.")
    
    def _download_results_from_s3(self, s3_bucket_name: str, tasks: List[Dict], run_id: str, batch_idx: int) -> List[Dict]:
        """Download results from GCS bucket.
        
        Note: Function name still says 's3' for backward compatibility,
        but now only uses GCS.
        """
        try:
            from google.cloud import storage as gcs_storage
            gcs_client = gcs_storage.Client()
            bucket_name = s3_bucket_name  # Use the same bucket name
            
            logger.info(f"📥 Downloading results from GCS bucket: {bucket_name}")
            
            # Create local temp directory for results
            local_results_dir = tempfile.mkdtemp(prefix=f"results_{bucket_name}_")
            self.temp_dirs.append(local_results_dir)
            
            # GCS logic
            try:
                bucket = gcs_client.bucket(bucket_name)
                blobs = list(bucket.list_blobs(prefix=f"{run_id}/", max_results=10))
                if blobs:
                    logger.info(f"🔍 Found {len(blobs)} objects in GCS bucket under {run_id}/")
                    for blob in blobs[:5]:  # Show first 5
                        logger.info(f"   - {blob.name} ({blob.size} bytes)")
                else:
                    logger.warning(f"⚠️  No objects found under {run_id}/ in GCS bucket {bucket_name}")
            except Exception as e:
                logger.error(f"❌ Failed to list GCS bucket contents: {e}")
                return [{**task, "cost": float('nan'), "error": f"GCS bucket access error: {e}"} for task in tasks]

            # Download batch_summary.json first
            summary_key = f'{run_id}/batch_{batch_idx}/batch_summary.json'
            summary_file = os.path.join(local_results_dir, 'batch_summary.json')
            
            try:
                bucket = gcs_client.bucket(bucket_name)
                blob = bucket.blob(summary_key)
                blob.download_to_filename(summary_file)
                logger.info(f"✅ Downloaded batch summary from GCS: {bucket_name}/{summary_key}")
                
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                    results = summary.get('results', [])
                    logger.info(f"✅ Retrieved {len(results)} results from batch summary")
                    return results
            
            except Exception as e:
                if 'NoSuchKey' in str(e) or 'NotFound' in str(e):
                    logger.warning(f"No batch_summary.json found in bucket {bucket_name}, trying individual files")
                else:
                    logger.warning(f"Error downloading batch_summary.json: {e}, trying individual files")
                
                # Fallback: download individual result files
                results = []
                for i in range(len(tasks)):
                    # Adjust key to include run_id path
                    result_key = f'{run_id}/batch_{batch_idx}/result_{i}.json'
                    result_file = os.path.join(local_results_dir, f'result_{i}.json')
                    
                    try:
                        bucket = gcs_client.bucket(bucket_name)
                        blob = bucket.blob(result_key)
                        blob.download_to_filename(result_file)
                        
                        with open(result_file, 'r') as f:
                            results.append(json.load(f))
                        logger.debug(f"Downloaded individual result: {result_key}")
                    except Exception as file_error:
                        if 'NotFound' in str(file_error) or '404' in str(file_error):
                            logger.warning(f"Missing result file: {result_key}")
                        else:
                            logger.warning(f"Error downloading {result_key}: {file_error}")
                        results.append({**tasks[i], "cost": float('nan'), "error": "Result file not found"})
                
                if results:
                    # Cost values should be scalar from corrected batch_worker
                    
                    valid_results = [r for r in results if not pd.isna(r.get('cost', float('nan')))]
                    logger.info(f"✅ Retrieved {len(valid_results)}/{len(results)} valid results from individual files")
                    return results
                else:
                    logger.error(f"No results found in bucket {bucket_name}")
                    return [{**task, "cost": float('nan'), "error": "No results found"} for task in tasks]
        
        except Exception as e:
            logger.error(f"Error downloading from bucket {bucket_name}: {e}")
            return [{**task, "cost": float('nan'), "error": f"Download error: {e}"} for task in tasks]
    
    def _terminate_cluster(self, cluster_name: str):
        """Terminate a SkyPilot cluster using Python API."""
        try:
            # sky.down returns a request ID
            down_request_id = sky.down(cluster_name)
            try:
                sky.get(down_request_id)  # Wait for completion
            except Exception as e:
                logger.warning(f"Error during cluster termination: {e}")
            
            if cluster_name in self.active_clusters:
                self.active_clusters.remove(cluster_name)
            logger.info(f"Terminated cluster {cluster_name}")
        except Exception as e:
            logger.error(f"Failed to terminate cluster {cluster_name}: {e}")
    
    def cleanup(self):
        """Clean up all active clusters and temporary directories."""
        # Clean up clusters
        for cluster_name in self.active_clusters[:]:
            if self.auto_down:
                self._terminate_cluster(cluster_name)
            else:
                logger.info(f"⚠️  Cluster {cluster_name} is still running for debugging. Run 'sky down {cluster_name}' to terminate it.")
            
        # Clean up temporary directories
        for temp_dir in self.temp_dirs:
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            except Exception as e:
                logger.error(f"Failed to clean up temp dir {temp_dir}: {e}")


def execute_tasks_with_skypilot(
    tasks: List[Dict], 
    cache_dir: Path, 
    params: Dict,
    auto_down: bool = True
) -> List[Dict]:
    """Execute tasks in parallel using SkyPilot on GCP."""
    
    # Enable GCS cache for SkyPilot execution
    os.environ['USE_GCS_CACHE'] = 'true'
    
    # Use GCP as the default cloud provider
    executor = SkyPilotExecutor(
        max_parallel_clusters=20,  # Increased for faster execution
        cloud='gcp',  # Always use GCP
        instance_type=None,  # Auto-select based on requirements
        auto_down=auto_down
    )
    
    try:
        results = executor.execute_batch(tasks, cache_dir, params)
        return results
    finally:
        executor.cleanup()