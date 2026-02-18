"""Cache management for benchmark simulations - self-contained module."""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)

# Check if we should use GCS backend
USE_GCS = os.environ.get('USE_GCS_CACHE', '').lower() == 'true'
GCS_BUCKET_NAME = 'skypilot-benchmark-results'

if USE_GCS:
    try:
        from google.cloud import storage as gcs_storage
        _gcs_client = gcs_storage.Client()
        _bucket = _gcs_client.bucket(GCS_BUCKET_NAME)
        logger.info(f"Using GCS cache backend: {GCS_BUCKET_NAME}")
    except Exception as e:
        logger.warning(f"Failed to initialize GCS client: {e}, falling back to local cache")
        USE_GCS = False
        _bucket = None
else:
    _bucket = None


def generate_cache_filename(
    strategy: str, 
    env: str, 
    traces: List[str], 
    checkpoint_size: float = 50.0, 
    restart_overhead: float = 0.2,
    deadline_hours: Optional[float] = None
) -> str:
    """Generates a cache filename, using hash if too long."""
    trace_descs = []
    for trace_path_str in traces:
        trace_path = Path(trace_path_str)
        if "union" in trace_path.name:
            # For union traces, the name itself is descriptive
            trace_descs.append(trace_path.stem)
        else:
            trace_descs.append(f"{trace_path.parent.name}_{trace_path.stem}")

    trace_identifier = "+".join(sorted(trace_descs))
    safe_strategy_name = strategy.replace("/", "_").replace("\\", "_")
    
    # Include deadline in filename if specified
    if deadline_hours is not None:
        filename = f"{safe_strategy_name}_{env}_{trace_identifier}_ckpt{checkpoint_size}gb_ro{restart_overhead}h_ddl{deadline_hours}h.json"
    else:
        filename = f"{safe_strategy_name}_{env}_{trace_identifier}_ckpt{checkpoint_size}gb_ro{restart_overhead}h.json"
    
    # If filename is too long, use hash
    if len(filename) > 200:  # Conservative limit to avoid filesystem issues
        # Include deadline in hash calculation
        hash_input = f"{strategy}_{env}_{trace_identifier}_ckpt{checkpoint_size}gb_ro{restart_overhead}h"
        if deadline_hours is not None:
            hash_input += f"_ddl{deadline_hours}h"
        content_hash = hashlib.md5(hash_input.encode()).hexdigest()
        # Keep strategy name but shorten traces part
        short_strategy = safe_strategy_name[-30:] if len(safe_strategy_name) > 30 else safe_strategy_name
        filename = f"{short_strategy}_{env}_{content_hash}.json"
    
    return filename


def load_from_cache(cache_file: Path) -> Optional[float]:
    """Load cost from cache file if it exists."""
    if USE_GCS and _bucket is not None:
        # GCS backend
        try:
            blob_name = f"cache/{cache_file.name}"
            blob = _bucket.blob(blob_name)
            
            if blob.exists():
                content = blob.download_as_text()
                data = json.loads(content)
                logger.debug(f"GCS cache HIT: {blob_name}")
                return data.get("mean_cost")
            else:
                logger.debug(f"GCS cache MISS: {blob_name}")
                return None
        except Exception as e:
            logger.warning(f"Failed to load from GCS cache {cache_file.name}: {e}")
            return None
    else:
        # Local file backend
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                    return data.get("mean_cost")
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_file}: {e}")
        return None


def save_to_cache(cache_file: Path, mean_cost: float) -> None:
    """Save cost to cache file."""
    try:
        data = {"mean_cost": mean_cost}
        
        if USE_GCS and _bucket is not None:
            # GCS backend
            blob_name = f"cache/{cache_file.name}"
            blob = _bucket.blob(blob_name)
            blob.upload_from_string(
                json.dumps(data),
                content_type='application/json'
            )
            logger.debug(f"Saved to GCS cache: {blob_name}")
        else:
            # Local file backend
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w") as f:
                json.dump(data, f)
    except Exception as e:
        logger.warning(f"Failed to save cache {cache_file}: {e}")