"""Load Cloudcast configuration files as GEPA dataset samples."""

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Config files that define the broadcast scenarios
_CONFIG_FILES = [
    "intra_aws.json",
    "intra_azure.json",
    "intra_gcp.json",
    "inter_agz.json",
    "inter_gaz2.json",
]


def load_config_dataset(
    config_dir: str | Path,
    num_vms: int = 2,
) -> list[dict[str, Any]]:
    """Return a list of dataset samples, one per configuration file.

    Each sample is a dict with keys ``config_file`` and ``num_vms``.

    Args:
        config_dir: Directory that contains the JSON configuration files.
        num_vms: Number of VMs per cloud region used during simulation.
    """
    config_dir = Path(config_dir)
    samples: list[dict[str, Any]] = []
    for filename in _CONFIG_FILES:
        path = config_dir / filename
        if path.exists():
            samples.append({"config_file": str(path), "num_vms": num_vms})
        else:
            logger.warning(f"Config file not found, skipping: {path}")
    return samples
