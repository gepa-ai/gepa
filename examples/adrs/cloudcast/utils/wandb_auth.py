"""Check for Weights & Biases credentials."""

import os
from netrc import NetrcParseError, netrc


def has_wandb_credentials() -> bool:
    """Return True if W&B credentials are available via env var or ~/.netrc."""
    if os.environ.get("WANDB_API_KEY"):
        return True
    return _netrc_has_wandb_entry()


def _netrc_has_wandb_entry() -> bool:
    netrc_path = os.path.expanduser("~/.netrc")
    if not os.path.exists(netrc_path):
        return False
    try:
        nrc = netrc(netrc_path)
    except (NetrcParseError, OSError):
        return False
    for host in ("api.wandb.ai", "wandb.ai"):
        auth = nrc.authenticators(host)
        if auth is not None and auth[2]:  # auth[2] is the password/token
            return True
    return False
