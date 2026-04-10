"""Built-in Terrarium tasks.

Importing this package registers all bundled tasks with the registry.
"""

from terrarium.tasks import aime_math, arc_agi, circle_packing, optuna_blackbox

__all__ = ["circle_packing", "optuna_blackbox", "aime_math", "arc_agi"]
