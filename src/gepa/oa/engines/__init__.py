"""Built-in engines. Importing this package registers each engine by name."""

from gepa.oa.engines.autoresearch import AutoResearchEngine
from gepa.oa.engines.best_of_n import BestOfNEngine
from gepa.oa.engines.gepa import GepaEngine
from gepa.oa.engines.meta_harness import MetaHarnessEngine
from gepa.oa.registry import register_engine

register_engine("gepa", GepaEngine)
register_engine("autoresearch", AutoResearchEngine)
register_engine("meta_harness", MetaHarnessEngine)
register_engine("best_of_n", BestOfNEngine)

__all__ = ["AutoResearchEngine", "BestOfNEngine", "GepaEngine", "MetaHarnessEngine"]
