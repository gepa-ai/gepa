"""
Phase II reproduction notes for GEPA #97.

Goal:
Inspect how DspyAdapter.make_reflective_dataset builds Program Trace entries
from eval_batch.trajectories and confirm that it includes multiple trace entries
instead of only the final/full trajectory.
"""

from gepa.adapters.dspy_full_program_adapter.full_program_adapter import DspyAdapter

print("Imported DspyAdapter successfully")
print("Target function: DspyAdapter.make_reflective_dataset")
print("Issue: GEPA #97 - redundant trajectory data in reflective dataset")
