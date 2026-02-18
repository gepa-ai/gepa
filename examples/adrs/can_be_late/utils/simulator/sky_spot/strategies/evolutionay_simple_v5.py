import configargparse
import logging
import typing

from sky_spot.strategies.strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType

if typing.TYPE_CHECKING:  # pragma: no cover
    from sky_spot import env, task  # noqa: F401

logger = logging.getLogger(__name__)


# EVOLVE-BLOCK-START
class EvolutionaryStrategy(MultiRegionStrategy):
    """
    A *lean* adaptive multi-region strategy.

    Design goals
    ------------
    1. Finish before the deadline.
    2. Spend as little money as possible.
       • Prefer SPOT whenever it is available/reliable.
       • Terminate expensive ON_DEMAND capacity first.
    3. Keep the control logic short and readable.
    """

    NAME = "evolutionary_simple_v5"

    # ------------------------------------------------------------------ #
    # Framework life-cycle hooks                                         #
    # ------------------------------------------------------------------ #
    def __init__(self, args: configargparse.Namespace):
        super().__init__(args)
        self._region_cnt: int = 0
        self._next_region: int = 0
        self._consecutive_spot_failures: int = 0

    def reset(self, env: "env.Env", task: "task.Task") -> None:
        super().reset(env, task)
        # Use trace-file count as *hint* for the number of regions;
        # fall back to 2 which is always safe.
        self._region_cnt = len(getattr(env, "trace_files", [])) or 2
        self._next_region = 0
        self._consecutive_spot_failures = 0

        logger.info("%s initialised – %d regions detected.",
                    self.NAME, self._region_cnt)

    # ------------------------------------------------------------------ #
    # Helper utilities                                                   #
    # ------------------------------------------------------------------ #
    def _rr_region(self) -> int:
        """Return the next region (round-robin)."""
        region = self._next_region
        self._next_region = (self._next_region + 1) % self._region_cnt
        return region

    # We add a small tolerance (3 %) so that tiny fluctuations do not
    # trigger costly scaling actions.
    _BEHIND_EPSILON = 0.03

    def _behind_schedule(self) -> bool:
        """Return True iff realised progress < ideal progress – ε."""
        if self.deadline <= 0:
            return False  # Safety guard (shouldn’t happen).

        done = sum(self.task_done_time)
        expected_done = self.task_duration * (
            self.env.elapsed_seconds / self.deadline
        )

        # 1 – ε  ≙ we allow a 3 % lag before we call it “behind”.
        return (done + 1e-6) < expected_done * (1 - self._BEHIND_EPSILON)

    # We only call the situation “urgent” if we are both *behind* and
    # > 75 % through the deadline.  Earlier escalation proved too costly.
    def _urgency(self) -> bool:
        if self.deadline <= 0:
            return False
        return (
            self._behind_schedule()
            and self.env.elapsed_seconds / self.deadline > 0.75
        )

    # ------------------------------------------------------------------ #
    # Core decision generator                                             #
    # ------------------------------------------------------------------ #
    def _step_multi(
        self,
    ) -> typing.Generator[
        typing.Union["TryLaunch", "Terminate"],
        typing.Optional["LaunchResult"],
        None,
    ]:
        """
        Event-driven controller with exactly **one** action per tick.

        Logic (summarised)
        ------------------
        1. Task finished       → terminate everything, stop.
        2. No live instances   → launch (SPOT preferred).
        3. Instances exist
           3a) Behind schedule
               • < 3 instances → launch helper (SPOT preferred).
           3b) On/ahead
               • > 1 instances → terminate surplus
                                   (prefer ON_DEMAND termination!).
        """
        # Local import avoids heavy dependencies during module import time.
        from sky_spot.multi_region_types import Terminate, TryLaunch

        # -------------------------------------------------------------- #
        # 1. Task completed?                                             #
        # -------------------------------------------------------------- #
        if sum(self.task_done_time) >= self.task_duration - 1e-3:
            for r in self.env.get_active_instances():
                yield Terminate(region=r)
            return  # Controller finished.

        # -------------------------------------------------------------- #
        # Snapshot                                                       #
        # -------------------------------------------------------------- #
        active = self.env.get_active_instances()  # {region: ClusterType}
        active_cnt = len(active)

        # -------------------------------------------------------------- #
        # 2. Bootstrap (nothing alive)                                   #
        # -------------------------------------------------------------- #
        if active_cnt == 0:
            prefer_on_demand = (
                self._urgency() and self._consecutive_spot_failures >= 3
            )
            cluster = (
                ClusterType.ON_DEMAND if prefer_on_demand else ClusterType.SPOT
            )
            region = self._rr_region()
            result = yield TryLaunch(region=region, cluster_type=cluster)

            # Simulation may end mid-tick – be defensive.
            if result is None:
                return

            # Track SPOT reliability.
            if cluster is ClusterType.SPOT:
                self._consecutive_spot_failures = (
                    0 if result.success else self._consecutive_spot_failures + 1
                )
            else:
                self._consecutive_spot_failures = 0
            return

        # -------------------------------------------------------------- #
        # 3. Scaling decisions                                           #
        # -------------------------------------------------------------- #
        # 3a) Behind schedule – consider scaling *out*.
        if self._behind_schedule() and active_cnt < 3:
            prefer_on_demand = (
                self._urgency() and self._consecutive_spot_failures >= 3
            )
            cluster = (
                ClusterType.ON_DEMAND if prefer_on_demand else ClusterType.SPOT
            )
            region = self._rr_region()
            yield TryLaunch(region=region, cluster_type=cluster)
            return  # Action taken.

        # 3b) On / ahead – consider scaling *in*.
        if not self._behind_schedule() and active_cnt > 1:
            # Terminate the most *expensive* instance first → ON_DEMAND,
            # otherwise fall back to any auxiliary region except the first.
            ondemand_regions = [
                r for r, t in active.items() if t is ClusterType.ON_DEMAND
            ]
            candidates = ondemand_regions or list(active.keys())[1:]
            region = sorted(candidates)[0]
            yield Terminate(region=region)
            return

        # No action this tick.
        return  # pragma: no cover

    # ------------------------------------------------------------------ #
    # CLI convenience                                                    #
    # ------------------------------------------------------------------ #
    @classmethod
    def _from_args(cls, parser):
        return cls(parser.parse_args())


# EVOLVE-BLOCK-END
