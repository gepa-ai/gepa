import configargparse
import logging
import typing

from sky_spot.strategies.strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType

if typing.TYPE_CHECKING:  # pragma: no cover
    from sky_spot import env, task

logger = logging.getLogger(__name__)


# EVOLVE-BLOCK-START
class EvolutionaryStrategy(MultiRegionStrategy):
    """
    A simpler – yet more reliable – multi–region strategy.

    Design goals
    -------------
    1. Never miss a deadline             -> keep at least one instance alive.
    2. Minimise cost                     -> strongly prefer SPOT.
    3. React quickly to failures         -> retry *other* regions first.
    4. Keep the implementation concise   -> easier to reason about / improve.
    """

    NAME = "evolutionary_simple_v2"

    # --------------------------------------------------------------------- #
    # Framework hooks
    # --------------------------------------------------------------------- #
    def __init__(self, args: configargparse.Namespace):
        super().__init__(args)

        # These attributes are (re-)initialised in reset().
        self.region_stats: typing.Dict[int, typing.Dict[str, int]] = {}
        self.region_ids: typing.List[int] = []
        self.initialised: bool = False

    def reset(self, env: "env.Env", task: "task.Task"):
        """Called once at the beginning of every simulation run."""
        super().reset(env, task)

        # Determine how many regions are available to us.
        region_count = getattr(env, "num_regions", None)
        if region_count is None:
            # Fallback – older versions expose `trace_files`.
            region_count = len(getattr(env, "trace_files", [])) or 2

        self.region_ids = list(range(region_count))

        # Per-region counters (very small & fast).
        self.region_stats = {
            r: {"success": 0, "fail": 0, "last_try": -1} for r in self.region_ids
        }

        self.initialised = True
        logger.debug("%s reset with %d regions", self.NAME, region_count)

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _progress(self) -> float:
        """Return fraction of the task already completed (0‒1)."""
        # `MultiRegionStrategy` exposes `self.task_done_time`.
        done = sum(self.task_done_time)
        return 0.0 if not self.task_duration else done / self.task_duration

    def _time_used(self) -> float:
        """Return fraction of time already elapsed (0‒1)."""
        return (
            0.0
            if not self.deadline
            else self.env.elapsed_seconds / float(self.deadline)
        )

    def _urgent(self) -> bool:
        """
        Decide whether we are behind schedule.

        We define urgency as *progress < time_used – 0.05*.
        The 5 % slack allows minor fluctuations without
        triggering expensive ON_DEMAND launches too early.
        """
        return self._progress() < self._time_used() - 0.05

    def _time_critical(self) -> bool:
        """
        Hard urgency – we will certainly miss the deadline
        unless we launch something that is guaranteed to run.
        """
        remaining_work = self.task_duration * (1.0 - self._progress())
        time_left = self.deadline - self.env.elapsed_seconds
        return time_left <= remaining_work

    # Region selection ---------------------------------------------------- #
    def _spot_score(self, region: int) -> float:
        """Heuristic: higher value → better place to try SPOT."""
        stats = self.region_stats[region]
        attempts = stats["success"] + stats["fail"]
        success_rate = stats["success"] / attempts if attempts else 0.5

        # Small bonus for recency (helps cluster churn).
        recency = self.env.elapsed_seconds - stats["last_try"]
        recency_bonus = max(0.0, 1.0 - recency / 3600.0)  # linearly decays over 1 h

        return success_rate + 0.1 * recency_bonus

    def _next_region_for_spot(self) -> int:
        """Pick region with highest SPOT score."""
        return max(self.region_ids, key=self._spot_score)

    # --------------------------------------------------------------------- #
    # Main generator
    # --------------------------------------------------------------------- #
    def _step_multi(
        self,
    ) -> typing.Generator[
        typing.Union["TryLaunch", "Terminate"], typing.Optional["LaunchResult"], None
    ]:
        """
        Yield‐based control loop.

        High-level logic
        ----------------
        1. If work is done              → terminate everything.
        2. If no active instances:
              a. Try best SPOT region(s) sequentially.
              b. If *time-critical*     → fall back to ON_DEMAND.
        3. If active SPOTs but urgent   → launch a *second* SPOT elsewhere.
        4. If still likely to miss the deadline
              → fall back to ON_DEMAND as insurance.
        """
        from sky_spot.multi_region_types import Terminate, TryLaunch

        if not self.initialised:
            return  # Safety net (should never happen).

        # ---------------------------------------------------------------- #
        # 1. Finish early
        # ---------------------------------------------------------------- #
        remaining_task_seconds = self.task_duration - sum(self.task_done_time)
        if remaining_task_seconds <= 1e-3:
            for r in getattr(self.env, "get_active_instances", lambda: [])():
                yield Terminate(region=r)
            return

        active = getattr(self.env, "get_active_instances", lambda: [])()
        active_spot = [r for r, t in active.items() if t == ClusterType.SPOT]
        active_ondemand = [r for r, t in active.items() if t == ClusterType.ON_DEMAND]

        # ---------------------------------------------------------------- #
        # 2. No running instances → we *must* launch something
        # ---------------------------------------------------------------- #
        if not active:
            launched = False

            # 2a. Try SPOT(s)
            for _ in self.region_ids:  # at most one attempt per region
                region = self._next_region_for_spot()
                res = yield TryLaunch(region=region, cluster_type=ClusterType.SPOT)
                assert res is not None

                self.region_stats[region]["last_try"] = self.env.elapsed_seconds
                if res.success:
                    self.region_stats[region]["success"] += 1
                    launched = True
                    break
                self.region_stats[region]["fail"] += 1

            # 2b. Fallback to ON_DEMAND if necessary
            if not launched:
                # Choose region with fewest ON_DEMAND launches so far (cheap heuristic)
                region = min(self.region_ids, key=lambda r: self.region_stats[r]["success"])
                res = yield TryLaunch(region=region, cluster_type=ClusterType.ON_DEMAND)
                assert res is not None
                # No further book-keeping necessary for ON_DEMAND.
            return  # End of decision block – wait for next env step.

        # ---------------------------------------------------------------- #
        # 3. We have at least one instance running
        # ---------------------------------------------------------------- #
        if self._urgent():
            # 3a. Behind schedule – launch an *extra* SPOT if we only have one.
            if len(active_spot) == 1 and not active_ondemand:
                backup_region = self._next_region_for_spot()
                if backup_region not in active:
                    res = yield TryLaunch(
                        region=backup_region, cluster_type=ClusterType.SPOT
                    )
                    assert res is not None
                    self.region_stats[backup_region]["last_try"] = self.env.elapsed_seconds
                    if res.success:
                        self.region_stats[backup_region]["success"] += 1
                    else:
                        self.region_stats[backup_region]["fail"] += 1

            # 3b. If truly time-critical & still only SPOTs → add ON_DEMAND.
            if self._time_critical() and not active_ondemand:
                ondemand_region = self._next_region_for_spot()
                res = yield TryLaunch(
                    region=ondemand_region, cluster_type=ClusterType.ON_DEMAND
                )
                assert res is not None
        # ---------------------------------------------------------------- #
        # Otherwise: keep current state – nothing to yield this tick.
        # ---------------------------------------------------------------- #

    # --------------------------------------------------------------------- #
    # CLI helper (required by framework)
    # --------------------------------------------------------------------- #
    @classmethod
    def _from_args(cls, parser):
        return cls(parser.parse_args())


# EVOLVE-BLOCK-END