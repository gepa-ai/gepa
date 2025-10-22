"""Multi-island dashboard for visualizing concurrent optimization."""

from __future__ import annotations

import sys
from typing import Dict, List

from .progress_chart import TermColors


class IslandDashboard:
    """
    Real-time dashboard showing progress across multiple islands.

    Displays:
    - Global best quality across all islands
    - Individual island progress with sparklines
    - Combined metrics and statistics
    """

    def __init__(self, n_islands: int = 4, width: int = 80, max_rounds: int = 10):
        self.n_islands = n_islands
        self.width = width
        self.max_rounds = max_rounds  # For progress calculation
        self.island_data: Dict[int, Dict] = {
            i: {
                "round": 0,
                "best_quality": 0.0,
                "avg_quality": 0.0,
                "pareto_size": 0,
                "history": [],
                "evals_done": 0,  # Evaluations completed in current round
                "evals_total": 0,  # Total evaluations for current round
            }
            for i in range(n_islands)
        }
        self.global_best = 0.0
        self.global_round = 0
        self.lines_printed = 0

    def update_island(
        self,
        island_id: int,
        round_num: int,
        best_quality: float,
        avg_quality: float,
        pareto_size: int,
        evals_done: int = 0,
        evals_total: int = 0,
    ) -> None:
        """Update data for a specific island."""
        data = self.island_data[island_id]
        data["round"] = round_num
        data["best_quality"] = best_quality
        data["avg_quality"] = avg_quality
        data["pareto_size"] = pareto_size
        data["evals_done"] = evals_done
        data["evals_total"] = evals_total
        data["history"].append(best_quality)

        # Update global metrics
        self.global_best = max(self.global_best, best_quality)
        self.global_round = max(self.global_round, round_num)

    def render(self) -> str:
        """Generate the dashboard visualization."""
        lines = []

        # Header
        lines.append(f"\n{TermColors.BOLD}{TermColors.CYAN}{'═' * self.width}{TermColors.RESET}")
        lines.append(f"{TermColors.BOLD}{TermColors.WHITE}🏝️  Multi-Island Optimization Dashboard{TermColors.RESET}")
        lines.append(f"{TermColors.CYAN}{'═' * self.width}{TermColors.RESET}\n")

        # Global metrics
        total_pareto = sum(d["pareto_size"] for d in self.island_data.values())
        avg_round = sum(d["round"] for d in self.island_data.values()) / self.n_islands

        lines.append(f"{TermColors.GREEN}🏆 Global Best:{TermColors.RESET} {TermColors.BOLD}{self.global_best:.2%}{TermColors.RESET}  "
                    f"{TermColors.CYAN}📊 Total Pareto:{TermColors.RESET} {total_pareto}  "
                    f"{TermColors.YELLOW}⚡ Avg Round:{TermColors.RESET} {avg_round:.1f}")
        lines.append("")

        # Island progress bars
        lines.append(f"{TermColors.BOLD}Island Progress:{TermColors.RESET}")
        lines.append("")

        for island_id in range(self.n_islands):
            data = self.island_data[island_id]
            best = data["best_quality"]
            avg = data["avg_quality"]
            round_num = data["round"]
            pareto = data["pareto_size"]
            history = data["history"]
            evals_done = data["evals_done"]
            evals_total = data["evals_total"]

            # Calculate progress through rounds
            if self.max_rounds > 0:
                round_progress = round_num / self.max_rounds
            else:
                round_progress = 0.0

            # Calculate progress within current round
            if evals_total > 0:
                round_internal_progress = evals_done / evals_total
            else:
                round_internal_progress = 0.0

            # Overall progress combines round number and within-round progress
            overall_progress = (round_num + round_internal_progress) / max(self.max_rounds, 1)

            # Island header with round and eval progress
            island_label = f"Island {island_id}"
            if evals_total > 0:
                status = f"Round {round_num}/{self.max_rounds} ({evals_done}/{evals_total} evals)"
            else:
                status = f"Round {round_num}/{self.max_rounds}"
            lines.append(f"{TermColors.BOLD}{island_label}{TermColors.RESET} {TermColors.GRAY}│{TermColors.RESET} {status}")

            # Progress bar showing overall progress through optimization
            bar_width = 40
            filled = int(overall_progress * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)

            # Color based on progress
            if overall_progress >= 0.8:
                bar_color = TermColors.GREEN
            elif overall_progress >= 0.5:
                bar_color = TermColors.YELLOW
            else:
                bar_color = TermColors.CYAN

            lines.append(f"  {bar_color}{bar}{TermColors.RESET} {overall_progress:.1%}")

            # Metrics row with quality sparkline
            sparkline = self._create_sparkline(history, width=30)
            lines.append(f"  {TermColors.GRAY}Quality:{TermColors.RESET} {best:.2%} (avg {avg:.2%})  "
                        f"{TermColors.GRAY}Pareto:{TermColors.RESET} {pareto}  "
                        f"{sparkline}")
            lines.append("")

        # Footer
        lines.append(f"{TermColors.CYAN}{'─' * self.width}{TermColors.RESET}")
        lines.append(f"{TermColors.GRAY}Press Ctrl+C to stop{TermColors.RESET}\n")

        return "\n".join(lines)

    def _create_sparkline(self, history: List[float], width: int = 20) -> str:
        """Create a mini sparkline visualization."""
        if not history:
            return ""

        # Sample if too long
        if len(history) > width:
            step = len(history) / width
            sampled = [history[int(i * step)] for i in range(width)]
        else:
            sampled = history

        min_val = min(sampled) if sampled else 0
        max_val = max(sampled) if sampled else 1

        if max_val == min_val:
            return f"{TermColors.GRAY}{'▄' * len(sampled)}{TermColors.RESET}"

        # Use unicode block characters
        blocks = "▁▂▃▄▅▆▇█"

        def get_block(val: float) -> str:
            ratio = (val - min_val) / (max_val - min_val)
            idx = int(ratio * (len(blocks) - 1))
            return blocks[idx]

        # Color based on trend
        trend = sampled[-1] - sampled[0] if len(sampled) > 1 else 0
        if trend > 0.05:
            color = TermColors.GREEN
        elif trend > 0:
            color = TermColors.YELLOW
        else:
            color = TermColors.GRAY

        sparkline = color + "".join(get_block(v) for v in sampled) + TermColors.RESET
        return sparkline

    def display(self) -> None:
        """Print the dashboard to terminal, updating in place."""
        # Clear previous output
        if self.lines_printed > 0:
            sys.stdout.write(f"\033[{self.lines_printed}A")  # Move up
            sys.stdout.write("\033[J")  # Clear from cursor down

        output = self.render()
        print(output)
        sys.stdout.flush()

        # Count lines for next clear
        self.lines_printed = output.count("\n") + 1


class IslandProgressAggregator:
    """
    Collects progress updates from multiple islands and maintains dashboard.

    Can be shared across async tasks to aggregate metrics.
    """

    def __init__(self, n_islands: int = 4, max_rounds: int = 10):
        self.dashboard = IslandDashboard(n_islands=n_islands, max_rounds=max_rounds)
        self.update_count = 0
        self.display_frequency = 1  # Display every N updates

    def update(
        self,
        island_id: int,
        round_num: int,
        best_quality: float,
        avg_quality: float,
        pareto_size: int,
        evals_done: int = 0,
        evals_total: int = 0,
    ) -> None:
        """Update and optionally display."""
        self.dashboard.update_island(
            island_id, round_num, best_quality, avg_quality, pareto_size,
            evals_done=evals_done, evals_total=evals_total
        )
        self.update_count += 1

        # Display based on frequency
        if self.update_count % self.display_frequency == 0:
            self.dashboard.display()

    def display(self) -> None:
        """Force display the dashboard."""
        self.dashboard.display()
