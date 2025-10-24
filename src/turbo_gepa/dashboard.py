"""
Terminal dashboard for TurboGEPA using plotext.

Simple, clean visualization of optimization progress.
"""

from __future__ import annotations

import sys
import time
from typing import List

try:
    import plotext as plt
except ImportError:
    plt = None  # Graceful degradation if plotext not installed

from .metrics import Metrics


class TerminalDashboard:
    """
    Live terminal dashboard for TurboGEPA optimization.

    Uses plotext for charts and refreshes at most once per 100ms to avoid flicker.
    """

    def __init__(self, refresh_interval: float = 0.1):
        """
        Initialize dashboard.

        Args:
            refresh_interval: Minimum seconds between redraws (default: 0.1s = 100ms)
        """
        if plt is None:
            raise ImportError(
                "plotext is required for the dashboard. Install with: pip install plotext"
            )

        self.refresh_interval = refresh_interval
        self.last_refresh = 0.0
        self.history: List[Metrics] = []
        self._first_render = True

    def update(self, metrics: Metrics) -> None:
        """
        Update dashboard with new metrics and redraw if enough time has passed.

        Args:
            metrics: Current orchestrator metrics
        """
        self.history.append(metrics)

        # Throttle refreshes to avoid flicker
        now = time.time()
        if now - self.last_refresh < self.refresh_interval:
            return

        self.last_refresh = now
        self._render()

    def _render(self) -> None:
        """Render the full dashboard."""
        if not self.history:
            return

        latest = self.history[-1]

        # Clear screen on first render, then use ANSI to return to top
        if self._first_render:
            plt.clear_terminal()
            self._first_render = False
        else:
            # Move cursor to top-left without clearing (reduces flicker)
            sys.stdout.write("\033[H")

        # Render components
        self._render_header(latest)
        self._render_quality_chart()
        self._render_stats(latest)
        self._render_rung_bars(latest)
        self._render_footer(latest)

        sys.stdout.flush()

    def _render_header(self, metrics: Metrics) -> None:
        """Render header with title."""
        print("=" * 80)
        print("  TurboGEPA Optimization Dashboard".center(80))
        print("=" * 80)

    def _render_quality_chart(self) -> None:
        """Render line chart of quality over time."""
        if len(self.history) < 2:
            return

        # Extract data
        rounds = [m.round for m in self.history]
        best_qualities = [m.best_quality * 100 for m in self.history]  # Convert to %
        avg_qualities = [m.avg_quality * 100 for m in self.history]

        # Configure plot with dark theme
        plt.clf()
        plt.theme("dark")  # Black background theme
        plt.plot_size(width=70, height=15)
        plt.title("Quality Over Time")
        plt.xlabel("Round")
        plt.ylabel("Quality (%)")

        # Plot lines with vibrant colors
        plt.plot(rounds, best_qualities, label="Best", marker="braille", color="cyan")
        plt.plot(rounds, avg_qualities, label="Avg", marker="braille", color="magenta")

        # Show legend and render
        plt.show()
        print()

    def _render_stats(self, metrics: Metrics) -> None:
        """Render statistics panel."""
        print("-" * 80)
        print("  Archive Statistics:".ljust(80))
        print(f"    Pareto Frontier: {metrics.pareto_size}".ljust(40), end="")
        print(f"QD Grid: {metrics.qd_size}".ljust(40))
        print(f"    Total Candidates: {metrics.total_candidates}".ljust(40), end="")
        print(f"Best: {metrics.best_quality:.1%}".ljust(40))
        print(f"    Average Quality: {metrics.avg_quality:.1%}".ljust(80))
        print("-" * 80)
        print()

    def _render_rung_bars(self, metrics: Metrics) -> None:
        """Render horizontal bar chart of rung activity."""
        if not metrics.rung_activity:
            return

        print("  Rung Activity (Inflight Candidates):".ljust(80))
        print()

        # Sort rungs by name for consistent display
        rungs = sorted(metrics.rung_activity.items())

        # Find max for scaling
        max_inflight = max(count for _, count in rungs) if rungs else 1
        bar_width = 50

        for rung_key, count in rungs:
            # Create bar
            filled = int((count / max(max_inflight, 1)) * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)

            # Display
            print(f"    {rung_key:15s} │{bar}│ {count}")

        print()

    def _render_footer(self, metrics: Metrics) -> None:
        """Render progress footer."""
        print("=" * 80)

        # Progress info
        round_info = f"Round: {metrics.round}"
        if metrics.max_rounds:
            round_info += f"/{metrics.max_rounds}"

        eval_info = f"Evals: {metrics.evaluations:,}"
        if metrics.max_evaluations:
            eval_info += f"/{metrics.max_evaluations:,}"

        # Calculate rate if we have history
        rate_info = ""
        if len(self.history) >= 2:
            time_delta = metrics.timestamp - self.history[0].timestamp
            eval_delta = metrics.evaluations - self.history[0].evaluations
            if time_delta > 0:
                rate = eval_delta / time_delta
                rate_info = f"Rate: {rate:.1f} eval/s"

        # Combine
        parts = [round_info, eval_info]
        if rate_info:
            parts.append(rate_info)

        footer = "  |  ".join(parts)
        print(f"  {footer}".ljust(80))
        print("=" * 80)
        print()
