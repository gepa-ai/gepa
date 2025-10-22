"""Live ASCII progress chart for terminal with colors."""

from __future__ import annotations

import sys
from typing import List, Optional


class TermColors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Colors
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"

    # Backgrounds
    BG_GREEN = "\033[102m"
    BG_BLUE = "\033[104m"
    BG_GRAY = "\033[100m"


class ProgressChart:
    """
    Real-time ASCII chart showing optimization progress.

    Displays quality improvement over rounds with color-coded sparklines.
    """

    def __init__(
        self,
        width: int = 60,
        height: int = 12,
        title: str = "Quality Progress",
    ):
        self.width = width
        self.height = height
        self.title = title
        self.quality_history: List[float] = []
        self.best_quality_history: List[float] = []
        self.round_labels: List[int] = []

    def update(
        self,
        round_num: int,
        current_quality: float,
        best_quality: float,
    ) -> None:
        """Record a new data point."""
        self.quality_history.append(current_quality)
        self.best_quality_history.append(best_quality)
        self.round_labels.append(round_num)

    def render(self, clear_screen: bool = True) -> str:
        """Generate the ASCII chart."""
        if not self.quality_history:
            return ""

        lines = []

        # Header with round info
        total_rounds = len(self.quality_history)
        if total_rounds == 1:
            subtitle = f"Baseline established (Round {self.round_labels[0]})"
        else:
            subtitle = f"{total_rounds} rounds completed"

        lines.append(f"\n{TermColors.BOLD}{TermColors.CYAN}{'=' * (self.width + 10)}{TermColors.RESET}")
        lines.append(f"{TermColors.BOLD}{TermColors.WHITE}📈 {self.title}{TermColors.RESET}  {TermColors.GRAY}{subtitle}{TermColors.RESET}")
        lines.append(f"{TermColors.CYAN}{'=' * (self.width + 10)}{TermColors.RESET}\n")

        # Stats
        initial = self.quality_history[0]
        current = self.quality_history[-1]
        best = self.best_quality_history[-1]
        delta = best - initial
        delta_pct = (delta / max(initial, 0.001)) * 100

        # Color code the improvement
        if delta > 0.1:
            delta_color = TermColors.GREEN
            delta_icon = "📈"
        elif delta > 0:
            delta_color = TermColors.YELLOW
            delta_icon = "↗️ "
        else:
            delta_color = TermColors.RED
            delta_icon = "→"

        lines.append(f"{TermColors.GRAY}Initial:{TermColors.RESET} {initial:.3f}  "
                    f"{TermColors.CYAN}Current:{TermColors.RESET} {current:.3f}  "
                    f"{TermColors.GREEN}Best:{TermColors.RESET} {TermColors.BOLD}{best:.3f}{TermColors.RESET}")
        lines.append(f"{delta_color}{delta_icon} Improvement: {delta:+.3f} ({delta_pct:+.1f}%){TermColors.RESET}\n")

        # Determine scale
        all_values = self.quality_history + self.best_quality_history
        min_val = min(all_values)
        max_val = max(all_values)

        # Add 10% padding
        value_range = max_val - min_val
        if value_range < 0.01:
            value_range = 0.1
        min_val -= value_range * 0.1
        max_val += value_range * 0.1

        # Build the chart
        chart_lines = self._build_chart_lines(min_val, max_val)
        lines.extend(chart_lines)

        # X-axis
        lines.append(f"{TermColors.GRAY}{'─' * (self.width + 2)}{TermColors.RESET}")

        # Round numbers on x-axis
        if len(self.round_labels) > 0:
            first_round = self.round_labels[0]
            last_round = self.round_labels[-1]
            mid_round = self.round_labels[len(self.round_labels) // 2]

            x_axis = f"{TermColors.GRAY}Round {first_round:2d}"
            x_axis += " " * (self.width - 20)
            x_axis += f"Round {mid_round:2d}"
            x_axis += " " * (self.width - 40)
            x_axis += f"Round {last_round:2d}{TermColors.RESET}"
            lines.append(x_axis)

        # Legend
        lines.append(f"\n{TermColors.BLUE}━━{TermColors.RESET} Best Quality  "
                    f"{TermColors.MAGENTA}━━{TermColors.RESET} Avg Quality  "
                    f"{TermColors.GREEN}▓{TermColors.RESET} Improvement Area")

        lines.append(f"{TermColors.CYAN}{'=' * (self.width + 10)}{TermColors.RESET}\n")

        return "\n".join(lines)

    def _build_chart_lines(self, min_val: float, max_val: float) -> List[str]:
        """Build the chart visualization with connected lines."""
        lines = []

        # Normalize data to chart height
        def normalize(val: float) -> int:
            if max_val == min_val:
                return self.height // 2
            ratio = (val - min_val) / (max_val - min_val)
            return int(ratio * (self.height - 1))

        # Sample data points to fit width
        step = max(1, len(self.quality_history) // self.width)
        sampled_current = [self.quality_history[i] for i in range(0, len(self.quality_history), step)]
        sampled_best = [self.best_quality_history[i] for i in range(0, len(self.best_quality_history), step)]

        # Build 2D grid
        grid = [[' ' for _ in range(len(sampled_current))] for _ in range(self.height)]

        # Draw connected lines for best quality
        for x in range(len(sampled_best) - 1):
            y1 = normalize(sampled_best[x])
            y2 = normalize(sampled_best[x + 1])

            # Bresenham-like line drawing
            if y1 == y2:
                # Horizontal line
                grid[self.height - 1 - y1][x] = 'B'
                grid[self.height - 1 - y1][x + 1] = 'B'
            else:
                # Diagonal line
                steps = abs(y2 - y1) + 1
                for step in range(steps):
                    t = step / max(steps - 1, 1)
                    y = int(y1 + t * (y2 - y1))
                    x_pos = x if step < steps / 2 else x + 1
                    if 0 <= x_pos < len(sampled_best):
                        grid[self.height - 1 - y][x_pos] = 'B'

        # Draw connected lines for current quality
        for x in range(len(sampled_current) - 1):
            y1 = normalize(sampled_current[x])
            y2 = normalize(sampled_current[x + 1])

            if y1 == y2:
                # Horizontal line - check if overlapping with best
                if grid[self.height - 1 - y1][x] != 'B':
                    grid[self.height - 1 - y1][x] = 'C'
                if grid[self.height - 1 - y1][x + 1] != 'B':
                    grid[self.height - 1 - y1][x + 1] = 'C'
                else:
                    grid[self.height - 1 - y1][x + 1] = 'X'  # Overlap
            else:
                # Diagonal line
                steps = abs(y2 - y1) + 1
                for step in range(steps):
                    t = step / max(steps - 1, 1)
                    y = int(y1 + t * (y2 - y1))
                    x_pos = x if step < steps / 2 else x + 1
                    if 0 <= x_pos < len(sampled_current):
                        if grid[self.height - 1 - y][x_pos] == 'B':
                            grid[self.height - 1 - y][x_pos] = 'X'  # Overlap
                        elif grid[self.height - 1 - y][x_pos] == ' ':
                            grid[self.height - 1 - y][x_pos] = 'C'

        # Fill area between lines (improvement zone)
        for x in range(len(sampled_current)):
            y_best = normalize(sampled_best[x])
            y_current = normalize(sampled_current[x])
            y_min = min(y_best, y_current)
            y_max = max(y_best, y_current)
            for y in range(y_min, y_max):
                if grid[self.height - 1 - y][x] == ' ':
                    grid[self.height - 1 - y][x] = '▒'  # Fill

        # Convert grid to colored strings
        for y in range(self.height):
            # Y-axis label
            value = min_val + (y / (self.height - 1)) * (max_val - min_val)
            if y % 3 == 0:  # Only show every 3rd label
                label = f"{TermColors.GRAY}{value:.2f}{TermColors.RESET} │ "
            else:
                label = f"{TermColors.GRAY}      │{TermColors.RESET} "

            # Chart line
            line = label
            for x in range(len(sampled_current)):
                char = grid[y][x]
                if char == 'B':
                    line += f"{TermColors.BLUE}━{TermColors.RESET}"
                elif char == 'C':
                    line += f"{TermColors.MAGENTA}━{TermColors.RESET}"
                elif char == 'X':
                    line += f"{TermColors.CYAN}╋{TermColors.RESET}"
                elif char == '▒':
                    line += f"{TermColors.GREEN}▓{TermColors.RESET}"
                else:
                    # Background grid
                    if x % 10 == 0:
                        line += f"{TermColors.GRAY}·{TermColors.RESET}"
                    else:
                        line += " "

            lines.append(line)

        # Reverse so highest values are at top
        return list(reversed(lines))

    def display(self) -> None:
        """Print the chart to terminal."""
        # Clear previous output (move cursor up and clear)
        if len(self.quality_history) > 1:
            # Move up by chart height + header/footer
            num_lines = self.height + 12
            sys.stdout.write(f"\033[{num_lines}A")  # Move up
            sys.stdout.write("\033[J")  # Clear from cursor down

        print(self.render(clear_screen=False))
        sys.stdout.flush()

    def sparkline(self, values: List[float], width: int = 20) -> str:
        """Generate a mini sparkline for inline display."""
        if not values:
            return ""

        min_val = min(values)
        max_val = max(values)
        if max_val == min_val:
            return "▄" * min(width, len(values))

        # Use unicode block characters for fine resolution
        blocks = "▁▂▃▄▅▆▇█"

        step = max(1, len(values) // width)
        sampled = [values[i] for i in range(0, len(values), step)][:width]

        def get_block(val: float) -> str:
            ratio = (val - min_val) / (max_val - min_val)
            idx = int(ratio * (len(blocks) - 1))
            return blocks[idx]

        # Color code based on trend
        trend = sampled[-1] - sampled[0] if len(sampled) > 1 else 0
        if trend > 0.05:
            color = TermColors.GREEN
        elif trend > 0:
            color = TermColors.YELLOW
        else:
            color = TermColors.GRAY

        line = color + "".join(get_block(v) for v in sampled) + TermColors.RESET
        return line
