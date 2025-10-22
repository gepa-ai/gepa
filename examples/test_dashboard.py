"""Test the multi-island dashboard visualization."""

import time
from turbo_gepa.multi_island_dashboard import IslandProgressAggregator

def main():
    print("Testing Multi-Island Dashboard\n")
    print("Simulating 4 islands running 10 rounds each...\n")

    # Create dashboard for 4 islands, 10 rounds max
    dashboard = IslandProgressAggregator(n_islands=4, max_rounds=10)

    # Simulate progress over 10 rounds
    for round_num in range(1, 11):
        for island_id in range(4):
            # Simulate different progress rates for each island
            quality = min(1.0, 0.2 + (round_num * 0.08) + (island_id * 0.02))
            avg_quality = quality * 0.9
            pareto_size = round_num + island_id

            # Simulate evaluations within the round
            total_evals = 20 + (island_id * 5)
            evals_done = total_evals if round_num < 10 else int(total_evals * 0.7)

            dashboard.update(
                island_id=island_id,
                round_num=round_num,
                best_quality=quality,
                avg_quality=avg_quality,
                pareto_size=pareto_size,
                evals_done=evals_done,
                evals_total=total_evals
            )

        time.sleep(0.5)  # Pause to show animation

    print("\n\nTest complete! Dashboard shows:")
    print("  • Round progress (X/10)")
    print("  • Evaluations within round (Y/Z evals)")
    print("  • Progress bar showing overall completion")
    print("  • Quality metrics and sparkline showing trend")

if __name__ == "__main__":
    main()
