# Multi-Region Scheduling Strategies README

This document outlines the design and evolution of multi-region scheduling strategies within this framework.

## 1. Core objective

The primary goal of the multi-region strategies is to minimize the monetary cost of completing a long-running, preemptible task before a fixed deadline, by dynamically leveraging spot instances across multiple geographic regions.

## 2. Evolution of the `multi_region_rc_cr_threshold` Strategy

### Version 1: Cost-Blind (The Old Way)

*   **Logic**: The initial strategy was only aware of spot instance *availability*. It decided to switch regions primarily when the current region's spot instance was preempted.
*   **Region Selection**: When multiple regions had available spot instances, it chose one arbitrarily (e.g., by the lowest region index).
*   **Weakness**: This approach was "cost-blind". It knew *where* resources were, but not how *expensive* they were. This often led to suboptimal decisions, sometimes performing worse than a good single-region strategy.

### Version 2: Cost-Aware (The New, Effective Way)

This is the current, high-performing version. The improvement came from two key changes:

1.  **Enhanced Environment API**: The `MultiTraceEnv` in `sky_spot/env.py` was upgraded with a `get_all_regions_spot_prices()` method. This provides the strategy with a global view of spot prices across all regions at every time step.

2.  **Cost-Driven Logic**: The strategy's core decision-making (`_step` method) was completely rewritten with a simple, powerful principle:
    > **At every decision point, find the cheapest available spot instance across all regions and use it.**

*   **Logic**:
    *   The strategy continuously scans the global price list.
    *   It identifies the region with the lowest spot price that currently has an available instance.
    *   If the strategy is not already in that region, it immediately initiates a switch.
    *   It only falls back to using expensive On-Demand instances as a last resort: when the task is behind schedule (`_condition() < 0`) or the deadline is imminent, and *no spot instances are available anywhere*.

*   **Strength**: This proactive, cost-driven approach ensures the task is always running on the most economical resource available globally, leading to significant cost reductions compared to both the cost-blind version and even the theoretical optimal for any single region.

## 3. Future Directions & Potential Improvements

The current cost-aware strategy is highly effective, but there is still room for more sophisticated logic:

*   **Region Prioritization**: Not all regions are equal. Some may have historically lower prices or better availability. A future strategy could incorporate a weighting system to prioritize regions that are historically more reliable or cheaper on average.
*   **Spot Quality Bar**: A region might offer a very low price, but if the instances are preempted every few minutes, the cost of frequent restarts (`restart_overhead_hours`) could negate the savings. A "quality bar" could be introduced, where a region is only considered if its spot price is below a certain percentage of the on-demand price, or if its historical stability meets a minimum threshold.
*   **Predictive Modeling**: A more advanced strategy could use historical trace data to predict near-term price fluctuations or availability, allowing it to make even smarter, forward-looking decisions. 