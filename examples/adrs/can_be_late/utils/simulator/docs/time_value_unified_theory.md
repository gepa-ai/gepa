# Unified Time-Value Theory

This document re-derives the marginal value of time used by the Can’t Be Late scheduler and extends it to a continuous setting. Everything is written in English so it can be shared publicly.

## Notation

| Symbol | Meaning | Unit/Notes |
|--------|---------|------------|
| $t$ | current wall-clock time | seconds |
| $p$ | cumulative completed work | seconds |
| $D$ | total required work | seconds |
| $T$ | deadline | seconds |
| $C_{OD}$ | on-demand price | $/hour |
| $C_{SPOT}$ | spot price | $/hour |
| $k$ | price ratio | $k = C_{OD} / C_{SPOT}$ |
| $d$ | restart overhead | hours |
| $o$ | spot lifetime | hours (random) |
| $V(t, p)$ | marginal value of time | $/hour |

## Motivation

Spot capacity across regions differs in both availability and cost. A shared value function lets us compare heterogeneous regions by converting the expected amount of additional work into dollars saved relative to on-demand usage.

## Deriving the Value Function

The required completion rate is $(D - p) / (T - t)$; the uniform-progress rate is $D / T$. Scaling the on-demand price by their ratio yields

$$V(t, p) = C_{OD} \times \frac{(D - p)/(T - t)}{D/T} = C_{OD} \times \frac{T}{D} \times \frac{D - p}{T - t}.$$

When $V(t, p) > C_{OD}$ the job is behind schedule relative to uniform progress and should switch to on-demand capacity. The discrete RC-CR rule therefore falls out of the same expression.

## Continuous Check

Running on-demand over $[t_0, t_0 + \Delta t]$ with initial progress $p_0$ gives a net benefit of

$$\frac{C_{OD} T}{D} \int_{0}^{\Delta t} \frac{A - u}{B - u} du - C_{OD} \Delta t,$$

where $A = D - p_0$ and $B = T - t_0$. On the uniform-progress boundary ($p_0 = D t_0 / T$) the logarithmic term vanishes and the net benefit is exactly zero, confirming consistency with the discrete rule.

## Multi-Region Policy

Given a region-specific lifetime distribution $P(o)$, the expected net value of picking that region is

$$E[\text{net}] = \int \left( \int_{0}^{o} V(t + \tau, p + \tau) d\tau - C_{SPOT} o \right) dP(o).$$

The scheduler simply chooses the region with the highest expected net value, providing a parameter-free policy that still respects the classic safety guarantees.

## Practical Advice

- Subtract restart overhead $d$ from the effective lifetime before integrating $V$.
- If spot prices fluctuate, substitute the expected time-averaged spot cost into $C_{SPOT}$.
- The framework is agnostic to the origin of $P(o)$; empirical traces or predictive models can be used interchangeably.

