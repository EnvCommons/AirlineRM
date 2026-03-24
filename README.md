# AirlineRM

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/airlinerm)

## Description

AirlineRM is a hyper-realistic airline network revenue management environment where an agent operates a hub-and-spoke carrier over a 30-day horizon. The agent makes daily decisions about fare class availability (opening/closing 8 nested fare buckets), overbooking limits, and disruption response (weather, mechanical failures, crew shortages, competitor fare wars). The simulation is calibrated against real-world airline industry parameters including DOT-regulated denied boarding compensation, FAR Part 117 crew legality constraints, and empirical booking curve shapes.

## Capabilities

- Dynamic fare class management across an 8-class nested inventory (Y/B/M/H/Q/V/T/L)
- Overbooking optimization balancing fill rates against denied boarding costs
- Disruption management: weather events, mechanical failures, crew shortages
- Aircraft swap decisions under range and capacity constraints
- Competitor fare war response and demand adaptation
- Multi-day strategy development with dense per-day reward feedback
- Long-horizon multi-turn execution (30+ advance_day calls plus analysis)

## Compute Requirements

Agents in AirlineRM are given a sandbox with 1 GB of RAM and 0.5 CPUs running a Python 3.12 data science image. Network access is enabled.

## License

[ORLv1](https://openreward.ai/orlv1.md).

## Tasks

There are 12 tasks across 4 seasonal scenarios:

**Training (9 tasks):**
- `summer_peak_v{1,2,3}`: Peak summer travel with high leisure demand and thunderstorm risk (demand multiplier 1.30x)
- `winter_holiday_v{1,2,3}`: Holiday season with mixed demand and snowstorm risk (demand multiplier 1.15x)
- `shoulder_spring_v{1,2,3}`: Shoulder season with moderate demand and competitor pressure (demand multiplier 0.90x)

**Test (3 tasks):**
- `fall_business_v{1,2,3}`: Fall business travel season with conference demand and occasional fog (demand multiplier 1.05x)

Each task simulates 30 operating days across a 12-route hub-and-spoke network with 29 daily departures serving business, mixed, and leisure markets using a fleet of E175 (76 seats), 737-700 (138 seats), and 737-800 (175 seats) aircraft. Variants within the same scenario share the same demand parameters but have different random seeds producing different disruption patterns and booking flows.

## Reward Structure

This is a dense, verifiable reward environment. Rewards are computed after each operating day (via the `advance_day` tool). The reward compares the agent's daily net revenue (revenue minus costs) against a naive baseline policy:

$$\text{reward}_t = \frac{\text{agent\_net}_t - \text{baseline\_net}_t}{\max(|\text{baseline\_net}_t|, 1000)}$$

The baseline policy opens all fare classes without strategic management, never overbooks, and handles disruptions with simple delay-or-cancel logic. A positive reward means the agent outperformed the baseline.

Revenue sources:
- Passenger fare revenue (fare class price x passengers boarded)

Cost sources:
- Denied boarding: $775/passenger (DOT 14 CFR 250)
- Flight cancellation: $200/passenger (rebooking + accommodation)
- Delay >1 hour: $50/passenger/hour
- Aircraft swap: $5,000 fixed cost

We do not use LLM graders for this environment.

## Data

All data is synthetically generated and deterministic given the task seed. No external datasets are required. The simulation generates:
- Network topology (12 routes with realistic distances, frequencies, and fare structures)
- Demand via parametric booking curves calibrated to industry data (business routes book late, leisure routes book early)
- Disruptions via seasonal stochastic models (thunderstorms in summer, snowstorms in winter, mechanical failures year-round)
- Competitor fare wars with demand reduction effects

Agents are given access to a sandbox filesystem where they can write analysis scripts and build models.

Simulation parameters are calibrated against: BTS LOADFACTORD (2019-2024 domestic load factor averages), DOT 14 CFR 250.5 (denied boarding compensation tiers), Smith, Leimkuhler & Darrow 1992 (no-show rates by fare class), Belobaba 1989 (booking curve shapes), FAA/DOT Form 41 Schedule P-5.2 (aircraft operating costs), and Gallego & van Ryzin 1994 (willingness-to-pay models).

## Tools

Agents have access to 6 domain-specific tools and 9 CLI tools:

**Domain-specific tools:**
| Tool | Description |
|------|-------------|
| `view_network_status` | Current day status: flights, bookings, disruptions, performance metrics |
| `view_flight_details` | Detailed view of a specific flight's bookings by fare class |
| `set_fare_availability` | Open/close fare classes for upcoming flights |
| `set_overbooking_limit` | Set overbooking authorization (0 to 15% of capacity) |
| `handle_disruption` | Respond to IRROPS: cancel, delay, swap aircraft, or do nothing |
| `advance_day` | Process departures, bookings, and disruptions; advance to next day |

**CLI tools:** `bash`, `glob`, `grep`, `ls`, `read`, `write`, `edit`, `multi_edit`, `todo_write`

## Time Horizon

AirlineRM is a multi-turn environment requiring at least 30 `advance_day` calls (one per operating day), plus additional calls for viewing network status, adjusting fares, setting overbooking limits, and handling disruptions.

## Environment Difficulty

A simple heuristic policy (closing discount fare classes near departure, 7% overbooking, smart disruption swaps) achieves positive cumulative reward across all four scenarios, demonstrating that the environment is solvable. However, performance varies significantly by scenario, indicating meaningful strategic differentiation is required:

| Scenario | Heuristic Reward | Baseline 30-day Net |
|----------|-----------------|---------------------|
| summer_peak | +6.3 | ~$3.3M |
| winter_holiday | +14.0 | ~$2.8M |
| shoulder_spring | +0.2 | ~$2.3M |
| fall_business | +0.02 | ~$2.7M |

A do-nothing agent (replicating baseline behavior) scores exactly 0.0. A deliberately destructive policy (closing all fares, cancelling all flights) scores below -5.0, confirming the reward signal correctly differentiates agent quality.

## Other Environment Requirements

There are no further environment requirements; AirlineRM works out of the box with the OpenReward endpoint without any external API keys.

## Safety

Agents in AirlineRM are told to maximize net revenue through revenue management decisions. The environment does not present direct safety risks as agents interact only with a synthetic simulation. No real airline operations, financial transactions, or passenger data are involved. All operations are sandboxed.

There may be indirect risks in that an agent optimizing purely for revenue may learn aggressive overbooking or cancellation strategies that would be ethically problematic in real operations. The cost structure is calibrated to penalize such behaviour (denied boarding costs $775/passenger), but multi-environment training should include environments that reinforce ethical decision-making.

## Citations

```bibtex
@dataset{GRAirlineRM,
  author    = {General Reasoning Inc. Team},
  title     = {AirlineRM: Airline Network Revenue Management Environment},
  year      = {2026},
  publisher = {OpenReward},
  url       = {https://openreward.ai/GeneralReasoning/airlinerm}
}
```
