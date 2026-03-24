"""
Local end-to-end test of the Airline RM environment.

Exercises the full 30-day simulation using a simple heuristic policy
(closing discount classes near departure, moderate overbooking, smart
disruption handling). Writes a JSONL trajectory for inspection.

Does NOT require OpenReward or OpenAI API keys.
"""

import json
import sys
from datetime import datetime

import numpy as np

from network import (
    FARE_CLASSES,
    FARE_CLASS_CODES,
    ROUTES,
    SCENARIOS,
    AIRCRAFT_TYPES,
    get_daily_flight_count,
    get_flights_for_day,
    route_key,
)
from simulation import SimulationState
from baseline import BaselinePolicy


def run_local_test(scenario_name: str = "summer_peak", variant: int = 1):
    """Run a full 30-day simulation locally with a heuristic policy."""

    import hashlib
    task_id = f"{scenario_name}_v{variant}"
    seed = int(hashlib.sha256(task_id.encode()).hexdigest(), 16) % (2**32)
    scenario = SCENARIOS[scenario_name]
    total_days = 30

    trajectory_file = f"trajectory_local_{scenario_name}_v{variant}.jsonl"

    def log(event_type: str, data: dict):
        entry = {"timestamp": datetime.now().isoformat(), "event_type": event_type, **data}
        with open(trajectory_file, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    print(f"=== LOCAL TEST: {task_id} ===")
    print(f"Scenario: {scenario.description}")
    print(f"Seed: {seed}")
    print(f"Trajectory: {trajectory_file}")
    print()

    # --- Run baseline ---
    baseline_rng = np.random.default_rng(seed)
    baseline = BaselinePolicy(baseline_rng, scenario, total_days)
    baseline_results = baseline.run_full_simulation()
    baseline_net_by_day = [r.net_revenue for r in baseline_results]
    print(f"Baseline total net: ${sum(baseline_net_by_day):,.0f}")

    # --- Run agent (heuristic) ---
    agent_rng = np.random.default_rng(seed)
    sim = SimulationState(agent_rng, scenario, total_days)
    sim.generate_disruptions(1)

    cumulative_reward = 0.0
    total_revenue = 0.0
    total_costs = 0.0
    total_denied = 0
    total_disruptions_handled = 0

    log("test_start", {
        "task_id": task_id,
        "scenario": scenario_name,
        "variant": variant,
        "seed": seed,
        "total_days": total_days,
        "daily_flights": get_daily_flight_count(),
    })

    print(f"\n{'Day':>3} | {'Revenue':>12} | {'Costs':>10} | {'Net':>12} | {'Bsl Net':>12} | {'Reward':>8} | {'Cumul':>8} | {'Denied':>6} | {'Disruptions':>11}")
    print("-" * 110)

    for day in range(1, total_days + 1):
        sim.current_day = day

        # --- HEURISTIC POLICY ---

        # 1. Fare management: close discount classes as departure approaches
        fare_changes = 0
        for flight in sim.flights:
            if flight.status in ("cancelled", "departed"):
                continue
            if flight.departure_day < day:
                continue
            days_before = flight.departure_day - day

            for fc in FARE_CLASSES:
                old_open = flight.fare_availability.get(fc.code, True)

                if days_before < fc.advance_purchase_days:
                    # Can't sell below advance purchase
                    new_open = False
                elif days_before < 3 and fc.code in ("T", "L"):
                    # Close deep discounts 3 days out
                    new_open = False
                elif days_before < 7 and fc.code in ("V",):
                    # Close V class 7 days out
                    new_open = False
                elif days_before < 1 and fc.code in ("Q",):
                    # Close Q day-of
                    new_open = False
                else:
                    new_open = True

                if new_open != old_open:
                    flight.fare_availability[fc.code] = new_open
                    fare_changes += 1

            # 2. Overbooking: moderate (~7% of capacity)
            flight.overbooking_limit = max(1, int(flight.capacity * 0.07))

        # 3. Handle disruptions
        day_disruptions = 0
        pending = sim.get_pending_disruptions()
        for d in pending:
            day_disruptions += 1
            for fid in d.affected_flight_ids:
                flight = sim.flights_by_id.get(fid)
                if not flight or flight.status in ("cancelled", "departed"):
                    continue

                if fid in d.cancel_flight_ids:
                    # Severe disruption — try swap first
                    cost, err = sim.apply_disruption_swap(d, fid, flight.aircraft_type)
                    if err:
                        sim.apply_disruption_cancel(d, fid)
                elif d.delay_hours < 1.5:
                    # Minor delay — accept it
                    sim.apply_disruption_delay(d, fid)
                elif d.delay_hours < 3.0:
                    # Moderate — try swap
                    cost, err = sim.apply_disruption_swap(d, fid, flight.aircraft_type)
                    if err:
                        sim.apply_disruption_delay(d, fid)
                else:
                    # Long delay — cancel
                    sim.apply_disruption_cancel(d, fid)

            d.resolved = True
            total_disruptions_handled += 1

        # 4. Advance day
        result = sim.advance_day()

        # Compute reward vs baseline
        baseline_net = baseline_net_by_day[day - 1]
        agent_net = result.net_revenue
        denominator = max(abs(baseline_net), 1000.0)
        daily_reward = (agent_net - baseline_net) / denominator
        cumulative_reward += daily_reward

        total_revenue += result.revenue
        total_costs += result.costs
        total_denied += result.denied_boardings

        print(
            f"{day:>3} | ${result.revenue:>10,.0f} | ${result.costs:>8,.0f} | "
            f"${result.net_revenue:>10,.0f} | ${baseline_net:>10,.0f} | "
            f"{daily_reward:>+7.4f} | {cumulative_reward:>+7.4f} | "
            f"{result.denied_boardings:>6} | {day_disruptions:>11}"
        )

        log("day_complete", {
            "day": day,
            "revenue": round(result.revenue, 2),
            "costs": round(result.costs, 2),
            "net_revenue": round(result.net_revenue, 2),
            "baseline_net": round(baseline_net, 2),
            "daily_reward": round(daily_reward, 4),
            "cumulative_reward": round(cumulative_reward, 4),
            "denied_boardings": result.denied_boardings,
            "passengers_boarded": result.passengers_boarded,
            "new_bookings": result.new_bookings,
            "spilled": result.spilled_customers,
            "fare_changes": fare_changes,
            "disruptions_handled": day_disruptions,
        })

    # --- Final summary ---
    print()
    print("=" * 60)
    print(f"SIMULATION COMPLETE — {total_days} DAYS")
    print(f"  Total revenue:         ${total_revenue:>12,.0f}")
    print(f"  Total costs:           ${total_costs:>12,.0f}")
    print(f"  Total net:             ${total_revenue - total_costs:>12,.0f}")
    print(f"  Baseline net:          ${sum(baseline_net_by_day):>12,.0f}")
    improvement_pct = ((total_revenue - total_costs) - sum(baseline_net_by_day)) / abs(sum(baseline_net_by_day)) * 100
    print(f"  Improvement:           {improvement_pct:>+11.2f}%")
    print(f"  Cumulative reward:     {cumulative_reward:>+12.4f}")
    print(f"  Total denied boardings: {total_denied}")
    print(f"  Disruptions handled:   {total_disruptions_handled}")
    print("=" * 60)

    log("test_complete", {
        "total_revenue": round(total_revenue, 2),
        "total_costs": round(total_costs, 2),
        "total_net": round(total_revenue - total_costs, 2),
        "baseline_net": round(sum(baseline_net_by_day), 2),
        "improvement_pct": round(improvement_pct, 2),
        "cumulative_reward": round(cumulative_reward, 4),
        "total_denied_boardings": total_denied,
        "total_disruptions": total_disruptions_handled,
    })

    print(f"\nTrajectory saved to: {trajectory_file}")

    # Assertions for sanity
    assert total_revenue > 0, "Revenue should be positive"
    assert cumulative_reward != 0.0 or total_days == 0, "Reward should be non-zero"
    print("\nAll assertions passed.")

    return cumulative_reward


if __name__ == "__main__":
    scenario = sys.argv[1] if len(sys.argv) > 1 else "summer_peak"
    variant = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    run_local_test(scenario, variant)
