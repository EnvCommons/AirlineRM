"""
Airline Network Revenue Management environment for OpenReward.

The agent manages a hub-and-spoke airline network over a 30-day operating
horizon, making daily decisions about fare class availability, overbooking
limits, and disruption (IRROPS) response.
"""

from __future__ import annotations

import json
import hashlib
from typing import Dict, List, Any, Optional

import numpy as np
from pydantic import BaseModel

from cli_environment import CLIEnvironment
from openreward.environments import tool, JSONObject, ToolOutput, TextBlock
from openreward import AsyncOpenReward, SandboxSettings

from network import (
    AIRCRAFT_TYPES,
    FARE_CLASSES,
    FARE_CLASS_CODES,
    ROUTES,
    SCENARIOS,
    ScheduledFlight,
    compute_fare_table,
    generate_flight_schedule,
    get_daily_flight_count,
    get_flights_for_day,
    route_key,
    DENIED_BOARDING_COST,
    CANCELLATION_COST_PER_PAX,
    DELAY_COST_PER_PAX_PER_HOUR,
    DELAY_THRESHOLD_HOURS,
    AIRCRAFT_SWAP_FIXED_COST,
)
from simulation import SimulationState, Disruption, DayResult
from baseline import BaselinePolicy


# ---------------------------------------------------------------------------
# Pydantic models for tool inputs
# ---------------------------------------------------------------------------

class TaskSpec(BaseModel):
    id: str

class ViewFlightDetailsParams(BaseModel, extra="forbid"):
    flight_id: str

class SetFareAvailabilityParams(BaseModel, extra="forbid"):
    flight_id: str
    fare_classes: Dict[str, bool]  # e.g. {"Y": true, "B": true, "Q": false, ...}

class SetOverbookingLimitParams(BaseModel, extra="forbid"):
    flight_id: str
    overbooking_limit: int

class HandleDisruptionParams(BaseModel, extra="forbid"):
    disruption_id: str
    action: str  # "cancel_flight", "delay_flight", "swap_aircraft", "do_nothing"
    swap_aircraft_type: Optional[str] = None  # required if action == "swap_aircraft"

class EmptyParams(BaseModel, extra="forbid"):
    pass


# ---------------------------------------------------------------------------
# Main environment class
# ---------------------------------------------------------------------------

class AirlineRM(CLIEnvironment):
    """
    Airline Network Revenue Management environment.

    The agent manages fare class availability, overbooking, and disruption
    response for a hub-and-spoke airline network over 30 operating days.
    """

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec, secrets=secrets)
        self.validated = TaskSpec.model_validate(task_spec)

        if not secrets.get("api_key"):
            raise ValueError("OpenReward API key is required")

        # Parse task ID: e.g. "summer_peak_v1" -> scenario="summer_peak", variant=1
        task_id = self.validated.id
        parts = task_id.rsplit("_v", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid task ID format: {task_id}. Expected 'scenario_vN'.")
        scenario_name = parts[0]
        variant = int(parts[1])

        if scenario_name not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}. Options: {list(SCENARIOS.keys())}")

        self.scenario = SCENARIOS[scenario_name]
        self.variant = variant
        self.total_days = 30

        # Deterministic seed from task ID
        seed = int(hashlib.sha256(task_id.encode()).hexdigest(), 16) % (2**32)

        # --- Run baseline simulation first ---
        baseline_rng = np.random.default_rng(seed)
        baseline_policy = BaselinePolicy(baseline_rng, self.scenario, self.total_days)
        self.baseline_results: List[DayResult] = baseline_policy.run_full_simulation()

        # --- Initialize agent simulation with the SAME seed ---
        agent_rng = np.random.default_rng(seed)
        self.sim = SimulationState(agent_rng, self.scenario, self.total_days)

        # Generate disruptions for day 1
        self.sim.generate_disruptions(1)

        # Tracking
        self.cumulative_reward = 0.0
        self.task_finished = False

        # Sandbox settings (no external data needed)
        self.sandbox_settings = SandboxSettings(
            environment="GeneralReasoning/airlinerm",
            image="generalreasoning/python-ds:3.12-tools",
            machine_size="0.5:1",
            block_network=False,
        )
        or_client = AsyncOpenReward(api_key=secrets.get("api_key"))
        self.sandbox = or_client.sandbox(self.sandbox_settings)

    # -------------------------------------------------------------------
    # Prompt
    # -------------------------------------------------------------------

    async def get_prompt(self) -> List[TextBlock]:
        daily_flights = get_daily_flight_count()

        # Build fare class table
        fc_table = "Class | Name                | Fare Mult | Adv Purch | No-Show | Refundable\n"
        fc_table += "------|---------------------|-----------|-----------|---------|----------\n"
        for fc in FARE_CLASSES:
            fc_table += (
                f"  {fc.code}   | {fc.name:<19} | "
                f"{fc.multiplier:>8.0%}  | {fc.advance_purchase_days:>5}d    | "
                f"{fc.no_show_rate:>5.0%}  | {'Yes' if fc.refundable else 'No'}\n"
            )

        # Build route summary
        route_table = "Route     | Category | Aircraft | Freq | Max Fare | Demand/Day\n"
        route_table += "----------|----------|----------|------|----------|----------\n"
        for r in ROUTES:
            route_table += (
                f"HUB-{r.destination:<4} | {r.category:<8} | {r.default_aircraft:<8} | "
                f"{r.daily_frequencies:>4} | ${r.max_fare:>6.0f}  | {r.base_demand_mean:>5.0f} pax\n"
            )

        prompt = f"""You are an airline revenue manager for SkyHub Airlines, operating a hub-and-spoke network out of HUB airport with 12 spoke routes and {daily_flights} daily departures.

SCENARIO: {self.scenario.name.replace('_', ' ').title()}
{self.scenario.description}
Operating horizon: 30 days. You are on Day 1.

YOUR NETWORK
{route_table}

FLEET: E175 (76 seats), 737-700 (138 seats), 737-800 (175 seats)

FARE CLASS STRUCTURE (8 classes, nested from highest to lowest)
{fc_table}
Fares are computed as: route max_fare x class multiplier. For example, HUB-BOS Y class = $620 x 100% = $620, L class = $620 x 13% = $80.60.

DAILY WORKFLOW
Each day you should:
1. Review network status: view_network_status()
2. Examine specific flights: view_flight_details(flight_id="HUB-BOS-F0-D5")
3. Set fare class availability: set_fare_availability(flight_id="...", fare_classes={{"Y": true, "M": true, "Q": false, ...}})
4. Set overbooking limits: set_overbooking_limit(flight_id="...", overbooking_limit=10)
5. Handle any disruptions: handle_disruption(disruption_id="...", action="delay_flight")
6. Advance to next day: advance_day()

REVENUE MANAGEMENT PRINCIPLES
- Close lower fare classes as departure approaches to protect seats for high-value late-booking business travelers
- Overbooking can increase revenue (filling no-show seats) but creates costly denied boardings if too aggressive
- Business routes (BOS, ORD, SFO) have more late bookings and higher no-show rates
- Leisure routes (MCO, FLL, CUN) book early with low no-show rates
- Competitor fare wars reduce demand on affected routes — consider matching with lower fares

DISRUPTION HANDLING
- Weather, mechanical failures, and crew shortages may affect flights
- Options: cancel_flight, delay_flight, swap_aircraft (to a different type), do_nothing (minor delays only)
- Aircraft swaps cost $5,000 but can avoid cancellation costs ($200/pax) or delay costs ($50/pax/hr)
- All pending disruptions MUST be resolved before you can advance_day()

COST STRUCTURE
- Denied boarding: ${DENIED_BOARDING_COST:.0f} per passenger (DOT regulations)
- Flight cancellation: ${CANCELLATION_COST_PER_PAX:.0f} per passenger (rebooking + accommodation)
- Delay >1 hour: ${DELAY_COST_PER_PAX_PER_HOUR:.0f} per passenger per hour
- Aircraft swap: ${AIRCRAFT_SWAP_FIXED_COST:.0f} fixed cost

REWARD
You earn a reward after each day comparing your net revenue (revenue - costs) against a naive baseline policy. Positive reward means you outperformed the baseline.

You also have access to CLI tools (bash, read, write, edit, glob, grep, ls) to write analysis scripts, build models, or track your strategies in the sandbox filesystem.

Your goal: maximize cumulative reward over the 30-day horizon through superior revenue management decisions.
"""
        return [TextBlock(text=prompt)]

    # -------------------------------------------------------------------
    # Tools
    # -------------------------------------------------------------------

    @tool
    async def view_network_status(self, params: EmptyParams) -> ToolOutput:
        """View current network status including flights, bookings, disruptions, and performance metrics."""
        if self.task_finished:
            return ToolOutput(
                metadata={"error": "Task already completed"},
                blocks=[TextBlock(text="Task is already finished.")],
                finished=True, reward=0.0,
            )

        day = self.sim.current_day
        summary = self.sim.get_network_summary(day)
        pending = self.sim.get_pending_disruptions()

        # Weather / disruption forecast
        disruption_text = ""
        if pending:
            disruption_text = "\nACTIVE DISRUPTIONS (must be resolved before advance_day):\n"
            for d in pending:
                disruption_text += (
                    f"  [{d.disruption_id}] {d.disruption_type.upper()}: "
                    f"{len(d.affected_flight_ids)} flights affected, "
                    f"proposed delay {d.delay_hours}h"
                )
                if d.cancel_flight_ids:
                    disruption_text += f", {len(d.cancel_flight_ids)} severe (recommend cancel)"
                disruption_text += "\n"
                for fid in d.affected_flight_ids:
                    flight = self.sim.flights_by_id.get(fid)
                    if flight:
                        disruption_text += (
                            f"    - {fid}: {flight.route.origin}-{flight.route.destination} "
                            f"({flight.aircraft_type}, {flight.total_booked}/{flight.capacity} booked)"
                        )
                        if fid in d.cancel_flight_ids:
                            disruption_text += " [SEVERE]"
                        disruption_text += "\n"
        else:
            disruption_text = "\nNo active disruptions.\n"

        # Fare war info
        fare_war_text = ""
        active_wars = summary.get("active_fare_wars", [])
        if active_wars:
            fare_war_text = "\nACTIVE COMPETITOR FARE WARS:\n"
            for fw in active_wars:
                fare_war_text += (
                    f"  Routes: {', '.join(fw['routes'])} | "
                    f"Demand reduction: {fw['demand_reduction']} | "
                    f"Days remaining: {fw['days_remaining']}\n"
                )

        # Category stats
        cat_text = "\nBOOKING SUMMARY BY CATEGORY (today's departures):\n"
        cat_text += "Category  | Flights | Booked | Capacity | Load Factor\n"
        cat_text += "----------|---------|--------|----------|------------\n"
        for cat in ("business", "mixed", "leisure"):
            cs = summary["category_stats"].get(cat, {})
            if cs:
                cat_text += (
                    f"{cat:<9} | {cs['flights']:>7} | {cs['total_booked']:>6} | "
                    f"{cs['total_capacity']:>8} | {cs['load_factor']:>8.1%}\n"
                )

        # Performance
        perf_text = (
            f"\nCUMULATIVE PERFORMANCE:\n"
            f"  Revenue: ${summary['cumulative_revenue']:>12,.2f}\n"
            f"  Costs:   ${summary['cumulative_costs']:>12,.2f}\n"
            f"  Net:     ${summary['cumulative_net']:>12,.2f}\n"
            f"  Denied boardings: {summary['total_denied_boardings']}\n"
            f"  Reward so far: {self.cumulative_reward:.4f}\n"
        )

        # Build upcoming flights summary
        upcoming_text = "\nUPCOMING DEPARTURES (next 3 days):\n"
        for d in range(day, min(day + 3, self.total_days + 1)):
            flights = get_flights_for_day(self.sim.flights, d)
            active = [f for f in flights if f.status not in ("cancelled", "departed")]
            if active:
                upcoming_text += f"  Day {d}: {len(active)} flights\n"
                for f in active[:5]:  # show first 5
                    upcoming_text += (
                        f"    {f.flight_id}: {f.route.origin}-{f.route.destination} "
                        f"({f.aircraft_type}) {f.total_booked}/{f.capacity} booked "
                        f"(LF {f.load_factor:.0%}) [{f.status}]\n"
                    )
                if len(active) > 5:
                    upcoming_text += f"    ... and {len(active) - 5} more\n"

        output = (
            f"=== DAY {day} of {self.total_days} ===\n"
            f"Season: {self.scenario.season.title()} | "
            f"Demand multiplier: {self.scenario.demand_multiplier:.2f}\n"
            f"{disruption_text}{fare_war_text}{cat_text}{perf_text}{upcoming_text}"
        )

        return ToolOutput(
            metadata=summary,
            blocks=[TextBlock(text=output)],
            reward=0.0, finished=False,
        )

    @tool
    async def view_flight_details(self, params: ViewFlightDetailsParams) -> ToolOutput:
        """View detailed booking and status information for a specific flight."""
        flight = self.sim.flights_by_id.get(params.flight_id)
        if not flight:
            return ToolOutput(
                metadata={"error": f"Flight not found: {params.flight_id}"},
                blocks=[TextBlock(text=f"Error: Flight '{params.flight_id}' not found.")],
                reward=0.0, finished=False,
            )

        route = flight.route
        days_until = flight.departure_day - self.sim.current_day

        # Fare class table
        fc_table = "Class | Fare    | Booked | Open  | Adv.Purch | No-Show Rate\n"
        fc_table += "------|---------|--------|-------|-----------|------------\n"
        total_revenue_if_full = 0.0
        current_revenue = 0.0
        for fc in FARE_CLASSES:
            fare = flight.fare_for_class(fc.code)
            booked = flight.bookings_by_class.get(fc.code, 0)
            is_open = flight.fare_availability.get(fc.code, False)
            can_open = days_until >= fc.advance_purchase_days
            status = "OPEN" if is_open else "CLOSED"
            if not can_open and is_open:
                status = "OPEN*"  # open but advance purchase not met for new bookings
            fc_table += (
                f"  {fc.code}   | ${fare:>6.2f} | {booked:>6} | {status:<5} | "
                f"{fc.advance_purchase_days:>5}d    | {fc.no_show_rate:>8.0%}\n"
            )
            current_revenue += booked * fare

        # Expected no-shows
        expected_no_shows = 0
        for fc in FARE_CLASSES:
            booked = flight.bookings_by_class.get(fc.code, 0)
            expected_no_shows += booked * fc.no_show_rate

        output = (
            f"=== FLIGHT {flight.flight_id} ===\n"
            f"Route: {route.origin}-{route.destination} ({route.category})\n"
            f"Distance: {route.distance_nm} nm | Block time: {route.block_time_hours}h\n"
            f"Aircraft: {flight.aircraft_type} ({flight.capacity} seats)\n"
            f"Departure: Day {flight.departure_day} ({days_until} days from now)\n"
            f"Status: {flight.status.upper()}"
        )
        if flight.status == "delayed":
            output += f" (delay: {flight.delay_hours}h)"
        output += "\n\n"

        output += f"BOOKINGS: {flight.total_booked} / {flight.capacity} "
        output += f"(Load Factor: {flight.load_factor:.1%})\n"
        output += f"Overbooking limit: {flight.overbooking_limit} "
        output += f"(effective capacity: {flight.capacity + flight.overbooking_limit})\n"
        output += f"Expected no-shows: {expected_no_shows:.1f}\n"
        output += f"Current booking revenue: ${current_revenue:,.2f}\n\n"
        output += fc_table

        # Check for disruptions affecting this flight
        disruptions = [
            d for d in self.sim.disruptions
            if flight.flight_id in d.affected_flight_ids and not d.resolved
        ]
        if disruptions:
            output += "\nACTIVE DISRUPTIONS:\n"
            for d in disruptions:
                output += f"  [{d.disruption_id}] {d.disruption_type}: delay {d.delay_hours}h"
                if flight.flight_id in d.cancel_flight_ids:
                    output += " [SEVERE — cancellation recommended]"
                output += "\n"

        metadata = {
            "flight_id": flight.flight_id,
            "route": f"{route.origin}-{route.destination}",
            "category": route.category,
            "aircraft": flight.aircraft_type,
            "capacity": flight.capacity,
            "total_booked": flight.total_booked,
            "load_factor": round(flight.load_factor, 3),
            "overbooking_limit": flight.overbooking_limit,
            "status": flight.status,
            "departure_day": flight.departure_day,
            "days_until": days_until,
            "expected_no_shows": round(expected_no_shows, 1),
            "bookings_by_class": dict(flight.bookings_by_class),
            "fare_availability": dict(flight.fare_availability),
        }

        return ToolOutput(
            metadata=metadata,
            blocks=[TextBlock(text=output)],
            reward=0.0, finished=False,
        )

    @tool
    async def set_fare_availability(self, params: SetFareAvailabilityParams) -> ToolOutput:
        """Open or close fare classes for a specific flight to control booking flow."""
        flight = self.sim.flights_by_id.get(params.flight_id)
        if not flight:
            return ToolOutput(
                metadata={"error": f"Flight not found: {params.flight_id}"},
                blocks=[TextBlock(text=f"Error: Flight '{params.flight_id}' not found.")],
                reward=0.0, finished=False,
            )

        if flight.status in ("cancelled", "departed"):
            return ToolOutput(
                metadata={"error": f"Flight {params.flight_id} is {flight.status}"},
                blocks=[TextBlock(text=f"Error: Cannot modify fares — flight is {flight.status}.")],
                reward=0.0, finished=False,
            )

        # Validate fare class codes
        invalid = [k for k in params.fare_classes if k not in FARE_CLASS_CODES]
        if invalid:
            return ToolOutput(
                metadata={"error": f"Invalid fare classes: {invalid}"},
                blocks=[TextBlock(text=f"Error: Invalid fare class codes: {invalid}. Valid: {FARE_CLASS_CODES}")],
                reward=0.0, finished=False,
            )

        # Apply changes
        changes = []
        for code, is_open in params.fare_classes.items():
            old = flight.fare_availability.get(code, True)
            flight.fare_availability[code] = is_open
            if old != is_open:
                action = "OPENED" if is_open else "CLOSED"
                fare = flight.fare_for_class(code)
                changes.append(f"  {code} (${fare:.2f}): {action}")

        if changes:
            output = f"Fare availability updated for {params.flight_id}:\n" + "\n".join(changes)
        else:
            output = f"No changes to fare availability for {params.flight_id}."

        return ToolOutput(
            metadata={
                "flight_id": params.flight_id,
                "fare_availability": dict(flight.fare_availability),
                "changes": len(changes),
            },
            blocks=[TextBlock(text=output)],
            reward=0.0, finished=False,
        )

    @tool
    async def set_overbooking_limit(self, params: SetOverbookingLimitParams) -> ToolOutput:
        """Set the overbooking authorization for a flight (extra seats beyond capacity)."""
        flight = self.sim.flights_by_id.get(params.flight_id)
        if not flight:
            return ToolOutput(
                metadata={"error": f"Flight not found: {params.flight_id}"},
                blocks=[TextBlock(text=f"Error: Flight '{params.flight_id}' not found.")],
                reward=0.0, finished=False,
            )

        if flight.status in ("cancelled", "departed"):
            return ToolOutput(
                metadata={"error": f"Flight {params.flight_id} is {flight.status}"},
                blocks=[TextBlock(text=f"Error: Cannot set overbooking — flight is {flight.status}.")],
                reward=0.0, finished=False,
            )

        max_overbook = int(flight.capacity * 0.15)
        if params.overbooking_limit < 0:
            return ToolOutput(
                metadata={"error": "Overbooking limit must be >= 0"},
                blocks=[TextBlock(text="Error: Overbooking limit cannot be negative.")],
                reward=0.0, finished=False,
            )
        if params.overbooking_limit > max_overbook:
            return ToolOutput(
                metadata={"error": f"Max overbooking for {flight.aircraft_type} is {max_overbook}"},
                blocks=[TextBlock(text=f"Error: Maximum overbooking limit for {flight.aircraft_type} ({flight.capacity} seats) is {max_overbook} (15% of capacity).")],
                reward=0.0, finished=False,
            )

        old_limit = flight.overbooking_limit
        flight.overbooking_limit = params.overbooking_limit

        # Estimate expected denied boardings
        expected_no_shows = 0
        for fc in FARE_CLASSES:
            booked = flight.bookings_by_class.get(fc.code, 0)
            expected_no_shows += booked * fc.no_show_rate
        expected_show_ups = flight.total_booked - expected_no_shows
        expected_denied = max(0, expected_show_ups - flight.capacity)

        output = (
            f"Overbooking limit for {params.flight_id}: {old_limit} -> {params.overbooking_limit}\n"
            f"Effective capacity: {flight.capacity + flight.overbooking_limit} "
            f"({flight.capacity} seats + {flight.overbooking_limit} overbook)\n"
            f"Current bookings: {flight.total_booked}\n"
            f"Expected no-shows: {expected_no_shows:.1f}\n"
            f"Expected show-ups: {expected_show_ups:.1f}\n"
            f"Estimated denied boardings: {expected_denied:.1f}\n"
            f"Potential denied boarding cost: ${expected_denied * DENIED_BOARDING_COST:,.0f}"
        )

        return ToolOutput(
            metadata={
                "flight_id": params.flight_id,
                "old_limit": old_limit,
                "new_limit": params.overbooking_limit,
                "effective_capacity": flight.capacity + flight.overbooking_limit,
                "expected_denied": round(expected_denied, 1),
            },
            blocks=[TextBlock(text=output)],
            reward=0.0, finished=False,
        )

    @tool
    async def handle_disruption(self, params: HandleDisruptionParams) -> ToolOutput:
        """Respond to an active disruption by choosing cancel, delay, swap, or do-nothing."""
        # Find the disruption
        disruption = None
        for d in self.sim.disruptions:
            if d.disruption_id == params.disruption_id:
                disruption = d
                break

        if not disruption:
            return ToolOutput(
                metadata={"error": f"Disruption not found: {params.disruption_id}"},
                blocks=[TextBlock(text=f"Error: Disruption '{params.disruption_id}' not found.")],
                reward=0.0, finished=False,
            )

        if disruption.resolved:
            return ToolOutput(
                metadata={"error": "Disruption already resolved"},
                blocks=[TextBlock(text=f"Disruption {params.disruption_id} has already been resolved.")],
                reward=0.0, finished=False,
            )

        valid_actions = {"cancel_flight", "delay_flight", "swap_aircraft", "do_nothing"}
        if params.action not in valid_actions:
            return ToolOutput(
                metadata={"error": f"Invalid action: {params.action}"},
                blocks=[TextBlock(text=f"Error: Invalid action '{params.action}'. Options: {valid_actions}")],
                reward=0.0, finished=False,
            )

        # do_nothing is only valid for delays < 1 hour
        if params.action == "do_nothing" and disruption.delay_hours >= 1.0:
            return ToolOutput(
                metadata={"error": "Cannot do nothing for delays >= 1 hour"},
                blocks=[TextBlock(text=f"Error: 'do_nothing' only valid for delays < 1 hour. This disruption has {disruption.delay_hours}h delay.")],
                reward=0.0, finished=False,
            )

        total_cost = 0.0
        details = []

        for fid in disruption.affected_flight_ids:
            flight = self.sim.flights_by_id.get(fid)
            if not flight or flight.status in ("cancelled", "departed"):
                continue

            if params.action == "cancel_flight":
                cost = self.sim.apply_disruption_cancel(disruption, fid)
                total_cost += cost
                details.append(f"  {fid}: CANCELLED ({flight.total_booked} pax, cost ${cost:,.0f})")

            elif params.action == "delay_flight":
                cost = self.sim.apply_disruption_delay(disruption, fid)
                total_cost += cost
                details.append(
                    f"  {fid}: DELAYED {disruption.delay_hours}h "
                    f"({flight.total_booked} pax, est. cost ${cost:,.0f})"
                )

            elif params.action == "swap_aircraft":
                if not params.swap_aircraft_type:
                    return ToolOutput(
                        metadata={"error": "swap_aircraft_type required for swap_aircraft action"},
                        blocks=[TextBlock(text="Error: Must specify swap_aircraft_type (E175, 737-700, or 737-800).")],
                        reward=0.0, finished=False,
                    )
                cost, err = self.sim.apply_disruption_swap(
                    disruption, fid, params.swap_aircraft_type
                )
                if err:
                    return ToolOutput(
                        metadata={"error": err},
                        blocks=[TextBlock(text=f"Error swapping aircraft for {fid}: {err}")],
                        reward=0.0, finished=False,
                    )
                total_cost += cost
                details.append(
                    f"  {fid}: SWAPPED to {params.swap_aircraft_type} (cost ${cost:,.0f})"
                )

            elif params.action == "do_nothing":
                details.append(f"  {fid}: no action (minor delay)")

        disruption.resolved = True
        disruption.resolution = params.action
        disruption.resolution_cost = total_cost

        output = (
            f"Disruption {disruption.disruption_id} ({disruption.disruption_type}) resolved: {params.action}\n"
            f"Total cost: ${total_cost:,.2f}\n"
            + "\n".join(details)
        )

        return ToolOutput(
            metadata={
                "disruption_id": disruption.disruption_id,
                "action": params.action,
                "total_cost": round(total_cost, 2),
                "flights_affected": len(disruption.affected_flight_ids),
            },
            blocks=[TextBlock(text=output)],
            reward=0.0, finished=False,
        )

    @tool
    async def advance_day(self, params: EmptyParams) -> ToolOutput:
        """Process the current day (departures, bookings, disruptions) and advance to the next day."""
        if self.task_finished:
            return ToolOutput(
                metadata={"error": "Task already completed"},
                blocks=[TextBlock(text="Task is already finished.")],
                finished=True, reward=0.0,
            )

        # Check for unresolved disruptions
        pending = self.sim.get_pending_disruptions()
        if pending:
            ids = [d.disruption_id for d in pending]
            return ToolOutput(
                metadata={"error": "Unresolved disruptions", "disruption_ids": ids},
                blocks=[TextBlock(text=f"Error: Must resolve all disruptions before advancing. Pending: {ids}")],
                reward=0.0, finished=False,
            )

        day = self.sim.current_day
        day_result = self.sim.advance_day()

        # Compare to baseline
        baseline_idx = day - 1
        if baseline_idx < len(self.baseline_results):
            baseline_net = self.baseline_results[baseline_idx].net_revenue
        else:
            baseline_net = 0.0

        agent_net = day_result.net_revenue

        # Reward: normalized difference vs baseline
        denominator = max(abs(baseline_net), 1000.0)  # floor to avoid division issues
        daily_reward = (agent_net - baseline_net) / denominator
        self.cumulative_reward += daily_reward

        # Check if finished
        is_last_day = day >= self.total_days
        if is_last_day:
            self.task_finished = True

        output = (
            f"=== DAY {day} COMPLETE ===\n\n"
            f"DEPARTURES:\n"
            f"  Flights departed: {day_result.flights_departed}\n"
            f"  Flights cancelled: {day_result.flights_cancelled}\n"
            f"  Flights delayed: {day_result.flights_delayed}\n"
            f"  Passengers boarded: {day_result.passengers_boarded}\n"
            f"  Passengers cancelled: {day_result.passengers_cancelled}\n"
            f"  Denied boardings: {day_result.denied_boardings}\n\n"
            f"FINANCIALS:\n"
            f"  Revenue: ${day_result.revenue:>12,.2f}\n"
            f"  Costs:   ${day_result.costs:>12,.2f}\n"
            f"  Net:     ${day_result.net_revenue:>12,.2f}\n\n"
            f"BOOKINGS:\n"
            f"  New bookings today: {day_result.new_bookings}\n"
            f"  Customers spilled: {day_result.spilled_customers}\n\n"
            f"BASELINE COMPARISON:\n"
            f"  Baseline net: ${baseline_net:>12,.2f}\n"
            f"  Your net:     ${agent_net:>12,.2f}\n"
            f"  Difference:   ${agent_net - baseline_net:>+12,.2f}\n"
            f"  Daily reward: {daily_reward:>+.4f}\n\n"
            f"CUMULATIVE:\n"
            f"  Total revenue: ${self.sim.cumulative_revenue:>12,.2f}\n"
            f"  Total costs:   ${self.sim.cumulative_costs:>12,.2f}\n"
            f"  Cumulative reward: {self.cumulative_reward:>+.4f}\n"
        )

        if is_last_day:
            output += (
                f"\n{'='*50}\n"
                f"SIMULATION COMPLETE — 30 DAYS\n"
                f"Final cumulative reward: {self.cumulative_reward:+.4f}\n"
                f"{'='*50}\n"
            )

        metadata = {
            "day": day,
            "revenue": round(day_result.revenue, 2),
            "costs": round(day_result.costs, 2),
            "net_revenue": round(day_result.net_revenue, 2),
            "baseline_net": round(baseline_net, 2),
            "daily_reward": round(daily_reward, 4),
            "cumulative_reward": round(self.cumulative_reward, 4),
            "denied_boardings": day_result.denied_boardings,
            "passengers_boarded": day_result.passengers_boarded,
            "new_bookings": day_result.new_bookings,
            "spilled": day_result.spilled_customers,
            "finished": is_last_day,
        }

        return ToolOutput(
            metadata=metadata,
            blocks=[TextBlock(text=output)],
            reward=self.cumulative_reward if is_last_day else daily_reward,
            finished=is_last_day,
        )

    # -------------------------------------------------------------------
    # Task listing
    # -------------------------------------------------------------------

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        if split == "train":
            tasks = []
            for scenario in ["summer_peak", "winter_holiday", "shoulder_spring"]:
                for variant in range(1, 4):
                    tasks.append({"id": f"{scenario}_v{variant}"})
            return tasks
        elif split == "test":
            tasks = []
            for variant in range(1, 4):
                tasks.append({"id": f"fall_business_v{variant}"})
            return tasks
        else:
            raise ValueError(f"Unknown split: {split}. Options: train, test")

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["train", "test"]
