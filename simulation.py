"""
Stochastic simulation engine for the Airline RM environment.

Handles demand generation, booking simulation, no-show simulation,
disruption generation, and day-processing logic. All randomness flows
through a numpy Generator for deterministic replay.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from network import (
    AIRCRAFT_TYPES,
    AIRCRAFT_SWAP_FIXED_COST,
    CANCELLATION_COST_PER_PAX,
    COMPETITOR_FARE_WAR,
    DELAY_COST_PER_PAX_PER_HOUR,
    DELAY_THRESHOLD_HOURS,
    DENIED_BOARDING_COST,
    DISRUPTION_TYPES,
    FARE_CLASSES,
    FARE_CLASS_CODES,
    AircraftType,
    Route,
    ROUTES,
    Scenario,
    SCENARIOS,
    ScheduledFlight,
    generate_flight_schedule,
    get_flights_for_day,
    route_key,
)


# ---------------------------------------------------------------------------
# Booking curve helpers
# ---------------------------------------------------------------------------

def _booking_curve_cdf(days_before_departure: int, category: str) -> float:
    """
    Return the cumulative fraction of total demand that has arrived
    by *days_before_departure* before the flight.

    Business routes book late; leisure routes book early.

    Uses a power-curve approximation (x^2.5 for business, 1-(1-x)^2.5
    for leisure, cubic Hermite for mixed). While the literature favors
    exponential forms (Shintani & Umeno 2023, "Average Booking Curves Draw
    Exponential Functions", Scientific Reports 13:15773) or nonhomogeneous
    Poisson arrival rates (Talluri & van Ryzin 2004, "The Theory and
    Practice of Revenue Management", Springer), the qualitative
    business-late / leisure-early asymmetry is well-established.

    Calibrated to Belobaba 1989 ("Application of a Probabilistic Decision
    Model to Airline Seat Inventory Control", Operations Research 37(2))
    and Embark Aviation booking-curve data.
    """
    # x = timeline progression from 0 (60 days out) to 1 (departure day)
    x = max(0.0, min(1.0, (60 - days_before_departure) / 60.0))

    if category == "business":
        # Heavy late booking: CDF is convex (slow early, fast late)
        return x ** 2.5
    elif category == "leisure":
        # Heavy early booking: CDF is concave (fast early, slow late)
        return 1.0 - (1.0 - x) ** 2.5
    else:  # mixed
        # Roughly S-shaped
        return 3 * x ** 2 - 2 * x ** 3


def _booking_curve_daily_fraction(days_before: int, category: str) -> float:
    """Fraction of total demand arriving exactly *days_before* departure."""
    if days_before >= 60:
        return 0.0
    cdf_today = _booking_curve_cdf(days_before, category)
    cdf_yesterday = _booking_curve_cdf(days_before + 1, category) if days_before < 59 else 0.0
    return max(0.0, cdf_today - cdf_yesterday)


# ---------------------------------------------------------------------------
# Disruption data structures
# ---------------------------------------------------------------------------

@dataclass
class Disruption:
    disruption_id: str
    disruption_type: str       # e.g. "thunderstorm"
    day: int
    affected_flight_ids: List[str]
    delay_hours: float         # proposed delay (if not cancelled)
    cancel_flight_ids: List[str]  # flights that *must* be cancelled (severity)
    resolved: bool = False
    resolution: str = ""       # "cancel_flight", "delay_flight", "swap_aircraft", "do_nothing"
    resolution_cost: float = 0.0


@dataclass
class ActiveFareWar:
    """Tracks an ongoing competitor fare war."""
    affected_route_keys: List[str]
    demand_reduction_pct: float
    start_day: int
    end_day: int


# ---------------------------------------------------------------------------
# Day result
# ---------------------------------------------------------------------------

@dataclass
class DayResult:
    day: int
    revenue: float = 0.0
    costs: float = 0.0
    denied_boardings: int = 0
    passengers_boarded: int = 0
    passengers_cancelled: int = 0
    flights_departed: int = 0
    flights_cancelled: int = 0
    flights_delayed: int = 0
    new_bookings: int = 0
    spilled_customers: int = 0

    @property
    def net_revenue(self) -> float:
        return self.revenue - self.costs


# ---------------------------------------------------------------------------
# Simulation State
# ---------------------------------------------------------------------------

class SimulationState:
    """
    Full mutable state of the airline network simulation.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        scenario: Scenario,
        total_days: int = 30,
    ):
        self.rng = rng
        self.scenario = scenario
        self.total_days = total_days

        # Generate flight schedule
        self.flights: List[ScheduledFlight] = generate_flight_schedule(total_days)
        self.flights_by_id: Dict[str, ScheduledFlight] = {
            f.flight_id: f for f in self.flights
        }

        # Pre-populate historical bookings (days before the simulation starts)
        self._populate_initial_bookings()

        # Disruption state
        self.disruptions: List[Disruption] = []
        self.active_fare_wars: List[ActiveFareWar] = []

        # Tracking
        self.current_day = 1
        self.day_results: List[DayResult] = []
        self.cumulative_revenue = 0.0
        self.cumulative_costs = 0.0
        self.total_denied_boardings = 0

    # -------------------------------------------------------------------
    # Initial booking population
    # -------------------------------------------------------------------

    def _populate_initial_bookings(self):
        """
        Simulate bookings that have already occurred before day 1.

        For a flight departing on day D, bookings from day (D-60) through day 0
        have already occurred. We simulate these with the default (all classes open)
        fare availability.
        """
        for flight in self.flights:
            days_until_departure = flight.departure_day  # from perspective of day 0
            # Bookings from 60 days out up to day 0
            booking_start = min(60, days_until_departure + 60)
            for days_before in range(booking_start, days_until_departure, -1):
                self._simulate_booking_arrivals_for_flight(
                    flight, days_before, demand_multiplier=1.0
                )

    # -------------------------------------------------------------------
    # Booking simulation
    # -------------------------------------------------------------------

    def _simulate_booking_arrivals_for_flight(
        self,
        flight: ScheduledFlight,
        days_before_departure: int,
        demand_multiplier: float = 1.0,
    ) -> Tuple[int, int]:
        """
        Simulate one day of booking arrivals for a single flight.

        Returns (bookings_made, customers_spilled).
        """
        if flight.status == "cancelled":
            return 0, 0

        route = flight.route
        category = route.category

        # Daily fraction of total demand for this point in booking curve
        daily_frac = _booking_curve_daily_fraction(days_before_departure, category)

        # Total demand for this flight = route demand / daily_frequencies
        flight_demand_mean = (
            route.base_demand_mean
            * self.scenario.demand_multiplier
            * demand_multiplier
            / route.daily_frequencies
        )

        # Expected arrivals today
        expected_arrivals = flight_demand_mean * daily_frac
        if expected_arrivals <= 0:
            return 0, 0

        # Draw actual number of arrivals (Poisson)
        n_arrivals = int(self.rng.poisson(max(0.01, expected_arrivals)))

        # Determine business vs leisure split
        n_business = int(self.rng.binomial(n_arrivals, route.business_traveler_pct))
        n_leisure = n_arrivals - n_business

        bookings = 0
        spills = 0

        # Willingness-to-pay (WTP) drawn from LogNormal distribution.
        # The standard in airline RM is exponential reservation prices
        # (Gallego & van Ryzin 1994, "Optimal Dynamic Pricing of
        # Inventories with Stochastic Demand", Management Science 40(8)).
        # LogNormal is defensible from the broader contingent valuation
        # literature and better captures the heavy right tail of business
        # traveler WTP while remaining strictly positive.
        for _ in range(n_business):
            wtp = self.rng.lognormal(
                math.log(max(1.0, 0.70 * route.max_fare)), 0.30
            )
            booked = self._try_book_passenger(flight, wtp, days_before_departure)
            if booked:
                bookings += 1
            else:
                spills += 1

        for _ in range(n_leisure):
            wtp = self.rng.lognormal(
                math.log(max(1.0, 0.25 * route.max_fare)), 0.40
            )
            booked = self._try_book_passenger(flight, wtp, days_before_departure)
            if booked:
                bookings += 1
            else:
                spills += 1

        return bookings, spills

    def _try_book_passenger(
        self,
        flight: ScheduledFlight,
        willingness_to_pay: float,
        days_before_departure: int,
    ) -> bool:
        """
        Try to book a single passenger on the lowest available fare class
        whose fare ≤ WTP and that is open with seats available.

        Fare classes are tried from lowest (L) to highest (Y) — the customer
        wants the cheapest acceptable option.
        """
        effective_capacity = flight.capacity + flight.overbooking_limit

        if flight.total_booked >= effective_capacity:
            return False

        # Try fare classes from cheapest to most expensive
        for fc in reversed(FARE_CLASSES):
            fare = flight.fare_for_class(fc.code)
            # Check: class is open, customer can afford, advance purchase met
            if (
                flight.fare_availability.get(fc.code, False)
                and fare <= willingness_to_pay
                and days_before_departure >= fc.advance_purchase_days
            ):
                flight.bookings_by_class[fc.code] += 1
                return True

        return False

    # -------------------------------------------------------------------
    # No-show and departure simulation
    # -------------------------------------------------------------------

    def simulate_departure(self, flight: ScheduledFlight) -> DayResult:
        """
        Simulate the departure of a single flight: no-shows, denied boardings,
        revenue calculation.
        """
        result = DayResult(day=flight.departure_day)

        if flight.status == "cancelled":
            # All passengers need rebooking
            total_pax = flight.total_booked
            result.passengers_cancelled = total_pax
            result.costs = total_pax * CANCELLATION_COST_PER_PAX
            result.flights_cancelled = 1
            return result

        # Simulate no-shows per fare class
        show_ups_by_class: Dict[str, int] = {}
        for fc in FARE_CLASSES:
            booked = flight.bookings_by_class.get(fc.code, 0)
            if booked > 0:
                no_shows = int(self.rng.binomial(booked, fc.no_show_rate))
                show_ups_by_class[fc.code] = booked - no_shows
            else:
                show_ups_by_class[fc.code] = 0

        total_show_ups = sum(show_ups_by_class.values())

        # Denied boardings (if show-ups exceed capacity)
        denied = max(0, total_show_ups - flight.capacity)
        boarded = min(total_show_ups, flight.capacity)

        # Revenue from boarded passengers (board highest-paying first)
        revenue = 0.0
        remaining_seats = flight.capacity
        for fc in FARE_CLASSES:  # highest fare first
            pax = show_ups_by_class.get(fc.code, 0)
            if pax <= 0:
                continue
            board_from_class = min(pax, remaining_seats)
            fare = flight.fare_for_class(fc.code)
            revenue += board_from_class * fare
            remaining_seats -= board_from_class
            if remaining_seats <= 0:
                break

        # Costs
        cost = denied * DENIED_BOARDING_COST

        # Delay costs
        if flight.status == "delayed" and flight.delay_hours > DELAY_THRESHOLD_HOURS:
            cost += boarded * DELAY_COST_PER_PAX_PER_HOUR * flight.delay_hours

        result.revenue = revenue
        result.costs = cost
        result.denied_boardings = denied
        result.passengers_boarded = boarded
        result.flights_departed = 1
        if flight.status == "delayed":
            result.flights_delayed = 1

        flight.status = "departed"
        return result

    # -------------------------------------------------------------------
    # Process all departures for a day
    # -------------------------------------------------------------------

    def process_departures(self, day: int) -> DayResult:
        """Process all flight departures for a given day."""
        day_flights = get_flights_for_day(self.flights, day)
        combined = DayResult(day=day)

        for flight in day_flights:
            if flight.status == "departed":
                continue
            fr = self.simulate_departure(flight)
            combined.revenue += fr.revenue
            combined.costs += fr.costs
            combined.denied_boardings += fr.denied_boardings
            combined.passengers_boarded += fr.passengers_boarded
            combined.passengers_cancelled += fr.passengers_cancelled
            combined.flights_departed += fr.flights_departed
            combined.flights_cancelled += fr.flights_cancelled
            combined.flights_delayed += fr.flights_delayed

        return combined

    # -------------------------------------------------------------------
    # Process new bookings for all future flights
    # -------------------------------------------------------------------

    def process_new_bookings(self, current_day: int) -> Tuple[int, int]:
        """
        Simulate one day of new booking arrivals for all future flights.
        Returns (total_bookings, total_spills).
        """
        total_bookings = 0
        total_spills = 0

        for flight in self.flights:
            if flight.departure_day <= current_day:
                continue  # already departed or departing today
            if flight.status == "cancelled":
                continue

            days_before = flight.departure_day - current_day

            # Check for active fare wars affecting this route
            demand_mult = 1.0
            rk = route_key(flight.route)
            for fw in self.active_fare_wars:
                if rk in fw.affected_route_keys and fw.start_day <= current_day <= fw.end_day:
                    demand_mult *= (1.0 - fw.demand_reduction_pct)

            bookings, spills = self._simulate_booking_arrivals_for_flight(
                flight, days_before, demand_multiplier=demand_mult
            )
            total_bookings += bookings
            total_spills += spills

        return total_bookings, total_spills

    # -------------------------------------------------------------------
    # Disruption generation
    # -------------------------------------------------------------------

    def generate_disruptions(self, day: int) -> List[Disruption]:
        """Generate disruptions for a given day based on scenario and season."""
        new_disruptions: List[Disruption] = []
        season = self.scenario.season
        day_flights = get_flights_for_day(self.flights, day)

        if not day_flights:
            return new_disruptions

        active_flight_ids = [
            f.flight_id for f in day_flights
            if f.status not in ("cancelled", "departed")
        ]

        if not active_flight_ids:
            return new_disruptions

        disruption_counter = len(self.disruptions)

        for dtype_name, dtype in DISRUPTION_TYPES.items():
            seasonal_mult = dtype.seasonal_multipliers.get(season, 1.0)
            prob = dtype.base_probability_per_day * seasonal_mult
            if prob <= 0:
                continue

            if self.rng.random() < prob:
                # Determine number of affected flights
                pct_lo, pct_hi = dtype.affected_flights_pct_range
                pct = self.rng.uniform(pct_lo, pct_hi)
                n_affected = max(1, int(round(len(active_flight_ids) * pct)))
                n_affected = min(n_affected, len(active_flight_ids))

                affected_ids = list(
                    self.rng.choice(active_flight_ids, size=n_affected, replace=False)
                )

                # Delay hours
                delay_lo, delay_hi = dtype.delay_hours_range
                delay = float(self.rng.uniform(delay_lo, delay_hi))

                # Which flights are so severely affected they should be cancelled
                cancel_ids = []
                for fid in affected_ids:
                    if self.rng.random() < dtype.cancellation_pct:
                        cancel_ids.append(fid)

                disruption_counter += 1
                d = Disruption(
                    disruption_id=f"DISRUPT-{day}-{disruption_counter}",
                    disruption_type=dtype_name,
                    day=day,
                    affected_flight_ids=affected_ids,
                    delay_hours=round(delay, 1),
                    cancel_flight_ids=cancel_ids,
                )
                new_disruptions.append(d)

        # Competitor fare wars
        fw = COMPETITOR_FARE_WAR
        fw_prob = fw.base_probability_per_day * fw.seasonal_multipliers.get(season, 1.0)
        if self.rng.random() < fw_prob:
            n_routes = int(self.rng.integers(fw.affected_routes_range[0], fw.affected_routes_range[1] + 1))
            route_keys_all = [route_key(r) for r in ROUTES]
            affected_rks = list(self.rng.choice(route_keys_all, size=min(n_routes, len(route_keys_all)), replace=False))
            reduction = float(self.rng.uniform(*fw.demand_reduction_pct_range))
            duration = int(self.rng.integers(fw.duration_days_range[0], fw.duration_days_range[1] + 1))

            fare_war = ActiveFareWar(
                affected_route_keys=affected_rks,
                demand_reduction_pct=round(reduction, 3),
                start_day=day,
                end_day=min(day + duration - 1, self.total_days),
            )
            self.active_fare_wars.append(fare_war)

        self.disruptions.extend(new_disruptions)
        return new_disruptions

    # -------------------------------------------------------------------
    # Disruption resolution helpers
    # -------------------------------------------------------------------

    def apply_disruption_cancel(self, disruption: Disruption, flight_id: str) -> float:
        """Cancel a flight. Returns cost."""
        flight = self.flights_by_id.get(flight_id)
        if not flight or flight.status in ("cancelled", "departed"):
            return 0.0
        flight.status = "cancelled"
        cost = flight.total_booked * CANCELLATION_COST_PER_PAX
        return cost

    def apply_disruption_delay(self, disruption: Disruption, flight_id: str) -> float:
        """Delay a flight. Returns estimated cost (actual cost at departure)."""
        flight = self.flights_by_id.get(flight_id)
        if not flight or flight.status in ("cancelled", "departed"):
            return 0.0
        flight.status = "delayed"
        flight.delay_hours = disruption.delay_hours
        # Cost estimate (actual cost computed at departure)
        if disruption.delay_hours > DELAY_THRESHOLD_HOURS:
            return flight.total_booked * DELAY_COST_PER_PAX_PER_HOUR * disruption.delay_hours
        return 0.0

    def apply_disruption_swap(
        self, disruption: Disruption, flight_id: str, new_aircraft_code: str
    ) -> Tuple[float, str]:
        """
        Swap aircraft for a flight. Returns (cost, error_message).
        Error message is empty on success.
        """
        flight = self.flights_by_id.get(flight_id)
        if not flight or flight.status in ("cancelled", "departed"):
            return 0.0, "Flight not available for swap"

        if new_aircraft_code not in AIRCRAFT_TYPES:
            return 0.0, f"Unknown aircraft type: {new_aircraft_code}"

        new_ac = AIRCRAFT_TYPES[new_aircraft_code]

        # Check range capability
        if flight.route.distance_nm > new_ac.range_nm:
            return 0.0, (
                f"{new_aircraft_code} has range {new_ac.range_nm} nm, "
                f"but route requires {flight.route.distance_nm} nm"
            )

        # Apply swap
        old_capacity = flight.capacity
        flight.aircraft_type = new_aircraft_code
        flight.capacity = new_ac.total_seats

        # If we downguaged and have more bookings than new capacity + overbook,
        # we don't force cancellations here — denied boardings happen at departure
        cost = AIRCRAFT_SWAP_FIXED_COST

        # Clear the delay since we're resolving with a swap
        flight.status = "scheduled"
        flight.delay_hours = 0.0

        return cost, ""

    # -------------------------------------------------------------------
    # Advance day
    # -------------------------------------------------------------------

    def advance_day(self) -> DayResult:
        """
        Process current day: departures, new bookings, disruptions for tomorrow.
        Returns DayResult for the current day.
        """
        day = self.current_day

        # 1. Process departures
        dep_result = self.process_departures(day)

        # 2. Process new bookings for future flights
        new_bookings, spills = self.process_new_bookings(day)
        dep_result.new_bookings = new_bookings
        dep_result.spilled_customers = spills

        # 3. Update cumulative tracking
        self.cumulative_revenue += dep_result.revenue
        self.cumulative_costs += dep_result.costs
        self.total_denied_boardings += dep_result.denied_boardings

        # 4. Store result
        self.day_results.append(dep_result)

        # 5. Generate disruptions for tomorrow (if not last day)
        if day < self.total_days:
            self.generate_disruptions(day + 1)

        # 6. Advance
        self.current_day += 1

        return dep_result

    # -------------------------------------------------------------------
    # Query helpers
    # -------------------------------------------------------------------

    def get_pending_disruptions(self) -> List[Disruption]:
        """Return unresolved disruptions for the current day."""
        return [
            d for d in self.disruptions
            if d.day == self.current_day and not d.resolved
        ]

    def get_network_summary(self, day: int) -> Dict:
        """Compute summary statistics for the network on a given day."""
        day_flights = get_flights_for_day(self.flights, day)
        total = len(day_flights)
        on_time = sum(1 for f in day_flights if f.status == "scheduled")
        delayed = sum(1 for f in day_flights if f.status == "delayed")
        cancelled = sum(1 for f in day_flights if f.status == "cancelled")
        departed = sum(1 for f in day_flights if f.status == "departed")

        # Booking summary by route category
        category_stats: Dict[str, Dict] = {}
        for cat in ("business", "mixed", "leisure"):
            cat_flights = [f for f in day_flights if f.route.category == cat]
            if cat_flights:
                total_booked = sum(f.total_booked for f in cat_flights)
                total_capacity = sum(f.capacity for f in cat_flights)
                avg_lf = total_booked / total_capacity if total_capacity > 0 else 0.0
                category_stats[cat] = {
                    "flights": len(cat_flights),
                    "total_booked": total_booked,
                    "total_capacity": total_capacity,
                    "load_factor": round(avg_lf, 3),
                }

        # Active fare wars
        active_wars = [
            fw for fw in self.active_fare_wars
            if fw.start_day <= day <= fw.end_day
        ]

        return {
            "day": day,
            "total_flights": total,
            "on_time": on_time,
            "delayed": delayed,
            "cancelled": cancelled,
            "departed": departed,
            "category_stats": category_stats,
            "active_fare_wars": [
                {
                    "routes": fw.affected_route_keys,
                    "demand_reduction": f"{fw.demand_reduction_pct:.1%}",
                    "days_remaining": fw.end_day - day + 1,
                }
                for fw in active_wars
            ],
            "cumulative_revenue": round(self.cumulative_revenue, 2),
            "cumulative_costs": round(self.cumulative_costs, 2),
            "cumulative_net": round(self.cumulative_revenue - self.cumulative_costs, 2),
            "total_denied_boardings": self.total_denied_boardings,
        }

    def deep_copy_flights(self) -> List[ScheduledFlight]:
        """Deep copy all flights for baseline simulation."""
        return [f.copy() for f in self.flights]
