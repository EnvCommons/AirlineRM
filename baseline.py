"""
Naive baseline revenue management policy.

Opens all fare classes (respecting advance-purchase rules), never overbooks,
and handles disruptions with simple delay-or-cancel logic. Used as the
reference against which agent performance is measured.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from network import (
    FARE_CLASSES,
    Scenario,
    ScheduledFlight,
    get_flights_for_day,
)
from simulation import DayResult, Disruption, SimulationState


class BaselinePolicy:
    """
    Run a full 30-day simulation with a naive RM policy.

    The baseline:
      1. Opens all fare classes whose advance-purchase is met (no strategic closure).
      2. Never overbooks (overbooking_limit = 0 on all flights).
      3. On disruption: delays if proposed delay < 3 hours, else cancels.
      4. Never swaps aircraft.
    """

    def __init__(self, rng: np.random.Generator, scenario: Scenario, total_days: int = 30):
        self.rng = rng
        self.scenario = scenario
        self.total_days = total_days

    def run_full_simulation(self) -> List[DayResult]:
        """
        Execute the baseline policy for the full horizon.
        Returns a list of DayResult, one per day.
        """
        sim = SimulationState(self.rng, self.scenario, self.total_days)

        results: List[DayResult] = []

        for day in range(1, self.total_days + 1):
            sim.current_day = day

            # Generate disruptions for this day (on day 1 we also generate)
            if day == 1:
                sim.generate_disruptions(day)

            # --- Baseline fare availability: open everything within advance-purchase ---
            for flight in sim.flights:
                if flight.status in ("cancelled", "departed"):
                    continue
                if flight.departure_day < day:
                    continue
                days_before = flight.departure_day - day
                for fc in FARE_CLASSES:
                    flight.fare_availability[fc.code] = (
                        days_before >= fc.advance_purchase_days
                    )

            # --- Baseline overbooking: never overbook ---
            # (default is 0, no action needed)

            # --- Baseline disruption handling ---
            pending = sim.get_pending_disruptions()
            for disruption in pending:
                for fid in disruption.affected_flight_ids:
                    if fid in disruption.cancel_flight_ids:
                        # Must cancel
                        sim.apply_disruption_cancel(disruption, fid)
                    elif disruption.delay_hours < 3.0:
                        sim.apply_disruption_delay(disruption, fid)
                    else:
                        sim.apply_disruption_cancel(disruption, fid)
                disruption.resolved = True

            # --- Advance the day ---
            day_result = sim.advance_day()
            results.append(day_result)

        return results
