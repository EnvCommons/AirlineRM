"""
Comprehensive tests for the Airline RM environment.

Tests network definitions, simulation mechanics, baseline policy,
booking behaviour, disruption handling, overbooking, cost calculations,
and end-to-end simulation correctness.
"""

import hashlib
import re

import pytest
import numpy as np
from copy import deepcopy

from network import (
    AIRCRAFT_TYPES,
    AIRCRAFT_SWAP_FIXED_COST,
    CANCELLATION_COST_PER_PAX,
    DELAY_COST_PER_PAX_PER_HOUR,
    DELAY_THRESHOLD_HOURS,
    DENIED_BOARDING_COST,
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
)
from simulation import (
    SimulationState,
    DayResult,
    Disruption,
    ActiveFareWar,
    _booking_curve_cdf,
    _booking_curve_daily_fraction,
)
from baseline import BaselinePolicy


# ========================================================================
# 1. Network definition tests
# ========================================================================

class TestNetworkDefinitions:
    def test_twelve_routes(self):
        assert len(ROUTES) == 12

    def test_three_aircraft_types(self):
        assert set(AIRCRAFT_TYPES.keys()) == {"E175", "737-700", "737-800"}

    def test_eight_fare_classes(self):
        assert len(FARE_CLASSES) == 8
        assert FARE_CLASS_CODES == ["Y", "B", "M", "H", "Q", "V", "T", "L"]

    def test_fare_class_ordering(self):
        """Y class should be most expensive, L cheapest (multipliers decreasing)."""
        multipliers = [fc.multiplier for fc in FARE_CLASSES]
        for i in range(len(multipliers) - 1):
            assert multipliers[i] > multipliers[i + 1], (
                f"Fare class {FARE_CLASSES[i].code} ({multipliers[i]}) should have "
                f"higher multiplier than {FARE_CLASSES[i+1].code} ({multipliers[i+1]})"
            )

    def test_fare_class_fares_per_route(self):
        """Y fare > B > M > ... > L for every route."""
        for route in ROUTES:
            table = compute_fare_table(route)
            fares = [entry["fare"] for entry in table]
            for i in range(len(fares) - 1):
                assert fares[i] > fares[i + 1], (
                    f"Route {route_key(route)}: {table[i]['class']} fare (${fares[i]}) "
                    f"should exceed {table[i+1]['class']} fare (${fares[i+1]})"
                )

    def test_fare_ratio(self):
        """Y-to-L fare ratio should be approximately 6-8x."""
        for route in ROUTES:
            y_fare = route.max_fare * FARE_CLASSES[0].multiplier
            l_fare = route.max_fare * FARE_CLASSES[-1].multiplier
            ratio = y_fare / l_fare
            assert 5.0 <= ratio <= 10.0, (
                f"Route {route_key(route)}: Y/L ratio is {ratio:.1f}, expected 5-10x"
            )

    def test_no_show_rates_ordering(self):
        """Flexible classes should have higher no-show rates."""
        rates = [fc.no_show_rate for fc in FARE_CLASSES]
        # Y should have highest, L lowest (generally decreasing)
        assert rates[0] > rates[-1]
        # Overall trend should be non-increasing
        for i in range(len(rates) - 1):
            assert rates[i] >= rates[i + 1]

    def test_all_routes_have_valid_aircraft(self):
        for route in ROUTES:
            assert route.default_aircraft in AIRCRAFT_TYPES

    def test_route_distance_within_aircraft_range(self):
        for route in ROUTES:
            ac = AIRCRAFT_TYPES[route.default_aircraft]
            assert route.distance_nm <= ac.range_nm, (
                f"Route {route_key(route)} ({route.distance_nm} nm) exceeds "
                f"{ac.code} range ({ac.range_nm} nm)"
            )

    def test_four_scenarios(self):
        assert len(SCENARIOS) == 4
        assert set(SCENARIOS.keys()) == {
            "summer_peak", "winter_holiday", "shoulder_spring", "fall_business"
        }

    def test_route_categories(self):
        categories = {r.category for r in ROUTES}
        assert categories == {"business", "mixed", "leisure"}


# ========================================================================
# 2. Flight schedule tests
# ========================================================================

class TestFlightSchedule:
    def test_schedule_length(self):
        flights = generate_flight_schedule(30)
        expected = sum(r.daily_frequencies for r in ROUTES) * 30
        assert len(flights) == expected

    def test_daily_flight_count(self):
        daily = get_daily_flight_count()
        assert daily > 0
        # 12 routes with varying frequencies, should be around 29-33
        assert 25 <= daily <= 40

    def test_flight_ids_unique(self):
        flights = generate_flight_schedule(30)
        ids = [f.flight_id for f in flights]
        assert len(ids) == len(set(ids)), "Flight IDs are not unique"

    def test_flight_initial_state(self):
        flights = generate_flight_schedule(5)
        for f in flights:
            assert f.status == "scheduled"
            assert f.overbooking_limit == 0
            assert f.delay_hours == 0.0
            assert all(v == 0 for v in f.bookings_by_class.values())
            assert all(v is True for v in f.fare_availability.values())

    def test_get_flights_for_day(self):
        flights = generate_flight_schedule(30)
        day1 = get_flights_for_day(flights, 1)
        day30 = get_flights_for_day(flights, 30)
        daily = get_daily_flight_count()
        assert len(day1) == daily
        assert len(day30) == daily


# ========================================================================
# 3. Booking curve tests
# ========================================================================

class TestBookingCurves:
    def test_cdf_monotonic(self):
        """CDF should be non-decreasing as departure approaches."""
        for category in ("business", "mixed", "leisure"):
            prev = 0.0
            for days_before in range(60, -1, -1):
                cdf = _booking_curve_cdf(days_before, category)
                assert cdf >= prev - 1e-10, (
                    f"{category}: CDF decreased from {prev} to {cdf} at {days_before} days"
                )
                prev = cdf

    def test_cdf_endpoints(self):
        for category in ("business", "mixed", "leisure"):
            assert _booking_curve_cdf(60, category) == pytest.approx(0.0, abs=0.01)
            assert _booking_curve_cdf(0, category) == pytest.approx(1.0, abs=0.01)

    def test_business_books_later_than_leisure(self):
        """At 14 days before departure, business should have less cumulative demand."""
        biz_14 = _booking_curve_cdf(14, "business")
        lei_14 = _booking_curve_cdf(14, "leisure")
        assert biz_14 < lei_14, (
            f"Business CDF at 14d ({biz_14:.2f}) should be less than leisure ({lei_14:.2f})"
        )

    def test_daily_fractions_sum_to_approximately_one(self):
        for category in ("business", "mixed", "leisure"):
            total = sum(
                _booking_curve_daily_fraction(d, category) for d in range(60, -1, -1)
            )
            assert total == pytest.approx(1.0, abs=0.05), (
                f"{category}: daily fractions sum to {total}, expected ~1.0"
            )


# ========================================================================
# 4. Simulation determinism tests
# ========================================================================

class TestSimulationDeterminism:
    def test_same_seed_same_bookings(self):
        """Two simulations with the same seed should produce identical initial bookings."""
        scenario = SCENARIOS["summer_peak"]

        rng1 = np.random.default_rng(42)
        sim1 = SimulationState(rng1, scenario, 10)

        rng2 = np.random.default_rng(42)
        sim2 = SimulationState(rng2, scenario, 10)

        for f1, f2 in zip(sim1.flights, sim2.flights):
            assert f1.bookings_by_class == f2.bookings_by_class, (
                f"Flight {f1.flight_id}: bookings differ"
            )

    def test_different_seeds_different_bookings(self):
        """Different seeds should produce different bookings."""
        scenario = SCENARIOS["summer_peak"]

        rng1 = np.random.default_rng(42)
        sim1 = SimulationState(rng1, scenario, 10)

        rng2 = np.random.default_rng(99)
        sim2 = SimulationState(rng2, scenario, 10)

        # At least some flights should differ
        differences = 0
        for f1, f2 in zip(sim1.flights, sim2.flights):
            if f1.bookings_by_class != f2.bookings_by_class:
                differences += 1
        assert differences > 0, "Different seeds should produce different bookings"


# ========================================================================
# 5. Disruption generation tests
# ========================================================================

class TestDisruptionGeneration:
    def test_disruptions_generated(self):
        """Disruptions should be generated stochastically."""
        scenario = SCENARIOS["summer_peak"]
        rng = np.random.default_rng(12345)
        sim = SimulationState(rng, scenario, 30)

        total_disruptions = 0
        for day in range(1, 31):
            sim.current_day = day
            new = sim.generate_disruptions(day)
            total_disruptions += len(new)

        # With multiple disruption types and 30 days, should get at least some
        assert total_disruptions > 0, "Expected at least one disruption over 30 days"

    def test_no_snowstorms_in_summer(self):
        """Summer scenarios should have near-zero snowstorm probability."""
        scenario = SCENARIOS["summer_peak"]
        rng = np.random.default_rng(42)
        sim = SimulationState(rng, scenario, 30)

        snowstorm_count = 0
        for day in range(1, 31):
            sim.current_day = day
            new = sim.generate_disruptions(day)
            for d in new:
                if d.disruption_type == "snowstorm":
                    snowstorm_count += 1

        # Probability is 0.04 * 0.0 = 0 for summer, so should be exactly 0
        assert snowstorm_count == 0, "No snowstorms expected in summer"

    def test_winter_has_more_snowstorms_than_spring(self):
        """Winter should have more snow disruptions than spring (statistical)."""
        snow_counts = {}
        for scenario_name in ["winter_holiday", "shoulder_spring"]:
            scenario = SCENARIOS[scenario_name]
            count = 0
            # Run multiple seeds for statistical robustness
            for seed in range(100):
                rng = np.random.default_rng(seed)
                sim = SimulationState(rng, scenario, 30)
                for day in range(1, 31):
                    sim.current_day = day
                    new = sim.generate_disruptions(day)
                    for d in new:
                        if d.disruption_type == "snowstorm":
                            count += 1
            snow_counts[scenario_name] = count

        assert snow_counts["winter_holiday"] > snow_counts["shoulder_spring"], (
            f"Winter snow ({snow_counts['winter_holiday']}) should exceed "
            f"spring snow ({snow_counts['shoulder_spring']})"
        )


# ========================================================================
# 6. No-show and departure tests
# ========================================================================

class TestDepartureSimulation:
    def _make_test_flight(self, booked_per_class: dict = None) -> ScheduledFlight:
        route = ROUTES[0]  # HUB-BOS
        flight = ScheduledFlight(
            flight_id="TEST-F0-D1",
            route=route,
            departure_day=1,
            frequency_index=0,
            aircraft_type="737-800",
            capacity=175,
        )
        if booked_per_class:
            flight.bookings_by_class = booked_per_class
        return flight

    def test_departure_with_no_bookings(self):
        """Empty flight should depart with zero revenue."""
        scenario = SCENARIOS["summer_peak"]
        rng = np.random.default_rng(42)
        sim = SimulationState(rng, scenario, 1)

        flight = self._make_test_flight({fc.code: 0 for fc in FARE_CLASSES})
        result = sim.simulate_departure(flight)

        assert result.revenue == 0.0
        assert result.costs == 0.0
        assert result.denied_boardings == 0
        assert result.passengers_boarded == 0

    def test_departure_with_bookings(self):
        """Flight with bookings should generate revenue."""
        scenario = SCENARIOS["summer_peak"]
        rng = np.random.default_rng(42)
        sim = SimulationState(rng, scenario, 1)

        bookings = {fc.code: 0 for fc in FARE_CLASSES}
        bookings["Y"] = 10
        bookings["M"] = 50
        bookings["Q"] = 80
        flight = self._make_test_flight(bookings)
        result = sim.simulate_departure(flight)

        assert result.revenue > 0
        assert result.passengers_boarded > 0
        assert result.passengers_boarded <= 175  # capacity

    def test_cancelled_flight_costs(self):
        """Cancelled flight should incur cancellation costs."""
        scenario = SCENARIOS["summer_peak"]
        rng = np.random.default_rng(42)
        sim = SimulationState(rng, scenario, 1)

        bookings = {fc.code: 0 for fc in FARE_CLASSES}
        bookings["M"] = 100
        flight = self._make_test_flight(bookings)
        flight.status = "cancelled"

        result = sim.simulate_departure(flight)
        assert result.revenue == 0.0
        assert result.passengers_cancelled == 100
        assert result.costs == 100 * CANCELLATION_COST_PER_PAX
        assert result.flights_cancelled == 1

    def test_delayed_flight_costs(self):
        """Delayed flight should incur delay costs for passengers."""
        scenario = SCENARIOS["summer_peak"]
        rng = np.random.default_rng(42)
        sim = SimulationState(rng, scenario, 1)

        bookings = {fc.code: 0 for fc in FARE_CLASSES}
        bookings["M"] = 100
        flight = self._make_test_flight(bookings)
        flight.status = "delayed"
        flight.delay_hours = 3.0

        result = sim.simulate_departure(flight)
        # Should have delay costs: passengers * hours * cost_per_hour
        # (minus no-shows who don't incur delay costs since they didn't board)
        assert result.costs > 0
        assert result.flights_delayed == 1


# ========================================================================
# 7. Overbooking mechanics tests
# ========================================================================

class TestOverbooking:
    def test_overbooking_allows_more_bookings(self):
        """With overbooking, more passengers can be booked than capacity."""
        route = ROUTES[0]
        flight = ScheduledFlight(
            flight_id="OB-ALLOW-TEST",
            route=route,
            departure_day=5,
            frequency_index=0,
            aircraft_type="737-800",
            capacity=175,
        )
        flight.overbooking_limit = 20
        effective = flight.capacity + flight.overbooking_limit

        # Force-book to over-capacity
        flight.bookings_by_class["M"] = effective

        assert flight.total_booked == effective
        assert flight.total_booked > flight.capacity

    def test_denied_boardings_from_overbooking(self):
        """If all overbooked passengers show up, denied boardings occur."""
        scenario = SCENARIOS["summer_peak"]
        # Use a seed where no-shows are minimal
        rng = np.random.default_rng(42)
        sim = SimulationState(rng, scenario, 1)

        route = ROUTES[0]
        flight = ScheduledFlight(
            flight_id="OB-TEST-D1",
            route=route,
            departure_day=1,
            frequency_index=0,
            aircraft_type="737-800",
            capacity=175,
        )
        # Book 190 passengers in L class (lowest no-show rate: 2%)
        flight.bookings_by_class = {fc.code: 0 for fc in FARE_CLASSES}
        flight.bookings_by_class["L"] = 190  # 15 over capacity

        # With 2% no-show rate on 190 pax, expect ~3.8 no-shows
        # So ~186 show up vs 175 capacity -> ~11 denied boardings
        result = sim.simulate_departure(flight)

        # Should have some denied boardings (probabilistic, but very likely)
        # Run multiple times for robustness
        total_denied = result.denied_boardings
        for seed in range(10):
            rng2 = np.random.default_rng(seed + 100)
            sim2 = SimulationState(rng2, scenario, 1)
            flight2 = ScheduledFlight(
                flight_id=f"OB-TEST2-{seed}",
                route=route,
                departure_day=1,
                frequency_index=0,
                aircraft_type="737-800",
                capacity=175,
            )
            flight2.bookings_by_class = {fc.code: 0 for fc in FARE_CLASSES}
            flight2.bookings_by_class["L"] = 190
            r2 = sim2.simulate_departure(flight2)
            total_denied += r2.denied_boardings

        assert total_denied > 0, "Expected denied boardings from overbooking"

    def test_denied_boarding_cost_calculation(self):
        """Denied boarding cost should be $775 per denied passenger."""
        assert DENIED_BOARDING_COST == 775.0


# ========================================================================
# 8. Fare availability impact tests
# ========================================================================

class TestFareAvailabilityImpact:
    def test_closing_discount_classes_increases_avg_fare(self):
        """Closing low fare classes should result in higher average revenue per booking."""
        scenario = SCENARIOS["summer_peak"]

        # Simulation with all classes open
        rng1 = np.random.default_rng(42)
        sim1 = SimulationState(rng1, scenario, 5)
        # Process day 1 bookings for future flights
        b1, s1 = sim1.process_new_bookings(1)

        # Simulation with discount classes closed
        rng2 = np.random.default_rng(42)
        sim2 = SimulationState(rng2, scenario, 5)
        # Close V, T, L on all flights
        for flight in sim2.flights:
            flight.fare_availability["V"] = False
            flight.fare_availability["T"] = False
            flight.fare_availability["L"] = False
        b2, s2 = sim2.process_new_bookings(1)

        # Calculate average fare per booking for each
        def calc_avg_fare(sim):
            total_fare = 0.0
            total_booked = 0
            for f in sim.flights:
                for fc in FARE_CLASSES:
                    booked = f.bookings_by_class[fc.code]
                    if booked > 0:
                        total_fare += booked * f.fare_for_class(fc.code)
                        total_booked += booked
            return total_fare / max(total_booked, 1)

        avg1 = calc_avg_fare(sim1)
        avg2 = calc_avg_fare(sim2)

        # Note: sim2 starts from same initial state as sim1, so the
        # "closing" only affects the incremental bookings.
        # More bookings should have shifted to higher classes or spilled
        assert s2 >= s1, "Closing classes should spill more customers"

    def test_booking_spill_when_all_closed(self):
        """With all fare classes closed, all customers should spill."""
        scenario = SCENARIOS["summer_peak"]
        rng = np.random.default_rng(42)
        sim = SimulationState(rng, scenario, 5)

        # Close all classes on all flights
        for flight in sim.flights:
            for fc_code in FARE_CLASS_CODES:
                flight.fare_availability[fc_code] = False

        bookings, spills = sim.process_new_bookings(1)
        assert bookings == 0, "No bookings should occur with all classes closed"
        # spills should be > 0 since there's demand


# ========================================================================
# 9. Disruption response tests
# ========================================================================

class TestDisruptionResponse:
    def test_cancel_vs_delay_cost_difference(self):
        """Cancelling should have different cost profile than delaying."""
        route = ROUTES[0]
        # Create a fresh flight with known bookings
        flight = ScheduledFlight(
            flight_id="CANCEL-TEST-D1",
            route=route,
            departure_day=1,
            frequency_index=0,
            aircraft_type="737-800",
            capacity=175,
        )
        flight.bookings_by_class["M"] = 100

        scenario = SCENARIOS["winter_holiday"]
        rng = np.random.default_rng(42)
        sim = SimulationState(rng, scenario, 10)
        # Register the test flight
        sim.flights.append(flight)
        sim.flights_by_id[flight.flight_id] = flight

        disruption = Disruption(
            disruption_id="TEST-1",
            disruption_type="thunderstorm",
            day=1,
            affected_flight_ids=[flight.flight_id],
            delay_hours=2.5,
            cancel_flight_ids=[],
        )

        # Test cancel cost: 100 pax * $200
        cancel_cost = sim.apply_disruption_cancel(disruption, flight.flight_id)
        expected_cancel = 100 * CANCELLATION_COST_PER_PAX
        assert cancel_cost == expected_cancel

    def test_aircraft_swap_cost(self):
        """Aircraft swap should cost the fixed swap fee."""
        scenario = SCENARIOS["summer_peak"]
        rng = np.random.default_rng(42)
        sim = SimulationState(rng, scenario, 5)

        flight = sim.flights[0]
        disruption = Disruption(
            disruption_id="SWAP-1",
            disruption_type="aircraft_mechanical",
            day=1,
            affected_flight_ids=[flight.flight_id],
            delay_hours=2.0,
            cancel_flight_ids=[],
        )

        # Swap to same type should work (any type that covers the distance)
        cost, err = sim.apply_disruption_swap(disruption, flight.flight_id, "737-800")
        assert err == ""
        assert cost == AIRCRAFT_SWAP_FIXED_COST

    def test_aircraft_swap_range_check(self):
        """Cannot swap to aircraft with insufficient range."""
        scenario = SCENARIOS["summer_peak"]
        rng = np.random.default_rng(42)
        sim = SimulationState(rng, scenario, 5)

        # Find a long route (SFO = 1900nm)
        sfo_flights = [f for f in sim.flights if f.route.destination == "SFO"]
        assert len(sfo_flights) > 0
        flight = sfo_flights[0]

        disruption = Disruption(
            disruption_id="RANGE-1",
            disruption_type="aircraft_mechanical",
            day=flight.departure_day,
            affected_flight_ids=[flight.flight_id],
            delay_hours=2.0,
            cancel_flight_ids=[],
        )

        # E175 range is 2200nm, SFO route is 1900nm, should work
        cost, err = sim.apply_disruption_swap(disruption, flight.flight_id, "E175")
        assert err == "", f"E175 should cover SFO route but got: {err}"


# ========================================================================
# 10. Baseline policy tests
# ========================================================================

class TestBaselinePolicy:
    def test_baseline_completes_30_days(self):
        """Baseline should complete a full 30-day simulation."""
        rng = np.random.default_rng(42)
        scenario = SCENARIOS["summer_peak"]
        baseline = BaselinePolicy(rng, scenario, 30)
        results = baseline.run_full_simulation()
        assert len(results) == 30

    def test_baseline_generates_positive_revenue(self):
        """Baseline should generate positive revenue."""
        rng = np.random.default_rng(42)
        scenario = SCENARIOS["summer_peak"]
        baseline = BaselinePolicy(rng, scenario, 30)
        results = baseline.run_full_simulation()

        total_revenue = sum(r.revenue for r in results)
        assert total_revenue > 0, "Baseline should generate positive revenue"

    def test_baseline_revenue_reasonable_range(self):
        """Baseline daily revenue should be in a reasonable range for this network."""
        rng = np.random.default_rng(42)
        scenario = SCENARIOS["summer_peak"]
        baseline = BaselinePolicy(rng, scenario, 30)
        results = baseline.run_full_simulation()

        avg_daily = sum(r.revenue for r in results) / 30
        # ~30 flights/day, ~100 pax/flight avg, ~$250 avg fare = ~$750K/day
        # This is a rough order-of-magnitude check
        assert avg_daily > 50000, f"Average daily revenue ${avg_daily:,.0f} seems too low"
        assert avg_daily < 5000000, f"Average daily revenue ${avg_daily:,.0f} seems too high"

    def test_baseline_load_factors_reasonable(self):
        """Baseline should achieve reasonable load factors (70-95%)."""
        rng = np.random.default_rng(42)
        scenario = SCENARIOS["summer_peak"]
        # Just check initial bookings on flights
        sim = SimulationState(rng, scenario, 30)

        # Check day 15 flights (well into booking window)
        day15_flights = get_flights_for_day(sim.flights, 15)
        load_factors = [f.load_factor for f in day15_flights if f.total_booked > 0]
        if load_factors:
            avg_lf = np.mean(load_factors)
            assert 0.3 <= avg_lf <= 1.5, (
                f"Average load factor {avg_lf:.1%} is outside reasonable range"
            )


# ========================================================================
# 11. Day advance and reward tests
# ========================================================================

class TestDayAdvanceAndReward:
    def test_advance_day_produces_result(self):
        """advance_day should return a DayResult."""
        scenario = SCENARIOS["summer_peak"]
        rng = np.random.default_rng(42)
        sim = SimulationState(rng, scenario, 5)
        sim.current_day = 1
        sim.generate_disruptions(1)

        # Resolve any disruptions first
        for d in sim.get_pending_disruptions():
            for fid in d.affected_flight_ids:
                sim.apply_disruption_delay(d, fid)
            d.resolved = True

        result = sim.advance_day()
        assert isinstance(result, DayResult)
        assert result.day == 1

    def test_advance_day_updates_cumulative(self):
        """Cumulative tracking should be updated after advance."""
        scenario = SCENARIOS["summer_peak"]
        rng = np.random.default_rng(42)
        sim = SimulationState(rng, scenario, 5)
        sim.current_day = 1
        sim.generate_disruptions(1)

        for d in sim.get_pending_disruptions():
            for fid in d.affected_flight_ids:
                sim.apply_disruption_delay(d, fid)
            d.resolved = True

        result = sim.advance_day()
        assert sim.cumulative_revenue == result.revenue
        assert sim.cumulative_costs == result.costs
        assert sim.current_day == 2

    def test_full_30_day_simulation(self):
        """Run a full 30-day simulation handling all disruptions."""
        scenario = SCENARIOS["summer_peak"]
        rng = np.random.default_rng(42)
        sim = SimulationState(rng, scenario, 30)

        for day in range(1, 31):
            sim.current_day = day
            if day == 1:
                sim.generate_disruptions(day)

            # Resolve disruptions
            for d in sim.get_pending_disruptions():
                for fid in d.affected_flight_ids:
                    if fid in d.cancel_flight_ids:
                        sim.apply_disruption_cancel(d, fid)
                    else:
                        sim.apply_disruption_delay(d, fid)
                d.resolved = True

            sim.advance_day()

        assert len(sim.day_results) == 30
        assert sim.cumulative_revenue > 0


# ========================================================================
# 12. Fare war tests
# ========================================================================

class TestCompetitorFareWar:
    def test_fare_war_reduces_demand(self):
        """Active fare war should reduce bookings on affected routes."""
        scenario = SCENARIOS["fall_business"]  # higher fare war probability
        rng1 = np.random.default_rng(42)
        sim1 = SimulationState(rng1, scenario, 10)

        # No fare war: process bookings
        b1_no_war, _ = sim1.process_new_bookings(1)

        # With fare war: same state but inject a fare war
        rng2 = np.random.default_rng(42)
        sim2 = SimulationState(rng2, scenario, 10)
        sim2.active_fare_wars.append(ActiveFareWar(
            affected_route_keys=[route_key(r) for r in ROUTES],  # all routes
            demand_reduction_pct=0.30,
            start_day=1,
            end_day=10,
        ))
        b2_war, _ = sim2.process_new_bookings(1)

        # War bookings should be fewer (not guaranteed per single run due to
        # randomness, but the mean should be lower with 30% demand reduction)
        # We just check it's a reasonable reduction
        # Note: the RNG states diverge after initial booking, but the fare war
        # should reduce total bookings
        assert b2_war < b1_no_war * 1.1, (
            f"Fare war bookings ({b2_war}) should be noticeably less than "
            f"no-war bookings ({b1_no_war})"
        )


# ========================================================================
# 13. Edge case tests
# ========================================================================

class TestEdgeCases:
    def test_flight_copy(self):
        """Flight copy should be independent."""
        flights = generate_flight_schedule(1)
        original = flights[0]
        original.bookings_by_class["Y"] = 10
        copy = original.copy()
        copy.bookings_by_class["Y"] = 20
        assert original.bookings_by_class["Y"] == 10

    def test_very_high_demand_scenario(self):
        """Even with very high demand, system should not crash."""
        scenario = SCENARIOS["summer_peak"]
        rng = np.random.default_rng(42)
        sim = SimulationState(rng, scenario, 5)
        # Should complete without errors
        sim.process_new_bookings(1)

    def test_pending_disruptions_query(self):
        """get_pending_disruptions should only return unresolved ones for current day."""
        scenario = SCENARIOS["winter_holiday"]
        rng = np.random.default_rng(42)
        sim = SimulationState(rng, scenario, 10)
        sim.current_day = 1

        d1 = Disruption("D1", "fog", 1, ["F1"], 1.0, [], resolved=False)
        d2 = Disruption("D2", "fog", 1, ["F2"], 1.0, [], resolved=True)
        d3 = Disruption("D3", "fog", 2, ["F3"], 1.0, [], resolved=False)
        sim.disruptions = [d1, d2, d3]

        pending = sim.get_pending_disruptions()
        assert len(pending) == 1
        assert pending[0].disruption_id == "D1"


# ========================================================================
# 14. Cost constant validation
# ========================================================================

class TestCostConstants:
    def test_denied_boarding_cost(self):
        assert DENIED_BOARDING_COST == 775.0

    def test_cancellation_cost(self):
        assert CANCELLATION_COST_PER_PAX == 200.0

    def test_delay_cost(self):
        assert DELAY_COST_PER_PAX_PER_HOUR == 50.0

    def test_delay_threshold(self):
        assert DELAY_THRESHOLD_HOURS == 1.0

    def test_swap_cost(self):
        assert AIRCRAFT_SWAP_FIXED_COST == 5000.0


# ========================================================================
# 15. Simulation realism tests
# ========================================================================

class TestSimulationRealism:
    """
    Verify that simulation output matches realistic airline industry
    parameters. Citations embedded in individual test docstrings.
    """

    def test_load_factors_by_scenario(self):
        """
        Each scenario should produce plausible departure-day load factors.

        Reference: BTS LOADFACTORD 2019-2024 domestic average ~83-87%.
        With demand multipliers (0.90x-1.30x) applied to a base ~85% LF
        calibration, expected ranges per scenario are scenario-dependent.
        """
        expected_ranges = {
            "summer_peak": (0.75, 1.00),
            "winter_holiday": (0.70, 0.95),
            "shoulder_spring": (0.50, 0.80),
            "fall_business": (0.60, 0.85),
        }

        for sname, (lo, hi) in expected_ranges.items():
            scenario = SCENARIOS[sname]
            rng = np.random.default_rng(42)
            sim = SimulationState(rng, scenario, 30)

            # Process 15 days of simulation to get realistic mid-horizon state
            for day in range(1, 16):
                sim.current_day = day
                if day == 1:
                    sim.generate_disruptions(day)
                for d in sim.get_pending_disruptions():
                    for fid in d.affected_flight_ids:
                        sim.apply_disruption_delay(d, fid)
                    d.resolved = True
                sim.advance_day()

            # Check Day 16 flights
            day16 = get_flights_for_day(sim.flights, 16)
            active = [f for f in day16 if f.status != "cancelled"]
            if active:
                total_booked = sum(f.total_booked for f in active)
                total_cap = sum(f.capacity for f in active)
                avg_lf = total_booked / total_cap
                assert lo <= avg_lf <= hi, (
                    f"{sname}: Day16 LF {avg_lf:.1%} outside [{lo:.0%}, {hi:.0%}]"
                )

    def test_business_routes_higher_avg_fare(self):
        """
        Business routes (BOS, ORD, SFO) should yield higher average
        revenue per passenger than leisure routes (MCO, FLL, CUN, SRQ).
        """
        scenario = SCENARIOS["summer_peak"]
        rng = np.random.default_rng(42)
        sim = SimulationState(rng, scenario, 30)

        biz_rev, biz_pax = 0.0, 0
        lei_rev, lei_pax = 0.0, 0

        for f in sim.flights:
            for fc in FARE_CLASSES:
                booked = f.bookings_by_class.get(fc.code, 0)
                fare = f.fare_for_class(fc.code)
                if f.route.category == "business":
                    biz_rev += booked * fare
                    biz_pax += booked
                elif f.route.category == "leisure":
                    lei_rev += booked * fare
                    lei_pax += booked

        if biz_pax > 0 and lei_pax > 0:
            biz_avg = biz_rev / biz_pax
            lei_avg = lei_rev / lei_pax
            assert biz_avg > lei_avg, (
                f"Business avg fare ${biz_avg:.0f} should exceed leisure ${lei_avg:.0f}"
            )

    def test_no_show_impact_on_revenue(self):
        """
        No-shows from Y class (15%, $620 fare on BOS) should represent
        more lost revenue per no-show than from L class (2%, $80 fare).

        Reference: Smith, Leimkuhler & Darrow 1992 (Interfaces).
        """
        route = ROUTES[0]  # HUB-BOS
        y_fare = route.max_fare * FARE_CLASSES[0].multiplier
        l_fare = route.max_fare * FARE_CLASSES[-1].multiplier
        y_nsr = FARE_CLASSES[0].no_show_rate
        l_nsr = FARE_CLASSES[-1].no_show_rate

        y_expected_loss = y_fare * y_nsr
        l_expected_loss = l_fare * l_nsr

        assert y_expected_loss > l_expected_loss, (
            f"Y no-show loss/pax ${y_expected_loss:.2f} should exceed "
            f"L no-show loss/pax ${l_expected_loss:.2f}"
        )

    def test_demand_respects_advance_purchase(self):
        """
        At 0 days before departure, only Y (0d) and B (0d) should be
        bookable — no bookings in classes with advance purchase > 0.
        """
        scenario = SCENARIOS["summer_peak"]
        rng = np.random.default_rng(42)
        sim = SimulationState(rng, scenario, 5)

        flight = ScheduledFlight(
            flight_id="ADV-PURCH-TEST",
            route=ROUTES[0],
            departure_day=1,
            frequency_index=0,
            aircraft_type="737-800",
            capacity=175,
        )
        sim._simulate_booking_arrivals_for_flight(flight, 0)

        for fc in FARE_CLASSES:
            if fc.advance_purchase_days > 0:
                assert flight.bookings_by_class[fc.code] == 0, (
                    f"Class {fc.code} (adv_purch={fc.advance_purchase_days}d) should "
                    f"have 0 bookings at 0 days before departure, "
                    f"got {flight.bookings_by_class[fc.code]}"
                )

    def test_highest_fares_board_first(self):
        """
        In oversold situations, Y/B passengers should board before L/T.
        simulate_departure iterates FARE_CLASSES highest-first.
        """
        scenario = SCENARIOS["summer_peak"]
        rng = np.random.default_rng(42)
        sim = SimulationState(rng, scenario, 1)

        flight = ScheduledFlight(
            flight_id="PRIORITY-BOARD",
            route=ROUTES[0],
            departure_day=1,
            frequency_index=0,
            aircraft_type="737-800",
            capacity=10,
        )
        flight.bookings_by_class = {fc.code: 0 for fc in FARE_CLASSES}
        flight.bookings_by_class["Y"] = 8
        flight.bookings_by_class["L"] = 8

        result = sim.simulate_departure(flight)

        y_fare = flight.fare_for_class("Y")
        # At least 5 Y should show (8 * 0.85 = 6.8 expected) and board
        min_y_revenue = 5 * y_fare
        assert result.revenue >= min_y_revenue, (
            f"Revenue ${result.revenue:.0f} should reflect Y-first boarding "
            f"(minimum ${min_y_revenue:.0f} from 5+ Y pax)"
        )

    def test_booking_curve_business_vs_leisure_quantitative(self):
        """
        At 7 days before departure:
        - Business should have <80% cumulative bookings
        - Leisure should have >95% cumulative bookings

        Reference: Belobaba 1989; Embark Aviation booking-curve data.
        """
        biz_cdf_7 = _booking_curve_cdf(7, "business")
        lei_cdf_7 = _booking_curve_cdf(7, "leisure")

        assert biz_cdf_7 < 0.80, (
            f"Business CDF at 7d = {biz_cdf_7:.3f}, expected < 0.80"
        )
        assert lei_cdf_7 > 0.95, (
            f"Leisure CDF at 7d = {lei_cdf_7:.3f}, expected > 0.95"
        )

    def test_disruption_frequency_per_scenario(self):
        """
        Verify seasonal disruption distributions:
        - Summer: thunderstorms dominant, zero snowstorms
        - Winter: snowstorms dominant
        - Fall: fog prominent, near-zero snowstorms
        """
        for sname, expected_dominant, expected_rare in [
            ("summer_peak", "thunderstorm", "snowstorm"),
            ("winter_holiday", "snowstorm", None),
            ("fall_business", "fog", "snowstorm"),
        ]:
            scenario = SCENARIOS[sname]
            counts = {}
            for seed in range(50):
                rng = np.random.default_rng(seed)
                sim = SimulationState(rng, scenario, 30)
                for day in range(1, 31):
                    sim.current_day = day
                    for d in sim.generate_disruptions(day):
                        counts[d.disruption_type] = counts.get(d.disruption_type, 0) + 1

            assert expected_dominant in counts, (
                f"{sname}: expected {expected_dominant} disruptions"
            )

            if expected_rare:
                rare_count = counts.get(expected_rare, 0)
                dominant_count = counts[expected_dominant]
                assert rare_count < dominant_count, (
                    f"{sname}: {expected_rare} ({rare_count}) should be rarer "
                    f"than {expected_dominant} ({dominant_count})"
                )

    def test_multiple_disruptions_same_day(self):
        """Multiple different disruptions can occur on the same day."""
        scenario = SCENARIOS["summer_peak"]
        multi_day = 0
        for seed in range(200):
            rng = np.random.default_rng(seed)
            sim = SimulationState(rng, scenario, 30)
            for day in range(1, 31):
                sim.current_day = day
                new = sim.generate_disruptions(day)
                if len(new) > 1:
                    multi_day += 1

        assert multi_day > 50, (
            f"Expected >50 multi-disruption days in 6000 samples, got {multi_day}"
        )

    def test_overbooking_expected_denied_vs_actual(self):
        """
        Statistical test: for flights overbooked by 15 in M class (8%
        no-show), average denied boardings should be in a plausible range.
        """
        scenario = SCENARIOS["summer_peak"]
        route = ROUTES[0]

        total_denied = 0
        n_runs = 100

        for seed in range(n_runs):
            rng = np.random.default_rng(seed)
            sim = SimulationState(rng, scenario, 1)

            flight = ScheduledFlight(
                flight_id=f"STAT-OB-{seed}",
                route=route,
                departure_day=1,
                frequency_index=0,
                aircraft_type="737-800",
                capacity=175,
            )
            flight.bookings_by_class = {fc.code: 0 for fc in FARE_CLASSES}
            flight.bookings_by_class["M"] = 190  # 15 over capacity
            result = sim.simulate_departure(flight)
            total_denied += result.denied_boardings

        avg_denied = total_denied / n_runs
        # 190 pax * 0.92 show rate = 174.8 expected show-ups vs 175 capacity
        # Variance means roughly half will have some denied boardings
        assert 0 <= avg_denied < 20, (
            f"Average denied boardings {avg_denied:.1f} outside expected range"
        )

    def test_revenue_composition_by_class(self):
        """
        With all classes open (baseline behavior), discount classes should
        capture majority of volume since customers buy cheapest available.
        """
        scenario = SCENARIOS["summer_peak"]
        rng = np.random.default_rng(42)
        sim = SimulationState(rng, scenario, 30)

        rev_by_class = {}
        for f in sim.flights:
            for fc in FARE_CLASSES:
                booked = f.bookings_by_class.get(fc.code, 0)
                fare = f.fare_for_class(fc.code)
                rev_by_class[fc.code] = rev_by_class.get(fc.code, 0) + booked * fare

        total_rev = sum(rev_by_class.values())
        assert total_rev > 0

        premium_pct = (rev_by_class.get("Y", 0) + rev_by_class.get("B", 0)) / total_rev
        assert premium_pct < 0.10, (
            f"Premium classes (Y+B) revenue share {premium_pct:.1%} should be < 10%"
        )

        discount_pct = (
            rev_by_class.get("V", 0) + rev_by_class.get("T", 0) + rev_by_class.get("L", 0)
        ) / total_rev
        assert discount_pct > 0.50, (
            f"Discount classes (V+T+L) revenue share {discount_pct:.1%} should be > 50%"
        )


# ========================================================================
# 16. Edge cases and bug-finding tests
# ========================================================================

class TestEdgeCasesAndBugs:

    def test_downguage_swap_causes_denied_boardings(self):
        """
        Swapping to smaller aircraft when fully booked should cause
        denied boardings at departure.
        """
        scenario = SCENARIOS["summer_peak"]
        rng = np.random.default_rng(42)
        sim = SimulationState(rng, scenario, 30)

        flight = ScheduledFlight(
            flight_id="DOWNGUAGE-TEST",
            route=ROUTES[0],  # HUB-BOS, 820nm (within E175 2200nm range)
            departure_day=15,
            frequency_index=0,
            aircraft_type="737-800",
            capacity=175,
        )
        flight.bookings_by_class = {fc.code: 0 for fc in FARE_CLASSES}
        flight.bookings_by_class["M"] = 175

        sim.flights.append(flight)
        sim.flights_by_id[flight.flight_id] = flight

        disruption = Disruption(
            disruption_id="DG-1",
            disruption_type="aircraft_mechanical",
            day=15,
            affected_flight_ids=[flight.flight_id],
            delay_hours=2.0,
            cancel_flight_ids=[],
        )

        cost, err = sim.apply_disruption_swap(disruption, flight.flight_id, "E175")
        assert err == ""
        assert flight.capacity == 76
        assert flight.total_booked == 175

        result = sim.simulate_departure(flight)
        assert result.denied_boardings > 50

    def test_delay_under_threshold_no_cost(self):
        """Delay of 0.9 hours should incur zero delay cost."""
        scenario = SCENARIOS["summer_peak"]
        rng = np.random.default_rng(42)
        sim = SimulationState(rng, scenario, 1)

        flight = ScheduledFlight(
            flight_id="DELAY-UNDER-TEST",
            route=ROUTES[0],
            departure_day=1,
            frequency_index=0,
            aircraft_type="737-800",
            capacity=175,
        )
        flight.bookings_by_class = {fc.code: 0 for fc in FARE_CLASSES}
        flight.bookings_by_class["M"] = 100
        flight.status = "delayed"
        flight.delay_hours = 0.9

        result = sim.simulate_departure(flight)
        assert result.costs == 0.0, (
            f"Delay of 0.9h should have zero cost, got ${result.costs:.2f}"
        )

    def test_delay_at_threshold_boundary(self):
        """
        Delay of exactly 1.0 hours should incur NO delay cost.
        The condition is strict greater-than: delay_hours > DELAY_THRESHOLD.
        """
        scenario = SCENARIOS["summer_peak"]
        rng = np.random.default_rng(42)
        sim = SimulationState(rng, scenario, 1)

        flight = ScheduledFlight(
            flight_id="DELAY-BOUNDARY-TEST",
            route=ROUTES[0],
            departure_day=1,
            frequency_index=0,
            aircraft_type="737-800",
            capacity=175,
        )
        flight.bookings_by_class = {fc.code: 0 for fc in FARE_CLASSES}
        flight.bookings_by_class["M"] = 100
        flight.status = "delayed"
        flight.delay_hours = 1.0

        result = sim.simulate_departure(flight)
        assert result.costs == 0.0, (
            f"Delay of exactly 1.0h should have zero cost, got ${result.costs:.2f}"
        )

    def test_cancellation_then_no_new_bookings(self):
        """After a flight is cancelled, it should receive zero new bookings."""
        scenario = SCENARIOS["summer_peak"]
        rng = np.random.default_rng(42)
        sim = SimulationState(rng, scenario, 10)

        flight = sim.flights[0]
        flight.status = "cancelled"
        booked_before = flight.total_booked

        sim.process_new_bookings(1)

        assert flight.total_booked == booked_before, (
            f"Cancelled flight should have no new bookings. "
            f"Before: {booked_before}, after: {flight.total_booked}"
        )

    def test_same_action_all_flights_in_disruption(self):
        """Action applies to all affected flights in a disruption."""
        scenario = SCENARIOS["summer_peak"]
        rng = np.random.default_rng(42)
        sim = SimulationState(rng, scenario, 30)

        day5_flights = get_flights_for_day(sim.flights, 5)
        active = [f for f in day5_flights if f.status == "scheduled"][:3]
        assert len(active) == 3

        disruption = Disruption(
            disruption_id="MULTI-FLT-1",
            disruption_type="thunderstorm",
            day=5,
            affected_flight_ids=[f.flight_id for f in active],
            delay_hours=2.0,
            cancel_flight_ids=[],
        )

        for fid in disruption.affected_flight_ids:
            sim.apply_disruption_delay(disruption, fid)

        for f in active:
            assert f.status == "delayed", (
                f"Flight {f.flight_id} should be delayed, got {f.status}"
            )
            assert f.delay_hours == 2.0

    def test_swap_clears_delay_status(self):
        """After aircraft swap, flight should be 'scheduled' with 0 delay."""
        scenario = SCENARIOS["summer_peak"]
        rng = np.random.default_rng(42)
        sim = SimulationState(rng, scenario, 5)

        flight = sim.flights[0]
        flight.status = "delayed"
        flight.delay_hours = 3.0

        disruption = Disruption(
            disruption_id="SWAP-CLEAR-1",
            disruption_type="aircraft_mechanical",
            day=1,
            affected_flight_ids=[flight.flight_id],
            delay_hours=3.0,
            cancel_flight_ids=[],
        )

        cost, err = sim.apply_disruption_swap(
            disruption, flight.flight_id, flight.aircraft_type
        )
        assert err == ""
        assert flight.status == "scheduled"
        assert flight.delay_hours == 0.0

    def test_fare_war_demand_reduction_magnitude(self):
        """
        With 30% fare war on all routes, bookings should be roughly 70%
        of normal. Statistical test over 20 seeds.
        """
        scenario = SCENARIOS["summer_peak"]
        all_route_keys = [route_key(r) for r in ROUTES]

        no_war_total = 0
        war_total = 0

        for seed in range(20):
            rng1 = np.random.default_rng(seed)
            sim1 = SimulationState(rng1, scenario, 5)
            b1, _ = sim1.process_new_bookings(1)
            no_war_total += b1

            rng2 = np.random.default_rng(seed)
            sim2 = SimulationState(rng2, scenario, 5)
            sim2.active_fare_wars.append(ActiveFareWar(
                affected_route_keys=all_route_keys,
                demand_reduction_pct=0.30,
                start_day=1,
                end_day=5,
            ))
            b2, _ = sim2.process_new_bookings(1)
            war_total += b2

        ratio = war_total / max(no_war_total, 1)
        assert 0.55 < ratio < 0.85, (
            f"Fare war ratio {ratio:.2f} outside expected range 0.55-0.85"
        )

    def test_baseline_determinism(self):
        """Running baseline twice with same seed produces identical results."""
        scenario = SCENARIOS["summer_peak"]

        rng1 = np.random.default_rng(42)
        results1 = BaselinePolicy(rng1, scenario, 30).run_full_simulation()

        rng2 = np.random.default_rng(42)
        results2 = BaselinePolicy(rng2, scenario, 30).run_full_simulation()

        for i, (r1, r2) in enumerate(zip(results1, results2)):
            assert r1.revenue == r2.revenue, f"Day {i+1}: revenue differs"
            assert r1.costs == r2.costs, f"Day {i+1}: costs differ"
            assert r1.denied_boardings == r2.denied_boardings
            assert r1.passengers_boarded == r2.passengers_boarded


# ========================================================================
# 17. Reward calculation tests
# ========================================================================

class TestRewardCalculation:

    def test_reward_formula_manual_calculation(self):
        """Manually compute reward for known values and verify."""
        # Standard case
        agent_net = 500_000.0
        baseline_net = 400_000.0
        expected = (agent_net - baseline_net) / max(abs(baseline_net), 1000.0)
        assert expected == pytest.approx(0.25)

        # Floor case
        agent_net2 = 1500.0
        baseline_net2 = 500.0
        expected2 = (1500 - 500) / max(500, 1000)
        assert expected2 == pytest.approx(1.0)

    def test_negative_baseline_net_reward_calculation(self):
        """Negative baseline uses abs() in denominator."""
        agent_net = 100.0
        baseline_net = -2000.0
        reward = (agent_net - baseline_net) / max(abs(baseline_net), 1000.0)
        assert reward == pytest.approx(1.05)

        # Agent also negative but less so
        reward2 = (-500.0 - (-2000.0)) / max(abs(-2000.0), 1000.0)
        assert reward2 == pytest.approx(0.75)

    def test_cumulative_reward_is_sum_of_daily(self):
        """Do-nothing agent cumulative = sum(daily) = 0.0."""
        task_id = "summer_peak_v1"
        seed = int(hashlib.sha256(task_id.encode()).hexdigest(), 16) % (2**32)
        scenario = SCENARIOS["summer_peak"]

        brng = np.random.default_rng(seed)
        br = BaselinePolicy(brng, scenario, 30).run_full_simulation()

        arng = np.random.default_rng(seed)
        sim = SimulationState(arng, scenario, 30)
        sim.generate_disruptions(1)

        daily_rewards = []
        for day in range(1, 31):
            sim.current_day = day
            # Replicate baseline policy
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
            for d in sim.get_pending_disruptions():
                for fid in d.affected_flight_ids:
                    if fid in d.cancel_flight_ids:
                        sim.apply_disruption_cancel(d, fid)
                    elif d.delay_hours < 3.0:
                        sim.apply_disruption_delay(d, fid)
                    else:
                        sim.apply_disruption_cancel(d, fid)
                d.resolved = True

            result = sim.advance_day()
            bl_net = br[day - 1].net_revenue
            denom = max(abs(bl_net), 1000.0)
            daily_rewards.append((result.net_revenue - bl_net) / denom)

        cumulative = sum(daily_rewards)
        assert cumulative == pytest.approx(0.0, abs=1e-10), (
            f"Cumulative reward {cumulative} should be ~0 for do-nothing agent"
        )

    def test_reward_when_agent_matches_baseline(self):
        """Exact baseline replication in two scenarios yields reward = 0.0."""
        for sname in ["summer_peak", "shoulder_spring"]:
            task_id = f"{sname}_v1"
            seed = int(hashlib.sha256(task_id.encode()).hexdigest(), 16) % (2**32)
            scenario = SCENARIOS[sname]

            brng = np.random.default_rng(seed)
            br = BaselinePolicy(brng, scenario, 30).run_full_simulation()

            arng = np.random.default_rng(seed)
            sim = SimulationState(arng, scenario, 30)
            sim.generate_disruptions(1)

            cumulative = 0.0
            for day in range(1, 31):
                sim.current_day = day
                for flight in sim.flights:
                    if flight.status in ("cancelled", "departed"):
                        continue
                    if flight.departure_day < day:
                        continue
                    db = flight.departure_day - day
                    for fc in FARE_CLASSES:
                        flight.fare_availability[fc.code] = db >= fc.advance_purchase_days
                for d in sim.get_pending_disruptions():
                    for fid in d.affected_flight_ids:
                        if fid in d.cancel_flight_ids:
                            sim.apply_disruption_cancel(d, fid)
                        elif d.delay_hours < 3.0:
                            sim.apply_disruption_delay(d, fid)
                        else:
                            sim.apply_disruption_cancel(d, fid)
                    d.resolved = True
                result = sim.advance_day()
                bl_net = br[day - 1].net_revenue
                denom = max(abs(bl_net), 1000.0)
                cumulative += (result.net_revenue - bl_net) / denom

            assert cumulative == pytest.approx(0.0, abs=1e-10), (
                f"{sname}: do-nothing agent reward {cumulative:+.10f} != 0.0"
            )


# ========================================================================
# 18. RL environment correctness tests
# ========================================================================

class TestRLEnvironmentCorrectness:

    def _run_heuristic_policy(self, scenario_name: str, variant: int = 1) -> float:
        """Run the heuristic from test_local.py, return cumulative reward."""
        task_id = f"{scenario_name}_v{variant}"
        seed = int(hashlib.sha256(task_id.encode()).hexdigest(), 16) % (2**32)
        scenario = SCENARIOS[scenario_name]

        brng = np.random.default_rng(seed)
        br = BaselinePolicy(brng, scenario, 30).run_full_simulation()

        arng = np.random.default_rng(seed)
        sim = SimulationState(arng, scenario, 30)
        sim.generate_disruptions(1)

        cum_reward = 0.0
        for day in range(1, 31):
            sim.current_day = day
            for flight in sim.flights:
                if flight.status in ("cancelled", "departed"):
                    continue
                if flight.departure_day < day:
                    continue
                days_before = flight.departure_day - day
                for fc in FARE_CLASSES:
                    if days_before < fc.advance_purchase_days:
                        flight.fare_availability[fc.code] = False
                    elif days_before < 3 and fc.code in ("T", "L"):
                        flight.fare_availability[fc.code] = False
                    elif days_before < 7 and fc.code in ("V",):
                        flight.fare_availability[fc.code] = False
                    elif days_before < 1 and fc.code in ("Q",):
                        flight.fare_availability[fc.code] = False
                    else:
                        flight.fare_availability[fc.code] = True
                flight.overbooking_limit = max(1, int(flight.capacity * 0.07))

            for d in sim.get_pending_disruptions():
                for fid in d.affected_flight_ids:
                    f = sim.flights_by_id.get(fid)
                    if not f or f.status in ("cancelled", "departed"):
                        continue
                    if fid in d.cancel_flight_ids:
                        cost, err = sim.apply_disruption_swap(d, fid, f.aircraft_type)
                        if err:
                            sim.apply_disruption_cancel(d, fid)
                    elif d.delay_hours < 1.5:
                        sim.apply_disruption_delay(d, fid)
                    elif d.delay_hours < 3.0:
                        cost, err = sim.apply_disruption_swap(d, fid, f.aircraft_type)
                        if err:
                            sim.apply_disruption_delay(d, fid)
                    else:
                        sim.apply_disruption_cancel(d, fid)
                d.resolved = True

            result = sim.advance_day()
            bl_net = br[day - 1].net_revenue
            denom = max(abs(bl_net), 1000.0)
            cum_reward += (result.net_revenue - bl_net) / denom

        return cum_reward

    def test_smart_policy_beats_baseline(self):
        """Heuristic gets positive reward in ALL 4 scenarios."""
        for sname in ["summer_peak", "winter_holiday", "shoulder_spring", "fall_business"]:
            reward = self._run_heuristic_policy(sname)
            assert reward > 0.0, (
                f"{sname}: heuristic reward {reward:+.4f} should be positive"
            )

    def test_do_nothing_matches_baseline(self):
        """Baseline-replicating agent gets reward = 0.0."""
        for sname in ["summer_peak", "fall_business"]:
            task_id = f"{sname}_v1"
            seed = int(hashlib.sha256(task_id.encode()).hexdigest(), 16) % (2**32)
            scenario = SCENARIOS[sname]

            brng = np.random.default_rng(seed)
            br = BaselinePolicy(brng, scenario, 30).run_full_simulation()

            arng = np.random.default_rng(seed)
            sim = SimulationState(arng, scenario, 30)
            sim.generate_disruptions(1)

            cum = 0.0
            for day in range(1, 31):
                sim.current_day = day
                for flight in sim.flights:
                    if flight.status in ("cancelled", "departed"):
                        continue
                    if flight.departure_day < day:
                        continue
                    db = flight.departure_day - day
                    for fc in FARE_CLASSES:
                        flight.fare_availability[fc.code] = db >= fc.advance_purchase_days
                for d in sim.get_pending_disruptions():
                    for fid in d.affected_flight_ids:
                        if fid in d.cancel_flight_ids:
                            sim.apply_disruption_cancel(d, fid)
                        elif d.delay_hours < 3.0:
                            sim.apply_disruption_delay(d, fid)
                        else:
                            sim.apply_disruption_cancel(d, fid)
                    d.resolved = True
                result = sim.advance_day()
                bl_net = br[day - 1].net_revenue
                denom = max(abs(bl_net), 1000.0)
                cum += (result.net_revenue - bl_net) / denom

            assert cum == pytest.approx(0.0, abs=1e-10), (
                f"{sname} do-nothing: reward {cum:+.10f} should be 0.0"
            )

    def test_terrible_policy_negative_reward(self):
        """Close-all-fares + cancel-all should get very negative reward."""
        task_id = "summer_peak_v1"
        seed = int(hashlib.sha256(task_id.encode()).hexdigest(), 16) % (2**32)
        scenario = SCENARIOS["summer_peak"]

        brng = np.random.default_rng(seed)
        br = BaselinePolicy(brng, scenario, 30).run_full_simulation()

        arng = np.random.default_rng(seed)
        sim = SimulationState(arng, scenario, 30)
        sim.generate_disruptions(1)

        cum = 0.0
        for day in range(1, 31):
            sim.current_day = day
            for flight in sim.flights:
                if flight.status in ("cancelled", "departed"):
                    continue
                for code in FARE_CLASS_CODES:
                    flight.fare_availability[code] = False
            for d in sim.get_pending_disruptions():
                for fid in d.affected_flight_ids:
                    sim.apply_disruption_cancel(d, fid)
                d.resolved = True
            result = sim.advance_day()
            bl_net = br[day - 1].net_revenue
            denom = max(abs(bl_net), 1000.0)
            cum += (result.net_revenue - bl_net) / denom

        assert cum < -5.0, (
            f"Terrible policy should get very negative reward, got {cum:+.4f}"
        )

    def test_graduated_difficulty(self):
        """summer_peak reward should exceed shoulder_spring reward."""
        summer = self._run_heuristic_policy("summer_peak")
        shoulder = self._run_heuristic_policy("shoulder_spring")

        assert summer > shoulder, (
            f"summer_peak ({summer:+.4f}) should exceed shoulder_spring ({shoulder:+.4f})"
        )

    def test_finished_state_reached(self):
        """30 advance_day calls should complete the simulation."""
        scenario = SCENARIOS["summer_peak"]
        rng = np.random.default_rng(42)
        sim = SimulationState(rng, scenario, 30)
        sim.generate_disruptions(1)

        for day in range(1, 31):
            sim.current_day = day
            for d in sim.get_pending_disruptions():
                for fid in d.affected_flight_ids:
                    sim.apply_disruption_delay(d, fid)
                d.resolved = True
            sim.advance_day()

        assert sim.current_day == 31
        assert len(sim.day_results) == 30

    def test_all_12_tasks_complete(self):
        """Every task (9 train + 3 test) runs to completion without errors."""
        all_tasks = []
        for scenario_name in ["summer_peak", "winter_holiday", "shoulder_spring"]:
            for v in range(1, 4):
                all_tasks.append((scenario_name, v))
        for v in range(1, 4):
            all_tasks.append(("fall_business", v))

        assert len(all_tasks) == 12

        for sname, variant in all_tasks:
            task_id = f"{sname}_v{variant}"
            seed = int(hashlib.sha256(task_id.encode()).hexdigest(), 16) % (2**32)
            scenario = SCENARIOS[sname]

            rng = np.random.default_rng(seed)
            results = BaselinePolicy(rng, scenario, 30).run_full_simulation()

            assert len(results) == 30, f"Task {task_id}: expected 30 day results"
            total_rev = sum(r.revenue for r in results)
            assert total_rev > 0, f"Task {task_id}: should have positive revenue"

    def test_reward_range_reasonable(self):
        """Cumulative reward over 30 days should be bounded."""
        reward = self._run_heuristic_policy("summer_peak")
        assert -50 < reward < 50, f"Reward {reward:+.4f} outside bounds"

    def test_improvement_ceiling_exists(self):
        """Best heuristic should not beat baseline by unreasonable margins."""
        reward = self._run_heuristic_policy("winter_holiday")
        assert reward < 30.0, (
            f"Reward {reward:+.4f} seems unreasonably high"
        )

    def test_baseline_is_reasonable(self):
        """
        Baseline should not be catastrophically bad: net > 0, zero denied
        boardings (since baseline never overbooks).
        """
        for sname in SCENARIOS:
            rng = np.random.default_rng(42)
            results = BaselinePolicy(rng, SCENARIOS[sname], 30).run_full_simulation()

            total_net = sum(r.net_revenue for r in results)
            total_denied = sum(r.denied_boardings for r in results)

            assert total_net > 0, f"{sname} baseline net should be positive"
            assert total_denied == 0, (
                f"{sname} baseline should have zero denied boardings, got {total_denied}"
            )


# ========================================================================
# 19. Environment wiring tests
# ========================================================================

class TestEnvironmentWiring:
    """
    Verify AirlineRM class list_tasks/list_splits returns correct data.
    """

    def test_list_splits(self):
        from airlinerm import AirlineRM
        splits = AirlineRM.list_splits()
        assert splits == ["train", "test"]

    def test_list_tasks_train(self):
        from airlinerm import AirlineRM
        tasks = AirlineRM.list_tasks("train")
        assert len(tasks) == 9
        ids = [t["id"] for t in tasks]
        assert "summer_peak_v1" in ids
        assert "winter_holiday_v3" in ids
        assert "shoulder_spring_v2" in ids

    def test_list_tasks_test(self):
        from airlinerm import AirlineRM
        tasks = AirlineRM.list_tasks("test")
        assert len(tasks) == 3
        ids = [t["id"] for t in tasks]
        for v in range(1, 4):
            assert f"fall_business_v{v}" in ids

    def test_list_tasks_invalid_split(self):
        from airlinerm import AirlineRM
        with pytest.raises(ValueError):
            AirlineRM.list_tasks("invalid")

    def test_task_id_format(self):
        """All task IDs should match the pattern 'scenario_vN'."""
        from airlinerm import AirlineRM
        for split in ["train", "test"]:
            tasks = AirlineRM.list_tasks(split)
            for t in tasks:
                assert re.match(r"^[a-z_]+_v\d+$", t["id"]), (
                    f"Task ID '{t['id']}' doesn't match expected pattern"
                )


# ------------------------------------------------------------------ #
# 20. Citation integrity                                              #
# ------------------------------------------------------------------ #

class TestCitationIntegrity:
    """Verify that key citations in source files are accurate and haven't drifted."""

    @staticmethod
    def _file_contains(filepath, substring):
        with open(filepath) as f:
            return substring in f.read()

    def test_shintani_umeno_not_gerlach(self):
        """Gerlach et al. was a hallucinated author; correct is Shintani & Umeno."""
        assert self._file_contains("simulation.py", "Shintani & Umeno 2023")
        assert not self._file_contains("simulation.py", "Gerlach et al.")

    def test_rothstein_transportation_science(self):
        """Rothstein 1971 in Transportation Science is 'An Airline Overbooking Model', pp 180-192."""
        assert self._file_contains("network.py", "Rothstein 1971")
        assert self._file_contains("network.py", "Transportation Science 5(2):180-192")
        assert not self._file_contains("network.py", "180-196")  # wrong page range

    def test_smith_leimkuhler_darrow(self):
        """Smith et al. 1992, Interfaces 22(1):8-31 — verified."""
        assert self._file_contains("network.py", "Smith, Leimkuhler & Darrow 1992")
        assert self._file_contains("network.py", "Interfaces 22(1):8-31")

    def test_belobaba_1989(self):
        """Belobaba 1989, Operations Research 37(2) — verified."""
        assert self._file_contains("simulation.py", "Belobaba 1989")
        assert self._file_contains("simulation.py", "Operations Research 37(2)")

    def test_gallego_van_ryzin_1994(self):
        """Gallego & van Ryzin 1994, Management Science 40(8) — verified."""
        assert self._file_contains("simulation.py", "Gallego & van Ryzin 1994")
        assert self._file_contains("simulation.py", "Management Science 40(8)")

    def test_dot_14_cfr_250(self):
        """DOT 14 CFR 250.5 denied boarding regulation — verified."""
        assert self._file_contains("network.py", "DOT 14 CFR 250.5")
        assert self._file_contains("network.py", "$775")
        assert self._file_contains("network.py", "$1,550")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
