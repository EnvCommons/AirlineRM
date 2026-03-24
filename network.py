"""
Static network definitions for the Airline Revenue Management environment.

Defines hub-and-spoke network topology, aircraft types, fare classes, routes,
scenarios, and flight schedule generation. All parameters are grounded in
real-world airline industry data.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Aircraft types
#
# Seat counts reflect typical domestic single-class configurations.
# Operating costs approximated from FAA/DOT Form 41 (Schedule P-5.2)
# direct operating cost ranges for regional and narrowbody aircraft.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AircraftType:
    code: str
    total_seats: int
    operating_cost_per_block_hour: float  # USD
    range_nm: int
    turnaround_minutes: int  # minimum ground time

AIRCRAFT_TYPES: Dict[str, AircraftType] = {
    "E175": AircraftType(
        code="E175",
        total_seats=76,
        operating_cost_per_block_hour=2800.0,
        range_nm=2200,
        turnaround_minutes=35,
    ),
    "737-700": AircraftType(
        code="737-700",
        total_seats=138,
        operating_cost_per_block_hour=4200.0,
        range_nm=3000,
        turnaround_minutes=40,
    ),
    "737-800": AircraftType(
        code="737-800",
        total_seats=175,
        operating_cost_per_block_hour=4800.0,
        range_nm=3100,
        turnaround_minutes=45,
    ),
}

# ---------------------------------------------------------------------------
# Fare classes (ordered highest to lowest)
#
# No-show rates (2%-15%) calibrated from:
#   - Smith, Leimkuhler & Darrow 1992, "Yield Management at American
#     Airlines", Interfaces 22(1):8-31 — reported ~15% no-show for sold-out
#     flights without overbooking.
#   - Lawrence, Hong & Cherrier 2003, "Passenger-Based Predictive Modeling
#     of Airline No-show Rates", Proc. KDD — 6-10% average, varying by
#     fare class, frequent-flyer status, and ticketing channel.
# Gradient: refundable/flexible tickets have higher no-show rates (no
# financial penalty); non-refundable deep-discount tickets have lowest.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FareClass:
    code: str
    name: str
    multiplier: float        # fraction of route max_fare
    advance_purchase_days: int
    no_show_rate: float      # probability of no-show
    refundable: bool
    change_fee: float        # USD, 0 means free changes
    restrictions: str

FARE_CLASSES: List[FareClass] = [
    FareClass("Y", "Full Fare Economy", 1.00, 0,  0.15, True,  0,   "None — fully flexible"),
    FareClass("B", "Flex Economy",      0.82, 0,  0.12, True,  0,   "Minimal — free changes"),
    FareClass("M", "Standard Economy",  0.62, 3,  0.08, False, 75,  "3-day advance purchase"),
    FareClass("H", "Restricted Econ",   0.47, 7,  0.06, False, 150, "7-day advance, change fee"),
    FareClass("Q", "Discount Economy",  0.35, 14, 0.04, False, 200, "14-day advance, Saturday stay"),
    FareClass("V", "Deep Discount",     0.25, 21, 0.03, False, 200, "21-day advance, non-refundable"),
    FareClass("T", "Promo",             0.18, 30, 0.03, False, 200, "30-day advance, limited availability"),
    FareClass("L", "Basic Economy",     0.13, 45, 0.02, False, 200, "45-day advance, no changes, last to board"),
]

FARE_CLASS_CODES = [fc.code for fc in FARE_CLASSES]

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Route:
    origin: str
    destination: str
    distance_nm: int
    category: str            # "business", "mixed", "leisure"
    default_aircraft: str    # aircraft type code
    daily_frequencies: int
    max_fare: float          # Y-class one-way fare (USD)
    base_demand_mean: float  # mean total daily one-way pax demand
    base_demand_std: float   # std deviation of daily demand
    business_traveler_pct: float  # fraction of pax who are business travelers
    block_time_hours: float  # average one-way block time

ROUTES: List[Route] = [
    # --- Business-heavy corridors ---
    # base_demand_mean = total daily pax demand across all frequencies on this route.
    # Calibrated for ~85% base load factor matching BTS LOADFACTORD 2019-2024
    # domestic average (83-87%). Demand ≈ 1.05 * seats * freq to account for
    # spill from WTP below lowest fare and stochastic variance.
    Route("HUB", "BOS", 820, "business", "737-800", 3, 620, 555, 55, 0.55, 2.1),
    Route("HUB", "ORD", 520, "business", "737-800", 4, 540, 740, 65, 0.50, 1.6),
    Route("HUB", "SFO", 1900, "business", "737-800", 2, 780, 370, 45, 0.50, 4.5),
    # --- Mixed markets ---
    Route("HUB", "DFW", 780, "mixed", "737-800", 3, 520, 555, 55, 0.35, 2.0),
    Route("HUB", "ATL", 430, "mixed", "737-700", 3, 440, 435, 45, 0.35, 1.3),
    Route("HUB", "DEN", 1040, "mixed", "737-700", 2, 510, 290, 35, 0.30, 2.6),
    Route("HUB", "SEA", 1820, "mixed", "737-800", 2, 650, 370, 40, 0.30, 4.2),
    # --- Leisure-heavy markets ---
    Route("HUB", "MCO", 690, "leisure", "737-800", 2, 380, 370, 45, 0.15, 1.8),
    Route("HUB", "FLL", 780, "leisure", "737-700", 2, 350, 290, 35, 0.15, 2.0),
    Route("HUB", "CUN", 1300, "leisure", "737-800", 1, 520, 185, 30, 0.10, 3.3),
    # --- Thin regional ---
    Route("HUB", "SRQ", 610, "leisure", "E175", 2, 310, 160, 22, 0.20, 1.7),
    Route("HUB", "GRR", 350, "mixed", "E175", 3, 280, 240, 25, 0.30, 1.2),
]


def route_key(route: Route) -> str:
    return f"{route.origin}-{route.destination}"


# ---------------------------------------------------------------------------
# Scenarios
#
# Demand multipliers reflect BTS seasonal load factor patterns:
# summer_peak 1.30x (June-August peak), shoulder_spring 0.90x (off-peak),
# winter_holiday 1.15x (holiday surge), fall_business 1.05x (conference
# season). See BTS LOADFACTORD series.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Scenario:
    name: str
    season: str          # "summer", "winter", "spring", "fall"
    demand_multiplier: float
    description: str

SCENARIOS: Dict[str, Scenario] = {
    "summer_peak": Scenario(
        "summer_peak", "summer", 1.30,
        "Peak summer travel season with high leisure demand, thunderstorm risk, and strong load factors."
    ),
    "winter_holiday": Scenario(
        "winter_holiday", "winter", 1.15,
        "Holiday travel season with mixed business/leisure demand and significant snowstorm risk."
    ),
    "shoulder_spring": Scenario(
        "shoulder_spring", "spring", 0.90,
        "Shoulder season with moderate demand, competitor pressure, and variable weather."
    ),
    "fall_business": Scenario(
        "fall_business", "fall", 1.05,
        "Fall business travel season with conference-driven demand spikes and occasional fog."
    ),
}


# ---------------------------------------------------------------------------
# Disruption type definitions
# ---------------------------------------------------------------------------

@dataclass
class DisruptionType:
    name: str
    base_probability_per_day: float
    seasonal_multipliers: Dict[str, float]
    affected_flights_pct_range: Tuple[float, float]
    delay_hours_range: Tuple[float, float]
    cancellation_pct: float  # fraction of affected flights that get cancelled

DISRUPTION_TYPES: Dict[str, DisruptionType] = {
    "thunderstorm": DisruptionType(
        "thunderstorm", 0.08,
        {"summer": 2.0, "winter": 0.2, "spring": 1.0, "fall": 0.7},
        (0.15, 0.40), (1.0, 4.0), 0.15,
    ),
    "snowstorm": DisruptionType(
        "snowstorm", 0.04,
        {"summer": 0.0, "winter": 3.0, "spring": 0.3, "fall": 0.1},
        (0.20, 0.60), (2.0, 6.0), 0.25,
    ),
    "fog": DisruptionType(
        "fog", 0.06,
        {"summer": 0.4, "winter": 1.5, "spring": 1.2, "fall": 1.3},
        (0.05, 0.20), (0.5, 2.5), 0.05,
    ),
    "crew_shortage": DisruptionType(
        "crew_shortage", 0.05,
        {"summer": 1.2, "winter": 1.0, "spring": 0.8, "fall": 0.8},
        (0.02, 0.08), (1.0, 3.0), 0.30,
    ),
    "aircraft_mechanical": DisruptionType(
        "aircraft_mechanical", 0.10,
        {"summer": 1.0, "winter": 1.1, "spring": 1.0, "fall": 1.0},
        (0.01, 0.05), (0.5, 4.0), 0.20,
    ),
}

# Competitor fare war is modelled separately (affects demand, not flights).
# Max overbooking of 15% (enforced in airlinerm.py) exceeds typical
# industry practice of 5-10% (Rothstein 1971, "Airline Overbooking: The
# State of the Art", Transportation Science 5(2):180-196) but provides
# exploration headroom for RL agents.
@dataclass
class CompetitorFareWar:
    base_probability_per_day: float
    seasonal_multipliers: Dict[str, float]
    affected_routes_range: Tuple[int, int]
    demand_reduction_pct_range: Tuple[float, float]
    duration_days_range: Tuple[int, int]

COMPETITOR_FARE_WAR = CompetitorFareWar(
    base_probability_per_day=0.03,
    seasonal_multipliers={"summer": 0.8, "winter": 1.2, "spring": 1.0, "fall": 1.5},
    affected_routes_range=(1, 3),
    demand_reduction_pct_range=(0.15, 0.35),
    duration_days_range=(3, 10),
)


# ---------------------------------------------------------------------------
# Cost parameters (grounded in DOT / industry data)
#
# DENIED_BOARDING_COST: DOT 14 CFR 250.5, 200% tier maximum (pre-Jan 2025
#     values). The actual regulation is tiered: $0 if re-accommodated within
#     1h, 200% of one-way fare (max $775) if 1-2h late, 400% (max $1,550)
#     if >2h late. We use $775 as a flat simplification representing the
#     most common tier. (Updated to $1,075/$2,150 effective Jan 22, 2025.)
# CANCELLATION_COST_PER_PAX: Estimated $150-$250 range from DOT airline
#     customer service commitments (rebooking + hotel/meal vouchers).
# DELAY_COST_PER_PAX_PER_HOUR: Estimated from airline operations literature.
# AIRCRAFT_SWAP_FIXED_COST: Estimated from airline operations literature
#     (crew repositioning, provisioning, gate reassignment).
# ---------------------------------------------------------------------------

DENIED_BOARDING_COST = 775.0        # DOT 14 CFR 250.5 — 200% tier max (pre-2025)
CANCELLATION_COST_PER_PAX = 200.0   # Rebooking + accommodation (DOT commitments)
DELAY_COST_PER_PAX_PER_HOUR = 50.0  # >1h delay compensation estimate
DELAY_THRESHOLD_HOURS = 1.0         # Delays under this incur no per-pax cost
AIRCRAFT_SWAP_FIXED_COST = 5000.0   # Crew repositioning, provisioning, etc.


# ---------------------------------------------------------------------------
# Flight schedule generation
# ---------------------------------------------------------------------------

@dataclass
class ScheduledFlight:
    """A single scheduled flight in the network."""
    flight_id: str
    route: Route
    departure_day: int       # day within the 30-day horizon (1-indexed)
    frequency_index: int     # which departure of the day (0-indexed)
    aircraft_type: str       # code, may change via swap
    capacity: int            # seats, derived from aircraft type

    # Mutable state (initialized at creation, modified during simulation)
    bookings_by_class: Dict[str, int] = field(default_factory=dict)
    fare_availability: Dict[str, bool] = field(default_factory=dict)
    overbooking_limit: int = 0
    status: str = "scheduled"  # scheduled, delayed, cancelled, departed
    delay_hours: float = 0.0

    def __post_init__(self):
        if not self.bookings_by_class:
            self.bookings_by_class = {fc.code: 0 for fc in FARE_CLASSES}
        if not self.fare_availability:
            self.fare_availability = {fc.code: True for fc in FARE_CLASSES}

    @property
    def total_booked(self) -> int:
        return sum(self.bookings_by_class.values())

    @property
    def load_factor(self) -> float:
        if self.capacity == 0:
            return 0.0
        return self.total_booked / self.capacity

    def fare_for_class(self, fare_class_code: str) -> float:
        """Get the dollar fare for a given class on this flight."""
        for fc in FARE_CLASSES:
            if fc.code == fare_class_code:
                return round(self.route.max_fare * fc.multiplier, 2)
        return 0.0

    def copy(self) -> "ScheduledFlight":
        """Deep copy for baseline simulation."""
        sf = ScheduledFlight(
            flight_id=self.flight_id,
            route=self.route,
            departure_day=self.departure_day,
            frequency_index=self.frequency_index,
            aircraft_type=self.aircraft_type,
            capacity=self.capacity,
        )
        sf.bookings_by_class = dict(self.bookings_by_class)
        sf.fare_availability = dict(self.fare_availability)
        sf.overbooking_limit = self.overbooking_limit
        sf.status = self.status
        sf.delay_hours = self.delay_hours
        return sf


def generate_flight_schedule(total_days: int = 30) -> List[ScheduledFlight]:
    """Generate the complete flight schedule for the operating horizon."""
    flights = []
    for route in ROUTES:
        rk = route_key(route)
        ac = AIRCRAFT_TYPES[route.default_aircraft]
        for day in range(1, total_days + 1):
            for freq_idx in range(route.daily_frequencies):
                fid = f"{rk}-F{freq_idx}-D{day}"
                flight = ScheduledFlight(
                    flight_id=fid,
                    route=route,
                    departure_day=day,
                    frequency_index=freq_idx,
                    aircraft_type=ac.code,
                    capacity=ac.total_seats,
                )
                flights.append(flight)
    return flights


def get_daily_flight_count() -> int:
    """Total number of departures per day across the network."""
    return sum(r.daily_frequencies for r in ROUTES)


def get_flights_for_day(flights: List[ScheduledFlight], day: int) -> List[ScheduledFlight]:
    """Return all flights scheduled for a given day."""
    return [f for f in flights if f.departure_day == day]


def compute_fare_table(route: Route) -> List[Dict]:
    """Return a fare table for a route showing all classes and fares."""
    table = []
    for fc in FARE_CLASSES:
        table.append({
            "class": fc.code,
            "name": fc.name,
            "fare": round(route.max_fare * fc.multiplier, 2),
            "advance_purchase": fc.advance_purchase_days,
            "no_show_rate": fc.no_show_rate,
            "refundable": fc.refundable,
            "restrictions": fc.restrictions,
        })
    return table
