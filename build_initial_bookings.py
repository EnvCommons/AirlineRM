"""
Pre-compute the per-task initial-bookings state at image build time.

Running ``_populate_initial_bookings`` in ``SimulationState.__init__`` performs
~tens of thousands of synchronous numpy RNG draws per session under the GIL.
Under high session concurrency this serialises setup across sessions on the
same pod and starves the asyncio loop, pushing setup past the platform's
session deadline.

For each task_id we precompute and serialise:

  * ``bookings`` — ``{flight_id: {fare_class: count}}`` after
    ``_populate_initial_bookings`` has run.
  * ``rng_state`` — the rng's ``bit_generator.state`` after the bookings
    simulation, so subsequent draws (``generate_disruptions``, ``advance_day``)
    remain bit-exact with the non-cached path.

Run during docker build (see Dockerfile).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from network import SCENARIOS
from simulation import SimulationState


def _seed_for_task(task_id: str) -> int:
    return int(hashlib.sha256(task_id.encode()).hexdigest(), 16) % (2**32)


def _task_ids() -> List[str]:
    # Mirror AirlineRM.list_tasks(): train + test splits.
    ids: List[str] = []
    for scen in ("summer_peak", "winter_holiday", "shoulder_spring"):
        for v in (1, 2, 3):
            ids.append(f"{scen}_v{v}")
    for v in (1, 2, 3):
        ids.append(f"fall_business_v{v}")
    return ids


def _build_one(task_id: str, total_days: int = 30) -> Dict[str, Any]:
    scenario_name = task_id.rsplit("_v", 1)[0]
    scenario = SCENARIOS[scenario_name]
    rng = np.random.default_rng(_seed_for_task(task_id))

    sim = SimulationState(rng, scenario, total_days)  # runs _populate_initial_bookings

    bookings = {f.flight_id: dict(f.bookings_by_class) for f in sim.flights}
    return {
        "bookings": bookings,
        "rng_state": rng.bit_generator.state,
    }


def main() -> None:
    out: Dict[str, Dict[str, Any]] = {tid: _build_one(tid) for tid in _task_ids()}
    target = Path(__file__).parent / "initial_bookings.json"
    target.write_text(json.dumps(out))
    print(f"wrote {len(out)} initial-booking states to {target} ({target.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
