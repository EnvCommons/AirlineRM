"""
Pre-compute baseline simulation results for every task at image build time.

The baseline is a deterministic 30-day simulation derived from a
sha256(task_id) seed, so it is fully reproducible and identical across
sessions. Computing it once at build time and serialising to baselines.json
removes ~300ms of synchronous CPU work from AirlineRM.__init__, which under
high concurrency was serialising through the GIL and pushing setup past the
platform's session deadline.

Run during docker build (see Dockerfile).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np

from baseline import BaselinePolicy
from network import SCENARIOS


def _seed_for_task(task_id: str) -> int:
    return int(hashlib.sha256(task_id.encode()).hexdigest(), 16) % (2**32)


def main() -> None:
    task_ids: list[str] = []
    # Mirror AirlineRM.list_tasks(): train + test splits.
    for scen in ("summer_peak", "winter_holiday", "shoulder_spring"):
        for v in (1, 2, 3):
            task_ids.append(f"{scen}_v{v}")
    for v in (1, 2, 3):
        task_ids.append(f"fall_business_v{v}")

    out: dict[str, list[dict]] = {}
    for tid in task_ids:
        scenario_name = tid.rsplit("_v", 1)[0]
        rng = np.random.default_rng(_seed_for_task(tid))
        results = BaselinePolicy(rng, SCENARIOS[scenario_name], 30).run_full_simulation()
        out[tid] = [dataclasses.asdict(r) for r in results]

    target = Path(__file__).parent / "baselines.json"
    target.write_text(json.dumps(out))
    print(f"wrote {len(out)} baselines to {target} ({target.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
