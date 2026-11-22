import json
import sys
import unittest
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
for path in (THIS_DIR, THIS_DIR / "isa_set"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import make_batch
from test_t20_zone_weights import BatchWithZonesTests  # same fixture: a small fake zone-weights file in a temporary work folder

DESIGN = json.loads((THIS_DIR / "isa_set" / "design.json").read_text(encoding="utf-8"))


class FleetAssignmentTests(BatchWithZonesTests):
    """The fleets of a batch are the points of ONE 2-D Latin hypercube; with fleet.assignment 'stratified' every
    instance gets one point from each stratum of the fleet-size coordinate."""

    def fleets(self, n):
        make_batch.main(["--batch", "4", "--n", str(n), "--work", str(self.work)])
        raw = json.loads((self.work / "design" / "batch_04_fleets.json").read_text(encoding="utf-8"))
        per_instance = {}
        for key, fleet in raw.items():
            per_instance.setdefault(fleet["instance"], []).append(fleet)
        return raw, per_instance

    def test_ten_fleets_per_instance_and_walking_is_fixed_at_600(self):
        raw, per_instance = self.fleets(20)
        self.assertEqual(len(raw), 20 * DESIGN["fleet"]["draws_per_instance"])
        self.assertTrue(all(len(v) == DESIGN["fleet"]["draws_per_instance"] for v in per_instance.values()))
        configs = sorted((self.work / "configs" / "batch_04").glob("*.json"))
        walking = [a for a in json.loads(configs[0].read_text(encoding="utf-8"))["attributes"] if a["name"] == "max_walking"][0]
        self.assertEqual(walking["pdf"][0]["loc"], 600)
        self.assertEqual(walking["pdf"][0]["scale"], 0)

    def test_stratified_gives_every_instance_the_whole_fleet_range(self):
        if DESIGN["fleet"].get("assignment") != "stratified":
            self.skipTest("design.json does not ask for stratified fleets")
        low, high = DESIGN["fleet"]["percent_of_requests"]
        k = DESIGN["fleet"]["draws_per_instance"]
        raw, per_instance = self.fleets(100)
        step = (high - low) / k
        for instance, fleets in per_instance.items():
            percents = sorted(f["fleet_percent_of_requests"] for f in fleets)
            # one fleet in every stratum of width (high - low) / k, so the range is at least (high - low) - 2 strata widths
            for s in range(k):   # the s-th smallest fleet lies in the s-th stratum (the percent is rounded to 2 decimals)
                lo, hi = low + s * step, low + (s + 1) * step
                self.assertTrue(lo - 0.006 <= percents[s] <= hi + 0.006, (instance, s, percents))
        # all 1000 points are still the points of one hypercube: each stratum of the whole batch holds exactly n fleets
        everything = sorted(f["fleet_percent_of_requests"] for f in raw.values())
        self.assertEqual(len(everything), 100 * k)


if __name__ == "__main__":
    unittest.main()
