"""
demo_config.py — allowlist, caps and run mode for the public demo

Block 1 of DEMO_HOSTING_PLAN.md. This module is pure configuration plus a
readiness check: it does not import Streamlit, REQreate or osmnx, and nothing
in ``app.py`` uses it yet. Wiring happens in a later block.

Run it directly to see which allowlisted cities are actually prepared:

    python demo_config.py

Two modes, selected by the ``REQREATE_MODE`` environment variable:

    demo    picklist of prepared cities, capped sizes, no live OSM downloads
    local   free-text city, no caps (default; current behaviour, unchanged)
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent

ENV_MODE = "REQREATE_MODE"
MODE_DEMO = "demo"
MODE_LOCAL = "local"
VALID_MODES = (MODE_DEMO, MODE_LOCAL)


# ---------------------------------------------------------------------------
# Caps (demo mode only)
# ---------------------------------------------------------------------------
# PROVISIONAL. These are deliberately conservative and have not yet been tuned
# against measured generation times on the Replit VM. Re-tune once phase 2 has
# real timings; they are meant to be edited here and nowhere else.

@dataclass(frozen=True)
class DemoCaps:
    """Hard limits applied only when running in demo mode."""

    max_requests: int = 500
    max_replicates: int = 3
    generation_timeout_s: int = 300
    result_retention_days: int = 7

    def describe(self) -> List[str]:
        return [
            f"max requests per instance : {self.max_requests}",
            f"max replicates            : {self.max_replicates}",
            f"generation timeout        : {self.generation_timeout_s} s",
            f"results kept for          : {self.result_retention_days} days",
        ]


CAPS = DemoCaps()


# ---------------------------------------------------------------------------
# Allowlist
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class City:
    """One allowlisted city.

    ``folder`` is the directory name REQreate uses, which is also the
    ``place_name`` string passed to the generator.
    """

    key: str
    label: str
    folder: str
    country: str
    note: str = ""

    @property
    def path(self) -> Path:
        return REPO_ROOT / self.folder

    @property
    def pickle_dir(self) -> Path:
        return self.path / "pickle"

    def network_pickle(self) -> Optional[Path]:
        """The prepared network file, or None when the city is not ready."""
        if not self.pickle_dir.is_dir():
            return None
        matches = sorted(self.pickle_dir.glob("*.network.class.pkl"))
        return matches[0] if matches else None

    def is_prepared(self) -> bool:
        return self.network_pickle() is not None

    def size_mb(self) -> float:
        """Size of what the VM needs for this city, in MB (0.0 if absent)."""
        total = 0
        for relative in ("pickle", "csv", "travel_time_matrix"):
            directory = self.path / relative
            if not directory.is_dir():
                continue
            for item in directory.rglob("*"):
                # csv/ holds preparation scaffolding; only stations/zones are
                # needed at generation time.
                if relative == "csv" and not item.name.endswith(
                    (".stations.csv", ".zones.csv")
                ):
                    continue
                if item.is_file():
                    total += item.stat().st_size
        return total / 1048576


ALLOWED_CITIES: Tuple[City, ...] = (
    City(
        key="copenhagen",
        label="Copenhagen Municipality",
        folder="Copenhagen Municipality, Denmark",
        country="Denmark",
        note="Largest prepared network; 803 stations.",
    ),
    City(
        key="antwerp",
        label="Antwerp",
        folder="Antwerp, Belgium",
        country="Belgium",
        note="Reference city of the ODBRP papers.",
    ),
    City(
        key="salvador",
        label="Salvador",
        folder="Salvador, Brazil",
        country="Brazil",
        note="Latin American case; night-service motivation.",
    ),
    City(
        key="maastricht",
        label="Maastricht",
        folder="Maastricht, Netherlands",
        country="Netherlands",
        note="Small network; fastest generation.",
    ),
)


# ---------------------------------------------------------------------------
# Mode
# ---------------------------------------------------------------------------

def current_mode() -> str:
    """Resolve the run mode; anything unrecognised falls back to local."""
    raw = (os.environ.get(ENV_MODE) or "").strip().lower()
    return raw if raw in VALID_MODES else MODE_LOCAL


def is_demo() -> bool:
    return current_mode() == MODE_DEMO


# ---------------------------------------------------------------------------
# Lookup helpers (used by the UI in a later block)
# ---------------------------------------------------------------------------

def city_by_key(key: str) -> Optional[City]:
    for city in ALLOWED_CITIES:
        if city.key == key:
            return city
    return None


def city_by_folder(folder: str) -> Optional[City]:
    wanted = (folder or "").strip().strip('"').strip("'")
    for city in ALLOWED_CITIES:
        if city.folder == wanted:
            return city
    return None


def available_cities() -> List[City]:
    """Allowlisted cities that are actually prepared on this machine."""
    return [city for city in ALLOWED_CITIES if city.is_prepared()]


def missing_cities() -> List[City]:
    """Allowlisted cities that still need preparing (phase 2)."""
    return [city for city in ALLOWED_CITIES if not city.is_prepared()]


def is_allowed(folder: str) -> bool:
    """True when this place name may be generated in the current mode.

    In local mode every place is allowed, preserving today's behaviour.
    """
    if not is_demo():
        return True
    city = city_by_folder(folder)
    return city is not None and city.is_prepared()


def clamp_requests(value: int) -> int:
    return min(int(value), CAPS.max_requests) if is_demo() else int(value)


def clamp_replicates(value: int) -> int:
    return min(int(value), CAPS.max_replicates) if is_demo() else int(value)


# ---------------------------------------------------------------------------
# Readiness report
# ---------------------------------------------------------------------------

def readiness() -> Dict[str, object]:
    ready = available_cities()
    missing = missing_cities()
    return {
        "mode": current_mode(),
        "prepared": ready,
        "missing": missing,
        "total_mb": sum(city.size_mb() for city in ready),
        "demo_ready": bool(ready),
    }


def _report() -> int:
    state = readiness()
    print(f"mode: {state['mode']}  (set {ENV_MODE}=demo to enable demo limits)")
    print()

    print("Allowlisted cities")
    print("------------------")
    for city in ALLOWED_CITIES:
        if city.is_prepared():
            print(f"  [ready]   {city.label:24s} {city.size_mb():8.1f} MB  {city.folder}")
        else:
            print(f"  [missing] {city.label:24s} {'-':>8}     {city.folder}")

    print()
    print(f"prepared      : {len(state['prepared'])}/{len(ALLOWED_CITIES)}")
    print(f"VM footprint  : {state['total_mb']:.1f} MB for the prepared ones")
    print()

    print("Demo caps")
    print("---------")
    for line in CAPS.describe():
        print(f"  {line}")

    missing = state["missing"]
    if missing:
        print()
        print("Not yet prepared - phase 2 must build a bundle for each:")
        for city in missing:
            print(f"  - {city.label} ({city.folder})")

    return 0 if state["demo_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(_report())
