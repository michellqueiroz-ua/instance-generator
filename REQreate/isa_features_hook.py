"""
isa_features_hook.py — T12 (ISA-HODBRP)

Bridge between the REQreate generation pipeline and the ISA feature export in
``isa-features-plan-implementation/``.

That folder is not an importable package (its name contains dashes), so this
module puts it on ``sys.path`` and exposes a single, failure-tolerant call for
``input_json.py``:

    export_generated_instance_features(json_path, instance_ref, network_graph)

Feature export must never break instance generation, so every failure is
logged and swallowed; the function returns True only when a row was written.

The destination is ``<repo_root>/data/results/features.csv`` by default and can
be redirected with the ``ISA_FEATURES_CSV`` environment variable.
"""

import logging
import os
import sys


LOGGER = logging.getLogger(__name__)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ISA_FEATURES_DIR points an installed (non-checkout) package at the ISA code,
# e.g. in CI where REQreate runs from site-packages.
_ISA_DIR = os.environ.get("ISA_FEATURES_DIR") or os.path.join(_REPO_ROOT, "isa-features-plan-implementation")


def _ensure_isa_on_path():
    if os.path.isdir(_ISA_DIR) and _ISA_DIR not in sys.path:
        sys.path.insert(0, _ISA_DIR)


def export_generated_instance_features(
    json_path,
    instance_ref=None,
    network_graph=None,
    vehicles=None,
    csv_path=None,
    on_duplicate="replace",
):
    """
    Compute the 21 ISA features for one generated instance and append its row.

    Parameters
    ----------
    json_path : str
        Path to the instance JSON just written by the generator.
    instance_ref : str, optional
        Key written to the ``instance`` column; defaults to the JSON stem.
        Pass the final CSV name when the pipeline renames instances afterwards,
        so features.csv keys match the distributed instance files.
    network_graph : networkx graph, optional
        Drive graph (``inst.network.G_drive``) used to measure the service area
        from reachable geometry instead of the bounding-box fallback.
    vehicles : list of dict, optional
        Fleet definition. When omitted the fleet is resolved from
        ``isa-features-plan-implementation/fleet_config.json`` (or
        ``$ISA_FLEET_CONFIG``); fleet features stay NaN if neither is present.
    csv_path : str, optional
        Destination features.csv; defaults to the resolved project path.
    on_duplicate : str
        ``"replace"`` (default), ``"skip"``, or ``"error"``.

    Returns
    -------
    bool
        True when a row was written, False when the export was skipped.
    """
    try:
        _ensure_isa_on_path()

        from features_export import append_features_row
        from compute_features_scaffold import compute_features
        from fleet_config import resolve_fleet
        from reqreate_adapter import load_reqreate_instance, service_area_km2_from_graph

        service_area_km2 = None
        if network_graph is not None:
            area = service_area_km2_from_graph(network_graph)
            if area == area and area > 0.0:  # not NaN
                service_area_km2 = area

        instance = load_reqreate_instance(
            json_path,
            service_area_km2=service_area_km2,
            network_graph=network_graph,
        )

        if vehicles is None:
            # Fleet is a solver input, not part of the instance; features 11-14
            # come from fleet_config.json.
            vehicles = resolve_fleet(
                instance_ref if instance_ref else json_path,
                n_requests=len(instance.get("requests", [])),
            )
        instance["vehicles"] = list(vehicles or [])

        features = compute_features(instance)
        target = append_features_row(
            instance_ref if instance_ref else json_path,
            features,
            csv_path=csv_path,
            on_duplicate=on_duplicate,
        )

        print(f"[features.csv] wrote row for {instance_ref or json_path} -> {target}")
        return True

    except Exception as exc:
        LOGGER.warning("feature export skipped for %s: %s", json_path, exc)
        print(f"[features.csv] export skipped for {json_path}: {exc}")
        return False
