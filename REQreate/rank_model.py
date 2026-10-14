"""Zone grid, POI density and zone travel probabilities (``rank_model``).

Extracted from the former trip_patterns_general module, whose remaining
real-data analysis code now lives in ``legacy/``.

The three steps cost very different amounts, so they run separately:

* ``build_zone_grid``: always, when the network is built (seconds).
* ``add_poi_density``: downloads POIs from OpenStreetMap (tens of minutes
  for a mid-sized city); only when a configuration uses ``method_pois``.
* ``add_zone_probabilities``: zone ranks and travel probabilities between
  zones; only when a configuration sets ``zone_probabilities``.
"""

import os

from .logger_utils import get_logger
from .retrieve_POIs import get_POIs_matrix_csv
from .retrieve_POIs import attribute_density_zones
from .retrieve_POIs import calc_rank_between_zones
from .retrieve_POIs import calc_probability_travel_between_zones

GRID_ROWS = 25
GRID_COLUMNS = 25
ALPHA = 0.86


def _log(message):
    logger = get_logger()
    if logger:
        logger.info(message)
    else:
        print(message)


def _save_dir(place_name):
    return os.getcwd()+'/'+place_name


def build_zone_grid(network, place_name):
    """Replace network.zones with a GRID_ROWS x GRID_COLUMNS grid over the city."""
    network.zones = network.divide_network_grid(GRID_ROWS, GRID_COLUMNS, _save_dir(place_name), place_name)


def has_poi_density(network):
    return 'density_pois' in network.zones.columns


def add_poi_density(network, place_name):
    """Download POIs and attach each zone's share of them (``density_pois``).

    Returns True when it did work, False when the density was already there.
    """
    if has_poi_density(network):
        return False
    _log('Downloading points of interest from OpenStreetMap (needed for method_pois / zone_probabilities)...')
    pois = get_POIs_matrix_csv(network.G_drive, place_name, _save_dir(place_name), place_name)
    attribute_density_zones(network, pois)
    return True


def add_zone_probabilities(network, place_name, alpha=ALPHA):
    """Compute zone ranks and travel probabilities between zones.

    Stores them on the network (``zone_ranks``, ``zone_probabilities``) and
    writes ``<place>/csv/<place>.zone_ranks.csv`` and
    ``<place>.zone_probabilities.csv``. Returns True when it did work.
    """
    if getattr(network, 'zone_probabilities', None) is not None:
        return False
    add_poi_density(network, place_name)

    _log('Calculating zone ranks and travel probabilities between zones...')
    zone_ranks = calc_rank_between_zones(network)
    network.zone_ranks = zone_ranks
    network.zone_probabilities = calc_probability_travel_between_zones(network, zone_ranks, alpha)

    save_dir_csv = os.path.join(_save_dir(place_name), 'csv')
    os.makedirs(save_dir_csv, exist_ok=True)
    zone_ranks.to_csv(os.path.join(save_dir_csv, place_name+'.zone_ranks.csv'))
    network.zone_probabilities.to_csv(os.path.join(save_dir_csv, place_name+'.zone_probabilities.csv'))
    _log(f'Zone probabilities calculated: {len(zone_ranks)} zones')
    return True


def rank_model(network, place_name):
    """Full pipeline: grid, POI density, zone ranks and probabilities."""
    build_zone_grid(network, place_name)
    add_poi_density(network, place_name)
    add_zone_probabilities(network, place_name)
