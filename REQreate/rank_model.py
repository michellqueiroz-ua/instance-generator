"""Rank-based spatial request distribution (``rank_model``).

Extracted from the former trip_patterns_general module, whose remaining
real-data analysis code now lives in ``legacy/``.
"""

import os

from .retrieve_POIs import get_POIs_matrix_csv
from .retrieve_POIs import attribute_density_zones
from .retrieve_POIs import calc_rank_between_zones
from .retrieve_POIs import calc_probability_travel_between_zones


def rank_model(network, place_name):

    output_folder_base = place_name

    save_dir = os.getcwd()+'/'+place_name

    rows = 25
    columns = 25
    print('smaller zones')
    network.zones = network.divide_network_grid(rows, columns, save_dir, output_folder_base)

    pois = get_POIs_matrix_csv(network.G_drive, place_name, save_dir, output_folder_base)
    attribute_density_zones(network, pois)
    zone_ranks = calc_rank_between_zones(network)
    alpha = 0.86
    calc_probability_travel_between_zones(network, zone_ranks, alpha)

    print(network.zones['density_pois'].head())
    network.zone_ranks = zone_ranks
