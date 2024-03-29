import datetime
import gc
import codecs
import geopandas as gpd
import glob
import json
import math
from math import sqrt
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
import numpy as np
import os
import osmapi as osm
import osmnx as ox
import pandas as pd
import pickle
from random import randint
from random import seed
from random import choices
import ray
#from streamlit import caching
import sys
from shapely.geometry import Point
from shapely.geometry import MultiPoint
from shapely.geometry import LineString
from shapely.geometry import MultiLineString
from shapely.geometry import MultiPolygon
from shapely.geometry import Polygon
import warnings
import time

try:
    import tkinter as tk
    from tkinter import filedialog
except:
    pass

from stops_locations import *
from fixed_lines import get_fixed_lines_deconet
from fixed_lines import get_fixed_lines_osm
from fixed_lines import plot_fixed_lines
from output_files import JsonConverter
from retrieve_bus_stations import filter_bus_stations
from retrieve_bus_stations import get_bus_stations_matrix_csv
from retrieve_bus_stations import plot_bus_stations
from retrieve_zones import retrieve_zones
from retrieve_schools import retrieve_schools
from compute_distance_matrix import _get_distance_matrix
from compute_distance_matrix import _update_distance_matrix_walk
from speed_info import _calc_mean_max_speed
from speed_info import _get_max_speed_road
from network_class import Network
from instance_class import Instance
from request_distribution_class import RequestDistributionTime
from trip_patterns_general import rank_model
       
def download_network_information(
    place_name,
    vehicle_speed_data="max", 
    vehicle_speed=None, 
    max_speed_factor=0.5, 
    get_fixed_lines=None,
    BBx=1000,
    BBy=1000,
    rows=10,
    columns=10,
    graph_from_point=False,
    lon=None,
    lat=None,
    dist=0
):

    warnings.filterwarnings(action="ignore")
    '''
    download and compute several information from "place_name"
    '''
    output_folder_base = place_name
    #directory of instance's saved information
    save_dir = os.getcwd()+'/'+output_folder_base
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    #retrieve network
    print(place_name)

    save_dir_json = os.path.join(save_dir, 'json_format')
    if not os.path.isdir(save_dir_json):
        os.mkdir(save_dir_json)

    save_dir_images = os.path.join(save_dir, 'images')
    if not os.path.isdir(save_dir_images):
        os.mkdir(save_dir_images)

    pickle_dir = os.path.join(save_dir, 'pickle')
    if not os.path.isdir(pickle_dir):
        os.mkdir(pickle_dir)

    graphml_dir = os.path.join(save_dir, 'graphml_format')
    if not os.path.isdir(graphml_dir):
        os.mkdir(graphml_dir)

    ttm_dir = os.path.join(save_dir, 'travel_time_matrix')
    if not os.path.isdir(ttm_dir):
        os.mkdir(ttm_dir)

    '''
    ‘drive’ – get drivable public streets (but not service roads)
    ‘drive_service’ – get drivable public streets, including service roads
    ‘walk’ – get all streets and paths that pedestrians can use (this network type ignores one-way directionality)
    ‘bike’ – get all streets and paths that cyclists can use
    ‘all’ – download all (non-private) OSM streets and paths
    ‘all_private’ – download all OSM streets and paths, including private-access ones
    '''


    if graph_from_point:

        center_point = (lat, lon)
        north, south, east, west = ox.utils_geo.bbox_from_point(center_point, dist)

        '''
        G = ox.graph_from_bbox(
            north,
            south,
            east,
            west,
            network_type='drive',
            retain_all=True
        )
        '''
        polygon = ox.utils_geo.bbox_to_poly(north, south, east, west)
    else:
        api_osm = osm.OsmApi()

        if isinstance(place_name, (str, dict)):
            # if it is a string (place name) or dict (structured place query), then
            # it is a single place
            gdf_place = ox.geocoder.geocode_to_gdf(
                place_name, which_result=None, buffer_dist=None
            )
        elif isinstance(place_name, list):
            # if it is a list, it contains multiple places to get
            gdf_place = ox.geocoder.geocode_to_gdf(place_name, buffer_dist=None)
        else:
            raise TypeError("query must be dict, string, or list of strings")

        # extract the geometry from the GeoDataFrame to use in API query
        polygon = gdf_place["geometry"].unary_union 

    
    G_walk = ox.graph_from_polygon(
        polygon,
        network_type='walk', 
        retain_all=True
    )

    G_drive = ox.graph_from_polygon(
        polygon,
        network_type='drive',
        retain_all=True
    )

    print('Now genarating network_data')
    
    print('num walk nodes', len(G_walk.nodes()))
    print('num drive nodes', len(G_drive.nodes()))

    G_walk = ox.utils_graph.get_largest_component(G_walk, strongly=True)
    G_drive = ox.utils_graph.get_largest_component(G_drive, strongly=True)

    print('scc num walk nodes', len(G_walk.nodes()))
    print('scc num drive nodes', len(G_drive.nodes()))
    
    if vehicle_speed_data != "max" and vehicle_speed_data != "set":
        avg_uber_speed_data, speed_mean_overall = get_uber_speed_data_mean(G_drive, vehicle_speed_data, day_of_the_week)
        avg_uber_speed_data = pd.DataFrame(avg_uber_speed_data)
        print(avg_uber_speed_data.head())
        print('speed mean overall', speed_mean_overall)
    
    print('Now retrieving bus stops')
    bus_stations = get_bus_stations_matrix_csv(G_walk, G_drive, place_name, save_dir, output_folder_base)
    print('number of bus stations: ', len(bus_stations))

    max_speed_mean_overall = 0
    counter_max_speeds = 0
    if vehicle_speed_data == "max":
        for (u,v,k) in G_drive.edges(data=True):    
            dict_edge = {}
            dict_edge = G_drive.get_edge_data(u, v)
            dict_edge = dict_edge[0]
            
            max_speed_mean_overall,  counter_max_speeds = _calc_mean_max_speed(dict_edge, max_speed_mean_overall, counter_max_speeds)

        max_speed_mean_overall = max_speed_mean_overall/counter_max_speeds

        #value to replace the missing max speed values
        #the mean of all input maximum speed values in the network
        replace_vehicle_speed = float(max_speed_mean_overall)

    #add attribute travel_time to edges
    for (u,v,k) in G_drive.edges(data=True): 

        if vehicle_speed_data == "max":
            dict_edge = {}
            dict_edge = G_drive.get_edge_data(u, v)
            dict_edge = dict_edge[0]
            max_speed = _get_max_speed_road(dict_edge)

            if not math.isnan(max_speed):   
                G_drive[u][v][0]['travel_time'] = int(math.ceil(G_drive[u][v][0]['length']/(max_speed * max_speed_factor)))
            else:
                #fill in missing speed with replace_vehicle_speed
                G_drive[u][v][0]['travel_time'] = int(math.ceil(G_drive[u][v][0]['length']/(replace_vehicle_speed * max_speed_factor)))
        
        elif vehicle_speed_data == "set":
            if not math.isnan(vehicle_speed):
                #print(G_drive[u][v][0]['length'])   
                #print(vehicle_speed)
                G_drive[u][v][0]['travel_time'] = int(math.ceil(G_drive[u][v][0]['length']/(vehicle_speed)))
            else:
                raise ValueError('please set attribute vehicle_speed')

        else: raise ValueError('attribute vehicle_speed_data must be either "set" or "max"')

    
    network = Network(place_name, G_drive, G_walk, polygon, bus_stations)
    network.vehicle_speed = vehicle_speed

    
    del G_drive
    del G_walk
    gc.collect()

    print('Downloading zones from location')
    
    #create graph to plot zones here
    zones = network.divide_network_grid(rows, columns, save_dir, output_folder_base)           
    print('number of zones', len(zones))

    print('Downloading schools from location')
    schools = retrieve_schools(network.G_walk, network.G_drive, place_name, save_dir, output_folder_base)
    schools = schools.reset_index(drop=True)
              
    print('number of schools', len(schools))

    network.zones = zones
    network.schools = schools
    
    print(dir())
    del zones 
    del schools
    gc.collect() 
    
    plot_bus_stations(network, save_dir_images)
    
    #computes distance matrix for drivig and walking network
    shortest_path_walk = []
    shortest_path_drive = []
    shortest_dist_drive = []
    unreachable_nodes = []
    
    shortest_path_walk, shortest_path_drive, shortest_dist_drive, unreachable_nodes = _get_distance_matrix(network.G_walk, network.G_drive, network.bus_stations, save_dir, output_folder_base)


    #getting fixed lines    
    if get_fixed_lines is not None:
        if get_fixed_lines == 'osm':
            pt_fixed_lines = get_fixed_lines_osm(network.G_walk, network.G_drive, polygon, save_dir, output_folder_base)
        elif get_fixed_lines == 'deconet':
                
                folder_path_deconet = output_folder_base+'/'+'deconet'

                if not os.path.isdir(folder_path_deconet):
                    os.mkdir(folder_path_deconet)

                print('getting fixed lines DECONET')
                get_fixed_lines_deconet(network, folder_path_deconet, save_dir, output_folder_base, place_name)
                extra_stops = network.deconet_network_nodes['osmid_walk'].tolist()
                print('extra stops')
                extra_stops_fr = []
                [extra_stops_fr.append(int(x)) for x in extra_stops if str(x) != 'nan'] 
                print(extra_stops_fr)
                shortest_path_walk = _update_distance_matrix_walk(G_walk, extra_stops_fr, save_dir, output_folder_base)
                        
        else: raise ValueError('get_fixed_lines method argument must be either "osm" or "deconet"')


        num_removed = filter_bus_stations(network, shortest_path_drive, save_dir, output_folder_base)

        network.bus_stations = network.bus_stations.reset_index(drop=True)

        for lp in range(len(network.linepieces)):
            for s in range(len(network.linepieces[lp])): 
                network.linepieces[lp][s] -= num_removed

        for s in range(len(network.connecting_nodes)):
            network.connecting_nodes[s] -= num_removed

        for s in range(len(network.transfer_nodes)):
            network.transfer_nodes[s] -= num_removed

        for lp in range(len(network.direct_lines)):
            for s in range(len(network.direct_lines[lp])): 
                network.direct_lines[lp][s] -= num_removed

        for node in network.nodes_covered_fixed_lines:

            network.deconet_network_nodes.loc[int(node), 'bindex'] -= num_removed

        #testing if all remain as one
        count = 0
        for node in network.nodes_covered_fixed_lines:

            bn = network.deconet_network_nodes.loc[int(node), 'bindex']

            osm_w1 = network.deconet_network_nodes.loc[int(node), 'osmid_walk']
            osm_d1 = network.deconet_network_nodes.loc[int(node), 'osmid_drive']

            osm_w2 = network.bus_stations.loc[int(bn), 'osmid_walk']
            osm_d2 = network.bus_stations.loc[int(bn), 'osmid_drive']

            if int(network.bus_stations.loc[int(bn), 'type']) == 1:
                if ((osm_w1 == osm_w2) and (osm_d1 == osm_d2)):
                    count += 1

        print(len(network.nodes_covered_fixed_lines))

    else:
        pass 
        
        num_removed = filter_bus_stations(network, shortest_path_drive, save_dir, output_folder_base)

        network.bus_stations = network.bus_stations.reset_index(drop=True)
    
    #remove these comments
    
    network.bus_stations_ids = []
    for index, stop_node in network.bus_stations.iterrows():
        if index not in network.bus_stations_ids:
            network.bus_stations_ids.append(index)
            
    network.num_stations = len(network.bus_stations)

    network.shortest_path_walk = shortest_path_walk
    network.shortest_path_drive = shortest_path_drive
    network.shortest_dist_drive = shortest_dist_drive

    #save Points of Interest information
    rank_model(network, place_name)
    print('zone ranks')
    print(network.zone_ranks)

    print('successfully retrieved network')

    network_class_file = pickle_dir+'/'+output_folder_base+'.network.class.pkl'
    
    output_network_class = open(network_class_file, 'wb')
    pickle.dump(network, output_network_class, pickle.HIGHEST_PROTOCOL)
    
    output_network_class.close()
    #caching.clear_cache()

    return network
