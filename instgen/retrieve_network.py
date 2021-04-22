import datetime
import matplotlib.pyplot as plt
import numpy as np
import codecs, json
import math
import geopandas as gpd
import glob
from math import sqrt
from multiprocessing import cpu_count
import os
import osmapi as osm
import osmnx as ox
import pandas as pd
import pickle
from random import randint
from random import seed
from random import choices
import ray
from shapely.geometry import Point
from shapely.geometry import MultiPoint
from shapely.geometry import LineString
from shapely.geometry import MultiLineString
from shapely.geometry import MultiPolygon
from shapely.geometry import Polygon
from streamlit import caching
import sys
import time
import warnings

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
from speed_info import _calc_mean_max_speed
from speed_info import _get_max_speed_road

from network_class import Network
from instance_class import Instance
from request_distribution_class import RequestDistributionTime
       
def network_stats(network):
    pass
    #print('used vehicle speed: ', network.vehicle_speed*3.6, ' kmh')
    #print("average dist 2 stops (driving network):", network.travel_time_matrix["dist"].mean())
    #print("average travel time between 2 stops:", network.travel_time_matrix["eta"].mean())

def download_network_information(
    place_name,
    vehicle_speed_data="max", 
    vehicle_speed=None, 
    max_speed_factor=0.5, 
    get_fixed_lines=None,
    BBx=1000,
    BBy=1000
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

    '''
    ‘drive’ – get drivable public streets (but not service roads)
    ‘drive_service’ – get drivable public streets, including service roads
    ‘walk’ – get all streets and paths that pedestrians can use (this network type ignores one-way directionality)
    ‘bike’ – get all streets and paths that cyclists can use
    ‘all’ – download all (non-private) OSM streets and paths
    ‘all_private’ – download all OSM streets and paths, including private-access ones
    '''
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
    #G_walk = ox.graph_from_place(place_name, network_type='walk', retain_all=True)
    #G_walk, polygon_walk = ox.graph_from_place(place_name, network_type='walk', retain_all=True, buffer_dist=3000)
    #fig, ax = ox.plot_graph(G_walk, save=True, file_format='svg', filename='walk_network')
    #G_drive = ox.graph_from_place(place_name, network_type='drive', retain_all=True)
    #G_drive, polygon_drive = ox.graph_from_place(place_name, network_type='drive', retain_all=True, buffer_dist=3000)
    #fig, ax = ox.plot_graph(G_drive, save=True, filename='cincinnati_drive')
    
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
        #replace_vehicle_speed is the mean of all input maximum speed values in the network
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
                G_drive[u][v][0]['travel_time'] = int(math.ceil(G_drive[u][v][0]['length']/(vehicle_speed)))
            else:
                raise ValueError('please set attribute vehicle_speed')

        else: raise ValueError('attribute vehicle_speed_data must be either "set" or "max"')

    
    print('Downloading zones from location')
    zones = retrieve_zones(G_walk, G_drive, place_name, save_dir, output_folder_base, BBx, BBy)
    #create graph to plot zones here           
    print('number of zones', len(zones))

    print('Downloading schools from location')
    schools = retrieve_schools(G_walk, G_drive, place_name, save_dir, output_folder_base)
    schools = schools.reset_index(drop=True)
    #create graph to plot zones here           
    print('number of schools', len(schools))


    network = Network(place_name, G_drive, G_walk, polygon, bus_stations, zones, schools)
    #network.update_travel_time_matrix(travel_time_matrix)
    
    plot_bus_stations(network, save_dir_images)
    network_stats(network)

    '''
    list_bus_stations = []
    for index, stop_node in network.bus_stations.iterrows():
        list_bus_stations.append(index)

    network.list_bus_stations = list_bus_stations
    '''

    #get fixed lines
    if get_fixed_lines is not None:
        if get_fixed_lines == 'osm':
            pt_fixed_lines = get_fixed_lines_osm(G_walk, G_drive, polygon, save_dir, output_folder_base)
        elif get_fixed_lines == 'deconet':
                
                folder_path_deconet = output_folder_base+'/'+'deconet'

                if not os.path.isdir(folder_path_deconet):
                    os.mkdir(folder_path_deconet)

                print('getting fixed lines DECONET')
                get_fixed_lines_deconet(network, folder_path_deconet, save_dir, output_folder_base, place_name)

                #if not os.path.isdir(folder_path_deconet):
                #    raise ValueError('DECONET data files do not exist. Make sure you passed the correct path to the folder')
                #else:         
        else: raise ValueError('get_fixed_lines method argument must be either "osm" or "deconet"')



    #computes distance matrix for drivig and walking network
    shortest_path_walk, shortest_path_drive, unreachable_nodes = _get_distance_matrix(G_walk, G_drive, network.bus_stations, save_dir, output_folder_base)

    #print(unreachable_nodes)
    #network.G_drive.remove_nodes_from(unreachable_nodes)
    
    #removes unreacheable stops
    #filter_bus_stations(network, shortest_path_drive, save_dir, output_folder_base)

    network.bus_stations = network.bus_stations.reset_index(drop=True)

    network.bus_stations_ids = []
    for index, stop_node in network.bus_stations.iterrows():
        if index not in network.bus_stations_ids:
            network.bus_stations_ids.append(index)
            
    network.num_stations = len(network.bus_stations)

    network.shortest_path_walk = shortest_path_walk
    network.shortest_path_drive = shortest_path_drive

    print('successfully retrieved network')

    network_class_file = pickle_dir+'/'+output_folder_base+'.network.class.pkl'
    
    output_network_class = open(network_class_file, 'wb')
    pickle.dump(network, output_network_class, pickle.HIGHEST_PROTOCOL)
    
    output_network_class.close()
    caching.clear_cache()

    return network
