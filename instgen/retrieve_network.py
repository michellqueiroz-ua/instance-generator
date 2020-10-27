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
from retrieve_bus_stops import filter_bus_stops
from retrieve_bus_stops import get_bus_stops_matrix_csv
from retrieve_bus_stops import plot_bus_stops
from retrieve_zones import retrieve_zones
from retrieve_schools import retrieve_schools
from compute_distance_matrix import get_distance_matrix_csv
from speed_info import calc_mean_max_speed
from compute_travel_time import get_travel_time_matrix_osmnx_csv

from network_class import Network
from instance_class import Instance
from request_distribution_class import RequestDistributionTime
       
def network_stats(network):

    print('used vehicle speed: ', network.vehicle_speed*3.6, ' kmh')
    print("average dist 2 stops (driving network):", network.travel_time_matrix["dist"].mean())
    print("average travel time between 2 stops:", network.travel_time_matrix["eta"].mean())

def download_network_information(
    place_name,
    vehicle_speed_data="max", 
    vehicle_speed=None, 
    max_speed_factor=0.5, 
    get_fixed_lines=None
):
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
    
    if vehicle_speed_data != "max" and vehicle_speed_data != "set":
        avg_uber_speed_data, speed_mean_overall = get_uber_speed_data_mean(G_drive, vehicle_speed_data, day_of_the_week)
        avg_uber_speed_data = pd.DataFrame(avg_uber_speed_data)
        print(avg_uber_speed_data.head())
        print('speed mean overall', speed_mean_overall)
    
    print('Now retrieving bus stops')
    bus_stops = get_bus_stops_matrix_csv(G_walk, G_drive, place_name, save_dir, output_folder_base)
    
    #computes distance matrix for drivig and walking network
    shortest_path_walk, shortest_path_drive = get_distance_matrix_csv(G_walk, G_drive, bus_stops, save_dir, output_folder_base)

    #removes unreacheable stops
    filter_bus_stops(bus_stops, shortest_path_drive, save_dir, output_folder_base)

    
    

    print('Downloading zones from location')
    zones = retrieve_zones(G_walk, G_drive, place_name, save_dir, output_folder_base)
    #create graph to plot zones here           
    print('number of zones', len(zones))

    print('Downloading schools from location')
    schools = retrieve_schools(G_walk, G_drive, place_name, save_dir, output_folder_base)
    #create graph to plot zones here           
    print('number of schools', len(schools))

    print('Now genarating time travel data')
    #this considers max speed
    max_speed_mean_overall = 0
    counter_max_speeds = 0
    
    for (u,v,k) in G_drive.edges(data=True):    
        dict_edge = {}
        dict_edge = G_drive.get_edge_data(u, v)
        dict_edge = dict_edge[0]
        max_speed_mean_overall,  counter_max_speeds = calc_mean_max_speed(dict_edge, max_speed_mean_overall, counter_max_speeds)

    max_speed_mean_overall = max_speed_mean_overall/counter_max_speeds

    if vehicle_speed_data == "max":
        vehicle_speed = float(max_speed_mean_overall*max_speed_factor)

    #COME BACK TO THIS LATER - TIME DEPENDENT TIME TRAVEL
    #colocar range ser o time window do request generation?
    #for hour in range(24):

    '''
    for hour in range(1):
        for (u,v,k) in G_drive.edges(data=True):
            hour_key = 'travel_time_' + str(hour)

            #0 after [u][v] is necessary to access the edge data
            edge_length = G_drive[u][v][0]['length']
            
            if speed_data != "max":
                try:
                    edge_speed = avg_uber_speed_data.loc[(u,v,hour), 'speed_mph_mean'] 
                except KeyError:
                    edge_speed = speed_mean_overall

                #convert to m/s
                #speeds in the uber database are in mph
                edge_speed = edge_speed/2.237
            else:
                dict_edge = {}
                dict_edge = G_drive.get_edge_data(u, v)
                dict_edge = dict_edge[0]
                
                edge_speed = get_max_speed_road(dict_edge)
                
                if math.isnan(edge_speed):
                    edge_speed = max_speed_mean_overall
                
                #max_speed_factor - value between 0 and 1
                edge_speed = edge_speed*max_speed_factor

            #calculates the eta travel time for the given edge at 'hour'
            eta =  int(math.ceil(edge_length/edge_speed))

            G_drive[u][v][0][hour_key] = eta
    '''

    #itid = 0
    #updates the 'itid in bus_stops'
    #for index, stop in bus_stops.iterrows():
    #    bus_stops.loc[index, 'itid'] = int(itid)
    #    itid = itid + 1

    travel_time_matrix = get_travel_time_matrix_osmnx_csv(vehicle_speed, bus_stops, shortest_path_drive, shortest_path_walk, save_dir, output_folder_base)

    #param.update_network(G_drive, polygon_drive, shortest_path_drive, G_walk, polygon_walk, shortest_path_walk, bus_stops, zones, walk_speed)
    network = Network(G_drive, shortest_path_drive, G_walk, shortest_path_walk, polygon, bus_stops, zones, schools, vehicle_speed)
    network.update_travel_time_matrix(travel_time_matrix)
    
    plot_bus_stops(network, save_dir_images)
    network_stats(network)

    list_bus_stops = []
    for index, stop_node in network.bus_stops.iterrows():
        list_bus_stops.append(index)

    network.list_bus_stops = list_bus_stops

    #get fixed lines
    if get_fixed_lines == 'osm':
        pt_fixed_lines = get_fixed_lines_osm(param, G_walk, G_drive, polygon)
    else:
        if get_fixed_lines == 'deconet':
            #this could be changed for a server or something else
            folder_path_deconet = output_folder_base+'/'+'deconet'
            if not os.path.isdir(folder_path_deconet):
                raise ValueError('DECONET data files do not exist. Make sure you passed the correct path to the folder')
            else:
                get_fixed_lines_deconet(network, folder_path_deconet, save_dir, output_folder_base)

    print('successfully retrieved network')

    network_class_file = pickle_dir+'/'+output_folder_base+'.network.class.pkl'
    
    output_network_class = open(network_class_file, 'wb')
    pickle.dump(network, output_network_class, pickle.HIGHEST_PROTOCOL)
    
    output_network_class.close()
    caching.clear_cache()

    return network


if __name__ == '__main__':

    caching.clear_cache()
    
    is_network_generation = False
    is_request_generation = False

    for i in range(len(sys.argv)):
        
        if sys.argv[i] == "--is_request_generation":
            is_request_generation = True
            is_network_generation = False

        if sys.argv[i] == "--is_network_generation":
            is_network_generation = True
            is_request_generation = False
    
    set_seed = 0
    seed(set_seed)
    np.random.seed(set_seed)

    if is_network_generation:
        #retrieve the instance's network
        network = download_network_information(place_name='Rennes, France')
        
    if is_request_generation:
        
        inst1 = Instance(place_name='Rennes, France')
        inst1.set_problem_type(problem_type="SBRP", school_id=0)
        inst1.set_number_replicas(number_replicas=1)
        inst1.set_time_window(min_early_departure=7, max_early_departure=11, time_unit="h")
        inst1.set_lead_time(min_lead_time=0, max_lead_time=5, time_unit="min")
        #inst1.add_request_demand_uniform(max_time=8, min_time=10, number_of_requests=100, time_unit="h")
        inst1.add_request_demand_normal(mean=8, std=0.5, number_of_requests=100, time_unit="h")
        inst1.set_range_walk_speed(min_walk_speed=4, max_walk_speed=5, speed_unit="kmh")
        inst1.set_range_max_walking(lb_max_walking=300, ub_max_walking=600, time_unit="s")
        inst1.set_delay_vehicle_factor(delay_vehicle_factor=2)
        inst1.set_inbound_outbound_factor(inbound_outbound_factor=0.5)
        inst1.generate_requests()

        caching.clear_cache()
        
        # convert instances from json to cpp and localsolver formats
        save_dir_cpp = os.path.join(inst1.save_dir, 'cpp_format')
        if not os.path.isdir(save_dir_cpp):
            os.mkdir(save_dir_cpp)

        save_dir_localsolver = os.path.join(inst1.save_dir, 'localsolver_format')
        if not os.path.isdir(save_dir_localsolver):
            os.mkdir(save_dir_localsolver)

        for instance in os.listdir(os.path.join(inst1.save_dir, 'json_format')):
            
            if instance != ".DS_Store":
                input_name = os.path.join(inst1.save_dir, 'json_format', instance)
                output_name_cpp = instance.split('.')[0] + '_cpp.pass'
                output_name_ls = instance.split('.')[0] + '_ls.pass'

                converter = JsonConverter(file_name=input_name)
                converter.convert_normal(output_file_name=os.path.join(save_dir_cpp, output_name_cpp), network=inst1.network)
                converter.convert_localsolver(output_file_name=os.path.join(save_dir_localsolver, output_name_ls))

        #print('placement of stops - testing')
        #cluster_travel_demand(param, network)

