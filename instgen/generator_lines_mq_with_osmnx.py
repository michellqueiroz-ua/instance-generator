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

from classes import *
from passenger_requests import generate_requests
from stops_locations import *
from fixed_lines import get_fixed_lines_deconet
from fixed_lines import get_fixed_lines_osm
from fixed_lines import plot_fixed_lines
from output_files import JsonConverter
from retrieve_bus_stops import filter_bus_stops
from retrieve_bus_stops import get_bus_stops_matrix_csv
from retrieve_bus_stops import plot_bus_stops
from retrieve_zones import get_zones_csv
from compute_distance_matrix import get_distance_matrix_csv
from speed_info import calc_mean_max_speed
from compute_travel_time import get_travel_time_matrix_osmnx_csv
       
def network_stats(network):

    print('used vehicle speed: ', network.vehicle_speed*3.6, ' kmh')
    print("average dist 2 stops (driving network):", network.travel_time_matrix["dist"].mean())
    print("average travel time between 2 stops:", network.travel_time_matrix["eta"].mean())

def retrieve_network(
    place_name, 
    output_folder_base,
    vehicle_speed_data="max", 
    vehicle_speed=None, 
    max_speed_factor=0.5,  
):
    #directory of instance's saved information
    save_dir = os.getcwd()+'/'+output_folder_base
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    #retrieve network
    print(place_name)

    #creating object that has the instance input information
    #param = Parameter(max_walking, min_early_departure, max_early_departure, [], day_of_the_week, num_replicates, bus_factor, get_fixed_lines, vehicle_speed_data, vehicle_speed, max_speed_factor, save_dir, output_folder_base, num_of_cpu=cpu_count())
    #param.average_waiting_time = average_waiting_time

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

    print('Now genarating network_data')
    G_walk = ox.graph_from_place(place_name, network_type='walk', retain_all=True)
    #G_walk, polygon_walk = ox.graph_from_place(place_name, network_type='walk', retain_all=True, buffer_dist=3000)
    #fig, ax = ox.plot_graph(G_walk, save=True, file_format='svg', filename='walk_network')
    G_drive = ox.graph_from_place(place_name, network_type='drive', retain_all=True)
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

    print('Getting zones')
    zones = get_zones_csv(G_walk, G_drive, place_name, save_dir, output_folder_base)
    #create graph to plot zones here           
    print('number of zones', len(zones))

    
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

    #print('used vehicle speed:' , vehicle_speed)

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

    travel_time_matrix = get_travel_time_matrix_osmnx_csv(bus_stops, shortest_path_drive, shortest_path_walk, save_dir, output_file_base)

    #param.update_network(G_drive, polygon_drive, shortest_path_drive, G_walk, polygon_walk, shortest_path_walk, bus_stops, zones, walk_speed)
    network = Network(G_drive, polygon_drive, shortest_path_drive, G_walk, polygon_walk, shortest_path_walk, bus_stops, zones, walk_speed, vehicle_speed)
    network.update_travel_time_matrix(travel_time_matrix)
    
    plot_bus_stops(network, save_dir_images)
    network_stats(network)

    list_bus_stops = []
    for index, stop_node in network.bus_stops.iterrows():
        list_bus_stops.append(index)

    network.list_bus_stops = list_bus_stops

    print('successfully retrieved network')
    #print("total time", time.process_time() - start)

    network_class_file = pickle_dir+'/'+output_folder_base+'.network.class.pkl'
    #parameter_class_file = pickle_dir+'/'+param.output_folder_base+'.parameter.class.pkl'
    
    output_network_class = open(network_class_file, 'wb')
    #output_parameter_class = open(parameter_class_file, 'wb')
    pickle.dump(network, output_network_class, pickle.HIGHEST_PROTOCOL)
    #pickle.dump(param, output_parameter_class, pickle.HIGHEST_PROTOCOL)
    
    output_network_class.close()
    #output_parameter_class.close()
    caching.clear_cache()

    return network

def add_fixed_line_data(
    get_fixed_lines=None 
):
    print('Trying to get fixed transport routes')
    if get_fixed_lines == 'osm':

        pt_fixed_lines = get_fixed_lines_csv(param, G_walk, G_drive, polygon_drive)
        print('number of routes', len(pt_fixed_lines))
    else:
        if get_fixed_lines == 'deconet':

            #this could be changed for a server or something else
            folder_path_deconet = output_folder_base+'/'+'deconet'
            if not os.path.isdir(folder_path_deconet):
                print('ERROR: deconet data files do not exist')
            else:
                get_fixed_lines_deconet(param, network, folder_path_deconet)

def instance_requests(
    output_folder_base,
    request_demand,
    num_replicates,
    min_early_departure,
    max_early_departure,
    min_walk_speed,
    max_walk_speed,
    max_walking,
    bus_factor,
    problem_type

):
    #start = time.process_time()

    save_dir = os.getcwd()+'/'+output_folder_base
    pickle_dir = os.path.join(save_dir, 'pickle')

    save_dir_json = os.path.join(save_dir, 'json_format')
    
    #param_class_file = pickle_dir+'/'+output_folder_base+'.parameter.class.pkl'
    network_class_file = pickle_dir+'/'+output_folder_base+'.network.class.pkl'
    
    #generate the instance's requests
    #with open(param_class_file, 'rb') as input_inst_class:
        
        #load class from binary file
        #param = pickle.load(input_inst_class)
        
    request_demand = request_demand
    num_replicates = num_replicates
    min_early_departure = min_early_departure
    max_early_departure = max_early_departure

    with open(network_class_file, 'rb') as network_class_file:
        network = pickle.load(network_class_file)

    if problem_type=="DARP":
        pass

    if problem_type=="ODBRP":
        for replicate_num in range(num_replicates):
            generate_requests_ODBRP(network, request_demand, min_early_departure, max_early_departure, max_walking, bus_factor, replicate_num, save_dir_json, output_folder_base)

    del param
    print("total time", time.process_time() - start)
    caching.clear_cache()
        
    #generate instances in json output folder
    #generate_instances_json(param)

    # convert instances from json to normal and localsolver format
    save_dir_cpp = os.path.join(save_dir, 'cpp_format')
    if not os.path.isdir(save_dir_cpp):
        os.mkdir(save_dir_cpp)

    save_dir_localsolver = os.path.join(save_dir, 'localsolver_format')
    if not os.path.isdir(save_dir_localsolver):
        os.mkdir(save_dir_localsolver)

    for instance in os.listdir(os.path.join(save_dir, 'json_format')):
        input_name = os.path.join(save_dir, 'json_format', instance)
        output_name_cpp = instance.split('.')[0] + '_cpp.pass'
        output_name_ls = instance.split('.')[0] + '_ls.pass'

        converter = JsonConverter(file_name=input_name)
        converter.convert_normal(output_file_name=os.path.join(save_dir_cpp, output_name_cpp), network=network)
        converter.convert_localsolver(output_file_name=os.path.join(save_dir_localsolver, output_name_ls))

def generate_instance():
    pass

if __name__ == '__main__':

    caching.clear_cache()
    request_demand = []
    vehicle_fleet = []

    #default for some parameters
    get_fixed_lines = None
    
    num_replicates = 1
    
    set_seed = 0
    
    vehicle_speed_data = "max"
    vehicle_speed = -1
    
    min_walk_speed = 4/3.6 #m/s
    max_walk_speed = 5/3.6 #m/s

    max_walking = 10*30 #seconds

    min_early_departure = 0
    max_early_departure = 24*3600

    day_of_the_week = 0 #monday

    max_speed_factor = 0.5

    is_network_generation = False
    is_request_generation = False

    network_class_file = None

    average_waiting_time = 120

    #num_of_cpu = 

    #INSTANCE PARAMETER INPUT INFORMATION
    for i in range(len(sys.argv)):
        if sys.argv[i] == "--base_file_name":
           output_folder_base = sys.argv[i+1].split('.')[0]

        if sys.argv[i] == "--is_request_generation":
            is_request_generation = True
            is_network_generation = False

        if sys.argv[i] == "--is_network_generation":
            is_network_generation = True
            is_request_generation = False

        #if sys.argv[i] == "--network_class_file":
        #    i += 1
        #    network_class_file = str(sys.argv[i])

        #if sys.argv[i] == "--param_class_file":
        #    i += 1
        #    param_class_file = str(sys.argv[i])
            
        if sys.argv[i] == "--place_name":
           place_name = sys.argv[i+1]

        if sys.argv[i] == "--vehicle_speed_data":
            i += 1
            vehicle_speed_data = str(sys.argv[i])

            if vehicle_speed_data == "set":
                i += 1
                vehicle_speed = float(sys.argv[i])

                i += 1
                if sys.argv[i] == "kmh":
                    vehicle_speed = vehicle_speed/3.6

                if sys.argv[i] == "mph":
                    vehicle_speed = vehicle_speed/2.237

            if vehicle_speed_data == "max":
                i += 1
                max_speed_factor = float(sys.argv[i])

        if sys.argv[i] == "--walking_threshold":
            max_walking = int(sys.argv[i+1])

            if sys.argv[i+2] == "min":
                max_walking = max_walking*60

            if sys.argv[i+2] == "h":
                max_walking = max_walking*3600

        if sys.argv[i] == "--get_fixed_lines":
            #get_fixed_lines = True
            
            if sys.argv[i+1] == "osm":
                get_fixed_lines = "osm"

            if sys.argv[i+1] == "deconet":
                get_fixed_lines = "deconet"


        if sys.argv[i] == "--time_window":
            
            i += 1
            min_early_departure = int(sys.argv[i])

            i += 1
            max_early_departure = int(sys.argv[i])

            i += 1
            if sys.argv[i] == "h":
                min_early_departure = min_early_departure*3600
                max_early_departure = max_early_departure*3600

            if sys.argv[i] == "min":
                min_early_departure = min_early_departure*60
                max_early_departure = max_early_departure*60

        if sys.argv[i] == "--walk_speed":

            i += 1
            walk_speed = float(sys.argv[i])
            
            i += 1
            if sys.argv[i] == "kmh":
                walk_speed = walk_speed/3.6

            if sys.argv[i] == "mph":
                walk_speed = walk_speed/2.237

        if sys.argv[i] == "--add_fleet":
            num_vehicles = int(sys.argv[i+1])
            capacity_vehicles = int(sys.argv[i+2])  

            vf = VehicleFleet(num_vehicles, capacity_vehicles)
            vehicle_fleet.append(vf)
        
        if sys.argv[i] == "--seed":
            set_seed = int(sys.argv[i+1])

        if sys.argv[i] == "--num_replicates":
            num_replicates = int(sys.argv[i+1])

        if sys.argv[i] == "--day_of_the_week":
            day_of_the_week = sys.argv[i+1]

            if day_of_the_week == "monday":
                day_of_the_week = 0

            if day_of_the_week == "tuesday":
                day_of_the_week = 1

            if day_of_the_week == "wednesday":
                day_of_the_week = 2

            if day_of_the_week == "thrusday":
                day_of_the_week = 3

            if day_of_the_week == "friday":
                day_of_the_week = 4

            if day_of_the_week == "saturday":
                day_of_the_week = 5

            if day_of_the_week == "sunday":
                day_of_the_week = 6

            day_of_the_week = int(day_of_the_week)

        #request demand that comes from anywhere in the city
        #origin and destination are random

        if sys.argv[i] == "--request_demand":
            
            i += 1
            pdf = sys.argv[i]
            
            if pdf == "normal":
                
                i += 1
                mean = sys.argv[i]
                mean = float(mean)
                
                i += 1
                if sys.argv[i] == "h":
                    mean = mean*3600

                if sys.argv[i] == "min":
                    mean = mean*60

                i += 1
                std = sys.argv[i]
                std = float(std)

                i += 1
                if sys.argv[i] == "h":
                    std = std*3600

                if sys.argv[i] == "min":
                    std = std*60

                i += 1
                num_req = int(sys.argv[i])

                
                origin_zones = []
                destination_zones = []
                i += 1
                if sys.argv[i] == "--origin":

                    i += 1
                    if sys.argv[i] == "random":
                
                        is_random_origin_zones = True
                        
                        i += 1
                        num_origins = int(sys.argv[i])
                        
                    else:
                        if sys.argv[i] == "set":
                
                            is_random_origin_zones = False
                            
                            i += 1
                            num_origins = int(sys.argv[i])
                            for k in range(num_origins):
                                i += 1
                                origin_zones.append(int(sys.argv[i]))
                i += 1           
                if sys.argv[i] == "--destination":

                    i += 1
                    if sys.argv[i] == "random":
                
                        is_random_destination_zones = True
                                        
                        i += 1
                        num_destinations = int(sys.argv[i])

                        i += 1
                        time_type = sys.argv[i]

                    else:
                        if sys.argv[i] == "set":
                
                            is_random_destination_zones = False
                            
                            i += 1
                            num_destinations = int(sys.argv[i])
                            for k in range(num_destinations):
                                i += 1
                                destination_zones.append(int(sys.argv[i]))

                            i += 1
                            time_type = sys.argv[i]

                dnd = RequestDistribution(mean, std, num_req, pdf, num_origins, num_destinations, time_type, is_random_origin_zones, is_random_destination_zones, origin_zones, destination_zones)
                request_demand.append(dnd)

            if pdf == "uniform":
                
                i += 1
                min_time = sys.argv[i]
                min_time = float(min_time)
                
                i += 1
                if sys.argv[i] == "h":
                    min_time = min_time*3600

                if sys.argv[i] == "min":
                    min_time = min_time*60

                i += 1
                max_time = sys.argv[i]
                max_time = float(max_time)

                i += 1
                if sys.argv[i] == "h":
                    max_time = max_time*3600

                if sys.argv[i] == "min":
                    max_time = max_time*60

                i += 1
                num_req = int(sys.argv[i])

                
                origin_zones = []
                destination_zones = []
                i += 1
                if sys.argv[i] == "--origin":

                    i += 1
                    if sys.argv[i] == "random":
                
                        is_random_origin_zones = True
                        
                        i += 1
                        num_origins = int(sys.argv[i])
                        
                    else:
                        if sys.argv[i] == "set":
                
                            is_random_origin_zones = False
                            
                            i += 1
                            num_origins = int(sys.argv[i])
                            for k in range(num_origins):
                                i += 1
                                origin_zones.append(int(sys.argv[i]))
                i += 1           
                if sys.argv[i] == "--destination":

                    i += 1
                    if sys.argv[i] == "random":
                
                        is_random_destination_zones = True
                                        
                        i += 1
                        num_destinations = int(sys.argv[i])

                        i += 1
                        time_type = sys.argv[i]

                    else:
                        if sys.argv[i] == "set":
                
                            is_random_destination_zones = False
                            
                            i += 1
                            num_destinations = int(sys.argv[i])
                            for k in range(num_destinations):
                                i += 1
                                destination_zones.append(int(sys.argv[i]))

                            i += 1
                            time_type = sys.argv[i]

                dnd = RequestDistribution(min_time, max_time, num_req, pdf, num_origins, num_destinations, time_type, is_random_origin_zones, is_random_destination_zones, origin_zones, destination_zones)
                request_demand.append(dnd)   
        #if sys.argv[i] == "--max_speed_factor":
        #    max_speed_factor = float(sys.argv[i+1])

    bus_factor = 2
    seed(set_seed)
    np.random.seed(set_seed)

    if is_network_generation:

        #retrieve the instance's network
        network = retrieve_network(place_name, output_folder_base)
        
    problem_type = "ODBRP"
    #problem_type = "DARP"
    
    if is_request_generation:
        instance_requests(output_folder_base, request_demand, num_replicates, min_early_departure, max_early_departure, min_walk_speed, max_walk_speed, max_walking, bus_factor, problem_type)
        
        #print('placement of stops - testing')
        #cluster_travel_demand(param, network)

