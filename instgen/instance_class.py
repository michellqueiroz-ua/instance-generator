import math
import osmapi as osm
import osmnx as ox
import networkx as nx
import numpy as np
import random
from shapely.geometry import Point

            
class Instance:

    def __init__(self, place_name):

        self.place_name = place_name
        self.output_folder_base = output_folder_base

         save_dir = os.getcwd()+'/'+output_folder_base
        pickle_dir = os.path.join(save_dir, 'pickle')

        save_dir_json = os.path.join(save_dir, 'json_format')
        
        network_class_file = pickle_dir+'/'+output_folder_base+'.network.class.pkl'

        with open(network_class_file, 'rb') as network_class_file:
            self.network = pickle.load(network_class_file)

        self.request_demand = []

        self.min_early_departure = None
        self.max_early_departure = None
    
    def add_request_demand_uniform(self, 
        min_time, 
        max_time, 
        number_of_requests, 
        time_unit,
        umber_origin_zones, 
        number_destination_zones, 
        origin_zones_list, 
        destination_zones_list, 
        which_sampled_time
):

        '''
        add request demand that sample earliest departure time/latest arrival time using uniform distribution
        '''
        min_time = float(min_time)
        
        if time_unit == "h":
            min_time = min_time*3600

        if time_unit == "min":
            min_time = min_time*60

        max_time = float(max_time)

        if time_unit == "h":
            max_time = max_time*3600

        if time_unit == "min":
            max_time = max_time*60

        dnd = RequestDistribution(min_time, max_time, number_of_requests, "uniform", num_origins, num_destinations, which_sampled_time, is_random_origin_zones, is_random_destination_zones, origin_zones_list, destination_zones_list)
        self.request_demand.append(dnd)

    def add_request_demand_normal(self, 
        mean, 
        std, 
        number_of_requests, 
        time_unit,
        number_origin_zones, 
        number_destination_zones, 
        origin_zones_list, 
        destination_zones_list, 
        which_sampled_time
):

        '''
        add request demand that sample earliest departure time/latest arrival time using normal distribution
        '''
        
        mean = float(mean)
        
        if time_unit == "h":
            mean = mean*3600

        if time_unit == "min":
            mean = mean*60

        std = float(std)

        if time_unit == "h":
            std = std*3600

        if time_unit == "min":
            std = std*60

        dnd = RequestDistribution(mean, std, number_of_requests, "normal", num_origins, num_destinations, which_sampled_time, is_random_origin_zones, is_random_destination_zones, origin_zones_list, destination_zones_list)
        self.request_demand.append(dnd)

    def set_time_window(min_early_departure, max_early_departure, time_unit):

        if time_unit == "h":
            self.min_early_departure = min_early_departure*3600
            self.max_early_departure = max_early_departure*3600

        if time_unit == "min":
            self.min_early_departure = min_early_departure*60
            self.max_early_departure = max_early_departure*60
