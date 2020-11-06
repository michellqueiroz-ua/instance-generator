import math
import osmapi as osm
import os
import osmnx as ox
import networkx as nx
import numpy as np
import pickle
import random
from request_distribution_class import RequestDistributionTime
from shapely.geometry import Point
from passenger_requests import _generate_requests_DARP
from passenger_requests import _generate_requests_ODBRP
from passenger_requests import _generate_requests_ODBRPFL
from passenger_requests import _generate_requests_SBRP

 
class Instance:

    def __init__(self, folder_to_network):

        self.folder_to_network = folder_to_network
        self.output_folder_base = self.folder_to_network

        self.save_dir = os.getcwd()+'/'+self.output_folder_base
        self.pickle_dir = os.path.join(self.save_dir, 'pickle')
        self.save_dir_json = os.path.join(self.save_dir, 'json_format')
        
        self.network_class_file = self.pickle_dir+'/'+self.output_folder_base+'.network.class.pkl'

        with open(self.network_class_file, 'rb') as self.network_class_file:
            self.network = pickle.load(self.network_class_file)


        #problem for which the instance is being created
        self.problem_type = None

        self.request_demand = []

        self.num_origins = -1
        self.num_destinations = -1
        self.is_random_origin_zones = True
        self.is_random_destination_zones = True
        self.origin_zones = []
        self.destination_zones = []

        #time window
        self.min_early_departure = None
        self.max_early_departure = None

        #interval walk speed
        self.min_walk_speed = None
        self.max_walk_speed = None

        #interval of maximum walking threshold of the user
        self.lb_max_walking = None
        self.ub_max_walking = None

        #interval of lead time
        self.min_lead_time = None
        self.max_lead_time = None

        #number of replicas of the instance with randomized characteristics 
        self.number_replicas = None

        #school id in case of SBRP
        self.school_id = None
        self.school_station = None

        #factor to compute delay travel time by the vehicle
        self.delay_vehicle_factor = None

        #probability of each request having a return
        self.return_factor = None

    
    def add_request_demand_uniform(self, 
        min_time, 
        max_time, 
        number_of_requests, 
        time_unit
):

        '''
        add request demand that sample earliest departure time/latest arrival time using uniform distribution
        '''
        if time_unit == "s":
            min_time = int(min_time)
        
        elif time_unit == "h":
            min_time = int(min_time*3600)

        elif time_unit == "min":
            min_time = int(min_time*60)

        else: raise ValueError('time_unit method argument must be either "h" or "min", or "s"')

        if time_unit == "s":
            max_time = int(max_time)

        elif time_unit == "h":
            max_time = int(max_time*3600)

        elif time_unit == "min":
            max_time = int(max_time*60)

        else: raise ValueError('time_unit method argument must be either "h" or "min", or "s"')

        dnd = RequestDistributionTime(min_time, max_time, number_of_requests, "uniform")
        self.request_demand.append(dnd)

    def add_request_demand_normal(self, 
        mean, 
        std, 
        number_of_requests, 
        time_unit
):

        '''
        add request demand that sample earliest departure time/latest arrival time using normal distribution
        '''
        if time_unit == "s":
            mean = int(mean)
        
        elif time_unit == "h":
            mean = int(mean*3600)

        elif time_unit == "min":
            mean = int(mean*60)

        else: raise ValueError('time_unit method argument must be either "h" or "min", or "s"')

        if time_unit == "s":
            std = int(std)

        elif time_unit == "h":
            std = int(std*3600)

        elif time_unit == "min":
            std = int(std*60)

        else: raise ValueError('time_unit method argument must be either "h" or "min", or "s"')

        dnd = RequestDistributionTime(mean, std, number_of_requests, "normal")
        self.request_demand.append(dnd)

    def set_number_origins(self, num_origins):

        self.num_origins = int(num_origins)

    def set_number_destinations(self, num_destinations):

        self.num_destinations = int(num_destinations)

    def add_origin_zone(self, zone_id):

        self.is_random_origin_zones = False
        self.origin_zones.append(int(zone_id))

    def add_destination_zone(self, zone_id):

        self.is_random_destination_zones = False
        self.destination_zones.append(int(zone_id))
    
    def randomly_sample_origin_zones(self, num_zones):

        self.origin_zones = []

        if self.num_origins != -1:
            self.origin_zones = np.random.randint(0, num_zones, self.num_origins)

    def randomly_sample_destination_zones(self, num_zones):

        self.destination_zones = []

        if self.num_destinations != -1:
            self.destination_zones = np.random.randint(0, num_zones, self.num_destinations)

    def set_time_window(self, min_early_departure, max_early_departure, time_unit):

        if time_unit == "s":
            self.min_early_departure = int(min_early_departure)
            self.max_early_departure = int(max_early_departure)

        elif time_unit == "h":
            self.min_early_departure = int(min_early_departure*3600)
            self.max_early_departure = int(max_early_departure*3600)

        elif time_unit == "min":
            self.min_early_departure = int(min_early_departure*60)
            self.max_early_departure = int(max_early_departure*60)

        else: raise ValueError('time_unit method argument must be either "h" or "min", or "s"')

    def set_interval_lead_time(self, min_lead_time, max_lead_time, time_unit):
    
        if time_unit == "s":
            self.min_lead_time = int(min_lead_time)
            self.max_lead_time = int(max_lead_time)

        elif time_unit == "h":
            self.min_lead_time = int(min_lead_time*3600)
            self.max_lead_time = int(max_lead_time*3600)

        elif time_unit == "min":
            self.min_lead_time = int(min_lead_time*60)
            self.max_lead_time = int(max_lead_time*60)

        else: raise ValueError('time_unit method argument must be either "h" or "min", or "s"')

    def set_interval_walk_speed(self, min_walk_speed, max_walk_speed, speed_unit):

        '''
        set the walking speed considering during computation of travel times
        value is randomized for each user
        '''


        if speed_unit == "mps":
            self.min_walk_speed = float(min_walk_speed)
            
        elif speed_unit == "kmh":
            self.min_walk_speed = float(min_walk_speed/3.6)

        elif speed_unit == "mph":
            self.min_walk_speed = float(min_walk_speed/2.237)

        else: raise ValueError('speed_unit method argument must be either "kmh", "mph" or "mps"')

        if speed_unit == "mps":
            self.max_walk_speed = float(max_walk_speed)
            
        elif speed_unit == "kmh":
            self.max_walk_speed = float(max_walk_speed/3.6)

        elif speed_unit == "mph":
            self.max_walk_speed = float(max_walk_speed/2.237)

        else: raise ValueError('speed_unit method argument must be either "kmh", "mph" or "mps"')



    def set_interval_max_walking(self, lb_max_walking, ub_max_walking, time_unit):

        '''
        interval for max desired walk by the user
        lb - lower bound
        ub - upper bound
        '''

        if time_unit == "s":
            self.lb_max_walking = int(lb_max_walking)

        elif time_unit == "min":
            self.lb_max_walking = int(lb_max_walking*60)

        elif time_unit == "h":
            self.lb_max_walking = int(lb_max_walking*3600)

        else: raise ValueError('time_unit method argument must be either "h" or "min", or "s"') 

        if time_unit == "s":
            self.ub_max_walking = int(ub_max_walking)

        elif time_unit == "min":
            self.ub_max_walking = int(ub_max_walking*60)

        elif time_unit == "h":
            self.ub_max_walking = int(ub_max_walking*3600)

        else: raise ValueError('time_unit method argument must be either "h" or "min", or "s"') 

    def set_problem_type(self, problem_type, school_id=None):

        if (problem_type == "ODBRP") or (problem_type == "ODBRPFL") or (problem_type == "SBRP") or (problem_type == "DARP"):
            self.problem_type = problem_type

        else: raise ValueError('problem_type method argument must be either "ODBRP",  "ODBRPFL", "DARP" or "SBRP"') 

        if problem_type == "SBRP":
            self.school_id = school_id
            
            if school_id is None:
                raise ValueError('problem SBRP requires school as parameter. please provide school ID')

    def set_number_replicas(self, number_replicas):

        self.number_replicas = int(number_replicas)

    def set_delay_vehicle_factor(self, delay_vehicle_factor):

        self.delay_vehicle_factor = float(delay_vehicle_factor)

    def set_delay_walk_factor(self, delay_walk_factor):

        self.delay_walk_factor = float(delay_walk_factor)

    def set_return_factor(self, return_factor):

        self.return_factor = float(return_factor)


    def generate_requests(self):

        if self.problem_type == "DARP":
            for replicate_num in range(self.number_replicas):
                _generate_requests_DARP(self, replicate_num)  

        if self.problem_type == "ODBRP":
            for replicate_num in range(self.number_replicas):
                _generate_requests_ODBRP(self, replicate_num) 

        if self.problem_type == "ODBRPFL":
            for replicate_num in range(self.number_replicas):
                _generate_requests_ODBRPFL(self, replicate_num) 

        if self.problem_type == "SBRP":
            for replicate_num in range(self.number_replicas):
                _generate_requests_SBRP(self, replicate_num)

          
        



    '''
    sets the interval of walking time (in units of time), that the user is willing to walk to reach a pre defined location, such as bus stations
    value is randomized for each user
    '''




