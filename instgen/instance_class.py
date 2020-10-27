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

    def __init__(self, place_name):

        self.place_name = place_name
        self.output_folder_base = self.place_name

        self.save_dir = os.getcwd()+'/'+self.output_folder_base
        self.pickle_dir = os.path.join(self.save_dir, 'pickle')
        self.save_dir_json = os.path.join(self.save_dir, 'json_format')
        
        self.network_class_file = self.pickle_dir+'/'+self.output_folder_base+'.network.class.pkl'

        with open(self.network_class_file, 'rb') as self.network_class_file:
            self.network = pickle.load(self.network_class_file)


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

        #range walk speed
        self.min_walk_speed = None
        self.max_walk_speed = None

        #range of maximum walking threshold of the user
        self.lb_max_walking = None
        self.ub_max_walking = None

        #range of lead time
        self.min_lead_time = None
        self.max_lead_time = None

        #number of replicas of the instance with randomized characteristics 
        self.number_replicas = None

        #problem for which the instance is being created
        self.problem_type = None

        #school id in case of SBRP
        self.school_id = None
        self.school_station = None

        #factor to compute delay travel time by the vehicle
        self.delay_vehicle_factor = None

        #probability of each request having a return
        self.inbound_outbound_factor = None

    
    def add_request_demand_uniform(self, 
        min_time, 
        max_time, 
        number_of_requests, 
        time_unit
):

        '''
        add request demand that sample earliest departure time/latest arrival time using uniform distribution
        '''
        min_time = float(min_time)
        
        if time_unit == "h":
            min_time = min_time*3600

        elif time_unit == "min":
            min_time = min_time*60

        else: raise ValueError('time_unit method argument must be either "h" or "min", or "s"')

        max_time = float(max_time)

        if time_unit == "h":
            max_time = max_time*3600

        elif time_unit == "min":
            max_time = max_time*60

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
        
        mean = float(mean)
        
        if time_unit == "h":
            mean = mean*3600

        elif time_unit == "min":
            mean = mean*60

        else: raise ValueError('time_unit method argument must be either "h" or "min", or "s"')

        std = float(std)

        if time_unit == "h":
            std = std*3600

        elif time_unit == "min":
            std = std*60

        else: raise ValueError('time_unit method argument must be either "h" or "min", or "s"')

        dnd = RequestDistributionTime(mean, std, number_of_requests, "normal")
        self.request_demand.append(dnd)

    def set_number_origins(self, num_origins):

        self.num_origins = num_origins

    def set_number_destinations(self, num_destinations):

        self.num_destinations = num_destinations

    def add_origin_zone(self, zone_id):

        self.is_random_origin_zones = False
        self.origin_zones.append(zone_id)

    def add_destination_zone(self, zone_id):

        self.is_random_destination_zones = False
        self.destination_zones.append(zone_id)
    
    def randomly_sample_origin_zones(self, num_zones):

        self.origin_zones = []

        if self.num_origins != -1:
            self.origin_zones = np.random.randint(0, num_zones, self.num_origins)

    def randomly_sample_destination_zones(self, num_zones):

        self.destination_zones = []

        if self.num_destinations != -1:
            self.destination_zones = np.random.randint(0, num_zones, self.num_destinations)

    def set_time_window(self, min_early_departure, max_early_departure, time_unit):

        self.min_early_departure = min_early_departure
        self.max_early_departure = max_early_departure

        if time_unit == "h":
            self.min_early_departure = min_early_departure*3600
            self.max_early_departure = max_early_departure*3600

        elif time_unit == "min":
            self.min_early_departure = min_early_departure*60
            self.max_early_departure = max_early_departure*60

        else: raise ValueError('time_unit method argument must be either "h" or "min", or "s"')

    def set_lead_time(self, min_lead_time, max_lead_time, time_unit):
    
        self.min_lead_time = min_lead_time
        self.max_lead_time = max_lead_time

        if time_unit == "h":
            self.min_lead_time = min_lead_time*3600
            self.max_lead_time = max_lead_time*3600

        elif time_unit == "min":
            self.min_lead_time = min_lead_time*60
            self.max_lead_time = max_lead_time*60

        else: raise ValueError('time_unit method argument must be either "h" or "min", or "s"')

    def set_range_walk_speed(self, min_walk_speed, max_walk_speed, speed_unit):

        '''
        set the walking speed considering during computation of travel times
        value is randomized for each user
        '''

        self.min_walk_speed = float(min_walk_speed)
            
        if speed_unit == "kmh":
            self.min_walk_speed = min_walk_speed/3.6

        if speed_unit == "mph":
            self.min_walk_speed = min_walk_speed/2.237

        self.max_walk_speed = float(max_walk_speed)
            
        if speed_unit == "kmh":
            self.max_walk_speed = max_walk_speed/3.6

        if speed_unit == "mph":
            self.max_walk_speed = max_walk_speed/2.237

    def set_range_max_walking(self, lb_max_walking, ub_max_walking, time_unit):

        '''
        range for max desired walk by the user
        lb - lower bound
        ub - upper bound
        '''

        self.lb_max_walking = int(lb_max_walking)

        if time_unit == "min":
            self.lb_max_walking = lb_max_walking*60

        if time_unit == "h":
            self.lb_max_walking = lb_max_walking*3600

        self.ub_max_walking = int(ub_max_walking)

        if time_unit == "min":
            self.ub_max_walking = ub_max_walking*60

        if time_unit == "h":
            self.ub_max_walking = ub_max_walking*3600

    def set_problem_type(self, problem_type, school_id=None):

        self.problem_type = problem_type

        if problem_type == "SBRP":
            self.school_id = school_id
            
            if school_id is None:
                raise ValueError('problem SBRP requires school as parameter. please provide school ID')

    def set_number_replicas(self, number_replicas):

        self.number_replicas = number_replicas

    def set_delay_vehicle_factor(self, delay_vehicle_factor):

        self.delay_vehicle_factor = delay_vehicle_factor

    def set_inbound_outbound_factor(self, inbound_outbound_factor):

        self.inbound_outbound_factor = inbound_outbound_factor


    def generate_requests(self):

        if self.problem_type == "DARP":
            for replicate_num in range(self.number_replicas):
                _generate_requests_DARP(self, replicate_num)  

        if self.problem_type == "ODBRP":
            for replicate_num in range(self.number_replicas):
                _generate_requests_ODBRP(self, replicate_num)  

        if self.problem_type == "SBRP":
            for replicate_num in range(self.number_replicas):
                _generate_requests_SBRP(self, replicate_num)

          
        



    '''
    sets the range of walking time (in units of time), that the user is willing to walk to reach a pre defined location, such as bus stations
    value is randomized for each user
    '''




