import math
import osmapi as osm
import os
import osmnx as ox
import networkx as nx
import numpy as np
import pickle
import random
from request_distribution_class import RequestDistributionTime
from spatial_distribution_class import SpatialDistribution
from shapely.geometry import Point
from passenger_requests import _generate_requests
import ray
import gc
from multiprocessing import cpu_count

 
class Instance:

    def __init__(self, folder_to_network):

        self.folder_to_network = folder_to_network
        self.output_folder_base = self.folder_to_network

        self.save_dir = os.getcwd()+'/'+self.output_folder_base
        self.pickle_dir = os.path.join(self.save_dir, 'pickle')
        self.save_dir_json = os.path.join(self.save_dir, 'json_format')
        self.save_dir_graphml = os.path.join(self.save_dir, 'graphml_format')
        self.save_dir_ttm = os.path.join(self.save_dir, 'travel_time_matrix')
        
        self.network_class_file = self.pickle_dir+'/'+self.output_folder_base+'.network.class.pkl'

        print(self.network_class_file)
        with open(self.network_class_file, 'rb') as self.network_class_file:
            self.network = pickle.load(self.network_class_file)

        self.network_class_file2 = self.pickle_dir+'/'+self.output_folder_base+'.network.class.pkl'

        with open(self.network_class_file2, 'rb') as self.network_class_file:
            network2 = pickle.load(self.network_class_file)

        self.network.G_walk = network2.G_walk
        self.network.shortest_path_walk = network2.shortest_path_walk
        
        del network2
        #del self.network.shortest_path_drive
        gc.collect()

        #problem for which the instance is being created
        self.problem_type = None

        self.request_demand = []

        self.spatial_distribution = []

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

        #num depots DARP
        self.num_depots = 1
        self.depot_nodes_drive = []
        self.depot_nodes_walk = []
        self.can_set_random_depot = True
        self.can_set_address_depot = True

        #school id in case of SBRP
        self.num_schools = 1
        self.school_ids = []
        self.can_set_random_school = True
        self.can_set_address_school = True

        self.seed = 0
        self.increment_seed = 1

        self.parameters = {}

        self.properties = ['dynamism', 'urgency', 'geographic_dispersion']

    def set_seed(self, seed, increment_seed):

        self.seed = seed
        self.increment_seed = increment_seed

    def set_problem_type(self, problem_type, school_id=None):

        if (problem_type == "ODBRP") or (problem_type == "ODBRPFL") or (problem_type == "SBRP") or (problem_type == "DARP"):
            self.problem_type = problem_type

        else: raise ValueError('problem_type method argument must be either "ODBRP", "ODBRPFL", "DARP" or "SBRP"') 

        '''
        if problem_type == "SBRP":
            self.school_id = school_id
            
            if school_id is None:
                raise ValueError('problem SBRP requires school as parameter. please provide school ID')
        '''

    def set_number_replicas(self, number_replicas):

        self.number_replicas = int(number_replicas)

    def generate_requests(self, base_save_folder_name, inst_directory):

        for replicate_num in range(self.number_replicas):
            
            self.seed += self.increment_seed

            _generate_requests(self, replicate_num, base_save_folder_name, inst_directory)
