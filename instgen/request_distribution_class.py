import math
import osmapi as osm
import osmnx as ox
import networkx as nx
import numpy as np
import random
from shapely.geometry import Point

class RequestDistribution:

    def __init__(self, x, y, num_requests, pdf, num_origins, num_destinations, time_type, is_random_origin_zones, is_random_destination_zones, origin_zones=[], destination_zones=[]):
        
        self.pdf = pdf

        if self.pdf == "normal":
            self.mean = x
            self.std = y

        if self.pdf == "uniform":
            self.min_time = x
            self.max_time = y
        
        self.num_requests = num_requests
        
        self.demand = []
        
        self.num_origins = num_origins
        self.num_destinations = num_destinations
        
        self.time_type = time_type

        self.is_random_origin_zones = is_random_origin_zones
        self.is_random_destination_zones = is_random_destination_zones

        self.origin_zones = origin_zones
        self.destination_zones = destination_zones


    def set_demand(self):

        if self.pdf == "normal":
            self.demand = np.random.normal(self.mean, self.std, self.num_requests)

        if self.pdf == "uniform":
            self.demand = np.random.uniform(self.min_time, self.max_time, self.num_requests)

    def randomly_set_origin_zones(self, num_zones):

        self.origin_zones = []

        if self.num_origins != -1:
            self.origin_zones = np.random.randint(0, num_zones, self.num_origins)

    def randomly_set_destination_zones(self, num_zones):

        self.destination_zones = []

        if self.num_destinations != -1:
            self.destination_zones = np.random.randint(0, num_zones, self.num_destinations)
