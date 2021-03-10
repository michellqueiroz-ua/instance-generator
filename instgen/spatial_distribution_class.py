import math
import numpy as np
import random

class SpatialDistribution:

    def __init__(self, num_origins, num_destinations, prob, origin_zones=[], destination_zones=[], is_random_origin_zones=False, is_random_destination_zones=False):
        
        self.num_origins = int(num_origins)

        self.num_destinations = int(num_destinations)
        
        self.origin_zones = origin_zones
        
        self.destination_zones = destination_zones

        self.is_random_origin_zones = is_random_origin_zones

        self.is_random_destination_zones = is_random_destination_zones

        self.prob = prob
        

    def set_number_origins(self, num_origins):

        self.num_origins = int(num_origins)

    def set_number_destinations(self, num_destinations):

        self.num_destinations = int(num_destinations)

    '''
    def add_origin_zone(self, zone_id):

        self.is_random_origin_zones = False
        self.origin_zones.append(int(zone_id))

    def add_destination_zone(self, zone_id):

        self.is_random_destination_zones = False
        self.destination_zones.append(int(zone_id))
    '''

    def randomly_sample_origin_zones(self, num_zones):

        self.origin_zones = []

        if self.num_origins != -1:
            self.origin_zones = np.random.randint(0, num_zones, self.num_origins)

    def randomly_sample_destination_zones(self, num_zones):

        self.destination_zones = []

        if self.num_destinations != -1:
            self.destination_zones = np.random.randint(0, num_zones, self.num_destinations)

