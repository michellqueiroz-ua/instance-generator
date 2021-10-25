import math
import osmapi as osm
import osmnx as ox
import networkx as nx
import numpy as np
import random
from shapely.geometry import Point

class RequestDistributionTime:

    def __init__(self, x, y, num_requests, pdf):
        
        self.pdf = pdf

        if self.pdf == "normal":
            self.mean = x
            self.std = y

        if self.pdf == "uniform":
            self.min_time = x
            self.max_time = y
        
        self.num_requests = num_requests
        
        self.demand = []

    def sample_times(self):

        #sample time stamp according to normal or uniform distributions

        if self.pdf == "normal":
            self.demand = np.random.normal(self.mean, self.std, self.num_requests)

        if self.pdf == "uniform":
            self.demand = np.random.uniform(self.min_time, self.max_time, self.num_requests)