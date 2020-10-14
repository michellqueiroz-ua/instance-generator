import math
import networkx as nx
import numpy as np
from shapely.geometry import Point


class VehicleFleet:

    def __init__(self, num_vehicles, capacity_vehicles):

        self.num_vehicles = num_vehicles
        self.capacity_vehicles = capacity_vehicles

class RequestDistribution:

    def __init__(self, x, y, num_requests, pdf, num_origins, num_destinations, time_type, is_random_origin_zones, is_random_destination_zones, origin_zones=[], destination_zones=[]):
        
        self.pdf = pdf

        if self.pdf == "normal":
            self.mean = x
            self.std = y

        if self.pdf == "uniform":
            self.min_time = x
            self.max_time = y

        if self.pdf == "poisson":
            self.average_min = x
        
        self.num_requests = num_requests
        
        self.demand = []
        
        self.num_origins = num_origins
        self.num_destinations = num_destinations
        
        #self.origin_zones = []
        #self.destination_zones = []

        self.time_type = time_type

        self.is_random_origin_zones = is_random_origin_zones
        self.is_random_destination_zones = is_random_destination_zones

        self.origin_zones = origin_zones
        self.destination_zones = destination_zones

        print('origin_zones', origin_zones)
        print('destination_zones', destination_zones)

    def set_demand(self):

        if self.pdf == "normal":
            self.demand = np.random.normal(self.mean, self.std, self.num_requests)

        if self.pdf == "uniform":
            self.demand = np.random.uniform(self.min_time, self.max_time, self.num_requests)

        if self.pdf == "poisson":
            #for poisson distribution the number of requests is related to the size of the time window
            self.demand = np.random.poisson(self.average_min, self.num_requests)

    def randomly_set_origin_zones(self, num_zones):

        self.origin_zones = []
        #self.destination_zones = []

        if self.num_origins != -1:
            self.origin_zones = np.random.randint(0, num_zones, self.num_origins)

        #if self.num_destinations != -1:
        #    self.destination_zones = np.random.randint(0, num_zones, self.num_destinations)

    def randomly_set_destination_zones(self, num_zones):

        #self.origin_zones = []
        self.destination_zones = []

        #if self.num_origins != -1:
        #    self.origin_zones = np.random.randint(0, num_zones, self.num_origins)

        if self.num_destinations != -1:
            self.destination_zones = np.random.randint(0, num_zones, self.num_destinations)
            
class Network:

    def __init__(self, G_drive, polygon_drive, shortest_path_drive, G_walk, polygon_walk, shortest_path_walk, bus_stops, zones, walk_speed):
        
        #network graphs
        self.G_drive = G_drive
        self.polygon_drive = polygon_drive
        self.G_walk = G_walk
        self.polygon_walk = polygon_walk
        
        #indicates which nodes in the network are specifically bus stops
        self.bus_stops = bus_stops 
        self.num_stations = len(bus_stops)

        #used in request generation to avoid using iterrows
        self.bus_stops_ids = []
        for index, stop_node in self.bus_stops.iterrows():
            self.bus_stops_ids.append(index)
       
        self.zones = zones

        self.walk_speed = walk_speed

        self.shortest_path_drive = shortest_path_drive
        self.shortest_path_walk = shortest_path_walk
        
        self.all_requests = {}

    def get_eta_walk(self, origin_node, destination_node):

        #try:
        #    sdestination_node = str(destination_node)
        #    distance_walk = self.shortest_path_walk.loc[origin_node, sdestination_node]
        #except KeyError:

        #returns estimated time walking in seconds from origin_node to destination_node
        try:
            distance_walk = nx.dijkstra_path_length(self.G_walk, origin_node, destination_node, weight='length')
        except nx.NetworkXNoPath:
            distance_walk = np.nan

        speed = self.walk_speed
        if math.isnan(distance_walk):
            eta_walk = -1
        else:
            eta_walk = int(math.ceil(distance_walk/speed))


        ''' 
        try:
            distance_walk = nx.dijkstra_path_length(self.G_walk, origin_node, destination_node, weight='length')
            #destination_node = str(destination_node)
            #distance_walk = self.shortest_path_walk.loc[origin_node, destination_node]
            #distance_walk = self.dict_shortest_path_length_walk[origin_node][destination_node]
            #get the speed of the given travel mode
            #i = self.get_travel_mode_index("walking")
            #speed = self.travel_modes[i].speed
            speed = self.walk_speed
            #calculates the estimated travel time by walking
            if math.isnan(distance_walk):
                eta_walk = -1
            else:  
                eta_walk = int(math.ceil(distance_walk/speed))
        except (nx.NetworkXNoPath, KeyError):
            eta_walk = -1
        '''

        return eta_walk

    def return_estimated_arrival_bus_osmnx(self, stops_origin, stops_destination, hour):
        max_eta_bus = -1
        avg_eta_bus = -1
        min_eta_bus = 1000000000

        #hour = 0
        for origin in stops_origin:
            for destination in stops_destination:

                #distance_bus = self.distance_matrix.loc[origin, destination]
                #travel_time = self.travel_time_matrix.loc[(origin, destination,hour), 'travel_time']

                #curr_weight = 'travel_time_' + str(hour)
                
                #origin_osmid = self.bus_stops.loc[origin, 'osmid_drive']
                #destination_osmid = self.bus_stops.loc[destination, 'osmid_drive']
                
                #i = self.bus_stops.loc[origin, 'itid']
                #j = self.bus_stops.loc[destination, 'itid']
                
                #travel_time = nx.dijkstra_path_length(self.G_drive, origin_osmid, destination_osmid, weight=curr_weight)
                #travel_time = self.travel_time_matrix.loc[(i, j, hour), 'eta']
                travel_time = self.travel_time_matrix.loc[(origin, destination), 'eta']
                #print(travel_time2)
                
                eta_bus = travel_time
                
                #i = self.get_travel_mode_index("bus")
                #speed = self.travel_modes[i].speed
                #eta_bus = int(math.ceil(distance_bus/speed))
                if not math.isnan(eta_bus):
                    if eta_bus >= 0:
                        if (eta_bus > max_eta_bus):
                            max_eta_bus = eta_bus

                        if (eta_bus < min_eta_bus):
                            min_eta_bus = eta_bus

        return max_eta_bus, min_eta_bus

    def update_travel_time_matrix(self, travel_time_matrix):

        self.travel_time_matrix = travel_time_matrix

    def get_travel_time_matrix_osmnx(self, travel_mode_string, interval):
        
        travel_time = np.ndarray((len(self.bus_stops), len(self.bus_stops)), dtype=np.int64)
        tuple_dtype = np.dtype([('r', np.int64), ('g', np.int64)])
        #travel_time = np.ndarray((len(self.bus_stops), len(self.bus_stops)), dtype=tuple_dtype)
        hour = 0
        #loop for computing the travel time matrix
        for index_o, origin_stop in self.bus_stops.iterrows():
            for index_d, destination_stop in self.bus_stops.iterrows():
                if travel_mode_string == "bus":
                    #i = int(self.distance_matrix.loc[(index_o,index_d), 'origin_id'])
                    #j = int(self.distance_matrix.loc[(index_o,index_d), 'destination_id'])
                    #i = int(origin_stop['itid'])
                    #j = int(destination_stop['itid'])
                    
                    #setting the speed value for the travel time matrix
                    #itm = self.get_travel_mode_index("bus")
                    #speed_bus = self.travel_modes[itm].speed
                    #speed_bus = int(self.avg_curr_speed_matrix.loc[index_o, index_d])
                    #if speed_bus <= 0:
                    #    speed_bus = 30
                    #speed_bus = speed_bus/3.6 #convert to m/s
                    
                    #getting the distance value
                    #distance_bus = self.distance_matrix.loc[index_o, index_d]
                    #travel_time = self.travel_time_matrix.loc[(origin, destination,hour), 'travel_time']
                    #curr_weight = 'travel_time_' + str(hour)
                    #origin = origin_stop['osmid_drive']
                    #destination = destination_stop['osmid_drive']
                    #od_travel_time = nx.dijkstra_path_length(self.G_drive, origin, destination, weight=curr_weight)
                    od_travel_time = self.travel_time_matrix.loc[(index_o, index_d), 'eta']

                    #calculating travel time and storing in travel_time matrix
                    if not math.isnan(od_travel_time):
                        if od_travel_time >= 0:
                            #travel_time[i][j] = int(math.ceil(distance_bus/speed_bus))
                            travel_time[i][j] = od_travel_time
                            #travel_time[i][j] =  (distance_bus, speed_bus)
                        else:
                            travel_time[i][j] = -1
                            #travel_time[i][j] = (-1, -1)

        return travel_time


    def get_random_coord(self, polygon):

        minx, miny, maxx, maxy = polygon.bounds
        
        counter = 0
        number = 1
        while counter < number:
            pnt = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
            if polygon.contains(pnt):
                return pnt

        return (-1, -1)

class Parameter:

    def __init__(self, max_walking, min_early_departure, max_early_departure, request_demand, day_of_the_week, num_replicates, bus_factor, get_fixed_lines, vehicle_speed_data, vehicle_speed, max_speed_factor, save_dir, output_file_base, num_of_cpu):
        
        self.num_requests = 0 
        #maximum time walking for a passenger
        self.max_walking = max_walking
        self.min_early_departure = min_early_departure
        self.max_early_departure = max_early_departure
        self.request_demand = request_demand
        self.num_replicates = num_replicates
        self.bus_factor = bus_factor
        self.save_dir = save_dir
        self.output_file_base = output_file_base
        self.day_of_the_week = day_of_the_week
        self.get_fixed_lines = get_fixed_lines #bool
        self.max_speed_factor = max_speed_factor
        self.vehicle_speed_data = vehicle_speed_data
        self.vehicle_speed = vehicle_speed
        self.num_of_cpu = num_of_cpu
       
        #self.folder_path_deconet = None
        
    #def update_network(self, G_drive, polygon_drive, shortest_path_drive, G_walk, polygon_walk, shortest_path_walk, bus_stops, zones, walk_speed):
        #print('creating network')
        #self.network = Network(G_drive, polygon_drive, shortest_path_drive, G_walk, polygon_walk, shortest_path_walk, bus_stops, zones, walk_speed)
        #self.network.add_graphs(G_drive, polygon_drive, G_walk, polygon_walk)
        #self.network.add_bus_stops(bus_stop_nodes)
        #self.network.add_distance_matrix(distance_matrix)