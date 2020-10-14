import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import codecs, json
from random import randint
from random import seed
from random import choices
import math
from scipy.stats import norm
#import scipy.stats

try:
    import tkinter as tk
    from tkinter import filedialog
except:
    pass

class RequestDistribution:

    def __init__(self, mean, std, num_requests):
        self.mean = mean
        self.std = std
        self.num_requests = num_requests
        self.demand = []
        self.demand = np.random.normal(self.mean, self.std, self.num_requests)
        self.origin_neighborhood = None
        self.destination_neighborhood = None

    def set_origin_neighborhood(self, origin_neighborhood):
        self.origin_neighborhood = origin_neighborhood

    def set_destination_neighborhood(self, destination_neighborhood):
        self.destination_neighborhood = destination_neighborhood

class TravelMode:

    def __init__(self, mean_transportation, speed):
        self.mean_transportation = mean_transportation
        self.speed = speed
   
#also adds more information on the arcs like -> situation of the road => traffic (slower), accident etc
#function with parameter: time and returns the current speed of the road
class Arc:

    def __init__(self, origin, destination):
        self.origin = origin
        self.destination = destination
        self.connection_type = "indirect"
        self.distance_vehicle = np.iinfo(np.int64).max
        self.distance_walking = np.iinfo(np.int64).max
        self.fastest_time = np.iinfo(np.int64).max
        self.fastest_path = []
        self.traffic_flow = []
        self.max_flow = 0
        #self.paths -> store all possible paths between two nodes?
        self.max_speed = -1
        # normal : average speed
        # traffic : 1/2 average speed ?
        # accident : 1/4 average speed ?
        
    def set_vehicle_distance(self, distance):
        self.distance_vehicle = distance
        self.connection_type = "direct"

    #add current time as a parameter maybe?
    def get_expected_travel_time(self, travel_mode):

        if self.connection_type == "direct":
            speed = travel_mode.speed
            if travel_mode.mean_transportation == "bus":
                distance = self.distance_vehicle
            expected_travel_time = int(math.ceil(distance/speed))
        else:
            expected_travel_time = np.iinfo(np.int64).max
        
        return expected_travel_time

    def set_walking_distance(self, distance):
        self.distance_walking = distance

    #also give different speeds such as bus speed, car speed, bike speed whatever
    def set_max_speed(self, max_speed):
        self.max_speed = max_speed
    
    #traffic        
    def add_traffic_flow(self, mean, std):
        self.traffic_flow.append({"mean": mean, "std": std})

    def add_accident(self, time):
        accident = True

        if accident is True:
            std = 600
            peak = 600 
            self.add_traffic_flow(time+peak, std)

    #traffic frequency at current time "time" is given by all pdfs in the arc
    def get_traffic_flow(self, time):
        mean = 1600 
        std = 800
        traffic_flow = norm.pdf(x=time, loc=mean, scale=std)*1000

    def return_current_speed(self, travel_mode, time):    
        current_speed = int(math.ceil(3*self.max_speed/4))

        return current_speed

class Node:

    def __init__(self, index, coord):
        self.index = index
        self.coord = coord
        #self.adj = []
        self.incoming_arcs = []
        self.outgoing_arcs = []

class Coordinate:

    def __init__(self, x, y):
        self.x = x
        self.y = y

class Shape:

    def __init__(self, shape_type):
        self.shape_type = shape_type

    def square(self, side_length):
        self.side_length = side_length

class Neighborhood:

    def __init__(self, name, coord, shape):
        self.name = name
        self.coord = coord
        self.shape = shape
        #self.radius = radius
        #self.area_type = area_type
        #self.origin_demand = origin_demand
        #self.destination_demand = destination_demand

    #returns a random coordinate in the given neighborhood
    def get_random_coord(self):
        if (self.shape.shape_type == "square"):
            x = randint(self.coord.x, self.coord.x + self.shape.side_length + 1) 
            y = randint(self.coord.y, self.coord.y + self.shape.side_length + 1)
        r_coord = Coordinate(x, y)
        return r_coord

class Network:

    def __init__(self, city_max, nodes, travel_modes, stop_area_spacing, neighborhoods):
        self.nodes = nodes
        self.travel_modes = travel_modes 
        self.city_max = city_max
        self.num_stations = 0
        self.stop_area_spacing = stop_area_spacing
        self.neighborhoods = neighborhoods

    def fastest_node(self, fastest_time, finalized):
        fastest_time_v = np.iinfo(np.int64).max
        fastest_index_v = -1

        for v in range(len(self.nodes)):
            if fastest_time[v] < fastest_time_v and finalized[v] == False:
                fastest_time_v = fastest_time[v]
                fastest_index_v = v

        return fastest_index_v

    def dijkstras(self, source, travel_mode):
        fastest_time = [np.iinfo(np.int64).max] * len(self.nodes)
        fastest_time[source] = 0
        finalized = [False] * len(self.nodes)

        for count in range(len(self.nodes)):

            src = self.fastest_node(fastest_time, finalized) 
            finalized[src] = True

            for dest in range(len(self.nodes)):
                expected_travel_time_src_dest = self.arcs[src][dest].get_expected_travel_time(travel_mode)
                if finalized[dest] == False and fastest_time[dest] > fastest_time[src] + expected_travel_time_src_dest:
                    fastest_time[dest] = fastest_time[src] + expected_travel_time_src_dest

        return fastest_time

    def update_fastest_time_bus(self):
        for source in range(len(self.nodes)):
            t = self.get_travel_mode_index("bus")
            fastest_time = self.dijkstras(source, self.travel_modes[t])
            for dest in range(len(self.nodes)):
                self.arcs[source][dest].fastest_time = fastest_time[dest]

    def calculate_distance_euclidean(self, node1, node2):
        return int(math.ceil(((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2) ** .5))  # was vroeger int, maar gaf afrondingsproblemen

    def calculate_distance(self, node1, node2):
        return int(math.ceil(((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2) ** .5))  # was vroeger int, maar gaf afrondingsproblemen
        
    def update_network(self, nodes):
        self.nodes = nodes
        #generate the arcs in the network
        self.generate_vehicle_arcs()
        #time dist matrix
        self.update_fastest_time_bus()

    def generate_vehicle_arcs(self):
        self.arcs = np.ndarray((len(self.nodes), len(self.nodes)), dtype=object)
        radius_direct_road = 2000
        #add the probability later
        for i in range(len(self.nodes)):
            max_value = np.iinfo(np.int64).max
            cn_id = []
            for j in range(len(self.nodes)):
                if i != j:
                    distance = self.calculate_distance_euclidean(self.nodes[i].coord, self.nodes[j].coord)
                    if (distance <= radius_direct_road):
                        cn_id.append(j) 

            #every i should leave this loop with at least one outgoing arc

            for k in range(len(cn_id)):
                adj = cn_id[k]
                distance = self.calculate_distance(self.nodes[i].coord, self.nodes[adj].coord)
                arc = Arc(self.nodes[i].index, self.nodes[adj].index)
                arc.set_vehicle_distance(distance)
                self.arcs[self.nodes[i].index][self.nodes[adj].index] = arc

        for i in range(len(self.nodes)):
            for j in range(len(self.nodes)):
                if self.arcs[self.nodes[i].index][self.nodes[j].index] is None:
                    self.arcs[self.nodes[i].index][self.nodes[j].index] = Arc(self.nodes[i].index, self.nodes[j].index)
                    #distance is infinite. dijkstra's algorithm calculate fastest time between nodes that are not directly connected

    def get_travel_mode_index(self, travel_mode_string):
        for i in range(len(self.travel_modes)):
            #print(travel_mode, self.travel_modes[i].mean_transportation)
            if travel_mode_string == self.travel_modes[i].mean_transportation:
                return int(i)

    def return_estimated_arrival_bus(self, origin, destination, stops_origin, stops_destination):
        max_eta_bus = -1

        for i in range(len(stops_origin)):
            for j in range(len(stops_destination)):
                if self.arcs[stops_origin[i]][stops_destination[j]].fastest_time > max_eta_bus:
                    max_eta_bus = self.arcs[stops_origin[i]][stops_destination[j]].fastest_time

        return max_eta_bus

    def return_estimated_arrival_walk(self, origin_coord, destination_coord):

        min_distance = self.calculate_distance_euclidean(origin_coord, destination_coord)
        horizon_coord = Coordinate(destination_coord.x, origin_coord.y)
        #vertical_coord = Coordinate(origin_coord.x, destination_coord.y)
        #test if max distance is the same for both vertical/horizontal coord
        max_distance = self.calculate_distance_euclidean(origin_coord, horizon_coord) + self.calculate_distance_euclidean(horizon_coord, destination_coord)
        i = self.get_travel_mode_index("walking")
        avg_distance = int(math.ceil((max_distance+min_distance)/2))
        speed = self.travel_modes[i].speed
        eta_walk = int(math.ceil(avg_distance/speed))

        return eta_walk

    def get_travel_time_matrix(self, travel_mode_string, interval):
        #k = self.get_travel_mode_index(travel_mode_string)
        #travel_time_k = [[None for j in range(len(self.nodes))] for i in range(len(self.nodes))]
        travel_time = np.ndarray((len(self.nodes), len(self.nodes)), dtype=np.int64)
        for i in range(len(self.nodes)):
            for j in range(len(self.nodes)):
                if travel_mode_string == "bus":
                    travel_time[i][j] = self.arcs[i][j].fastest_time

        return travel_time

    def update_neighborhoods(self):
        for i in range(self.num_city_center):
            #choose the coordinates for the city center
            cx = 1
            cy = 1 
            radius = 20 
            area_type = 1 #types of area, which could mean, center, high density populated area, low density populated area
            city_center = CityArea(cx, cy, radius, area_type)
            self.city_centers.append(city_center) 

class Instance:

    def __init__(self, city_max, max_walking, max_early_departure, request_demand, travel_modes, neighborhoods, stop_area_spacing, num_replicates, bus_factor, save_dir, output_file_base):
        #initially 0 requests
        self.num_requests = 0
        #self.max = Coordinate(max_x, max_y)
        #self.city_max = city_max
        #self.num_stations = num_stations
        self.max_walking = max_walking
        self.min_early_departure = 0
        self.max_early_departure = max_early_departure
        self.request_demand = request_demand
        self.hour_spacing = 3600 #by pattern the code is working in meters and seconds, so each hour is represented by 3600 seconds
        self.num_replicates = num_replicates
        self.bus_factor = bus_factor
        self.save_dir = save_dir
        #self.walk_speed = walk_speed
        #self.bus_speed = bus_speed
        nodes = [] #initially is an empty network, but this can change depending on how the instance is created
        self.network = Network(city_max, nodes, travel_modes, stop_area_spacing, neighborhoods)
        #probability of creating a bus stop in given area
        #self.probability_low = 40
        #self.probability_medium = 70 
        #self.probability_high = 100
        self.output_file_base = output_file_base
        #self.req_demand = Demand(1500, 1000, 500, self.min_early_departure, self.max_early_departure, self.hour_spacing)
        
        
class JsonConverter(object):

    def __init__(self, file_name):

        with open(file_name, 'r') as file:
            self.json_data = json.load(file)
            file.close()

    def convert_normal(self, output_file_name):

        with open(output_file_name, 'w') as file:

            # first line: number of stations
            file.write(str(self.json_data.get('num_stations')))
            file.write('\n')

            # second line - nr station: distance matrix
            dist_matrix = self.json_data.get('distance_matrix')
            for row in dist_matrix:
                for distance in row:
                    file.write(str(distance))
                    file.write('\t')
                file.write('\n')

            # start of request information
            requests = self.json_data.get('requests')
            num_requests = len(requests)

            # first line: number of requests
            file.write(str(num_requests))
            file.write('\n')

            # foreach request
            for request in requests.values():

                # origin coordinates
                file.write(str(request.get('originx')) + '\t' + str(request.get('originy')))
                file.write('\n')

                # destination coordinates
                file.write(str(request.get('destinationx')) + '\t' + str(request.get('destinationy')))
                file.write('\n')

                # num stops origin + stops origin
                file.write(str(request.get('num_stops_origin')) + '\t')
                for stop in request.get('stops_origin'):
                    file.write(str(stop) + '\t')

                file.write('\n')

                # num stops destination + stops destination
                file.write(str(request.get('num_stops_destination')) + '\t')
                for stop in request.get('stops_destination'):
                    file.write(str(stop) + '\t')

                file.write('\n')

                # earliest departure time
                file.write(str(request.get('dep_time')))
                file.write('\n')

                # latest arrival time
                file.write(str(request.get('arr_time')))
                file.write('\n')

    def convert_localsolver(self, output_file_name):

        with open(output_file_name, 'w') as file:

            # first line: number of stations
            file.write(str(self.json_data.get('num_stations')))
            file.write('\n')

            # second line - nr station: distance matrix
            dist_matrix = self.json_data.get('distance_matrix')
            for row in dist_matrix:
                for distance in row:
                    file.write(str(distance))
                    file.write('\t')
                file.write('\n')

            # start of request information
            requests = self.json_data.get('requests')
            num_requests = len(requests)

            # first line: number of requests
            file.write(str(num_requests))
            file.write('\n')

            # request information
            requests = self.json_data.get('requests')
            num_requests = len(requests)

            # line format: (index nb_stops stops demand=1 ear_dep_time lat_arr_time serve_time pick_ind deliv_ind)
            # first line: 0    0   0   0   1000    0   0   0
            # file.write('0\t0\t0\t0\t1000\t0\t0\t0')

            # foreach request - request split in pickup and delivery_pair
            index = 1
            for request in requests.values():

                index_pickup = index
                index_delivery = index_pickup + 1

                nb_stops_pickup = request.get('num_stops_origin')
                stops_pickup = request.get('stops_origin')

                nb_stops_delivery = request.get('num_stops_destination')
                stops_delivery = request.get('stops_destination')

                demand = 1
                serv_time = 0
                dep_time = request.get('dep_time')
                arr_time = request.get('arr_time')

                file.write('\n')

                # write pickup
                file.write(str(index_pickup) + '\t')
                file.write(str(nb_stops_pickup) + '\t')
                for stop in stops_pickup:
                    file.write(str(stop) + '\t')
                file.write(str(demand) + '\t')
                file.write(str(dep_time) + '\t')
                file.write(str(arr_time) + '\t')
                file.write(str(serv_time) + '\t')
                file.write('0' + '\t')  # pickup index always = 0 for pickups
                file.write(str(index_delivery) + '\t')

                file.write('\n')

                # write delivery
                demand = -1
                file.write(str(index_delivery) + '\t')
                file.write(str(nb_stops_delivery) + '\t')
                for stop in stops_delivery:
                    file.write(str(stop) + '\t')
                file.write(str(demand) + '\t')
                file.write(str(dep_time) + '\t')
                file.write(str(arr_time) + '\t')
                file.write(str(serv_time) + '\t')
                file.write(str(index_pickup) + '\t')
                file.write('0' + '\t')  # delivery index always 0 for deliveries

                index += 2


def generate_instances_json(inst):

    inst.save_dir_json = os.path.join(inst.save_dir, 'json_format')
    
    if not os.path.isdir(inst.save_dir_json):
        os.mkdir(inst.save_dir_json)

    #stationsx = np.ndarray(inst.num_stations)
    #stationsy = np.ndarray(inst.num_stations)

    for replicate in range(inst.num_replicates):

        output_file_json = os.path.join(inst.save_dir_json, inst.output_file_base + '_' + str(replicate) + '.json')
        instance_data = {}  # holds information for each request

        stations = []
        inst.network.num_stations = 0
        print("Now generating " + " stations_data")

        #generate stations
        for b in range(inst.network.stop_area_spacing.y, inst.network.city_max.y+1, inst.network.stop_area_spacing.y):
            for a in range(inst.network.stop_area_spacing.x, inst.network.city_max.x+1, inst.network.stop_area_spacing.x):
                x = randint(a-inst.network.stop_area_spacing.x, a)
                y = randint(b-inst.network.stop_area_spacing.y, b)
                #print(x, y)
                index = inst.network.num_stations
                coord = Coordinate(x,y)
                station = Node(index, coord)
                stations.append(station)
                inst.network.num_stations += 1

        #print("num stops")
        #print(inst.network.num_stations)
        #print(" ")

        nodes = stations #currently the only nodes in the network are the stations (bus stops)
        inst.network.update_network(nodes)

        print("Now generating " + " request_data")
        lines = []
        all_requests = {}  # holds all instance_data
        h = 0
        inst.num_requests = 0

        #generate requests according to the desired time horizon
        #for hour in range(inst.min_early_departure, inst.max_early_departure, inst.hour_spacing):
        for r in range(len(inst.request_demand)):    
            num_requests = inst.request_demand[r].num_requests
            print(num_requests)
            for i in range(num_requests): 
                
                #print(inst.num_requests)
                dep_time = inst.request_demand[r].demand[i]
                dep_time = int(dep_time)
                if (dep_time >= 0):
                    nok = True
                else:
                    nok = False
                request_data = {}  # holds information about this request
                #randomly choosing coordinates and repeated for a passenger until he can go to at least 1 stop (for both origin and destination)

                while nok:

                    nok = False
                    # print ("Passenger " + str(i))

                    if inst.request_demand[r].origin_neighborhood is None:
                        originx = randint(0, inst.network.city_max.x + 1) #that can create a coordinate 
                        originy = randint(0, inst.network.city_max.y + 1)
                        origin = Coordinate(originx, originy)
                    else:
                        origin = inst.request_demand[r].origin_neighborhood.get_random_coord()

                    if inst.request_demand[r].destination_neighborhood is None:
                        destinationx = randint(0, inst.network.city_max.x + 1)
                        destinationy = randint(0, inst.network.city_max.y + 1)
                        destination = Coordinate(destinationx, destinationy)
                    else:
                        #print(inst.num_requests)
                        destination = inst.request_demand[r].destination_neighborhood.get_random_coord()

                    stops_origin = []
                    stops_destination = []
                    
                    if inst.network.return_estimated_arrival_walk(origin, destination) > inst.max_walking: #if distance between origin and destination is too small the person just walks
                        #change j to station_id or something similar. node_id
                        #calculates the stations which are close enough to the origin and destination of the request
                        
                        for j in range(inst.network.num_stations):
                            #change the append for the "index"?
                            
                            if inst.network.return_estimated_arrival_walk(origin, stations[j].coord) <= inst.max_walking:
                                stops_origin.append(j)

                            if inst.network.return_estimated_arrival_walk(destination, stations[j].coord) <= inst.max_walking:
                                stops_destination.append(j)

                    # Check whether (1) each passenger can walk to a stop (origin + destination), the intersection of the
                    # origin and destination stop list is not empty (in which case they do not require a bus)
                    if len(stops_origin) > 0 and len(stops_destination) > 0:
                        if not (set(stops_origin) & set(stops_destination)): 

                            #prints in the json file if the request is 'viable'
                            request_data.update({'originx': int(originx)})
                            request_data.update({'originy': int(originy)})

                            request_data.update({'destinationx': int(destinationx)})
                            request_data.update({'destinationy': int(destinationy)})

                            request_data.update({'num_stops_origin': len(stops_origin)})

                            request_data.update({'stops_origin': stops_origin})
                            request_data.update({'num_stops_destination': len(stops_destination)})

                            request_data.update({'stops_destination': stops_destination})

                            #departure time
                            request_data.update({'dep_time': int(dep_time)})

                            #time window for the arrival time
                            eta_bus = inst.network.return_estimated_arrival_bus(origin, destination, stops_origin, stops_destination)
                            arr_time = (dep_time) + (inst.bus_factor * eta_bus) + (inst.max_walking * 2)
                            
                            request_data.update({'arr_time': int(arr_time)})

                            # add request_data to instance_data container
                            all_requests.update({inst.num_requests: request_data})
                            inst.num_requests += 1
                        else:
                            # print ("Passenger cannot walk to a stop")
                            nok = True
                    else:
                        # print ("Passenger cannot walk to a stop")
                        nok = True
                # if nok == False:

            #h += 1
        

        travel_time_json = inst.network.get_travel_time_matrix("bus", 0)
        travel_time_json = travel_time_json.tolist()
        instance_data.update({'requests': all_requests,
                              'num_stations': inst.network.num_stations,
                              'distance_matrix': travel_time_json})

        with open(output_file_json, 'w') as file:
            json.dump(instance_data, file, indent=4)
            file.close()


if __name__ == '__main__':

    # OPTIONS - INSTANCE INPUT INFORMATION
    for i in range(len(sys.argv)):
        if sys.argv[i] == "--file_name":
           output_file_base = sys.argv[i+1].split('.')[0]

        if sys.argv[i] == "--city_max":
            city_max = Coordinate(int(sys.argv[i+1]), int(sys.argv[i+2]))

            if sys.argv[i+3] == "km":
                city_max.x = city_max.x*1000
                city_max.y = city_max.y*1000

        if sys.argv[i] == "--walking_threshold":
            max_walking = int(sys.argv[i+1])

            if sys.argv[i+2] == "min":
                max_walking = max_walking*60

        if sys.argv[i] == "--max_early_departure":
            max_early_departure = int(sys.argv[i+1])

            if sys.argv[i+2] == "h":
                max_early_departure = max_early_departure*3600

            if sys.argv[i+2] == "min":
                max_early_departure = max_early_departure*60

        #if sys.argv[i] == "--walk_speed":
            #walk_speed = int(sys.argv[i+1])
            
            #if sys.argv[i+2] == "kmh":
                #walk_speed = walk_speed/3.6

        #if sys.argv[i] == "--bus_speed":
            #bus_speed = int(sys.argv[i+1])
            
            #if sys.argv[i+2] == "kmh":
                #bus_speed = bus_speed/3.6

        if sys.argv[i] == "--seed":
            set_seed = int(sys.argv[i+1])
            seed(set_seed)
            np.random.seed(set_seed)

        if sys.argv[i] == "--num_replicates":
            num_replicates = int(sys.argv[i+1])

        if sys.argv[i] == "--stop_area_spacing":
            stop_area_spacing = Coordinate(int(sys.argv[i+1]), int(sys.argv[i+2]))

            if sys.argv[i+3] == "km":
                stop_area_spacing.x = stop_area_spacing.x*1000
                stop_area_spacing.y = stop_area_spacing.y*1000

        if sys.argv[i] == "--travel_modes":
            num_travel_modes = int(sys.argv[i+1])

            travel_modes = []
            k = i+2
            for j in range(num_travel_modes): 
                mean_transportation = sys.argv[k]
                k = k+1
                speed = int(sys.argv[k])
                k = k+1
                if sys.argv[k] == "kmh":
                    speed = speed/3.6
                k = k+1
                travel_mode = TravelMode(mean_transportation, speed)
                travel_modes.append(travel_mode)

        #request demand that comes from anywhere in the city
        #origin and destination are random
        if sys.argv[i] == "--city_distribution_demand":
            num_curves = int(sys.argv[i+1])
            request_demand = []
            k = i+1
            for j in range(num_curves):
                dnd = RequestDistribution(int(sys.argv[k+1]), int(sys.argv[k+2]), int(sys.argv[k+3]))
                request_demand.append(dnd)
                k = k+3
    
        
        if sys.argv[i] == "--neighborhoods":
            num_neighborhoods = int(sys.argv[i+1])
            neighborhoods = []
            k=i+2
            for j in range(num_neighborhoods):
                name = sys.argv[k]
                k = k+1
                x = int(sys.argv[k])
                if x == -1:
                    x = int(city_max.x/2)
                k = k+1
                y = int(sys.argv[k])
                if y == -1:
                    y = int(city_max.y/2)
                coord = Coordinate(x, y)
                k = k+1
                shape = Shape(sys.argv[k])

                if shape.shape_type == "square":
                    k = k+1
                    side_length = int(sys.argv[k])
                    if side_length == -1:
                        side_length = city_max.x/4
                    shape.square(side_length)

                neighborhood = Neighborhood(name, coord, shape)
                neighborhoods.append(neighborhood)
                
                k = k+1
                mean = int(sys.argv[k])
                k = k+1
                std = int(sys.argv[k])
                k = k+1
                num_requests = int(sys.argv[k])
                #print(mean, std, num_requests)
                origin_demand = RequestDistribution(mean, std, num_requests)
                origin_demand.set_origin_neighborhood(neighborhood)
                request_demand.append(origin_demand)

                k = k+1
                mean = int(sys.argv[k])
                k = k+1
                std = int(sys.argv[k])
                k = k+1
                num_requests = int(sys.argv[k])
                #print(mean, std, num_requests)
                destination_demand = RequestDistribution(mean, std, num_requests)
                destination_demand.set_destination_neighborhood(neighborhood)
                request_demand.append(destination_demand)
 
            #print("neighborhoods size", len(neighborhoods))

    bus_factor = 2
    

    #for j in range(len(nd_samples)):
        #print(len(nd_samples[j].demand))
        #for k in range(len(nd_samples[j].demand)):
            #print(nd_samples[j].demand[k])

    #files are saved on the current directory
    save_dir = os.getcwd()

    #creating object that has the instance input information
    inst = Instance(city_max, max_walking, max_early_departure, request_demand, travel_modes, neighborhoods, stop_area_spacing, num_replicates, bus_factor, save_dir, output_file_base)

    # generate instances in json output folder
    generate_instances_json(inst)

    # convert instances from json to normal and localsolver format
    save_dir_cpp = os.path.join(save_dir, 'cpp_format')
    if not os.path.isdir(save_dir_cpp):
        os.mkdir(save_dir_cpp)

    save_dir_localsolver = os.path.join(save_dir, 'localsolver_format')
    if not os.path.isdir(save_dir_localsolver):
        os.mkdir(save_dir_localsolver)

    for instance in os.listdir(os.path.join(save_dir, 'json_format')):
        #print(instance)
        input_name = os.path.join(save_dir, 'json_format', instance)
        output_name_cpp = instance.split('.')[0] + '_cpp.pass'
        output_name_ls = instance.split('.')[0] + '_ls.pass'

        converter = JsonConverter(file_name=input_name)
        converter.convert_normal(output_file_name=os.path.join(save_dir_cpp, output_name_cpp))
        converter.convert_localsolver(output_file_name=os.path.join(save_dir_localsolver, output_name_ls))
