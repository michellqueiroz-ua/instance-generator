import math
import osmapi as osm
import osmnx as ox
import networkx as nx
import numpy as np
import random
from shapely.geometry import Point

            
class Network:

    def __init__(self, G_drive, shortest_path_drive, G_walk, shortest_path_walk, polygon, bus_stops, zones, schools, vehicle_speed):
        
        #network graphs
        self.G_drive = G_drive
        self.G_walk = G_walk
        
        self.polygon = polygon

        #indicates which nodes in the network are specifically bus stops
        self.bus_stops = bus_stops 
        self.num_stations = len(bus_stops)

        #used in request generation to avoid using iterrows
        self.bus_stops_ids = []
        for index, stop_node in self.bus_stops.iterrows():
            self.bus_stops_ids.append(index)
       
        self.zones = zones

        self.schools = schools

        self.shortest_path_drive = shortest_path_drive
        self.shortest_path_walk = shortest_path_walk
        
        self.vehicle_speed = vehicle_speed

    def get_eta_walk(self, u, v, walk_speed):
        
        
        #returns estimated time walking in seconds from origin_node to destination_node
        #in the case of walking, there is no difference between origin/destination
        try:
            sv = str(v)
            distance_walk = self.shortest_path_walk.loc[u, sv]
        except KeyError:

            try:
                su = str(u)
                distance_walk = self.shortest_path_walk.loc[v, su]
            except KeyError:
                try:

                    distance_walk = nx.dijkstra_path_length(self.G_walk, u, v, weight='length')
                except nx.NetworkXNoPath:
                    distance_walk = np.nan
 
        
        speed = walk_speed
        #print(u, v)
        #print(distance_walk)
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

    def return_estimated_travel_time_bus(self, stops_origin, stops_destination):
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

    def _return_estimated_travel_time_drive(self, origin_node, destination_node):

        eta = -1
        try:
            distance = self.shortest_path_drive.loc[int(origin_node), str(destination_node)]
            
            if str(distance) != 'nan':
                distance = int(distance)
                eta = int(math.ceil(distance/self.vehicle_speed))

        except KeyError:
            eta = -1

        return int(eta)

    def update_travel_time_matrix(self, travel_time_matrix):

        self.travel_time_matrix = travel_time_matrix

    def _get_travel_time_matrix(self, nodes):
        
        
        if nodes == "bus":
            '''
            return travel time matrix between all bus stations in the network
            '''
            #travel_time = np.ndarray((len(self.bus_stops), len(self.bus_stops)))
            travel_time = []
            
            #loop for computing the travel time matrix
            i = 0
            for index_o, origin_stop in self.bus_stops.iterrows():
                j=0
                for index_d, destination_stop in self.bus_stops.iterrows():    
                    od_travel_time = self.travel_time_matrix.loc[(index_o, index_d), 'eta']

                    #calculating travel time and storing in travel_time matrix
                    if not math.isnan(od_travel_time):
                        if od_travel_time >= 0:
                            element = (i, j, od_travel_time)
                            travel_time.append(element)
                        else:
                            od_travel_time = -1
                            od_travel_time = int(od_travel_time)
                            element = (i, j, -1)
                            travel_time.append(element)
                    j+=1
                i+=1

        if nodes == "all":
            '''
            return travel time matrix between all nodes in the network
            '''
            travel_time = []
            #loop for computing the travel time matrix
            i = 0
            for u in self.G_drive.nodes:
                j=0
                for v in self.G_drive.nodes:
                    od_travel_time = self._return_estimated_travel_time_drive(u, v)   
                    
                    #calculating travel time and storing in travel_time matrix
                    if not math.isnan(od_travel_time):
                        if od_travel_time >= 0:
                            element = (i, j, od_travel_time)
                            travel_time.append(element)
                        else:
                            od_travel_time = -1
                            od_travel_time = int(od_travel_time)
                            element = (i, j, od_travel_time)
                            travel_time.append(element)
                    j+=1
                i+=1

        return travel_time

    def _get_travel_time_from_stops_to_school(self, school_id):

        '''
        computes and returns a list of travel time between all bus stations to school (destination)
        '''
        v = self.schools.loc[school_id, 'osmid_drive']
        travel_time = []

        i = 0
        for index_o, origin_stop in self.bus_stops.iterrows():
            
            u = self.bus_stops.loc[index_o, 'osmid_drive']

            od_travel_time = self._return_estimated_travel_time_drive(u, v)
            
            if not math.isnan(od_travel_time):
                if od_travel_time >= 0:
                    element = (i, od_travel_time)
                    travel_time.append(element)
                else:
                    element = (i, -1)
                    travel_time.append(element)

            i += 1

        return travel_time  



    def _get_random_coord(self, polygon):

        '''
        returns random coordinate within the polygon bounds
        '''
        minx, miny, maxx, maxy = polygon.bounds
        
        counter = 0
        number = 1
        while counter < number:
            pnt = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
            if polygon.contains(pnt):
                return pnt

        return (-1, -1)


    def set_zone_bbox(self, zone_id, dist_lat, dist_lon):

        '''
        function to manually set the zone geometry
        default when downloading network information is dist_lat=1km and dist_lon=1km
        '''

        self.zones.loc[zone_id, 'dist_lat'] = dist_lat
        self.zones.loc[zone_id, 'dist_lon'] = dist_lon
        earth_radius = 6371009  # meters
        
        lat = zone['center_point_y']
        lng = zone['center_point_x']

        delta_lat = (dist_lat / earth_radius) * (180 / math.pi)
        delta_lng = (dist_lon / earth_radius) * (180 / math.pi) / math.cos(lat * math.pi / 180)
        
        north = lat + delta_lat
        south = lat - delta_lat
        east = lng + delta_lng
        west = lng - delta_lng
        
        polygon = Polygon([(west, south), (east, south), (east, north), (west, north)])
            
        #updates the polygon. used to generate coordinates within the zone
        self.zones.loc[index, 'polygon'] = polygon


