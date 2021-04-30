import math
import osmapi as osm
import osmnx as ox
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from shapely.geometry import Point
from shapely.geometry import Polygon
from geopy.distance import geodesic

            
class Network:

    def __init__(self, place_name, G_drive, G_walk, polygon, bus_stations, zones, schools):
        
        self.place_name = place_name

        #network graphs
        self.G_drive = G_drive
        self.G_walk = G_walk
        
        self.polygon = polygon

        #indicates which nodes in the network are specifically bus stops
        self.bus_stations = bus_stations 
        self.num_stations = len(bus_stations)

        #used in request generation to avoid using iterrows
        self.bus_stations_ids = []
        #for index, stop_node in self.bus_stations.iterrows():
        #    self.bus_stations_ids.append(index)

        self.node_list_darp = []
       
        self.zones = zones

        self.schools = schools

        #self.shortest_path_drive = shortest_path_drive
        #self.shortest_path_walk = shortest_path_walk

    def get_eta_walk(self, u, v, walk_speed):
        
        osm_api = osm.OsmApi()
        
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
                    #print(u,v)
                    distance_walk = nx.dijkstra_path_length(self.G_walk, u, v, weight='length')
                
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    distance_walk = np.nan
                    
                    #nx.exception.NodeNotFound
        
        if math.isnan(distance_walk):
            '''
            nodeu = osm_api.NodeGet(u)

            nodev = osm_api.NodeGet(v)

            origin = (nodeu['lat'], nodeu['lon'])
            dest = (nodev['lat'], nodev['lon'])

            distance_walk = geodesic(origin, dest).meters

            eta_walk = int(math.ceil(distance_walk/speed))
            '''

            eta_walk = -1
            
        else:
            eta_walk = int(math.ceil(distance_walk/walk_speed))

        return eta_walk

    def _return_estimated_travel_time_drive(self, origin_node, destination_node):

        '''
        returns estimated travel time in seconds between origin_node to destination_node
        '''

        eta = -1
        try:
            
            distance = self.shortest_path_drive.loc[int(origin_node), str(destination_node)]
            
            if str(distance) != 'nan':
                eta = int(distance)

        except KeyError:
            eta = -1

        '''
        if eta == -1:

            try:
                eta = nx.dijkstra_path_length(self.G_drive, int(origin_node), int(destination_node), weight='length')
            except nx.NetworkXNoPath:
                eta = -2
        '''

        return int(eta)
        

    def return_estimated_travel_time_bus(self, stops_origin, stops_destination):
        max_eta_bus = -1
        avg_eta_bus = -1
        min_eta_bus = 1000000000

        for origin in stops_origin:
            for destination in stops_destination:

                u = self.bus_stations.loc[origin, 'osmid_drive']
                v = self.bus_stations.loc[destination, 'osmid_drive']

                travel_time = self._return_estimated_travel_time_drive(u, v)
                
                eta_bus = travel_time
                
                if not math.isnan(eta_bus):
                    if eta_bus >= 0:
                        if (eta_bus > max_eta_bus):
                            max_eta_bus = eta_bus

                        if (eta_bus < min_eta_bus):
                            min_eta_bus = eta_bus

        return max_eta_bus, min_eta_bus


    def _get_travel_time_matrix(self, nodes, inst=None, node_list=None):
        
        if nodes == "bus":
            
            #return travel time matrix between all bus stations in the network
            
            travel_time = np.ndarray((len(self.bus_stations), len(self.bus_stations)))
            #travel_time = []
            
            
            #loop for computing the travel time matrix
            i=0
            for index_o, origin_stop in self.bus_stations.iterrows():
                j=0
                for index_d, destination_stop in self.bus_stations.iterrows(): 
                    #u = self.bus_stations.loc[int(index_o), 'osmid_drive']
                    #v = self.bus_stations.loc[int(index_d), 'osmid_drive']
                    u = int(origin_stop['osmid_drive'])
                    v = int(destination_stop['osmid_drive'])
                    
                    od_travel_time = self._return_estimated_travel_time_drive(u, v)

                    #calculating travel time and storing in travel_time matrix
                    if not math.isnan(od_travel_time):
                        if od_travel_time >= 0:
                            #element = (str(index_o), str(index_d), str(od_travel_time))
                            #travel_time.append(element)
                            travel_time[index_o][index_d] = od_travel_time

                            if (od_travel_time == 0):
                                if (index_o != index_d):
                                    print('here')
                            #print(od_travel_time)
                        '''
                        else:
                            od_travel_time = -1
                            od_travel_time = int(od_travel_time)
                            element = (i, j, -1)
                            travel_time.append(element)
                        '''
                    j+=1
                i+=1

        if nodes == "list":
            #travel_time = []
            travel_time = np.ndarray((len(node_list), len(node_list))) 

            for u in range(0, len(node_list)):
                for v in range(0, len(node_list)):

                    #print(int(node_list[u]), int(node_list[v]))
                    od_travel_time = self._return_estimated_travel_time_drive(int(node_list[u]), int(node_list[v]))  

                    if not math.isnan(od_travel_time):
                        travel_time[u][v] = int(od_travel_time)
                        #element = (str(u), str(v), str(od_travel_time))
                        #travel_time.append(element)
                        

        '''
        if nodes == "subway":
            
            travel_time = []

            for line_id in self.subway_lines:

                for u in self.nodes_covered_fixed_lines:
                    for v in self.nodes_covered_fixed_lines:

                        try:

                            eta = nx.dijkstra_path_length(self.subway_lines[line_id]['route_graph'], u, v, weight='duration_avg')
                            
                            element = (str(u), str(v), str(int(eta)), str(line_id))
                            travel_time.append(element)

                        except (nx.NetworkXNoPath, KeyError, nx.NodeNotFound):
                            pass
        '''

        '''
        if nodes == "hybrid":

            travel_time = []

            for u in self.bus_stations_ids:
                for v in self.nodes_covered_fixed_lines:

                    uwalk = self.bus_stations.loc[u, 'osmid_walk']
                    udrive = self.bus_stations.loc[u, 'osmid_drive']
                    vwalk = self.deconet_network_nodes.loc[int(v), 'osmid_walk']
                    avg_walk_speed = (inst.min_walk_speed+inst.max_walk_speed)/2
                    
                    #from u to v
                    eta1 = self.get_eta_walk(uwalk, vwalk, avg_walk_speed)
                    #from v to u
                    eta2 = self.get_eta_walk(vwalk, uwalk, avg_walk_speed)
                    if eta1 >= 0:
                        element = (str(udrive), str(v), str(eta1))
                        travel_time.append(element)

                    if eta2 >= 0:
                        element = (str(v), str(udrive), str(eta2))
                        travel_time.append(element)
        '''

        '''
        if nodes == "list":
            travel_time = []

            for u in node_list:
                for v in node_list:
                    od_travel_time = self._return_estimated_travel_time_drive(u, v)  

                    if not math.isnan(od_travel_time):
                        if od_travel_time >= 0:
                            element = (str(u), str(v), str(od_travel_time))
                            travel_time.append(element)

        if nodes == "all":
            
            #return travel time matrix between all nodes in the network
            
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
                            element = (str(u), str(v), str(od_travel_time))
                            travel_time.append(element)
                    
                    j+=1
                i+=1
        '''



        return travel_time

    def _get_travel_time_from_stops_to_school(self, school_id):

        '''
        computes and returns a list of travel time between all bus stations to school (destination)
        '''
        v = self.schools.loc[school_id, 'osmid_drive']
        travel_time = []

        i = 0
        for index_o, origin_stop in self.bus_stations.iterrows():
            
            u = self.bus_stations.loc[index_o, 'osmid_drive']

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

    def _get_random_coord_circle(self, R, clat, clon):

        '''
        returns random coordinate within the circle bounds
        '''

        fig, ax = ox.plot_graph(self.G_drive, show=False, close=False, node_color='#000000', node_size=6, bgcolor="#ffffff", edge_color="#999999")

        counter = 0
        number = 1
        while counter < 1000:
            r = R * math.sqrt(np.random.uniform(0, 1))  
            theta = np.random.uniform(0, 1) * 2 * math.pi

            lon = clon + r * math.cos(theta)
            lat = clat + r * math.sin(theta)

            ax.scatter(lon, lat, c='red', s=8, marker=",")

            pnt = Point(lon, lat)

            cdist = math.sqrt(((lon - clon) ** 2) + ((lat - clat) ** 2))

            if cdist <= R:
                count += 1
                #return pnt

        plt.show()
        plt.close(fig)

        return (-1, -1)


    def set_zone_bbox(self, zone_id, dist_lat, dist_lon):

        '''
        function to manually set the zone bounding box
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

    def divide_network_grid(self, rows, columns):

        fig, ax = ox.plot_graph(self.G_drive, show=False, close=False, node_color='#000000', node_size=6, bgcolor="#ffffff", edge_color="#999999")

        earth_radius = 6371009  # meters
        blocks = []

        minx, miny, maxx, maxy = self.polygon.bounds

        dist_lat = (maxy-miny)/rows
        dist_lng = (maxx-minx)/columns

        lat = miny
        lng = minx
        for y in range(rows):

            delta_lat = (dist_lat / earth_radius) * (180 / math.pi)
            delta_lat *= 2
            for x in range(columns):

                delta_lng = (dist_lng / earth_radius) * (180 / math.pi) / math.cos(lat * math.pi / 180)
                delta_lng *= 2

                north = lat + delta_lat
                south = lat
                east = lng + delta_lng
                west = lng

                polygon = Polygon([(west, south), (east, south), (east, north), (west, north)])

                d = {
                    'type': 1,
                    'polygon': polygon,
                    'radius': np.nan,
                    'center_y': np.nan,
                    'center_x': np.nan,
                    'origin_weigth': 0,
                    'destination_weigth': 0
                }

                blocks.append(d)
                ax.scatter(lng, lat, c='red', s=8, marker=",")

                lng += delta_lng
            lat += delta_lat            

        self.blocks = pd.DataFrame(blocks)
        plt.show()
        plt.close(fig)

    def add_new_zone(self, center_x, center_y, length_x=0, length_y=0, radius=0, origin_weigth=0, destination_weigth=0):


        for index, row in self.zones.iterrows():

            if (row['center_y'] == center_y) and (row['center_x'] == center_x):
                raise ValueError('another zone with same center coordinates was already added.')

        if (radius > 0) and (length_x > 0):
            raise ValueError('radius and length can not be specified at the same time for a zone')

        if not ((origin_weigth >= 0) and (origin_weigth <= 100)):
            raise ValueError('origin_weigth must be in the interval [0,100]')

        if not ((destination_weigth >= 0) and (destination_weigth <= 100)):
            raise ValueError('destination_weigth must be in the interval [0,100]')

        if (length_x > 0) or (length_y > 0):

            if (length_x == 0) or (length_y == 0):
                raise ValueError('please set both length in x and y coordinates (length_x and length_y)')

            earth_radius = 6371009  # meters
        
            lat = center_y
            lng = center_x

            delta_lat = (length_y / earth_radius) * (180 / math.pi)
            delta_lng = (length_x / earth_radius) * (180 / math.pi) / math.cos(lat * math.pi / 180)
            
            north = lat + delta_lat
            south = lat - delta_lat
            east = lng + delta_lng
            west = lng - delta_lng
            
            polygon = Polygon([(west, south), (east, south), (east, north), (west, north)])

            d = {
                'type': 0,
                'polygon': np.nan,
                'radius': radius,
                'center_y': center_y,
                'center_x': center_x,
                'origin_weigth': origin_weigth,
                'destination_weigth': destination_weigth
            }

        else:

            polygon = np.nan

            d = {
                'type': 1,
                'polygon': np.nan,
                'radius': radius,
                'center_y': center_y,
                'center_x': center_x,
                'origin_weigth': origin_weigth,
                'destination_weigth': destination_weigth
            }   

        

        





