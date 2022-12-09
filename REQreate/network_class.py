import math
import os
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
import geopy.distance

import pyproj
from functools import partial
from shapely import ops

            
class Network:

    def __init__(self, place_name, G_drive, G_walk, polygon, bus_stations):
        
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

        self.node_list_darp = []

       

    def get_eta_walk(self, u, v, walk_speed):
        
        osm_api = osm.OsmApi()
        u = int(u)
        v = int(v)
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
                    print('heeere')
                    distance_walk = nx.dijkstra_path_length(self.G_walk, u, v, weight='length')
                    
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    distance_walk = np.nan
         
        '''
        try:
            distance_walk = nx.dijkstra_path_length(self.G_walk, u, v, weight='length')
            
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            distance_walk = np.nan 
        '''          
            
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

    def _return_estimated_distance_drive(self, origin_node, destination_node):

        #returns estimated distance in meters between origin_node to destination_node
        distance_drive = -1
        try:
            #distance_drive = nx.dijkstra_path_length(self.G_drive, int(origin_node), int(destination_node), weight='length')
            distance_drive = self.shortest_dist_drive.loc[int(origin_node), str(destination_node)]
            
            if str(distance_drive) != 'nan':
                distance_drive = float(distance_drive)
    
        except KeyError:
            distance_drive = -1

        return float(distance_drive)

    def _return_estimated_travel_time_drive(self, origin_node, destination_node):

        #returns estimated travel time in seconds between origin_node to destination_node
        eta = -1
        try:
            
            #travel_time = nx.dijkstra_path_length(self.G_drive, int(origin_node), int(destination_node), weight='travel_time')
            travel_time = self.shortest_dist_drive.loc[int(origin_node), str(destination_node)]
            speed = 5.56 #20kmh remove this later maybe? (look two lines above)
            travel_time = travel_time/speed
           
            #print(travel_timex, travel_time)
            if str(travel_time) != 'nan':
                eta = int(travel_time)

        except KeyError:
            eta = -1

        return int(eta)
        
    def return_estimated_travel_time_bus(self, stops_origin, stops_destination):
        max_eta_bus = -1
        avg_eta_bus = -1
        min_eta_bus = 1000000000

        for origin in stops_origin:
            for destination in stops_destination:


                u = self.bus_stations.loc[origin, 'osmid_drive']
                v = self.bus_stations.loc[destination, 'osmid_drive']

                travel_time = self._return_estimated_travel_time_drive(int(u), int(v))

                eta_bus = travel_time
                
                if not math.isnan(eta_bus):
                    if eta_bus >= 0:
                        if (eta_bus > max_eta_bus):
                            max_eta_bus = eta_bus

                        if (eta_bus < min_eta_bus):
                            min_eta_bus = eta_bus

        return max_eta_bus, min_eta_bus


    def _get_travel_time_matrix(self, nodes, inst=None, node_list=None):
        
        if nodes == "list":
            travel_time = np.ndarray((len(node_list), len(node_list))) 

            for u in range(0, len(node_list)):
                for v in range(0, len(node_list)):

                    od_travel_time = self._return_estimated_travel_time_drive(int(node_list[u]), int(node_list[v]))  

                    if not math.isnan(od_travel_time):
                        travel_time[u][v] = int(od_travel_time)
                        
                        
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

    def _get_random_coord(self, polygon, seed_coord):

        '''
        returns random coordinate within the polygon bounds
        '''
        minx, miny, maxx, maxy = polygon.bounds
        np.random.seed(seed_coord)
        counter = 0
        number = 1
        while counter < number:
            pnt = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
            pnt2 = (pnt.y, pnt.x)
            osmid_drive = ox.nearest_nodes(self.G_drive, pnt.x, pnt.y)
            pnt_osmid = Point(self.G_drive.nodes[osmid_drive]['x'], self.G_drive.nodes[osmid_drive]['y'])
            pnt_osmid2 = (pnt_osmid.y, pnt_osmid.x)
            
            distancepts = geopy.distance.distance(pnt2, pnt_osmid2).m
            if distancepts < 500:
                if polygon.contains(pnt):
                    return pnt

        return Point(-1, -1)

    def _get_random_coord_circle(self, R, clat, clon, seed_coord):

        '''
        returns random coordinate within the circle bounds
        '''
        earth_radius = 6371009  # meters
        np.random.seed(seed_coord)
        counter = 0
        number = 1
        while counter < 1000:
            r = R * math.sqrt(np.random.uniform(0, 1))  
            theta = np.random.uniform(0, 1) * 2 * math.pi

            dist_lng = r * math.cos(theta)
            dist_lat = r * math.sin(theta)

            delta_lat = (dist_lat / earth_radius) * (180 / math.pi)
            delta_lng = (dist_lng / earth_radius) * (180 / math.pi) / math.cos(clat * math.pi / 180)
            
            lon = clon + delta_lng
            lat = clat + delta_lat

            pnt = Point(lon, lat)

            c1 = (clat, clon)
            c2 = (lat, lon)
            cdist = (geopy.distance.distance(c1, c2).km)*1000

            if cdist <= R:
                return pnt

        return (-1, -1)

    def _get_random_coord_radius(self, lat, lon, radius, polygon, seed_coord):

        R = 6378.1 #Radius of the Earth

        counter = 0
        number = 1
        np.random.seed(seed_coord)
        while counter < 1000:
            degree = random.randint(0, 360)
            brng = math.radians(degree)
            #Bearing is degrees converted to radians.
            d = radius/1000 #Distance in km

            lat1 = math.radians(lat) #Current lat point converted to radians
            lon1 = math.radians(lon) #Current long point converted to radians

            lat2 = math.asin( math.sin(lat1)*math.cos(d/R) +
                 math.cos(lat1)*math.sin(d/R)*math.cos(brng))

            lon2 = lon1 + math.atan2(math.sin(brng)*math.sin(d/R)*math.cos(lat1),
                         math.cos(d/R)-math.sin(lat1)*math.sin(lat2))

            lat2 = math.degrees(lat2)
            lon2 = math.degrees(lon2)

            pnt = Point(lon2, lat2)
            
            distancepts = geopy.distance.distance((lat, lon), (lat2, lon2)).m
            
            if polygon.contains(pnt):
                return pnt
            else:
                counter += 1

        return Point(-1, -1)
        
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

    def divide_network_grid(self, rows, columns, save_dir, output_folder_base):

        save_dir_csv = os.path.join(save_dir, 'csv')
        if not os.path.isdir(save_dir_csv):
            os.mkdir(save_dir_csv)

        save_dir_images = os.path.join(save_dir, 'images')
        zones_folder = os.path.join(save_dir_images, 'zones')
        
        if not os.path.isdir(zones_folder):
            os.mkdir(zones_folder)

        path_zones_csv_file = os.path.join(save_dir_csv, output_folder_base+'.zones.csv')

        fig, ax = ox.plot_graph(self.G_drive, show=False, close=False, node_color='#000000', node_size=6, bgcolor="#ffffff", edge_color="#999999")

        zones = []
        earth_radius = 6371009  # meters
        blocks = []

        minx = 9000
        miny = 9000
        maxx = -9000
        maxy = -9000

        for node in self.G_drive.nodes(): 

            lat = self.G_drive.nodes[node]['y']
            lng = self.G_drive.nodes[node]['x']

            if (lng < minx):
                minx = lng

            if (lng > maxx):
                maxx = lng

            if (lat < miny):
                miny = lat

            if (lat > maxy):
                maxy = lat

        lats = np.linspace(miny, maxy, num=rows+1)
        lngs = np.linspace(minx, maxx, num=columns+1)

        print(minx, maxx)
        print(miny, maxy)

        zone_id = 0
        for lat in range(len(lats)-1):
            for lng in range(len(lngs)-1):

                north = lats[lat+1]
                south = lats[lat]
                east = lngs[lng+1]
                west = lngs[lng]

                polygon = Polygon([(west, south), (east, south), (east, north), (west, north)])

                strname = 'zone'+str(zone_id)

                zone_center_point = (polygon.centroid.y, polygon.centroid.x)

                #osmid nearest node walk
                osmid_walk = ox.nearest_nodes(self.G_walk, polygon.centroid.x, polygon.centroid.y) 

                #osmid nearest node drive
                osmid_drive = ox.nearest_nodes(self.G_drive, polygon.centroid.x, polygon.centroid.y)

                pnt = Point(self.G_drive.nodes[osmid_drive]['x'],  self.G_drive.nodes[osmid_drive]['y'])

                if polygon.contains(pnt):
                    ax.scatter(west, south, c='red', s=8, marker=",")
                    ax.scatter(east, south, c='red', s=8, marker=",")
                    ax.scatter(east, north, c='red', s=8, marker=",")
                    ax.scatter(west, north, c='red', s=8, marker=",")

                    ax.scatter(polygon.centroid.x, polygon.centroid.y, c='green', s=8, marker=",")

                    d = {
                        "name": strname,
                        'id': zone_id,
                        'polygon': polygon,
                        'center_y': polygon.centroid.y,
                        'center_x': polygon.centroid.x
                    } 
                    zone_id += 1

                    zones.append(d)

        zones = pd.DataFrame(zones)
        zones.to_csv(path_zones_csv_file)

        if len(zones) > 0:
            zones.set_index(['id'], inplace=True)

        plt.close(fig)
        return zones

    def add_new_zone(self, name, center_x, center_y, length_x=0, length_y=0, radius=0):

        if (radius > 0) and (length_x > 0):
            raise ValueError('radius and length can not be specified at the same time for a zone')

        if (length_x > 0) or (length_y > 0):

            if (length_x == 0) or (length_y == 0):
                raise ValueError('please set both length in x and y coordinates (length_x and length_y)')

            earth_radius = 6371009  # meters
        
            lat = center_y
            lng = center_x

            length_y = length_y/2
            length_x = length_x/2
            delta_lat = (length_y / earth_radius) * (180 / math.pi)
            delta_lng = (length_x / earth_radius) * (180 / math.pi) / math.cos(lat * math.pi / 180)
            
            north = lat + delta_lat
            south = lat - delta_lat
            east = lng + delta_lng
            west = lng - delta_lng
            
            polygon = Polygon([(west, south), (east, south), (east, north), (west, north)])

            d = {
                'name': name,
                'type': 0,
                'polygon': polygon,
                'radius': radius,
                'center_y': center_y,
                'center_x': center_x
            }

            self.zones = self.zones.append(d, ignore_index=True)

        else:

            polygon = np.nan

            d = {
                'name': name,
                'type': 1,
                'polygon': np.nan,
                'radius': radius,
                'center_y': center_y,
                'center_x': center_x
            }

            self.zones = self.zones.append(d, ignore_index=True)   

        
    def add_new_school(self, name, x, y):

        idxs = self.schools.index[self.schools['school_name'] == name].tolist()

        if (len(idxs) > 1):
            raise ValueError('school name must be unique')

        u, v, key = ox.nearest_edges(self.G_walk, x, y)
        school_node_walk = min((u, v), key=lambda n: ox.distance.great_circle_vec(y, x, self.G_walk.nodes[n]['y'], self.G_walk.nodes[n]['x']))
    
        u, v, key = ox.nearest_edges(self.G_drive, x, y)
        school_node_drive = min((u, v), key=lambda n: ox.distance.great_circle_vec(y, x, self.G_drive.nodes[n]['y'], self.G_drive.nodes[n]['x']))

        d = {
            'school_name':name,
            'osmid_walk':school_node_walk,
            'osmid_drive':school_node_drive,
            'lat':y,
            'lon':x,
        }
        self.schools = self.schools.append(d, ignore_index=True)

    def add_new_stop(self, types, x, y):

        u, v, key = ox.nearest_edges(self.G_walk, x, y)
        stop_node_walk = min((u, v), key=lambda n: ox.distance.great_circle_vec(y, x, self.G_walk.nodes[n]['y'], self.G_walk.nodes[n]['x']))
    
        u, v, key = ox.nearest_edges(self.G_drive, x, y)
        stop_node_drive = min((u, v), key=lambda n: ox.distance.great_circle_vec(y, x, self.G_drive.nodes[n]['y'], self.G_drive.nodes[n]['x']))

        d = {
            'osmid_walk':stop_node_walk,
            'osmid_drive':stop_node_drive,
            'lat':y,
            'lon':x,
            'type': types
        }
        self.bus_stations = self.bus_stations.append(d, ignore_index=True)





