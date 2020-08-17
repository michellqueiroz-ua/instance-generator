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
import osmnx as ox
import networkx as nx
import geopandas as gpd
import pandas as pd
#import modin.pandas as pd
#import scipy.stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import osmapi as osm
from math import sqrt
import datetime
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasRegressor
import seaborn as sns
import tensorflow_docs as tfdocs
import glob
from shapely.geometry import Point
from shapely.geometry import MultiPoint
from shapely.geometry import LineString
from shapely.geometry import MultiLineString
from shapely.geometry import MultiPolygon
from shapely.geometry import Polygon
import pickle
from multiprocessing import Pool
from multiprocessing import cpu_count
import time
import gc
import ray
from streamlit import caching

try:
    import tkinter as tk
    from tkinter import filedialog
except:
    pass

#from imports import *
from passenger_requests import generate_requests
from classes import *

        
class JsonConverter(object):

    def __init__(self, file_name):

        with open(file_name, 'r') as file:
            self.json_data = json.load(file)
            file.close()

    def convert_normal(self, output_file_name):

        with open(output_file_name, 'w') as file:

            # first line: number of stations
            #file.write(str(self.json_data.get('num_stations')))
            #file.write('\n')

            # second line - nr station: distance matrix
            #dist_matrix = self.json_data.get('distance_matrix')
            #for row in dist_matrix:
            #    for distance in row:
            #        file.write(str(distance))
            #        file.write('\t')
            #    file.write('\n')

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
                file.write(str(request.get('num_stops_origin')) + '\n')
                for stop in request.get('stops_origin'):
                    file.write(str(stop) + '\t')
                file.write('\n')
                for walking_distance in request.get('walking_time_origin_to_stops'):
                    file.write(str(walking_distance) + '\t')

                file.write('\n')

                # num stops destination + stops destination
                file.write(str(request.get('num_stops_destination')) + '\n')
                for stop in request.get('stops_destination'):
                    file.write(str(stop) + '\t')
                file.write('\n')
                for walking_distance in request.get('walking_time_stops_to_destination'):
                    file.write(str(walking_distance) + '\t')

                file.write('\n')

                # earliest departure time
                file.write(str(request.get('dep_time')))
                file.write('\n')

                # latest arrival time
                file.write(str(request.get('arr_time')))
                file.write('\n')

                #writing the fixed lines
                file.write(str(request.get('num_subway_routes')) + '\n')
                for subway_line in request.get('subway_line_ids'):
                    line_id = str(subway_line)
                    option = request.get('option'+line_id)
                    file.write(str(option) + '\n')
                    file.write(str(request.get('eta_in_vehicle'+line_id)) + '\n')

                    if option == 1:
                        file.write(str(request.get('walking_time_to_pick_up'+line_id)) + '\n')
                        file.write(str(request.get('walking_time_from_drop_off'+line_id)) + '\n')

                    if option == 2:
                        file.write(str(request.get('num_stops_nearby_pick_up'+line_id)) + '\n')
                        file.write(str(request.get('stops_nearby_pick_up'+line_id)) + '\n')
                        file.write(str(request.get('walking_time_to_pick_up'+line_id)) + '\n')
                        file.write(str(request.get('walking_time_from_drop_off'+line_id)) + '\n')

                    if option == 3:
                        file.write(str(request.get('walking_time_to_pick_up'+line_id)) + '\n')
                        file.write(str(request.get('num_stops_nearby_drop_off'+line_id)) + '\n')
                        file.write(str(request.get('stops_nearby_drop_off'+line_id)) + '\n')
                        file.write(str(request.get('walking_time_from_drop_off'+line_id)) + '\n')

                    if option == 4:
                        file.write(str(request.get('num_stops_nearby_pick_up'+line_id)) + '\n')
                        file.write(str(request.get('stops_nearby_pick_up'+line_id)) + '\n')
                        file.write(str(request.get('walking_time_to_pick_up'+line_id)) + '\n')
                        file.write(str(request.get('num_stops_nearby_drop_off'+line_id)) + '\n')
                        file.write(str(request.get('stops_nearby_drop_off'+line_id)) + '\n')
                        file.write(str(request.get('walking_time_from_drop_off'+line_id)) + '\n')


    def convert_localsolver(self, output_file_name):

        with open(output_file_name, 'w') as file:

            # first line: number of stations
            file.write(str(self.json_data.get('num_stations')))
            file.write('\n')

            # second line - nr station: distance matrix
            #dist_matrix = self.json_data.get('distance_matrix')
            #for row in dist_matrix:
            #    for distance in row:
            #        file.write(str(distance))
            #        file.write('\t')
            #    file.write('\n')

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

#endstop locations
def check_adj_nodes(dimensionsu, dimensionsv):

    for i in range(5):
        if abs(dimensionsu[i] - dimensionsv[i]) == 1:
            for k in range(5): 
                if i != k:
                    if dimensionsu[k] == dimensionsv[k]:
                        return True

    return False

def find_new_stop_location(param, network, locations, unc_locations, bus_stops):

    min_total_distance = math.inf
    loc_min_dist = -1

    #print(bus_stops)
    for loc1 in range(len(locations)):
        u = locations[loc1]['osmid_walk']
        #locations[loc1]['total_distance'] = 0
        total_distance = 0
        #print(u)
        #print(loc1)
        if loc1 not in bus_stops:
            #print('here')
            for loc2 in range(len(unc_locations)):
                v = unc_locations[loc2]['osmid_walk']
                #print('fns:', u, v)
                #print(u, v)
                if u != v:
                    #print(u, v)
                    sv = str(v)
                    if str(network.shortest_path_walk.loc[u, sv]) != 'nan':
                        total_distance += int(network.shortest_path_walk.loc[u, sv])
                    else:
                        pass
                        #print('nnnn')
                        #print(network.shortest_path_walk.loc[u, sv])
            
            #print(total_distance)
            #print(min_total_distance)
            if min_total_distance > total_distance:
                min_total_distance = total_distance
                loc_min_dist = loc1

    return loc_min_dist

def assign_location_to_nearest_stop(param, network, locations, bus_stops):

    for loc in range(len(locations)):
        u = locations[loc]['osmid_drive']
        min_dist = math.inf 

        for stop in range(len(bus_stops)):
            loc_stop = bus_stops[stop]
            
            v = locations[loc_stop]['osmid_drive']
            sv = str(v)
            try:
                dist = network.shortest_path_drive.loc[u, sv]
                if str(dist) != 'nan':
                    dist = int(dist)
                    if dist < min_dist:
                        min_dist = dist
                        locations[loc]['nearest_stop'] = stop
            except KeyError:
                pass
    
    locations_assigned_to_stop = [[] for i in range(len(bus_stops))]

    for loc in range(len(locations)):
        stop = locations[loc]['nearest_stop']
        locations_assigned_to_stop[stop].append(loc)

    return locations, locations_assigned_to_stop

def reset_location_stop(param, network, locations, locations_assigned_to_stop, bus_stops):

    #For each stop, reset its location at the location that has the minimum total distance to the other locations that assigned to the stop
    #CHECK IF THE LOCATION IS THE SAME???
    min_total_distance = math.inf
    new_stop = -1

    for loc1 in range(len(locations)):
        u = locations[loc1]['osmid_drive']
        total_distance = 0
        
        if loc1 not in bus_stops:
            for loc2 in range(len(locations_assigned_to_stop)):  
                v = locations[loc2]['osmid_drive']
                if u != v:
                    sv = str(v)
                    try:
                        if str(network.shortest_path_drive.loc[u, sv]) != 'nan':
                            total_distance += int(network.shortest_path_drive.loc[u, sv])
                    except KeyError:
                        pass
            
            if total_distance < min_total_distance:
                min_total_distance = total_distance
                new_stop = loc1

    return new_stop

def k_medoids(param, network, locations, unc_locations, bus_stops):

    decrease_distance = True
    total_distance = math.inf

    locations, locations_assigned_to_stop = assign_location_to_nearest_stop(param, network, locations, bus_stops)

    while decrease_distance:

        for stop in range(len(bus_stops)):
            new_stop = reset_location_stop(param, network, locations, locations_assigned_to_stop[stop], bus_stops)
            bus_stops[stop] = new_stop

        locations, locations_assigned_to_stop = assign_location_to_nearest_stop(param, network, locations, bus_stops)

        #Calculate the sum of distances from all the locations to their nearest stops
        sum_distances = 0
        for loc in range(len(locations)):
            
            u = locations[loc]['osmid_drive']
            nearest_stop = locations[loc]['nearest_stop']
            loc_stop = bus_stops[nearest_stop]
            v = locations[loc_stop]['osmid_drive']
            sv = str(v)
            try:
                if str(network.shortest_path_drive.loc[u, sv]) != 'nan':
                    sum_distances += int(network.shortest_path_drive.loc[u, sv])
            except KeyError:
                pass

        if sum_distances < total_distance:
            total_distance = sum_distances
            decrease_distance = True
        else:
            decrease_distance = False

        print('k_medoids total dist:', total_distance)

    return bus_stops

def loc_is_covered(param, network, loc, locations, bus_stops):

    u = locations[loc]['osmid_walk']
    
    for stop in bus_stops:
        v = locations[stop]['osmid_walk']
        
        if u == v:
            return True

        try:
            
            sv = str(v)
            dist = network.shortest_path_walk.loc[u, sv]
            
            if str(dist) != 'nan':
                walk_time = int(math.ceil(dist/network.walk_speed))
                #print('walk time:', walk_time)
            else:
                walk_time = math.inf

            if walk_time <= param.max_walking:
                #print('covered loc:', walk_time)
                return True
        
        except KeyError:
            #print('kerror')
            pass

    return False

def update_unc_locations(param, network, locations, bus_stops):

    cov_locations = 0
    unc_locations = []

    for loc in range(len(locations)):
        
        if loc_is_covered(param, network, loc, locations, bus_stops):
            cov_locations += 1
        else:
            unc_locations.append(locations[loc])


    total_locations = len(locations)
    pct_cvr = (cov_locations/total_locations)*100

    return unc_locations, pct_cvr

def assign_stop_locations(param, network, cluster, G_clusters):

    bus_stops = []
    k = 1
    locations = []

    for u in cluster:
        
        iu = int(u)
        value = G_clusters.nodes[iu]['osmid_origin_walk']
        if not any(l.get('osmid_walk', None) == value for l in locations):
            lo = {
                'osmid_walk': G_clusters.nodes[iu]['osmid_origin_walk'],
                'osmid_drive': G_clusters.nodes[iu]['osmid_origin_drive'],
                'point': G_clusters.nodes[iu]['origin_point'],
                'total_distance': 0,
                'nearest_stop': -1,
            }
            locations.append(lo)

        value = G_clusters.nodes[iu]['osmid_destination_walk']
        if not any(l.get('osmid_walk', None) == value for l in locations):
            ld = {
                'osmid_walk': G_clusters.nodes[iu]['osmid_destination_walk'],
                'osmid_drive': G_clusters.nodes[iu]['osmid_destination_drive'],
                'point': G_clusters.nodes[iu]['destination_point'],
                'total_distance': 0,
                'nearest_stop': -1,
            }
            locations.append(ld)
                
    print('num of locations', len(locations))

    unc_locations = locations
    new_stop = find_new_stop_location(param, network, locations, unc_locations, bus_stops)
    bus_stops.append(new_stop)
    unc_locations, pct_cvr = update_unc_locations(param, network, locations, bus_stops)

    print('"%" cvr: ', pct_cvr)
    print('num of uncovered loc: ', len(unc_locations))
    print('xxx')

    num_iterations = 0

    while pct_cvr < 75 and num_iterations < 200:
        #num_iterations += 1

        k += 1
        new_stop = find_new_stop_location(param, network, locations, unc_locations, bus_stops)
        bus_stops.append(new_stop)
        
        #adjust location of stops
        bus_stops = k_medoids(param, network, locations, unc_locations, bus_stops)

        unc_locations, pct_cvr = update_unc_locations(param, network, locations, bus_stops)

        print('"%" cvr: ', pct_cvr)
        print('num of uncovered loc: ', len(unc_locations))
        #print(bus_stops)
        #print('xxx')
        num_iterations += 1

    print(bus_stops)
    stop_locations = []
    for stop in bus_stops:
        d = {
            'stop_id': locations[stop]['osmid_drive'],
            'osmid_walk': locations[stop]['osmid_walk'],
            'osmid_drive': locations[stop]['osmid_drive'],
            'lat': locations[stop]['point'].y,
            'lon': locations[stop]['point'].x,
        }
        #stop_locations_point.append(locations[stop]['point'])
        #osmid_walk_stop_locations.append(locations[stop]['osmid_walk'])
        #osmid_drive_stop_locations.append(locations[stop]['osmid_drive'])
        stop_locations.append(d)

    return stop_locations

def cluster_travel_demand(param, network, num_points=20, request_density_treshold=1):
    
    #zone_id = 0
    #for zone in network.zones:
    #distance = zone['center_distance']
    #zone_center_point = (zone['center_point_y'], zone['center_point_x'])
    #ax.scatter(zone['center_point_x'], zone['center_point_y'], c='red')
    #north, south, east, west = ox.bbox_from_point(zone_center_point, distance)
    #polygon = Polygon([(west, south), (east, south), (east, north), (west, north)])

    #SPACE PARTINIONING
    #fig, ax = ox.plot_graph(network.G_walk, show=False, close=False)
    polygon = network.polygon_walk
    minx, miny, maxx, maxy = polygon.bounds

    #ax.scatter(maxx, maxy, c='green')
    #ax.scatter(minx, miny, c='green')            
    #diffy = north - south
    #diffx = east - west

    x_points = np.linspace(minx, maxx, num=num_points)
    y_points = np.linspace(miny, maxy, num=num_points)

    network.space_partition = []
    
    #partinioning the city in smaller zones/areas
    #separate this part, because zone partitioning will only happen once, and the calculation of demand units may vary
    for curr_point_x in range(len(x_points)):
        next_point_x = curr_point_x + 1
        if next_point_x < len(x_points):
            for curr_point_y in range(len(y_points)):
                next_point_y = curr_point_y + 1

                if next_point_y < len(y_points):
                    
                    minx = x_points[curr_point_x]
                    miny = y_points[curr_point_y]

                    maxx = x_points[next_point_x]
                    maxy = y_points[next_point_y]

                    #ax.scatter(maxx, maxy, c='blue')
                    #ax.scatter(minx, miny, c='blue')

                    #polygon represents the mini area
                    mini_polygon = Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])
                    
                    network.space_partition.append(mini_polygon)

    network.units = []

    for origin_polygon in network.space_partition:
        for destination_polygon in network.space_partition:
            d = {
                'origin_polygon': origin_polygon,
                'destination_polygon': destination_polygon,
                'travel_demand': [0] * 24,
            }
            network.units.append(d)


    print('number of units:', len(network.units))

    for request in network.all_requests.values():
        
        hour = int(math.floor(request.get('dep_time')/(3600)))
        
        for i in range(len(network.units)):

            origin_polygon =  network.units[i]['origin_polygon']
            destination_polygon =  network.units[i]['destination_polygon']

            origin_point = Point(request.get('originx'), request.get('originy'))
            destination_point = Point(request.get('destinationx'), request.get('destinationy'))
            
            if origin_polygon.contains(origin_point):
                if destination_polygon.contains(destination_point):
                    network.units[i]['travel_demand'][hour] += 1 
        
    
    print('density threshold:', request_density_treshold)
    #creating graph to cluster
    G_clusters = nx.Graph()
    node_id = 0
    for unit in network.units:
        for hour in range(24):
            if unit['travel_demand'][hour] >= request_density_treshold:
                
                dimensions = []

                uminx, uminy, umaxx, umaxy = unit['origin_polygon'].bounds
                dimensions.append(uminx)
                dimensions.append(uminy)
                origin_point = (uminx, uminy)
                origin_point_inv = (uminy, uminx)
                osmid_origin_walk = ox.get_nearest_node(network.G_walk, origin_point_inv)
                osmid_origin_drive = ox.get_nearest_node(network.G_drive, origin_point_inv)
                #ax.scatter(uminx, uminy, c='red')
                
                uminx, uminy, umaxx, umaxy = unit['destination_polygon'].bounds
                dimensions.append(uminx)
                dimensions.append(uminy)
                destination_point = (uminx, uminy)
                destination_point_inv = (uminy, uminx)
                osmid_destination_walk = ox.get_nearest_node(network.G_walk, destination_point_inv)
                osmid_destination_drive = ox.get_nearest_node(network.G_drive, destination_point_inv)
                #ax.scatter(uminx, uminy, c='red')

                #print(osmid_origin_walk, osmid_destination_walk)

                dimensions.append(hour)

                G_clusters.add_node(node_id, origin_polygon=unit['origin_polygon'], origin_point=origin_point, osmid_origin_walk=osmid_origin_walk, osmid_origin_drive=osmid_origin_drive, destination_polygon=unit['destination_polygon'], destination_point=destination_point, osmid_destination_walk=osmid_destination_walk, osmid_destination_drive=osmid_destination_drive, hour=hour, dimensions=dimensions)
                node_id += 1
    
    for u in G_clusters.nodes():
        for v in G_clusters.nodes():
            
            if u != v:
                adjacent_nodes = check_adj_nodes(G_clusters.nodes[u]['dimensions'], G_clusters.nodes[v]['dimensions'])

                if adjacent_nodes:
                    G_clusters.add_edge(u, v)

    connected_components = nx.connected_components(G_clusters)
    print('connected components:', connected_components)

    stop_locations = []
    for cluster in connected_components:
        if len(cluster) > 1:
            print(cluster)
            print('assign stop locations')
            stop_locations = assign_stop_locations(param, network, cluster, G_clusters)
            print('number of stops', len(stop_locations))
            
            break

    osmid_walk_nodes = []
    osmid_drive_nodes = []
    for stop in stop_locations:
        osmid_walk_nodes.append(stop['osmid_walk'])
        osmid_drive_nodes.append(stop['osmid_drive'])

    stops_folder = os.path.join(param.save_dir_images, 'bus_stops')

    #plot network to show NEW bus stops  
    nc = ['r' if (node in osmid_walk_nodes) else '#336699' for node in network.G_walk.nodes()]
    ns = [12 if (node in osmid_walk_nodes) else 6 for node in network.G_walk.nodes()]
    fig, ax = ox.plot_graph(network.G_walk, show=False, node_size=ns, node_color=nc, node_zorder=2, save=True, filepath=stops_folder+'/_new_stops_walk.png')
        
    #plot network to show NEW bus stops  
    nc = ['r' if (node in osmid_drive_nodes) else '#336699' for node in network.G_drive.nodes()]
    ns = [12 if (node in osmid_drive_nodes) else 6 for node in network.G_drive.nodes()]
    fig, ax = ox.plot_graph(network.G_drive, show=False, node_size=ns, node_color=nc, node_zorder=2, save=True, filepath=stops_folder+'/_new_stops_drive.png')
    #plot the city graph here with the points to see if it is correct
    #center point - green
    #samples - red

    save_dir_csv = os.path.join(param.save_dir, 'csv')
    path_new_bus_stops = os.path.join(save_dir_csv, param.output_file_base+'.new.stops.csv')

    new_stops = pd.DataFrame(stop_locations)
    new_stops.set_index(['stop_id'], inplace=True)
    new_stops.to_csv(path_new_bus_stops)

    travel_time_matrix_new_stops = get_travel_time_matrix_osmnx_csv(param, new_stops, network.shortest_path_drive, network.shortest_path_walk, filename='.travel.time.new.stops.csv')
###start stop locations

def plot_pt_fixed_lines(param, G, pt_fixed_lines):
    
    pt_lines_folder = os.path.join(param.save_dir_images, 'pt_fixed_lines')

    if not os.path.isdir(pt_lines_folder):
        os.mkdir(pt_lines_folder)

    for index, lines in pt_fixed_lines.iterrows():
        #bus_stop_list_nodes.append(stop['osmid_walk'])

        nc = ['r' if (str(node) in lines['osm_nodes']) else '#336699' for node in G.nodes()]
        ns = [12 if (str(node) in lines['osm_nodes']) else 6 for node in G.nodes()]
        fig, ax = ox.plot_graph(G, node_size=ns, show=False, node_color=nc, node_zorder=2, save=True, filepath=pt_lines_folder+'/'+str(index)+'_'+str(lines['name'])+'.pt_fixed_lines.png')
        plt.close(fig)

def get_fixed_lines_csv(param, G_walk, G_drive, polygon):

    api_osm = osm.OsmApi()
    pt_fixed_lines = []

    save_dir_csv = os.path.join(param.save_dir, 'csv')

    if not os.path.isdir(save_dir_csv):
        os.mkdir(save_dir_csv)

    #pt = public transport
    path_pt_lines_csv_file = os.path.join(save_dir_csv, param.output_file_base+'.pt.lines.csv')

    if os.path.isfile(path_pt_lines_csv_file):
        print('is file pt routes')
        pt_fixed_lines = pd.read_csv(path_pt_lines_csv_file)

    else:
        index_line = 0
        print('creating file pt routes')

        tags = {
            #'route_master':'bus',
            'route':'subway',
            'route':'tram',
        }
        
        routes = ox.pois_from_polygon(polygon, tags=tags)

        #print('number of routes', len(routes))

        for index, poi in routes.iterrows():
            
            try:

                keys = poi.keys()
                    
                if str(poi['nodes']) != 'nan':
                    
                    name = "" 
                    ref = []
                    interval = ""
                    duration = ""
                    frequency = ""
                    #distance
                    #roundtrip
                    #operator
        
                    for key in keys:
                        #print(key)

                        if key == "name":
                            name = str(poi[key])

                        if "ref" in key:
                            stref = poi[key]
                            ref.append(stref)

                        if key == "interval":
                            interval = poi[key]

                        if key == "duration":
                            duration = poi[key]

                        if key == "frequency":
                            frequency = poi[key]
                            
                    filtered_nodes_osm = []

                    #fig, ax = ox.plot_graph(G_drive, show=False, close=False)

                    for u in poi['nodes']:
                        nodeu = api_osm.NodeGet(u)
                        node_point = (nodeu['lat'], nodeu['lon'])
                        
                        #ax.scatter(nodeu['lon'], nodeu['lat'], c='blue')
                        #print(node_point)
                        
                        nn = ox.get_nearest_node(G_drive, node_point)
                        
                        if nn not in filtered_nodes_osm:
                            filtered_nodes_osm.append(nn)

                    if len(filtered_nodes_osm) > 1:

                        d = {
                            'index_line': index_line,
                            'name': name,
                            'ref': ref,
                            'osm_nodes': filtered_nodes_osm,
                            'nodes': poi['nodes'],
                            'interval': interval,
                            'duration': duration,
                            'frequency': frequency,
                        }
                        
                        pt_fixed_lines.append(d)

                        index_line += 1
                        #plt.show()
                        #break               
            except KeyError:
                pass

        
        pt_fixed_lines = pd.DataFrame(pt_fixed_lines)
        pt_fixed_lines.to_csv(path_pt_lines_csv_file)

        #plot_pt_fixed_lines(param, G_drive, pt_fixed_lines)
    
    return pt_fixed_lines

def plot_fixed_lines(param, network):

    #plot all nodes in the network that have a fixed line passing by
    fl_stations_walk = [] 
    fl_stations_drive = []

    for node in network.nodes_covered_fixed_lines:

        fl_station_walk = network.deconet_network_nodes.loc[int(node), 'osmid_walk']
        fl_station_drive = network.deconet_network_nodes.loc[int(node), 'osmid_drive']
        
        fl_stations_walk.append(fl_station_walk)
        fl_stations_drive.append(fl_station_drive)

    stops_folder = os.path.join(param.save_dir_images, 'bus_stops')
    nc = ['r' if (node in fl_stations_drive) else '#336699' for node in network.G_drive.nodes()]
    ns = [12 if (node in fl_stations_drive) else 6 for node in network.G_drive.nodes()]
    fig, ax = ox.plot_graph(network.G_drive, node_size=ns, show=False, node_color=nc, node_zorder=2, save=True, filepath=stops_folder+'/fixed_lines_nodes_drive.png')
    plt.close(fig)

    nc = ['r' if (node in fl_stations_walk) else '#336699' for node in network.G_walk.nodes()]
    ns = [12 if (node in fl_stations_walk) else 6 for node in network.G_walk.nodes()]
    fig, ax = ox.plot_graph(network.G_walk, node_size=ns, show=False, node_color=nc, node_zorder=2, save=True, filepath=stops_folder+'/fixed_lines_nodes_walk.png')
    plt.close(fig)


@ray.remote
def find_shortest_path_fl(u, v, fixed_lines):
    #u = int(nodeu['stop_I'])
    #v = int(nodev['stop_I'])
    
    shortest_fixed_line_route = [-1, math.inf]

    for route_id in fixed_lines:
        
        #if (u in fixed_lines[route_id]['route_graph'].nodes()) and (v in fixed_lines[route_id]['route_graph'].nodes()):
        try:
            #calculate shortest path using fixed line of id "route_id" between nodes u and v
            shortest_travel_time = nx.dijkstra_path_length(fixed_lines[route_id]['route_graph'], u, v, weight='duration_avg')
            #print("travel time", shortest_travel_time)
            if shortest_travel_time < shortest_fixed_line_route[1]:
                shortest_fixed_line_route[0] = route_id
                shortest_fixed_line_route[1] = shortest_travel_time
            
        except (nx.NetworkXNoPath, KeyError, nx.NodeNotFound):
            #print("no path")
            pass

    return shortest_fixed_line_route

def get_all_shortest_paths_fix_lines(param, fixed_lines, network_nodes):
    
    ray.shutdown()
    ray.init(num_cpus=param.num_of_cpu)

    print('shortest route fixed lines')
    shortest_path_line = []
    graph_nodes = []

    for route_id in fixed_lines:
        #graph_nodes = fixed_lines[route_id]['route_graph'].nodes()
        for node in fixed_lines[route_id]['route_graph'].nodes():
            if node not in graph_nodes:
                graph_nodes.append(node)

    #print(graph_nodes)


    fixed_lines_id = ray.put(fixed_lines)

    for u in graph_nodes:
        #u = int(nodeu['stop_I'])
        all_shortest_fixed_line_route = ray.get([find_shortest_path_fl.remote(u, v, fixed_lines_id) for v in graph_nodes]) 

        j=0
        #u = int(nodeu['stop_I'])
        #print('current node', u)
        for v in graph_nodes:
            #v = int(nodev['stop_I'])
            
            if all_shortest_fixed_line_route[j][0] != -1:
                row = {}
                #network IDs
                row['origin_Id'] = u
                row['destination_Id'] = v
                row['line_id'] = all_shortest_fixed_line_route[j][0]
                row['eta'] = all_shortest_fixed_line_route[j][1]

                shortest_path_line.append(row)
                j+=1

    return shortest_path_line

@ray.remote
def get_nodes_osm(G_walk, G_drive, lat, lon):

    node_point = (lat, lon)
    #network_nodes.loc[index, 'lat']
                
    u, v, key = ox.get_nearest_edge(G_walk, node_point)
    node_walk = min((u, v), key=lambda n: ox.distance.great_circle_vec(lat, lon, G_walk.nodes[n]['y'], G_walk.nodes[n]['x']))
    
    u, v, key = ox.get_nearest_edge(G_drive, node_point)
    node_drive = min((u, v), key=lambda n: ox.distance.great_circle_vec(lat, lon, G_drive.nodes[n]['y'], G_drive.nodes[n]['x']))
    
    return (node_walk, node_drive)

def get_fixed_lines_deconet(param, network, folder_path):

    #num_of_cpu = cpu_count()
    nodes_covered_fixed_lines = []
    ray.shutdown()
    ray.init(num_cpus=param.num_of_cpu)

    save_dir_csv = os.path.join(param.save_dir, 'csv')

    if not os.path.isdir(folder_path):
        print('folder does not exist')
        return -1

    network_nodes_filename = folder_path+'/network_nodes.csv'
    if os.path.isfile(network_nodes_filename):
        deconet_network_nodes = pd.read_csv(network_nodes_filename, delimiter=";")
        #print(network_nodes.head())
        #print(network_nodes.keys())
        #map the network nodes to open street maps

        G_walk_id = ray.put(network.G_walk)
        G_drive_id = ray.put(network.G_drive)
        #for index, node in network_nodes.iterrows():
        #    all_nodes.append(node)

        osm_nodes = ray.get([get_nodes_osm.remote(G_walk_id, G_drive_id, node['lat'], node['lon']) for index, node in deconet_network_nodes.iterrows()])

        j=0
        deconet_network_nodes['osmid_walk'] = np.nan
        deconet_network_nodes['osmid_drive'] = np.nan
        for index, node in deconet_network_nodes.iterrows():
            
            node_walk = osm_nodes[j][0]
            node_drive = osm_nodes[j][1]

            deconet_network_nodes.loc[index, 'osmid_walk'] = node_walk
            deconet_network_nodes.loc[index, 'osmid_drive'] = node_drive
            j += 1
        
        deconet_network_nodes.set_index('stop_I', inplace=True)

        
        subway_lines_filename = folder_path+'/network_subway.csv'
        print('entering subway lines')
        if os.path.isfile(subway_lines_filename):
            subway_lines = pd.read_csv(subway_lines_filename, delimiter=";")
            #subway_lines.set_index(['from_stop_I', 'to_stop_I'], inplace=True)

            dict_subway_lines = {}

            for index, row in subway_lines.iterrows():
                
                rts = row['route_I_counts'].split(',')
                #print(rts)
                for r in rts:

                    rtuple = r.split(':')
                    route_id = int(rtuple[0]) #id
                    occur = int(rtuple[1]) #number of occurences
                    
                    if route_id not in dict_subway_lines:
                        dict_subway_lines[route_id] = {}
                        dict_subway_lines[route_id]['route_graph'] = nx.DiGraph() #creates a graph for the given line/route

                    if int(row['from_stop_I']) not in dict_subway_lines[route_id]['route_graph'].nodes():
                            dict_subway_lines[route_id]['route_graph'].add_node(row['from_stop_I'])

                    if int(row['to_stop_I']) not in dict_subway_lines[route_id]['route_graph'].nodes():
                        dict_subway_lines[route_id]['route_graph'].add_node(row['to_stop_I'])

                    if row['from_stop_I'] not in nodes_covered_fixed_lines:
                        nodes_covered_fixed_lines.append(int(row['from_stop_I']))

                    if row['to_stop_I'] not in nodes_covered_fixed_lines:
                        nodes_covered_fixed_lines.append(int(row['to_stop_I']))

                    dict_subway_lines[route_id]['route_graph'].add_edge(row['from_stop_I'], row['to_stop_I'], duration_avg=float(row['duration_avg']))

            
            #shortest_path_subway = get_all_shortest_paths_fix_lines(param, dict_subway_lines, deconet_network_nodes)

            #path_csv_file_subway_lines = os.path.join(save_dir_csv, param.output_file_base+'.subway.lines.csv')
            #shortest_path_subway = pd.DataFrame(shortest_path_subway)
            #shortest_path_subway.to_csv(path_csv_file_subway_lines)
            
            #shortest_path_subway.set_index(['origin_Id', 'destination_Id'], inplace=True)
            
        #add network nodes e shortest_path_subway para network file
       
        #network.shortest_path_subway = shortest_path_subway
        
        network.deconet_network_nodes = deconet_network_nodes
        network.nodes_covered_fixed_lines = nodes_covered_fixed_lines
        network.subway_lines = dict_subway_lines

        plot_fixed_lines(param, network)

        tram_lines_filename = folder_path+'/network_tram.csv'
        if os.path.isfile(tram_lines_filename):
            tram_lines = pd.read_csv(tram_lines_filename, delimiter=";")

        bus_lines_filename = folder_path+'/network_bus.csv'
        if os.path.isfile(bus_lines_filename):
            bus_lines = pd.read_csv(bus_lines_filename, delimiter=";")

def get_zones_csv(param, G_walk, G_drive, polygon):

    zones = []
    zone_id = 0

    save_dir_csv = os.path.join(param.save_dir, 'csv')

    if not os.path.isdir(save_dir_csv):
        os.mkdir(save_dir_csv)

    zones_folder = os.path.join(param.save_dir_images, 'zones')

    if not os.path.isdir(zones_folder):
        os.mkdir(zones_folder)

    path_zones_csv_file = os.path.join(save_dir_csv, param.output_file_base+'.zones.csv')

    if os.path.isfile(path_zones_csv_file):
        
        print('is file zones')
        zones = pd.read_csv(path_zones_csv_file)

        #updates the polygons
        print('updating polygon')
        for index, zone in zones.iterrows():

            distance = zone['center_point_distance'] 
            zone_center_point = (zone['center_point_y'], zone['center_point_x'])
                        
            north, south, east, west = ox.utils_geo.bbox_from_point(zone_center_point, distance)
            polygon = Polygon([(west, south), (east, south), (east, north), (west, north)])
            
            zones.loc[index, 'polygon'] = polygon
        
    else:

        print('creating file zones')

        tags = {
            'place':'borough',
            'place':'suburb',
            'place':'quarter',
            'place':'neighbourhood',
        }
        
        poi_zones = ox.pois_from_polygon(polygon, tags=tags)
        print('poi zones len', len(poi_zones))

        if len(poi_zones) > 0:

            for index, poi in poi_zones.iterrows():
                if str(poi['name']) != 'nan':
                    zone_name = str(poi['name'])
                    
                    if not any((z.get('name', None) == zone_name) for z in zones):
                       
                        #future: see what to do with geometries that are not points
                        if poi['geometry'].geom_type == 'Point':
 
                            distance = 1000 
                            zone_center_point = (poi.geometry.centroid.y, poi.geometry.centroid.x)
                            
                            #osmid nearest node walk
                            osmid_walk = ox.get_nearest_node(G_walk, zone_center_point) 

                            #osmid nearest node drive
                            osmid_drive = ox.get_nearest_node(G_drive, zone_center_point)

                            
                            north, south, east, west = ox.utils_geo.bbox_from_point(zone_center_point, distance)
                            polygon = Polygon([(west, south), (east, south), (east, north), (west, north)])

                            #plot here the center point zone in the walk network
                            nc = ['r' if (node == osmid_walk) else '#336699' for node in G_walk.nodes()]
                            ns = [16 if (node == osmid_walk) else 1 for node in G_walk.nodes()]
                            zone_filename = str(zone_id)+'_'+zone_name+'_walk.png'
                            fig, ax = ox.plot_graph(G_walk, node_size=ns, show=False, node_color=nc, node_zorder=2, save=True, filepath=zones_folder+'/'+zone_filename)
                            plt.close(fig)

                            #plot here the center point zone in the drive network
                            nc = ['r' if (node == osmid_drive) else '#336699' for node in G_drive.nodes()]
                            ns = [16 if (node == osmid_drive) else 1 for node in G_drive.nodes()]
                            zone_filename = str(zone_id)+'_'+zone_name+'_drive.png'
                            fig, ax = ox.plot_graph(G_drive, node_size=ns, show=False, node_color=nc, node_zorder=2, save=True, filepath=zones_folder+'/'+zone_filename)
                            plt.close(fig)

                            n = {
                                'index': index,
                                'id': zone_id,
                                'name': zone_name,
                                'polygon': polygon,
                                'center_point_y': poi.geometry.centroid.y,
                                'center_point_x': poi.geometry.centroid.x,
                                'osmid_walk': osmid_walk,
                                'osmid_drive': osmid_drive,
                                'center_point_distance': distance,
                            }

                            zone_id += 1

                            zones.append(n)
                
            zones = pd.DataFrame(zones)
            zones.to_csv(path_zones_csv_file)
    
    if len(zones) > 0:
        zones.set_index(['id'], inplace=True)

    return zones

@ray.remote
def shortest_path_nx(G, u, v):

    try:
        shortest_path_length = nx.dijkstra_path_length(G, u, v, weight='length')
        return shortest_path_length
    except nx.NetworkXNoPath:
        return -1

@ray.remote
def shortest_path_nx_ss(G, u):

    shortest_path_length_u = {}
    shortest_path_length_u = nx.single_source_dijkstra_path_length(G, u, weight='length')
    return shortest_path_length_u

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def get_distance_matrix_csv(param, G_walk, G_drive, bus_stops):
    shortest_path_walk = []
    shortest_path_drive = []
    
    ray.shutdown()
    ray.init(num_cpus=param.num_of_cpu)
   
    save_dir_csv = os.path.join(param.save_dir, 'csv')
    if not os.path.isdir(save_dir_csv):
        os.mkdir(save_dir_csv)
    
    path_dist_csv_file_walk = os.path.join(save_dir_csv, param.output_file_base+'.dist.walk.csv')
    path_dist_csv_file_drive = os.path.join(save_dir_csv, param.output_file_base+'.dist.drive.csv')
    
    shortest_path_drive = pd.DataFrame()
    shortest_path_walk = pd.DataFrame()

    '''
    if os.path.isfile(path_dist_csv_file_walk):
        print('is file dist walk')
        shortest_path_walk = pd.read_csv(path_dist_csv_file_walk)
        shortest_path_walk.set_index(['osmid_origin'], inplace=True)
    else:

        bus_stops_ids = bus_stops['osmid_walk'].tolist()

        print('calculating distance matrix walk network')
        count_divisions = 0

        list_nodes = list(G_walk.nodes)
        G_walk_id = ray.put(G_walk)

        for u in bus_stops_ids:
            shortest_path_length_walk = []
            results = ray.get([shortest_path_nx.remote(G_walk_id, u, v) for v in list_nodes])

        print(results)

        j=0
        for u in bus_stops_ids:
            d = {}
            d['osmid_origin'] = u
            for v in list_nodes:
                #dist_uv = -1
                #try:
                dist_uv = int(results[j])
                #except KeyError:
                #    pass
                if dist_uv != -1:
                    sv = str(v)
                    dist[sv] = dist_uv
            shortest_path_length_walk.append(d)
            j+=1

        shortest_path_walk = shortest_path_walk.append(shortest_path_length_walk, ignore_index=True)
        del shortest_path_length_walk
        del results
        gc.collect()

        for u in list_nodes:
            shortest_path_length_walk = []
            results = ray.get([shortest_path_nx.remote(G_walk_id, u, v) for v in bus_stops_ids])

        #print(results)

        j=0
        for u in list_nodes:
            d = {}
            d['osmid_origin'] = u
            for v in bus_stops_ids:
                #dist_uv = -1
                #try:
                dist_uv = int(results[j])
                #except KeyError:
                #    pass
                if dist_uv != -1:
                    sv = str(v)
                    dist[sv] = dist_uv
            shortest_path_length_walk.append(d)
            j+=1

        shortest_path_walk = shortest_path_walk.append(shortest_path_length_walk, ignore_index=True)
        del shortest_path_length_walk
        del results
        gc.collect()
        '''





    if os.path.isfile(path_dist_csv_file_drive):
        print('is file dist drive')
        shortest_path_drive = pd.read_csv(path_dist_csv_file_drive)
        shortest_path_drive.set_index(['osmid_origin'], inplace=True)
    else:

        print('calculating shortest paths drive network')

        
        list_nodes = list(G_drive.nodes)
        G_drive_id = ray.put(G_drive)
        start = time.process_time()

        shortest_path_length_drive = []
        results = ray.get([shortest_path_nx_ss.remote(G_drive_id, u) for u in list_nodes])

        j=0
        for u in list_nodes:
            d = {}
            d['osmid_origin'] = u
            for v in G_drive.nodes():
                
                dist_uv = -1
                try:
                    dist_uv = int(results[j][v])
                except KeyError:
                    pass
                if dist_uv != -1:
                    sv = str(v)
                    d[sv] = dist_uv
            shortest_path_length_drive.append(d)

            j+=1
            del d

        shortest_path_drive = pd.DataFrame(shortest_path_length_drive)
        del shortest_path_length_drive
        del results
        gc.collect()

        shortest_path_drive.to_csv(path_dist_csv_file_drive)
        shortest_path_drive.set_index(['osmid_origin'], inplace=True)

    return shortest_path_walk, shortest_path_drive

def filter_bus_stops(param, bus_stops, shortest_path_drive):

    save_dir_csv = os.path.join(param.save_dir, 'csv')
    path_bus_stops = os.path.join(save_dir_csv, param.output_file_base+'.stops.csv')
    print('number of bus stops before cleaning: ', len(bus_stops))

    useless_bus_stop = True
    while useless_bus_stop:
        useless_bus_stop = False
        for index1, stop1 in bus_stops.iterrows():
            unreachable_nodes = 0
            for index2, stop2 in bus_stops.iterrows():
                try:
                    osmid_origin_stop = stop1['osmid_drive']
                    osmid_destination_stop = stop2['osmid_drive']
                    sosmid_destination_stop = str(osmid_destination_stop)
                    if str(shortest_path_drive.loc[osmid_origin_stop, sosmid_destination_stop]) != 'nan':
                        path_length = int(shortest_path_drive.loc[osmid_origin_stop, sosmid_destination_stop])
                    else:
                        unreachable_nodes = unreachable_nodes + 1
                except KeyError:
                    unreachable_nodes = unreachable_nodes + 1
                
            if unreachable_nodes == len(bus_stops) - 1:
                bus_stops = bus_stops.drop(index1)
                useless_bus_stop = True
            
    print('number of bus stops after removal: ', len(bus_stops))

    bus_stops.to_csv(path_bus_stops)

@ray.remote
def get_bus_stop(G_walk, G_drive, index, poi):

    if poi['highway'] == 'bus_stop':
        bus_stop_point = (poi.geometry.centroid.y, poi.geometry.centroid.x)
        
        u, v, key = ox.get_nearest_edge(G_walk, bus_stop_point)
        bus_stop_node_walk = min((u, v), key=lambda n: ox.distance.great_circle_vec(poi.geometry.centroid.y, poi.geometry.centroid.x, G_walk.nodes[n]['y'], G_walk.nodes[n]['x']))
        
        u, v, key = ox.get_nearest_edge(G_drive, bus_stop_point)
        bus_stop_node_drive = min((u, v), key=lambda n: ox.distance.great_circle_vec(poi.geometry.centroid.y, poi.geometry.centroid.x, G_drive.nodes[n]['y'], G_drive.nodes[n]['x']))
        
        d = {
            'stop_id': index,
            'osmid_walk': bus_stop_node_walk,
            'osmid_drive': bus_stop_node_drive,
            'lat': poi.geometry.centroid.y,
            'lon': poi.geometry.centroid.x,
            #'itid': -1
        }

        return d

def get_bus_stops_matrix_csv(param, G_walk, G_drive, polygon_drive):

    ray.shutdown()
    ray.init(num_cpus=param.num_of_cpu)

    save_dir_csv = os.path.join(param.save_dir, 'csv')

    if not os.path.isdir(save_dir_csv):
        os.mkdir(save_dir_csv)

    path_bus_stops = os.path.join(save_dir_csv, param.output_file_base+'.stops.csv')

    if os.path.isfile(path_bus_stops):
        print('is file bus stops')
        bus_stops = pd.read_csv(path_bus_stops)
        bus_stops.set_index(['stop_id'], inplace=True)

    else:
        start = time.process_time()
        print('creating file bus stops')

        #retrieve bus stops
        tags = {
            'highway':'bus_stop',
        }
        poi_bus_stops = ox.pois_from_polygon(polygon_drive, tags=tags)

        G_walk_id = ray.put(G_walk)
        G_drive_id = ray.put(G_drive)
        bus_stops = ray.get([get_bus_stop.remote(G_walk_id, G_drive_id, index, poi) for index, poi in poi_bus_stops.iterrows()]) 
         
        '''
        bus_stops = []
        for index, poi in poi_bus_stops.iterrows():
            if poi['highway'] == 'bus_stop':
                bus_stop_point = (poi.geometry.centroid.y, poi.geometry.centroid.x)
                
                u, v, key = ox.get_nearest_edge(G_walk, bus_stop_point)
                bus_stop_node_walk = min((u, v), key=lambda n: ox.distance.great_circle_vec(poi.geometry.centroid.y, poi.geometry.centroid.x, G_walk.nodes[n]['y'], G_walk.nodes[n]['x']))
                
                u, v, key = ox.get_nearest_edge(G_drive, bus_stop_point)
                bus_stop_node_drive = min((u, v), key=lambda n: ox.distance.great_circle_vec(poi.geometry.centroid.y, poi.geometry.centroid.x, G_drive.nodes[n]['y'], G_drive.nodes[n]['x']))
                
                d = {
                    'stop_id': index,
                    'osmid_walk': bus_stop_node_walk,
                    'osmid_drive': bus_stop_node_drive,
                    'lat': poi.geometry.centroid.y,
                    'lon': poi.geometry.centroid.x,
                    #'itid': -1
                }
            
                bus_stops.append(d)
        '''

        bus_stops = pd.DataFrame(bus_stops)
        bus_stops.set_index(['stop_id'], inplace=True)
        #print("total time", time.process_time() - start)

        drop_index_list = []
        for index1, stop1 in bus_stops.iterrows():
            if index1 not in drop_index_list:
                for index2, stop2 in bus_stops.iterrows():
                    if index2 not in drop_index_list:
                        if index1 != index2:
                            if stop1['osmid_drive'] == stop2['osmid_drive'] and stop1['osmid_walk'] == stop2['osmid_walk']:
                                drop_index_list.append(index2)

        for index_to_drop in drop_index_list:
            bus_stops = bus_stops.drop(index_to_drop)

    return bus_stops

@ray.remote
def calc_travel_time_od(param, origin, destination, shortest_path_drive, bus_stops):

    #curr_weight = 'travel_time_' + str(hour)
    #curr_weight = 'travel_time' 
    row = {}
    row['origin_id'] = origin
    row['destination_id'] = destination
    eta = -1
    
    try:
        origin = bus_stops.loc[origin, 'osmid_drive']
        sdestination = str(bus_stops.loc[destination, 'osmid_drive'])

        distance = shortest_path_drive.loc[origin, sdestination]
        if str(distance) != 'nan':
            distance = int(distance)
            eta = int(math.ceil(distance/param.vehicle_speed))
    except KeyError:
        pass

    if eta >= 0:
        row['eta'] = eta
        row['dist'] = distance
        #return eta
    else:
        row['eta'] = np.nan
        row['dist'] = np.nan

    return row

#not time dependent. for a time dependent create other function later
def get_travel_time_matrix_osmnx_csv(param, bus_stops, shortest_path_drive, shortest_path_walk, filename=None): 
    
    travel_time_matrix = []
    counter = 0
    save_dir_csv = os.path.join(param.save_dir, 'csv')

   
    #ray.init(num_cpus=num_of_cpu)
    
    ray.shutdown()
    ray.init(num_cpus=param.num_of_cpu)

    if not os.path.isdir(save_dir_csv):
        os.mkdir(save_dir_csv)

    if filename is None:
        path_csv_file = os.path.join(save_dir_csv, param.output_file_base+'.travel.time.csv')
    else:
        path_csv_file = os.path.join(save_dir_csv, param.output_file_base+filename)


    if os.path.isfile(path_csv_file):
        print('is file travel time')
        travel_time_matrix = pd.read_csv(path_csv_file)
        #print('rows travel time', len(travel_time_matrix))
    else:
        print('creating file estimated travel time')
        start = time.process_time()
        #travel_time_matrix = pd.DataFrame()
        
        list_nodes = []
        for index, row in bus_stops.iterrows():
            list_nodes.append(index)

        #shortest_path_drive2 = pd2.DataFrame(shortest_path_drive)
        shortest_path_drive_id = ray.put(shortest_path_drive)

        param_id = ray.put(param)

        #bus_stop2 = pd2.DataFrame(bus_stops)
        bus_stops_id = ray.put(bus_stops)
        
        #for origin in list_nodes:

        for origin in list_nodes:            
            #for group_nodes in chunker(list_nodes, num_of_cpu*4):

            #not parallel
            '''
            for destination in list_nodes:
                counter += 1
                row = {}
                row['stop_origin_osmid'] = bus_stops.loc[origin, 'osmid_drive']
                row['stop_destination_osmid'] = bus_stops.loc[destination, 'osmid_drive']
                row['eta'] = calc_travel_time_od(param, origin, destination, shortest_path_drive)
                travel_time_matrix = travel_time_matrix.append(row, ignore_index=True)
                del row
            '''

            #with multiprocessing
            '''
            pool = Pool(processes=num_of_cpu)
            results = pool.starmap(calc_travel_time_od, [(param, origin, destination, shortest_path_drive, 0) for destination in list_nodes])
            pool.close()
            pool.join()

            j=0
            for destination in list_nodes:
                counter += 1
                row = {}
                #row['stop_origin_id'] = bus_stops.loc[origin, 'itid']
                #row['stop_destination_id'] = bus_stops.loc[destination, 'itid']
                row['stop_origin_osmid'] = bus_stops.loc[origin, 'osmid_drive']
                row['stop_destination_osmid'] = bus_stops.loc[destination, 'osmid_drive']
                row['eta'] = results[j]
                travel_time_matrix = travel_time_matrix.append(row, ignore_index=True)
                j += 1
            '''

            #with ray
            #print("here")
            results = ray.get([calc_travel_time_od.remote(param_id, origin, destination, shortest_path_drive_id, bus_stops_id) for destination in list_nodes])
            for row in results:
                #travel_time_matrix = travel_time_matrix.append(row, ignore_index=True)
                travel_time_matrix.append(row)
                counter += 1

            del results
            #print("out")

            '''
            j=0
            for destination in list_nodes:
                counter += 1
                row = {}
                row['origin_osmid'] = bus_stops.loc[origin, 'osmid_drive']
                row['destination_osmid'] = bus_stops.loc[destination, 'osmid_drive']
                row['eta'] = results[j]
                travel_time_matrix = travel_time_matrix.append(row, ignore_index=True)
                j += 1
                del row
            '''

            #print('paths so far', counter)
            #print("total time so far", time.process_time() - start)
            #del results
            gc.collect()    
                               
        travel_time_matrix = pd.DataFrame(travel_time_matrix)
        travel_time_matrix.to_csv(path_csv_file)
        #print("total time", time.process_time() - start)

    travel_time_matrix.set_index(['origin_id', 'destination_id'], inplace=True)
    return travel_time_matrix
       
def get_max_speed_road(dict_edge):
    #returns the max speed in m/s
    try:
        if type(dict_edge['maxspeed']) is not list:
            speed = dict_edge['maxspeed'].split(" ", 1)
            if speed[0].isdigit():
                max_speed = int(speed[0])

                try:
                    if speed[1] == 'mph':
                        #print('mph')
                        max_speed = max_speed/2.237
                    else:
                        if speed[1] == 'knots':
                            max_speed = max_speed/1.944

                except IndexError:
                    #kph
                    max_speed = max_speed/3.6

                return max_speed
            else:
                return np.nan
        else:
            max_speed_avg = 0
            for speed in dict_edge['maxspeed']:
                speed = speed.split(" ", 1)
                if speed[0].isdigit():

                    max_speed = int(speed[0])
                
                    try:
                        if speed[1] == 'mph':
                            #print('mph')
                            max_speed = max_speed/2.237
                        else:
                            if speed[1] == 'knots':
                                max_speed = max_speed/1.944

                    except IndexError:
                        #kph
                        max_speed = max_speed/3.6

                    max_speed_avg = max_speed_avg + max_speed

            max_speed_avg = int(max_speed_avg/len(dict_edge['maxspeed']))
            
            if max_speed_avg > 0:
                return max_speed_avg
            else:
                return np.nan
            
    except KeyError:
        return np.nan
        
    return np.nan

def calc_mean_max_speed(dict_edge, max_speed_mean_overall, counter_max_speeds):
    #returns the max speed in m/s
    try:
        if type(dict_edge['maxspeed']) is not list:
            speed = dict_edge['maxspeed'].split(" ", 1)
            if speed[0].isdigit():
                max_speed = int(speed[0])

                try:
                    if speed[1] == 'mph':
                        #print('mph')
                        max_speed = max_speed/2.237
                    else:
                        if speed[1] == 'knots':
                            max_speed = max_speed/1.944

                except IndexError:
                    #kph
                    max_speed = max_speed/3.6

                '''
                if speed[1] == 'mph':
                    max_speed = max_speed/2.237
                else:
                    if speed[1] == 'knots':
                        max_speed = max_speed/1.944
                    else:
                        #kph
                        max_speed = max_speed/3.6
                '''

                max_speed_mean_overall = max_speed_mean_overall + max_speed
                counter_max_speeds = counter_max_speeds + 1
            
        else:
            
            for speed in dict_edge['maxspeed']:
                speed = speed.split(" ", 1)
                if speed[0].isdigit():
                    max_speed = int(speed[0])
                    
                    try:
                        if speed[1] == 'mph':
                            #print('mph')
                            max_speed = max_speed/2.237
                        else:
                            if speed[1] == 'knots':
                                max_speed = max_speed/1.944

                    except IndexError:
                        #kph
                        max_speed = max_speed/3.6

                    max_speed_mean_overall = max_speed_mean_overall + max_speed
                    counter_max_speeds = counter_max_speeds + 1

            #max_speed_avg = int(max_speed_avg/len(dict_edge['maxspeed']))
            
            ##if max_speed_avg > 0:
            #    return max_speed_avg
            #else:
            #    return np.nan
            
    except KeyError:
        pass
        
    return max_speed_mean_overall, counter_max_speeds
    #return np.nan

#function for returning the neural network model
def create_nn_model(init_mode='normal', activation='relu', dropout_rate=0.0, weight_constraint=0):

    nn_model = Sequential()
    nn_model.add(Dense(64, input_dim=10, kernel_initializer=init_mode, activation=activation, kernel_constraint=maxnorm(weight_constraint)))
    nn_model.add(Dropout(dropout_rate))
    nn_model.add(Dense(1, kernel_initializer=init_mode))
    #nn_model.add(Dense(1, activation="linear")) #regressor neuron?
    nn_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'mse'])
    
    return nn_model

def get_num_lanes(dict_edge):

    num_lanes = np.nan
    try:

        if type(dict_edge['lanes']) is not list:
            num_lanes = dict_edge['lanes']
            if num_lanes.isdigit():
                num_lanes = int(num_lanes)
                return num_lanes
        else:
            avg_num_lanes = 0
            for lanes in dict_edge['lanes']:
                if lanes.isdigit():
                    avg_num_lanes = avg_num_lanes + int(lanes)
            avg_num_lanes = int(avg_num_lanes/len(dict_edge['lanes']))
            if avg_num_lanes > 0:
                num_lanes = avg_num_lanes
                return num_lanes

    except KeyError:
        pass

    return num_lanes

def get_highway_info(dict_edge):

    try:

        if type(dict_edge['highway']) is not list:
            if dict_edge['highway'] == 'motorway' or dict_edge['highway'] == 'motorway_link':
                return 1
            if dict_edge['highway'] == 'trunk' or dict_edge['highway'] == 'trunk_link':
                return 2
            if dict_edge['highway'] == 'primary' or dict_edge['highway'] == 'primary_link':
                return 3
            if dict_edge['highway'] == 'secondary' or dict_edge['highway'] == 'secondary_link': 
                return 4
            if dict_edge['highway'] == 'tertiary' or dict_edge['highway'] == 'tertiary_link':
                return 5
            if dict_edge['highway'] == 'unclassified':
                return 6
            if dict_edge['highway'] == 'residential':
                return 7
            if dict_edge['highway'] == 'living_street':
                return 8
            if dict_edge['highway'] == 'service':
                return 9
            if dict_edge['highway'] == 'pedestrian':
                return 10
            if dict_edge['highway'] == 'track':
                return 11
            if dict_edge['highway'] == 'road':
                return 12

        else:
            #print(dict_edge['highway'])
            pass

    except KeyError:
        pass

    return np.nan

def get_uber_speed_data_prediction_groupby(G_drive, speed_data):

    api = osm.OsmApi()
    #load speed data from csv files
    path = speed_data
    all_files = glob.glob(os.path.join(path, "*.csv"))
    df_from_each_file = (pd.read_csv(f) for f in all_files)
    uber_data = pd.concat(df_from_each_file, ignore_index=True)

    #uber_data = pd.read_csv("../uber_movement/movement-speeds-hourly-cincinnati-2019-10.csv")
    unique_nodes = pd.unique(uber_data[['osm_start_node_id', 'osm_end_node_id']].values.ravel('K'))
    
    #print(concatenated_df.head())
    #print(uber_data.head())
    
    print("start number of rows", len(uber_data))

    nodes_in_graph = [] #nodes that are mapped by the uber movement speed database
    for node in unique_nodes:
        if node in G_drive.nodes():
            if node not in nodes_in_graph:
                nodes_in_graph.append(node)

    uber_data = uber_data[uber_data['osm_start_node_id'].isin(nodes_in_graph) & uber_data['osm_end_node_id'].isin(nodes_in_graph)]
    
    print("mid number of rows", len(uber_data))

    #talvez pegar o dia e fazer aqui seg, terça etc e dar groupby com isso tb?
    
    #add day of the week (monday, tuesday, wednesday, thursday, friday, saturday, sunday etc) info
    #add new column? job day? weekend? holiday?
    #these columns are created based on info that might help, such as peak hours etc
    unique_days = pd.unique(uber_data[['day']].values.ravel('K'))
    unique_months = pd.unique(uber_data[['month']].values.ravel('K'))
    unique_years = pd.unique(uber_data[['year']].values.ravel('K'))

    #add day info
    uber_data["week_day"] = np.nan
    for year in unique_years:
        for month in unique_months:
            for day in unique_days:
                try:
                    ans = datetime.date(year, month, day).weekday()
                    uber_data.loc[(uber_data['day'] == day) & (uber_data['month'] == month) & (uber_data['year'] == year), 'week_day'] = ans
                except ValueError:
                    pass
    

    uber_data = uber_data.groupby(['osm_start_node_id','osm_end_node_id', 'hour', 'week_day'], as_index=False)['speed_mph_mean'].mean()
    uber_data = pd.DataFrame(uber_data)
    
    print("mid number of rows (after grouby)", len(uber_data))

    print(uber_data.head())
    print(list(uber_data.columns))

    #add lat/lon info and max speed
    #in this part info from openstreetmaps is added
    uber_data["start_node_y"] = np.nan
    uber_data["start_node_x"] = np.nan
    uber_data["end_node_y"] = np.nan
    uber_data["end_node_x"] = np.nan
    uber_data["max_speed_mph"] = np.nan
    uber_data["num_lanes"] = np.nan
    uber_data["highway"] = np.nan

    #unique_highway_str = pd.unique(uber_data[['highway']].values.ravel('K'))
    #print(unique_highway_str)

    for (u,v,k) in G_drive.edges(data=True):
        try:
            uber_data.loc[uber_data['osm_start_node_id'] == u, 'start_node_y'] = G_drive.nodes[u]['y']
            uber_data.loc[uber_data['osm_start_node_id'] == u, 'start_node_x'] = G_drive.nodes[u]['x']

            uber_data.loc[uber_data['osm_end_node_id'] == v, 'end_node_y'] = G_drive.nodes[v]['y']
            uber_data.loc[uber_data['osm_end_node_id'] == v, 'end_node_x'] = G_drive.nodes[v]['x']

            if (uber_data.loc[uber_data['osm_start_node_id'] == u, 'start_node_y'] == 0.0).all():
                nodeu = api.NodeGet(u)
                uber_data.loc[uber_data['osm_start_node_id'] == u, 'start_node_y'] = nodeu['lat']
                uber_data.loc[uber_data['osm_start_node_id'] == u, 'start_node_x'] = nodeu['lon']

            if (uber_data.loc[uber_data['osm_end_node_id'] == v, 'end_node_y'] == 0.0).all():
                nodev = api.NodeGet(v)
                uber_data.loc[uber_data['osm_end_node_id'] == v, 'end_node_y'] = nodev['lat']
                uber_data.loc[uber_data['osm_end_node_id'] == v, 'end_node_x'] = nodev['lon']

        except KeyError:
            pass

        #add atribute max speed 
        dict_edge = {}
        dict_edge = G_drive.get_edge_data(u, v)
        dict_edge = dict_edge[0]
        
        max_speed = get_max_speed_road(dict_edge)
        num_lanes = get_num_lanes(dict_edge)
        #highway
        highway_type = get_highway_info(dict_edge)
        uber_data.loc[(uber_data['osm_start_node_id'] == u) & (uber_data['osm_end_node_id'] == v), 'max_speed_mph'] = max_speed

        uber_data.loc[(uber_data['osm_start_node_id'] == u) & (uber_data['osm_end_node_id'] == v), 'num_lanes'] = num_lanes

        uber_data.loc[(uber_data['osm_start_node_id'] == u) & (uber_data['osm_end_node_id'] == v), 'highway'] = highway_type

    #print("min hour", uber_data["hour"].min())
    #print("max hour", uber_data["hour"].max())
    
    uber_data["period_day"] = np.nan
    uber_data.loc[(uber_data['hour'] >= 0) & (uber_data['hour'] <= 6), 'period_day'] = 1 #before peak hours - morning
    uber_data.loc[(uber_data['hour'] >= 7) & (uber_data['hour'] <= 9), 'period_day'] = 2 #peak hours - morning
    uber_data.loc[(uber_data['hour'] >= 10) & (uber_data['hour'] <= 16), 'period_day'] = 3 #before peak hours - afternoon
    uber_data.loc[(uber_data['hour'] >= 17) & (uber_data['hour'] <= 20), 'period_day'] = 4 #peak hours - afternoon 
    uber_data.loc[(uber_data['hour'] >= 21) & (uber_data['hour'] <= 23), 'period_day'] = 5 # night period
    
    #upstream and downstream roads??
    #uber_data["adjacent_roads_speed"] = np.nan

    print(uber_data.isna().sum())
    #clean NA values
    print("clean NA values")
    uber_data = uber_data.dropna()

    uber_data["num_lanes"] = pd.to_numeric(uber_data["num_lanes"])

    #print("min lanes", uber_data["num_lanes"].min())
    #print("max lanes", uber_data["num_lanes"].max())

    #print("min highway", uber_data["highway"].min())
    #print("max highway", uber_data["highway"].max())

    print("end number of rows", len(uber_data))

    print(list(uber_data.columns))
    #sns.pairplot(uber_data[["max_speed_mph", "num_lanes", "highway"]], diag_kind="kde")
    
    #print(uber_data.head())
    print(uber_data.dtypes)        
    
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    #scaler = MinMaxScaler(feature_range=(0, 1))

    #columns with groupby
    #0 ID
    #1 ID
    #2 'hour'
    #3 day of the week (monday...)
    #4 'speed' (TARGET)
    #5,6,7,8 lat/lon of nodes representing the road
    #9 - max_speed_mph
    #10 - number of lanes
    #11 - highway
    #12 -  period of the day
    
    #divide in attributes and labels
    X = uber_data.iloc[:, [2,3,5,6,7,8,9,10,11,12]].values
    y = uber_data.iloc[:, 4].values 
    scaler = StandardScaler()

    #columns without groupby
    #0 - year
    #1 - month
    #2 - day
    #3 - hour
    #4 - utc_timestamp
    #5,6,7,8,9,10 - IDs
    #11 - speed (TARGET)
    #12 - speed std deviation
    #13, 14, 15, 16 - lat/lon of nodes representing the road
    #17 - max_speed_mph
    #18 - number of lanes
    #19 - highway
    #20 - day of the week (monday...)
    #21 - period of the day
    
    #divide in attributes and labels
    #X = uber_data.iloc[:, [3,13,14,15,16,17,18,19,20,21]].values
    #y = uber_data.iloc[:, 11].values 
    #scaler = StandardScaler()
    
    '''
    #knn kfold
    print("knn start")
    k_scores = []
    best_score = 999999
    best_k = -1
    for k in range(40):
        k = k+1
        knn_reg = KNeighborsRegressor(n_neighbors=k)
        regressor = make_pipeline(scaler, knn_reg)
        scores_mean = -1*cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error").mean()
        k_scores.append(scores_mean)
        if scores_mean < best_score:
            best_k = k
            best_score = scores_mean
    print(k_scores)
    print("best k, best msqerror:", best_k, best_score)
    ''' 

    '''
    X = uber_data.iloc[:, [3,13,14,15,16,17,18,19,20,21]].values
    y = uber_data.iloc[:, 11].values 
    scaler = StandardScaler()
    #linear regression
    lin_reg = LinearRegression()
    regressor = make_pipeline(scaler, lin_reg)
    scores_mean = -1*cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error").mean()
    print('linear regression msqerror:', scores_mean)
    '''

    '''   
    #SVR
    #gamma - scale/auto/0.1
    print('SVR start')
    #srv_rbf = SVR(kernel='rbf', gamma='scale', C=1.57, epsilon=0.03)
    srv_rbf = SVR(kernel='rbf', gamma='auto')
    #srv_linear = SVR(kernel='linear')
    regressor = make_pipeline(scaler, srv_rbf)
    scores_mean = -1*cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error").mean()
    print('SVR msqerror:', scores_mean)
    

    
    X = uber_data.iloc[:, [3,13,14,15,16,17,18,19,20,21]].values
    y = uber_data.iloc[:, 11].values 
    scaler = StandardScaler()
    '''

    #neural network
    print('nn start')
    estimators = []
    estimators.append(('standardize', scaler))
    #validation_split=0.2 -> testar com validation split?
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    #estimators.append(('mlp', KerasRegressor(build_fn=create_nn_model, epochs=100, batch_size=5, verbose=0, callbacks=[early_stop, tfdocs.modeling.EpochDots()])))
    estimators.append(('mlp', KerasRegressor(build_fn=create_nn_model, epochs=100, batch_size=5, verbose=0)))
    regressor = Pipeline(estimators)
    scores_mean = -1*cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error").mean()
    print('neural network msqerror:', scores_mean)
    

    '''
    model = create_nn_model()
    print(model.summary())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    #model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=5, epochs=100)
    #validation_split=0.2
    history = model.fit(X_train, y_train, validation_split=0.2, batch_size=5, epochs=100)
    y_pred = model.predict(X_test)
    #rmserror = sqrt(mean_squared_error(y_test,y_pred)) #calculate rmse
    msqerror = mean_squared_error(y_test,y_pred) #calculate msqerror
    print('neural network msqerror:', msqerror)
    '''

    '''
    #hyperparameter optimization technique usind Grid Search
    #The best_score_ member provides access to the best score observed during the optimization procedure 
    #the best_params_ describes the combination of parameters that achieved the best results
    print('grid search SVM')
    svmr = SVR()
    pipe = Pipeline([('scale', scaler),('svm', svmr)])
    #define the grid search parameters
    param_grid = [{'svm__kernel': ['rbf', 'poly', 'sigmoid'],'svm__C': [0.1, 1, 10, 100],'svm__gamma': [1,0.1,0.01,0.001],},]
    #param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
    #param_grid = {'C': [1], 'gamma': [0.1],'kernel': ['rbf']}
    gd_svr = GridSearchCV(estimator=pipe,param_grid=param_grid,scoring="neg_mean_squared_error",cv=10,n_jobs=-1,return_train_score=False,refit=True)
    #pipe_svm = make_pipeline(scaler, gd_sr)
    grid_svr_result = gd_svr.fit(X,y)
    print(grid_svr_result.cv_results_)
    print(grid_svr_result.best_estimator_)
    '''

    '''
    #define the grid search parameters
    #Tune Batch Size and Number of Epochs
    batch_size = [5, 10, 20, 40]
    epochs = [10, 50, 100, 200, 400]
    #Tune the Training Optimization Algorithm => optimization algorithm used to train the network, each with default parameters.
    #often you will choose one approach a priori and instead focus on tuning its parameters on your problem
    #optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    #Tune Learning Rate and Momentum <- relacionado ao algoritmo selecionado anteriormente
    #Tune Network Weight Initialization
    init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    #Tune the Neuron Activation Function
    activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
    #Tune Dropout Regularization
    weight_constraint = [1, 2, 3, 4, 5]
    dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #Tune the Number of Neurons in the Hidden Layer
    neurons = [1, 5, 10, 15, 20, 25, 30]
    param_grid = dict(batch_size=batch_size, epochs=epochs)
    grid_nn = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=10)
    grid_nn_result = grid_nn.fit(X, Y)
    print(grid_svr_result.cv_results_)
    print(grid_svr_result.best_estimator_)
    '''
    

    '''
    plt.plot(y_test, color = 'red', label = 'Real data')
    plt.plot(y_pred, color = 'blue', label = 'Predicted data')
    plt.title('Prediction')
    plt.legend()
    plt.show()
    '''

    '''
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    ''' 

    '''
    plt.scatter(y_test, y_pred)

    plt.xlabel('True Values')

    plt.ylabel('Predictions')
    '''

    '''
    #logistic regression
    log_reg = LogisticRegression()
    regressor = make_pipeline(scaler, log_reg)
    scores_mean = -1*cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error").mean()
    print('logistic regression msqerror:', scores_mean)
    '''

    #other error calculating. but i think those are not good for knn
    #print(np.mean(y_pred != y_test))
    #print(confusion_matrix(y_test, y_pred))
    #print(classification_report(y_test, y_pred))
    
    #add new columns ->doutros atributos da aresta q tenha do osmnx => for this we need to deal with some roads that don't have the info on max speed, etc
    #do the prediction on the missing roads

def get_uber_speed_data_prediction(G_drive, speed_data):

    api = osm.OsmApi()
    #load speed data from csv files
    path = speed_data
    all_files = glob.glob(os.path.join(path, "*.csv"))
    df_from_each_file = (pd.read_csv(f) for f in all_files)
    uber_data = pd.concat(df_from_each_file, ignore_index=True)


    #uber_data = pd.read_csv("../uber_movement/movement-speeds-hourly-cincinnati-2019-10.csv")
    unique_nodes = pd.unique(uber_data[['osm_start_node_id', 'osm_end_node_id']].values.ravel('K'))
    
    #print(concatenated_df.head())
    #print(uber_data.head())
    
    print("start number of rows", len(uber_data))

    nodes_in_graph = [] #nodes that are mapped by the uber movement speed database
    for node in unique_nodes:
        if node in G_drive.nodes():
            if node not in nodes_in_graph:
                nodes_in_graph.append(node)

    uber_data = uber_data[uber_data['osm_start_node_id'].isin(nodes_in_graph) & uber_data['osm_end_node_id'].isin(nodes_in_graph)]
    
    print("mid number of rows", len(uber_data))

    #add lat/lon info and max speed
    #in this part info from openstreetmaps is added
    uber_data["start_node_y"] = np.nan
    uber_data["start_node_x"] = np.nan
    uber_data["end_node_y"] = np.nan
    uber_data["end_node_x"] = np.nan
    uber_data["max_speed_mph"] = np.nan
    uber_data["num_lanes"] = np.nan
    uber_data["highway"] = np.nan

    #unique_highway_str = pd.unique(uber_data[['highway']].values.ravel('K'))
    #print(unique_highway_str)

    for (u,v,k) in G_drive.edges(data=True):
        try:
            uber_data.loc[uber_data['osm_start_node_id'] == u, 'start_node_y'] = G_drive.nodes[u]['y']
            uber_data.loc[uber_data['osm_start_node_id'] == u, 'start_node_x'] = G_drive.nodes[u]['x']

            uber_data.loc[uber_data['osm_end_node_id'] == v, 'end_node_y'] = G_drive.nodes[v]['y']
            uber_data.loc[uber_data['osm_end_node_id'] == v, 'end_node_x'] = G_drive.nodes[v]['x']

            if (uber_data.loc[uber_data['osm_start_node_id'] == u, 'start_node_y'] == 0.0).all():
                nodeu = api.NodeGet(u)
                uber_data.loc[uber_data['osm_start_node_id'] == u, 'start_node_y'] = nodeu['lat']
                uber_data.loc[uber_data['osm_start_node_id'] == u, 'start_node_x'] = nodeu['lon']

            if (uber_data.loc[uber_data['osm_end_node_id'] == v, 'end_node_y'] == 0.0).all():
                nodev = api.NodeGet(v)
                uber_data.loc[uber_data['osm_end_node_id'] == v, 'end_node_y'] = nodev['lat']
                uber_data.loc[uber_data['osm_end_node_id'] == v, 'end_node_x'] = nodev['lon']

        except KeyError:
            pass

        #add atribute max speed 
        dict_edge = {}
        dict_edge = G_drive.get_edge_data(u, v)
        dict_edge = dict_edge[0]
        
        max_speed = get_max_speed_road(dict_edge)
        num_lanes = get_num_lanes(dict_edge)
        #highway
        highway_type = get_highway_info(dict_edge)
        uber_data.loc[(uber_data['osm_start_node_id'] == u) & (uber_data['osm_end_node_id'] == v), 'max_speed_mph'] = max_speed

        uber_data.loc[(uber_data['osm_start_node_id'] == u) & (uber_data['osm_end_node_id'] == v), 'num_lanes'] = num_lanes

        uber_data.loc[(uber_data['osm_start_node_id'] == u) & (uber_data['osm_end_node_id'] == v), 'highway'] = highway_type

    
    
    #add day of the week (monday, tuesday, wednesday, thursday, friday, saturday, sunday etc) info
    #add new column? job day? weekend? holiday?
    #these columns are created based on info that might help, such as peak hours etc
    unique_days = pd.unique(uber_data[['day']].values.ravel('K'))
    unique_months = pd.unique(uber_data[['month']].values.ravel('K'))
    unique_years = pd.unique(uber_data[['year']].values.ravel('K'))

    #add day info
    uber_data["week_day"] = np.nan
    for year in unique_years:
        for month in unique_months:
            for day in unique_days:
                try:
                    ans = datetime.date(year, month, day).weekday()
                    uber_data.loc[(uber_data['day'] == day) & (uber_data['month'] == month) & (uber_data['year'] == year), 'week_day'] = ans
                except ValueError:
                    pass

    #print("min hour", uber_data["hour"].min())
    #print("max hour", uber_data["hour"].max())
    
    uber_data["period_day"] = np.nan
    uber_data.loc[(uber_data['hour'] >= 0) & (uber_data['hour'] <= 6), 'period_day'] = 1 #before peak hours - morning
    uber_data.loc[(uber_data['hour'] >= 7) & (uber_data['hour'] <= 9), 'period_day'] = 2 #peak hours - morning
    uber_data.loc[(uber_data['hour'] >= 10) & (uber_data['hour'] <= 16), 'period_day'] = 3 #before peak hours - afternoon
    uber_data.loc[(uber_data['hour'] >= 17) & (uber_data['hour'] <= 20), 'period_day'] = 4 #peak hours - afternoon 
    uber_data.loc[(uber_data['hour'] >= 21) & (uber_data['hour'] <= 23), 'period_day'] = 5 # night period
    
    #upstream and downstream roads??
    #uber_data["adjacent_roads_speed"] = np.nan


    print(uber_data.isna().sum())
    #clean NA values
    print("clean NA values")
    uber_data = uber_data.dropna()

    uber_data["num_lanes"] = pd.to_numeric(uber_data["num_lanes"])

    #print("min lanes", uber_data["num_lanes"].min())
    #print("max lanes", uber_data["num_lanes"].max())

    #print("min highway", uber_data["highway"].min())
    #print("max highway", uber_data["highway"].max())

    print("end number of rows", len(uber_data))

    

    print(list(uber_data.columns))
    #sns.pairplot(uber_data[["max_speed_mph", "num_lanes", "highway"]], diag_kind="kde")
    
    print(uber_data.head())
    print(uber_data.dtypes)        
    
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    #scaler = MinMaxScaler(feature_range=(0, 1))

    #columns without groupby
    #0 - year
    #1 - month
    #2 - day
    #3 - hour
    #4 - utc_timestamp
    #5,6,7,8,9,10 - IDs
    #11 - speed (TARGET)
    #12 - speed std deviation
    #13, 14, 15, 16 - lat/lon of nodes representing the road
    #17 - max_speed_mph
    #18 - number of lanes
    #19 - highway
    #20 - day of the week (monday...)
    #21 - period of the day
    
    #divide in attributes and labels
    X = uber_data.iloc[:, [3,13,14,15,16,17,18,19,20,21]].values
    y = uber_data.iloc[:, 11].values 
    scaler = StandardScaler()
    
    '''
    #knn kfold
    k_scores = []
    best_score = 999999
    best_k = -1
    for k in range(40):
        k = k+1
        knn_reg = KNeighborsRegressor(n_neighbors=k)
        regressor = make_pipeline(scaler, knn_reg)
        scores_mean = -1*cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error").mean()
        k_scores.append(scores_mean)
        if scores_mean < best_score:
            best_k = k
            best_score = scores_mean
    print(k_scores)
    print("best k, best msqerror:", best_k, best_score)
    '''

    '''
    X = uber_data.iloc[:, [3,13,14,15,16,17,18,19,20,21]].values
    y = uber_data.iloc[:, 11].values 
    scaler = StandardScaler()
    #linear regression
    lin_reg = LinearRegression()
    regressor = make_pipeline(scaler, lin_reg)
    scores_mean = -1*cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error").mean()
    print('linear regression msqerror:', scores_mean)
    '''

    '''
    X = uber_data.iloc[:, [3,13,14,15,16,17,18,19,20,21]].values
    y = uber_data.iloc[:, 11].values 
    scaler = StandardScaler()
    #SVR
    #gamma - scale/auto/0.1
    #srv_rbf = SVR(kernel='rbf', gamma='scale', C=1.57, epsilon=0.03)
    srv_rbf = SVR(kernel='rbf', gamma='auto')
    #srv_linear = SVR(kernel='linear')
    regressor = make_pipeline(scaler, srv_rbf)
    scores_mean = -1*cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error").mean()
    print('SVR msqerror:', scores_mean)
    '''
    
    '''
    #X = uber_data.iloc[:, [3,13,14,15,16,17,18,19,20,21]].values
    #y = uber_data.iloc[:, 11].values 
    #scaler = StandardScaler()
    #neural network
    estimators = []
    estimators.append(('standardize', scaler))
    #validation_split=0.2 -> testar com validation split?
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    #estimators.append(('mlp', KerasRegressor(build_fn=create_nn_model, epochs=100, batch_size=5, verbose=0, callbacks=[early_stop, tfdocs.modeling.EpochDots()])))
    estimators.append(('mlp', KerasRegressor(build_fn=create_nn_model, epochs=100, batch_size=10, verbose=0)))
    regressor = Pipeline(estimators)
    scores_mean = -1*cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error").mean()
    print('neural network msqerror:', scores_mean)
    '''

    '''
    model = create_nn_model()
    print(model.summary())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    #model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=5, epochs=100)
    #validation_split=0.2
    history = model.fit(X_train, y_train, validation_split=0.2, batch_size=5, epochs=100)
    y_pred = model.predict(X_test)
    #rmserror = sqrt(mean_squared_error(y_test,y_pred)) #calculate rmse
    msqerror = mean_squared_error(y_test,y_pred) #calculate msqerror
    print('neural network msqerror:', msqerror)
    '''

    '''
    #hyperparameter optimization technique usind Grid Search
    #The best_score_ member provides access to the best score observed during the optimization procedure 
    #the best_params_ describes the combination of parameters that achieved the best results
    print('grid search SVM')
    svmr = SVR()
    pipe = Pipeline([('scale', scaler),('svm', svmr)])
    #define the grid search parameters
    param_grid = [{'svm__kernel': ['rbf', 'poly', 'sigmoid'],'svm__C': [0.1, 1, 10, 100],'svm__gamma': [1,0.1,0.01,0.001],},]
    #param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
    #param_grid = {'C': [1], 'gamma': [0.1],'kernel': ['rbf']}
    gd_svr = GridSearchCV(estimator=pipe,param_grid=param_grid,scoring="neg_mean_squared_error",cv=3,n_jobs=-1,return_train_score=False,refit=True)
    #pipe_svm = make_pipeline(scaler, gd_sr)
    grid_svr_result = gd_svr.fit(X,y)
    print(grid_svr_result.cv_results_)
    print(grid_svr_result.best_estimator_)
    '''

    
    print('NEURAL NETWORK GRID SEARCH - BATCH SIZE AND EPOCHS')
    #define the grid search parameters
    #Tune Batch Size and Number of Epochs
    
    #batch_size = [5, 8, 10, 16, 20]
    #epochs = [100, 200, 400, 800, 1600]
    batch_size = [16]
    epochs = [800]

    #Tune the Training Optimization Algorithm => optimization algorithm used to train the network, each with default parameters.
    #often you will choose one approach a priori and instead focus on tuning its parameters on your problem
    #optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    #Tune Learning Rate and Momentum <- relacionado ao algoritmo selecionado anteriormente
    #Tune Network Weight Initialization
    
    #init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    init_mode = ['he_uniform']

    #Tune the Neuron Activation Function
    activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
    #Tune Dropout Regularization
    weight_constraint = [1, 2, 3, 4, 5]
    dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #Tune the Number of Neurons in the Hidden Layer
    neurons = [1, 5, 10, 15, 20, 25, 30]
    
    param_grid = dict(batch_size=batch_size, epochs=epochs, init_mode=init_mode, activation=activation, weight_constraint=weight_constraint, dropout_rate=dropout_rate)
    nn_model = KerasRegressor(build_fn=create_nn_model, verbose=0)
    grid_nn = GridSearchCV(estimator=nn_model, param_grid=param_grid, n_jobs=-1, cv=3)
    X = scaler.fit_transform(X)
    grid_nn_result = grid_nn.fit(X, y)
    print(grid_nn_result.cv_results_)
    #print(grid_nn_result.best_estimator_)
    print("Best: %f using %s" % (grid_nn_result.best_score_, grid_nn_result.best_params_))
    
    '''
    plt.plot(y_test, color = 'red', label = 'Real data')
    plt.plot(y_pred, color = 'blue', label = 'Predicted data')
    plt.title('Prediction')
    plt.legend()
    plt.show()
    '''

    '''
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    ''' 

    '''
    plt.scatter(y_test, y_pred)

    plt.xlabel('True Values')

    plt.ylabel('Predictions')
    '''

    '''
    #logistic regression
    log_reg = LogisticRegression()
    regressor = make_pipeline(scaler, log_reg)
    scores_mean = -1*cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error").mean()
    print('logistic regression msqerror:', scores_mean)
    '''

    #other error calculating. but i think those are not good for knn
    #print(np.mean(y_pred != y_test))
    #print(confusion_matrix(y_test, y_pred))
    #print(classification_report(y_test, y_pred))
    
    #add new columns ->doutros atributos da aresta q tenha do osmnx => for this we need to deal with some roads that don't have the info on max speed, etc
    #do the prediction on the missing roads

def get_uber_speed_data_mean(G_drive, speed_data, day_of_the_week):
    
    #function returns the avg speed for each road

    #load speed data from csv files
    path = speed_data
    all_files = glob.glob(os.path.join(path, "*.csv"))
    df_from_each_file = (pd.read_csv(f) for f in all_files)
    uber_data = pd.concat(df_from_each_file, ignore_index=True)
    
    count_nodes = 0
    unique_nodes = pd.unique(uber_data[['osm_start_node_id', 'osm_end_node_id']].values.ravel('K'))
    
    nodes_in_graph = []
    for node in unique_nodes:
        if node in G_drive.nodes():
            if node not in nodes_in_graph:
                nodes_in_graph.append(node)

    uber_data = uber_data[uber_data['osm_start_node_id'].isin(nodes_in_graph) & uber_data['osm_end_node_id'].isin(nodes_in_graph)]
    unique_nodes = pd.unique(uber_data[['osm_start_node_id', 'osm_end_node_id']].values.ravel('K'))
    
    #uber_data_nd = uber_data[['osm_start_node_id','osm_end_node_id']].drop_duplicates()
    #unique_nodes_nd = pd.unique(uber_data_nd[['osm_start_node_id', 'osm_end_node_id']].values.ravel('K'))

    unique_days = pd.unique(uber_data[['day']].values.ravel('K'))
    unique_months = pd.unique(uber_data[['month']].values.ravel('K'))
    unique_years = pd.unique(uber_data[['year']].values.ravel('K'))

    #add day info
    uber_data["week_day"] = np.nan
    for year in unique_years:
        for month in unique_months:
            for day in unique_days:
                try:
                    ans = datetime.date(year, month, day).weekday()
                    uber_data.loc[(uber_data['day'] == day) & (uber_data['month'] == month) & (uber_data['year'] == year), 'week_day'] = ans
                except ValueError:
                    pass

    uber_data = uber_data.loc[uber_data["week_day"] == day_of_the_week]
    #this value is used to add to roads in which speed information is unkown
    speed_mean_overall = uber_data['speed_mph_mean'].mean()

    speed_avg_data = uber_data.groupby(['osm_start_node_id','osm_end_node_id', 'hour'], as_index=False)['speed_mph_mean'].mean()

    

    #speed_mean_overall = speed_avg_data['speed_mph_mean'].mean()

    return speed_avg_data, speed_mean_overall

    
    #speed_avg_data.columns = ['osm_start_node_id','osm_end_node_id', 'hour', 'avg_speed']
    #print(speed_avg_data.head())

    #plot network to show nodes that are in the uber speed data
    #nc = ['r' if (node in unique_nodes) else '#336699' for node in G_drive.nodes()]
    #ns = [12 if (node in unique_nodes) else 6 for node in G_drive.nodes()]
    #fig, ax = ox.plot_graph(G_drive, node_size=ns, node_color=nc, node_zorder=2, save=True, filename='cincinnati_with_nodes_speed')
    
    '''
    for (u,v,k) in G_drive.edges(data=True):
        #print (u,v,k)
        try:
            G_drive[u][v][0]['uberspeed'] = 0
            G_drive[u][v][0]['num_occur'] = 0
        except KeyError:
            pass

    
    for index, row in uber_data.iterrows():
        try:
            u = row['osm_start_node_id']
            v = row['osm_end_node_id']
            G_drive[u][v][0]['uberspeed'] = G_drive[u][v][0]['uberspeed'] + row['speed_mph_mean']
            G_drive[u][v][0]['num_occur'] = G_drive[u][v][0]['num_occur'] + 1
        except KeyError:
            pass
    
    for (u,v,k) in G_drive.edges(data=True):
        if G_drive[u][v][0]['num_occur'] > 0:
            G_drive[u][v][0]['uberspeed'] = (G_drive[u][v][0]['uberspeed']/G_drive[u][v][0]['num_occur'])
            #G_drive[u][v][0]['num_occur'] = 0
    '''

def plot_bus_stops(param, network):

    stops_folder = os.path.join(param.save_dir_images, 'bus_stops')

    if not os.path.isdir(stops_folder):
        os.mkdir(stops_folder)

    bus_stop_list_nodes = []
    for index, stop in network.bus_stops.iterrows():
        bus_stop_list_nodes.append(stop['osmid_walk'])

    nc = ['r' if (node in bus_stop_list_nodes) else '#336699' for node in network.G_walk.nodes()]
    ns = [12 if (node in bus_stop_list_nodes) else 6 for node in network.G_walk.nodes()]
    fig, ax = ox.plot_graph(network.G_walk, node_size=ns, show=False, node_color=nc, node_zorder=2, save=True, filepath=stops_folder+'/pre_existing_stops_walk.png')
    plt.close(fig)

    bus_stop_list_nodes = []
    for index, stop in network.bus_stops.iterrows():
        bus_stop_list_nodes.append(stop['osmid_drive'])

    nc = ['r' if (node in bus_stop_list_nodes) else '#336699' for node in network.G_drive.nodes()]
    ns = [12 if (node in bus_stop_list_nodes) else 6 for node in network.G_drive.nodes()]
    fig, ax = ox.plot_graph(network.G_drive, node_size=ns, show=False, node_color=nc, node_zorder=2, save=True, filepath=stops_folder+'/pre_existing_stops_drive.png')
    plt.close(fig)

def network_stats(param, network):
    print('used vehicle speed: ', param.vehicle_speed*3.6, ' kmh')
    print("average dist 2 stops (driving network):", network.travel_time_matrix["dist"].mean())
    print("average travel time between 2 stops:", network.travel_time_matrix["eta"].mean())

def create_network(place_name, walk_speed, param):

    '''
    ‘drive’ – get drivable public streets (but not service roads)
    ‘drive_service’ – get drivable public streets, including service roads
    ‘walk’ – get all streets and paths that pedestrians can use (this network type ignores one-way directionality)
    ‘bike’ – get all streets and paths that cyclists can use
    ‘all’ – download all (non-private) OSM streets and paths
    ‘all_private’ – download all OSM streets and paths, including private-access ones
    '''
    api_osm = osm.OsmApi() 

    print('Now genarating network_data')
    G_walk, polygon_walk = ox.graph_from_place(place_name, network_type='walk', retain_all=True)
    #G_walk, polygon_walk = ox.graph_from_place(place_name, network_type='walk', retain_all=True, buffer_dist=3000)
    #fig, ax = ox.plot_graph(G_walk, save=True, file_format='svg', filename='walk_network')
    G_drive, polygon_drive = ox.graph_from_place(place_name, network_type='drive', retain_all=True)
    #G_drive, polygon_drive = ox.graph_from_place(place_name, network_type='drive', retain_all=True, buffer_dist=3000)
    #fig, ax = ox.plot_graph(G_drive, save=True, filename='cincinnati_drive')
    #print('network size (walk):', len(G_walk.nodes()))
    #print('network size (drive):', len(G_drive.nodes()))
    print('num walk nodes', len(G_walk.nodes()))
    print('num drive nodes', len(G_drive.nodes()))
    
    if param.vehicle_speed_data != "max" and param.vehicle_speed_data != "set":
        avg_uber_speed_data, speed_mean_overall = get_uber_speed_data_mean(G_drive, param.vehicle_speed_data, param.day_of_the_week)
        avg_uber_speed_data = pd.DataFrame(avg_uber_speed_data)
        print(avg_uber_speed_data.head())
        print('speed mean overall', speed_mean_overall)
        #get_uber_speed_data_prediction(G_drive, speed_data)
    
    print('Now retrieving bus stops')
    bus_stops = get_bus_stops_matrix_csv(param, G_walk, G_drive, polygon_drive)
    
    shortest_path_walk, shortest_path_drive = get_distance_matrix_csv(param, G_walk, G_drive, bus_stops)

    filter_bus_stops(param, bus_stops, shortest_path_drive)

    print('Trying to get zones')
    zones = get_zones_csv(param, G_walk, G_drive, polygon_drive)
    #create graph to plot zones here           
    print('number of zones', len(zones))

    
    print('Now genarating time_travel_data')
    max_speed_mean_overall = 0
    counter_max_speeds = 0
    
    for (u,v,k) in G_drive.edges(data=True):    
        dict_edge = {}
        dict_edge = G_drive.get_edge_data(u, v)
        dict_edge = dict_edge[0]
        max_speed_mean_overall,  counter_max_speeds = calc_mean_max_speed(dict_edge, max_speed_mean_overall, counter_max_speeds)

    max_speed_mean_overall = max_speed_mean_overall/counter_max_speeds

    if param.vehicle_speed_data == "max":
        #print('overall mean max speed mph', max_speed_mean_overall*2.237)
        #print('overall mean max speed m/s', max_speed_mean_overall)
        param.vehicle_speed = float(max_speed_mean_overall*param.max_speed_factor)

    #print('used vehicle speed:' , param.vehicle_speed)

    #COME BACK TO THIS LATER - TIME DEPENDENT TIME TRAVEL
    #colocar range ser o time window do request generation?
    #for hour in range(24):

    '''
    for hour in range(1):
        for (u,v,k) in G_drive.edges(data=True):
            hour_key = 'travel_time_' + str(hour)

            #0 after [u][v] is necessary to access the edge data
            edge_length = G_drive[u][v][0]['length']
            
            if speed_data != "max":
                try:
                    edge_speed = avg_uber_speed_data.loc[(u,v,hour), 'speed_mph_mean'] 
                except KeyError:
                    edge_speed = speed_mean_overall

                #convert to m/s
                #speeds in the uber database are in mph
                edge_speed = edge_speed/2.237
            else:
                dict_edge = {}
                dict_edge = G_drive.get_edge_data(u, v)
                dict_edge = dict_edge[0]
                
                edge_speed = get_max_speed_road(dict_edge)
                
                if math.isnan(edge_speed):
                    edge_speed = max_speed_mean_overall
                
                #max_speed_factor - value between 0 and 1
                edge_speed = edge_speed*param.max_speed_factor

            #calculates the eta travel time for the given edge at 'hour'
            eta =  int(math.ceil(edge_length/edge_speed))

            G_drive[u][v][0][hour_key] = eta
    '''

    #itid = 0
    #updates the 'itid in bus_stops'
    #for index, stop in bus_stops.iterrows():
    #    bus_stops.loc[index, 'itid'] = int(itid)
    #    itid = itid + 1

    travel_time_matrix = get_travel_time_matrix_osmnx_csv(param, bus_stops, shortest_path_drive, shortest_path_walk)

    #param.update_network(G_drive, polygon_drive, shortest_path_drive, G_walk, polygon_walk, shortest_path_walk, bus_stops, zones, walk_speed)
    network = Network(G_drive, polygon_drive, shortest_path_drive, G_walk, polygon_walk, shortest_path_walk, bus_stops, zones, walk_speed)
    network.update_travel_time_matrix(travel_time_matrix)
    #network = Network()
    
    plot_bus_stops(param, network)
    network_stats(param, network)

    print('Trying to get fixed transport routes')
    if param.get_fixed_lines == 'osm':

        pt_fixed_lines = get_fixed_lines_csv(param, G_walk, G_drive, polygon_drive)
        print('number of routes', len(pt_fixed_lines))
    else:
        if param.get_fixed_lines == 'deconet':

            #this could be changed for a server or something else
            folder_path_deconet = param.output_file_base+'/'+'deconet'
            if not os.path.isdir(folder_path_deconet):
                print('ERROR: deconet files do not exist')
            else:
                get_fixed_lines_deconet(param, network, folder_path_deconet)

    list_bus_stops = []
    for index, stop_node in network.bus_stops.iterrows():
        list_bus_stops.append(index)

    network.list_bus_stops = list_bus_stops


    return network
    
if __name__ == '__main__':

    caching.clear_cache()

    request_demand = []
    
    vehicle_fleet = []
    
    #default for some parameters

    get_fixed_lines = None
    
    num_replicates = 1
    
    set_seed = 0
    
    vehicle_speed_data = "max"
    vehicle_speed = -1
    
    walk_speed = 5/3.6 #m/s

    max_walking = 10*60 #seconds

    min_early_departure = 0
    max_early_departure = 24*3600

    day_of_the_week = 0 #monday

    max_speed_factor = 0.5

    is_network_generation = False
    is_request_generation = False

    network_class_file = None

    average_waiting_time = 120

    num_of_cpu = cpu_count()

    #INSTANCE PARAMETER INPUT INFORMATION
    for i in range(len(sys.argv)):
        if sys.argv[i] == "--base_file_name":
           output_file_base = sys.argv[i+1].split('.')[0]

        if sys.argv[i] == "--is_request_generation":
            is_request_generation = True
            is_network_generation = False

        if sys.argv[i] == "--is_network_generation":
            is_network_generation = True
            is_request_generation = False

        if sys.argv[i] == "num_cpus":
            #ray.shutdown()
            #ray.init(num_cpus=num_of_cpu)
            i = i+1
            num_of_cpu = int(sys.argv[i])

        #if sys.argv[i] == "--network_class_file":
        #    i += 1
        #    network_class_file = str(sys.argv[i])

        #if sys.argv[i] == "--param_class_file":
        #    i += 1
        #    param_class_file = str(sys.argv[i])
            
        if sys.argv[i] == "--place_name":
           place_name = sys.argv[i+1]

        if sys.argv[i] == "--vehicle_speed_data":
            i += 1
            vehicle_speed_data = str(sys.argv[i])

            if vehicle_speed_data == "set":
                i += 1
                vehicle_speed = float(sys.argv[i])

                i += 1
                if sys.argv[i] == "kmh":
                    vehicle_speed = vehicle_speed/3.6

                if sys.argv[i] == "mph":
                    vehicle_speed = vehicle_speed/2.237

            if vehicle_speed_data == "max":
                i += 1
                max_speed_factor = float(sys.argv[i])

        if sys.argv[i] == "--walking_threshold":
            max_walking = int(sys.argv[i+1])

            if sys.argv[i+2] == "min":
                max_walking = max_walking*60

            if sys.argv[i+2] == "h":
                max_walking = max_walking*3600

        if sys.argv[i] == "--get_fixed_lines":
            #get_fixed_lines = True
            
            if sys.argv[i+1] == "osm":
                get_fixed_lines = "osm"

            if sys.argv[i+1] == "deconet":
                get_fixed_lines = "deconet"


        if sys.argv[i] == "--time_window":
            
            i += 1
            min_early_departure = int(sys.argv[i])

            i += 1
            max_early_departure = int(sys.argv[i])

            i += 1
            if sys.argv[i] == "h":
                min_early_departure = min_early_departure*3600
                max_early_departure = max_early_departure*3600

            if sys.argv[i] == "min":
                min_early_departure = min_early_departure*60
                max_early_departure = max_early_departure*60

        if sys.argv[i] == "--walk_speed":

            i += 1
            walk_speed = float(sys.argv[i])
            
            i += 1
            if sys.argv[i] == "kmh":
                walk_speed = walk_speed/3.6

            if sys.argv[i] == "mph":
                walk_speed = walk_speed/2.237

        if sys.argv[i] == "--add_fleet":
            num_vehicles = int(sys.argv[i+1])
            capacity_vehicles = int(sys.argv[i+2])  

            vf = VehicleFleet(num_vehicles, capacity_vehicles)
            vehicle_fleet.append(vf)
        
        if sys.argv[i] == "--seed":
            set_seed = int(sys.argv[i+1])

        if sys.argv[i] == "--num_replicates":
            num_replicates = int(sys.argv[i+1])

        if sys.argv[i] == "--day_of_the_week":
            day_of_the_week = sys.argv[i+1]

            if day_of_the_week == "monday":
                day_of_the_week = 0

            if day_of_the_week == "tuesday":
                day_of_the_week = 1

            if day_of_the_week == "wednesday":
                day_of_the_week = 2

            if day_of_the_week == "thrusday":
                day_of_the_week = 3

            if day_of_the_week == "friday":
                day_of_the_week = 4

            if day_of_the_week == "saturday":
                day_of_the_week = 5

            if day_of_the_week == "sunday":
                day_of_the_week = 6

            day_of_the_week = int(day_of_the_week)

        #request demand that comes from anywhere in the city
        #origin and destination are random

        if sys.argv[i] == "--request_demand":
            
            i += 1
            pdf = sys.argv[i]
            
            if pdf == "normal":
                
                i += 1
                mean = sys.argv[i]
                mean = float(mean)
                
                i += 1
                if sys.argv[i] == "h":
                    mean = mean*3600

                if sys.argv[i] == "min":
                    mean = mean*60

                i += 1
                std = sys.argv[i]
                std = float(std)

                i += 1
                if sys.argv[i] == "h":
                    std = std*3600

                if sys.argv[i] == "min":
                    std = std*60

                i += 1
                num_req = int(sys.argv[i])

                
                origin_zones = []
                destination_zones = []
                i += 1
                if sys.argv[i] == "--origin":

                    i += 1
                    if sys.argv[i] == "random":
                
                        is_random_origin_zones = True
                        
                        i += 1
                        num_origins = int(sys.argv[i])
                        
                    else:
                        if sys.argv[i] == "set":
                
                            is_random_origin_zones = False
                            
                            i += 1
                            num_origins = int(sys.argv[i])
                            for k in range(num_origins):
                                i += 1
                                origin_zones.append(int(sys.argv[i]))
                i += 1           
                if sys.argv[i] == "--destination":

                    i += 1
                    if sys.argv[i] == "random":
                
                        is_random_destination_zones = True
                                        
                        i += 1
                        num_destinations = int(sys.argv[i])

                        i += 1
                        time_type = sys.argv[i]

                    else:
                        if sys.argv[i] == "set":
                
                            is_random_destination_zones = False
                            
                            i += 1
                            num_destinations = int(sys.argv[i])
                            for k in range(num_destinations):
                                i += 1
                                destination_zones.append(int(sys.argv[i]))

                            i += 1
                            time_type = sys.argv[i]

                dnd = RequestDistribution(mean, std, num_req, pdf, num_origins, num_destinations, time_type, is_random_origin_zones, is_random_destination_zones, origin_zones, destination_zones)
                request_demand.append(dnd)

            if pdf == "uniform":
                
                i += 1
                min_time = sys.argv[i]
                min_time = float(min_time)
                
                i += 1
                if sys.argv[i] == "h":
                    min_time = min_time*3600

                if sys.argv[i] == "min":
                    min_time = min_time*60

                i += 1
                max_time = sys.argv[i]
                max_time = float(max_time)

                i += 1
                if sys.argv[i] == "h":
                    max_time = max_time*3600

                if sys.argv[i] == "min":
                    max_time = max_time*60

                i += 1
                num_req = int(sys.argv[i])

                
                origin_zones = []
                destination_zones = []
                i += 1
                if sys.argv[i] == "--origin":

                    i += 1
                    if sys.argv[i] == "random":
                
                        is_random_origin_zones = True
                        
                        i += 1
                        num_origins = int(sys.argv[i])
                        
                    else:
                        if sys.argv[i] == "set":
                
                            is_random_origin_zones = False
                            
                            i += 1
                            num_origins = int(sys.argv[i])
                            for k in range(num_origins):
                                i += 1
                                origin_zones.append(int(sys.argv[i]))
                i += 1           
                if sys.argv[i] == "--destination":

                    i += 1
                    if sys.argv[i] == "random":
                
                        is_random_destination_zones = True
                                        
                        i += 1
                        num_destinations = int(sys.argv[i])

                        i += 1
                        time_type = sys.argv[i]

                    else:
                        if sys.argv[i] == "set":
                
                            is_random_destination_zones = False
                            
                            i += 1
                            num_destinations = int(sys.argv[i])
                            for k in range(num_destinations):
                                i += 1
                                destination_zones.append(int(sys.argv[i]))

                            i += 1
                            time_type = sys.argv[i]

                dnd = RequestDistribution(min_time, max_time, num_req, pdf, num_origins, num_destinations, time_type, is_random_origin_zones, is_random_destination_zones, origin_zones, destination_zones)
                request_demand.append(dnd)
           
        #if sys.argv[i] == "--max_speed_factor":
        #    max_speed_factor = float(sys.argv[i+1])

    bus_factor = 2

    seed(set_seed)
    np.random.seed(set_seed)
            
    #print(stats['circuity_avg'])

    #files are saved on the current directory
    #save_dir = os.getcwd()
    #print(save_dir)

    start = time.process_time()
    if is_network_generation:

        
        save_dir = os.getcwd()+'/'+output_file_base
        print(save_dir)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        #input_dir = os.getcwd()+'/'+output_file_base

        #create network
        print(place_name)

        #creating object that has the instance input information
        param = Parameter(max_walking, min_early_departure, max_early_departure, [], day_of_the_week, num_replicates, bus_factor, get_fixed_lines, vehicle_speed_data, vehicle_speed, max_speed_factor, save_dir, output_file_base, num_of_cpu)
        param.average_waiting_time = average_waiting_time

        param.save_dir_json = os.path.join(param.save_dir, 'json_format')
        if not os.path.isdir(param.save_dir_json):
            os.mkdir(param.save_dir_json)

        param.save_dir_images = os.path.join(param.save_dir, 'images')
        if not os.path.isdir(param.save_dir_images):
            os.mkdir(param.save_dir_images)

        pickle_dir = os.path.join(param.save_dir, 'pickle')
        if not os.path.isdir(pickle_dir):
            os.mkdir(pickle_dir)

        #create the instance's network
        network = create_network(place_name, walk_speed, param)
        print('over network')
        print("total time", time.process_time() - start)

        network_class_file = pickle_dir+'/'+param.output_file_base+'.network.class.pkl'
        parameter_class_file = pickle_dir+'/'+param.output_file_base+'.parameter.class.pkl'
        output_network_class = open(network_class_file, 'wb')
        output_parameter_class = open(parameter_class_file, 'wb')
        pickle.dump(network, output_network_class, pickle.HIGHEST_PROTOCOL)
        pickle.dump(param, output_parameter_class, pickle.HIGHEST_PROTOCOL)
        del param
        del network
        output_network_class.close()
        output_parameter_class.close()
        caching.clear_cache()

    if is_request_generation:


        
        save_dir = os.getcwd()+'/'+output_file_base
        print(save_dir)
        
        pickle_dir = os.path.join(save_dir, 'pickle')
        
        param_class_file = pickle_dir+'/'+output_file_base+'.parameter.class.pkl'
        network_class_file = pickle_dir+'/'+output_file_base+'.network.class.pkl'
        
        #generate the instance's requests
        with open(param_class_file, 'rb') as input_inst_class:
            #load class from binary file
            param = pickle.load(input_inst_class)
            
            param.request_demand = request_demand
            param.num_replicates = num_replicates
            param.min_early_departure = min_early_departure
            param.max_early_departure = max_early_departure

        with open(network_class_file, 'rb') as network_class_file:

            network = pickle.load(network_class_file)

        for replicate in range(param.num_replicates):
            generate_requests(param, network, replicate)

        #print('placement of stops - testing')
        #cluster_travel_demand(param, network)

        del param
        print("total time", time.process_time() - start)
        caching.clear_cache()
            
    
    #generate instances in json output folder
    #generate_instances_json(param)

    # convert instances from json to normal and localsolver format
    save_dir_cpp = os.path.join(save_dir, 'cpp_format')
    if not os.path.isdir(save_dir_cpp):
        os.mkdir(save_dir_cpp)

    save_dir_localsolver = os.path.join(save_dir, 'localsolver_format')
    if not os.path.isdir(save_dir_localsolver):
        os.mkdir(save_dir_localsolver)

    for instance in os.listdir(os.path.join(save_dir, 'json_format')):
        input_name = os.path.join(save_dir, 'json_format', instance)
        output_name_cpp = instance.split('.')[0] + '_cpp.pass'
        output_name_ls = instance.split('.')[0] + '_ls.pass'

        converter = JsonConverter(file_name=input_name)
        converter.convert_normal(output_file_name=os.path.join(save_dir_cpp, output_name_cpp))
        converter.convert_localsolver(output_file_name=os.path.join(save_dir_localsolver, output_name_ls))
