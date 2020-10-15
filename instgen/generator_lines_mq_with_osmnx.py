import datetime
import matplotlib.pyplot as plt
import numpy as np
import codecs, json
import math
import gc
import geopandas as gpd
import glob
from math import sqrt
from multiprocessing import Pool
from multiprocessing import cpu_count
import networkx as nx
import os
import osmapi as osm
import osmnx as ox
import pandas as pd
import pickle
from random import randint
from random import seed
from random import choices
import ray
from shapely.geometry import Point
from shapely.geometry import MultiPoint
from shapely.geometry import LineString
from shapely.geometry import MultiLineString
from shapely.geometry import MultiPolygon
from shapely.geometry import Polygon
from streamlit import caching
import sys
import time




#import modin.pandas as pd
#import scipy.stats
#from scipy.stats import norm

try:
    import tkinter as tk
    from tkinter import filedialog
except:
    pass

#from imports import *
from classes import *
from passenger_requests import generate_requests

from stops_locations import *

from output_files import JsonConverter

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
        
        routes = ox.geometries_from_polygon(polygon, tags=tags)

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
                    
                    #creates a graph for the given line/route
                    if route_id not in dict_subway_lines:
                        dict_subway_lines[route_id] = {}
                        dict_subway_lines[route_id]['route_graph'] = nx.DiGraph() 

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
        
        poi_zones = ox.geometries_from_polygon(polygon, tags=tags)
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

#ss -> single source to all nodes
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

    #calculates the shortest paths between all nodes walk (takes long time)

    if os.path.isfile(path_dist_csv_file_walk):
        print('is file dist walk')
        shortest_path_walk = pd.read_csv(path_dist_csv_file_walk)
        shortest_path_walk.set_index(['osmid_origin'], inplace=True)
    else:

        print('calculating distance matrix walk network')
        count_divisions = 0

        list_nodes = list(G_walk.nodes)
        test_bus_stops_ids = bus_stops['osmid_walk'].tolist()

        #remove duplicates from list
        bus_stops_ids = [] 
        [bus_stops_ids.append(x) for x in test_bus_stops_ids if x not in bus_stops_ids] 

        
        G_walk_id = ray.put(G_walk)

        #calculate shortest path between nodes in the walking network to the bus stops
        shortest_path_length_walk = []
        results = ray.get([shortest_path_nx_ss.remote(G_walk_id, u) for u in bus_stops_ids])

        j=0
        for u in bus_stops_ids:
            d = {}
            d['osmid_origin'] = u
            for v in G_walk.nodes():
                
                dist_uv = -1
                try:
                    dist_uv = int(results[j][v])
                except KeyError:
                    pass
                if dist_uv != -1:
                    sv = str(v)
                    d[sv] = dist_uv
            shortest_path_length_walk.append(d)

            j+=1
            del d

        shortest_path_walk = pd.DataFrame(shortest_path_length_walk)
        del shortest_path_length_walk
        del results
        gc.collect()

        shortest_path_walk.to_csv(path_dist_csv_file_walk)
        shortest_path_walk.set_index(['osmid_origin'], inplace=True)
    

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
        #start = time.process_time()
        print('creating file bus stops')

        #retrieve bus stops
        tags = {
            'highway':'bus_stop',
        }
        poi_bus_stops = ox.geometries_from_polygon(polygon_drive, tags=tags)

        G_walk_id = ray.put(G_walk)
        G_drive_id = ray.put(G_drive)
        bus_stops = ray.get([get_bus_stop.remote(G_walk_id, G_drive_id, index, poi) for index, poi in poi_bus_stops.iterrows()]) 
         
        bus_stops = pd.DataFrame(bus_stops)
        bus_stops.set_index(['stop_id'], inplace=True)
        
        #drop repeated occurences of bus stops
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

def create_network(
    place_name, 
    walk_speed, 
    max_walking,
    min_early_departure,
    max_early_departure,
    day_of_the_week,
    num_replicates,
    bus_factor,
    get_fixed_lines, 
    vehicle_speed_data, 
    vehicle_speed, 
    max_speed_factor,  
    output_file_base,
    set_seed, 
    num_of_cpu
):

    seed(set_seed)
    np.random.seed(set_seed)

    #directory of instance's saved information
    save_dir = os.getcwd()+'/'+output_file_base
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

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
    
    print('num walk nodes', len(G_walk.nodes()))
    print('num drive nodes', len(G_drive.nodes()))
    
    if param.vehicle_speed_data != "max" and param.vehicle_speed_data != "set":
        avg_uber_speed_data, speed_mean_overall = get_uber_speed_data_mean(G_drive, param.vehicle_speed_data, param.day_of_the_week)
        avg_uber_speed_data = pd.DataFrame(avg_uber_speed_data)
        print(avg_uber_speed_data.head())
        print('speed mean overall', speed_mean_overall)
    
    print('Now retrieving bus stops')
    bus_stops = get_bus_stops_matrix_csv(param, G_walk, G_drive, polygon_drive)
    
    shortest_path_walk, shortest_path_drive = get_distance_matrix_csv(param, G_walk, G_drive, bus_stops)

    filter_bus_stops(param, bus_stops, shortest_path_drive)

    print('Getting zones')
    zones = get_zones_csv(param, G_walk, G_drive, polygon_drive)
    #create graph to plot zones here           
    print('number of zones', len(zones))

    
    print('Now genarating time travel data')
    #this considers max speed
    max_speed_mean_overall = 0
    counter_max_speeds = 0
    
    for (u,v,k) in G_drive.edges(data=True):    
        dict_edge = {}
        dict_edge = G_drive.get_edge_data(u, v)
        dict_edge = dict_edge[0]
        max_speed_mean_overall,  counter_max_speeds = calc_mean_max_speed(dict_edge, max_speed_mean_overall, counter_max_speeds)

    max_speed_mean_overall = max_speed_mean_overall/counter_max_speeds

    if param.vehicle_speed_data == "max":
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
                print('ERROR: deconet data files do not exist')
            else:
                get_fixed_lines_deconet(param, network, folder_path_deconet)

    list_bus_stops = []
    for index, stop_node in network.bus_stops.iterrows():
        list_bus_stops.append(index)

    network.list_bus_stops = list_bus_stops

    print('over network')
    #print("total time", time.process_time() - start)

    network_class_file = pickle_dir+'/'+param.output_file_base+'.network.class.pkl'
    parameter_class_file = pickle_dir+'/'+param.output_file_base+'.parameter.class.pkl'
    
    output_network_class = open(network_class_file, 'wb')
    output_parameter_class = open(parameter_class_file, 'wb')
    pickle.dump(network, output_network_class, pickle.HIGHEST_PROTOCOL)
    pickle.dump(param, output_parameter_class, pickle.HIGHEST_PROTOCOL)
    
    output_network_class.close()
    output_parameter_class.close()
    caching.clear_cache()

    return network

def instance_requests(
    output_file_base,
    request_demand,
    num_replicates,
    min_early_departure,
    max_early_departure
):
    start = time.process_time()
    save_dir = os.getcwd()+'/'+output_file_base
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
        converter.convert_normal(output_file_name=os.path.join(save_dir_cpp, output_name_cpp), network=network)
        converter.convert_localsolver(output_file_name=os.path.join(save_dir_localsolver, output_name_ls))

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
    
    min_walk_speed = 4/3.6 #m/s
    max_walk_speed = 5/3.6 #m/s

    max_walking = 10*30 #seconds

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
     
    if is_network_generation:

        #create the instance's network
        network = create_network(place_name, max_walk_speed, max_walking, min_early_departure, max_early_departure, day_of_the_week, num_replicates, bus_factor, get_fixed_lines, vehicle_speed_data, vehicle_speed, max_speed_factor, output_file_base, set_seed, num_of_cpu)
        
    if is_request_generation:

        instance_requests(output_file_base, request_demand, num_replicates, min_early_departure, max_early_departure)
        
        #print('placement of stops - testing')
        #cluster_travel_demand(param, network)

