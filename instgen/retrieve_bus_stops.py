import matplotlib.pyplot as plt
from multiprocessing import cpu_count
import os
import osmnx as ox
import pandas as pd
import ray

def filter_bus_stops(bus_stops, shortest_path_drive, save_dir, output_file_base):

    '''
    this function deletes useless bus stops, i.e., they are not reacheable from any node 
    '''

    save_dir_csv = os.path.join(save_dir, 'csv')
    path_bus_stops = os.path.join(save_dir_csv, output_file_base+'.stops.csv')
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

    stations_ids = range(0, len(bus_stops))
    bus_stops.index = stations_ids


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
            'station_id': index,
            'osmid_walk': bus_stop_node_walk,
            'osmid_drive': bus_stop_node_drive,
            'lat': poi.geometry.centroid.y,
            'lon': poi.geometry.centroid.x,
        }

        return d

def get_bus_stops_matrix_csv(G_walk, G_drive, place_name, save_dir, output_folder_base):

    '''
    retrieve the bus stops from the location
    '''

    ray.shutdown()
    ray.init(num_cpus=cpu_count())

    save_dir_csv = os.path.join(save_dir, 'csv')

    if not os.path.isdir(save_dir_csv):
        os.mkdir(save_dir_csv)

    path_bus_stops = os.path.join(save_dir_csv, output_folder_base+'.stops.csv')

    if os.path.isfile(path_bus_stops):
        print('is file bus stops')
        bus_stops = pd.read_csv(path_bus_stops)
        bus_stops.set_index(['station_id'], inplace=True)

    else:
        print('creating file bus stops')

        #retrieve bus stops
        tags = {
            'highway':'bus_stop',
        }
        #poi_bus_stops = ox.geometries_from_polygon(polygon_drive, tags=tags)
        poi_bus_stops = ox.geometries_from_place(place_name, tags=tags)

        G_walk_id = ray.put(G_walk)
        G_drive_id = ray.put(G_drive)
        bus_stops = ray.get([get_bus_stop.remote(G_walk_id, G_drive_id, index, poi) for index, poi in poi_bus_stops.iterrows()]) 
         
        bus_stops = pd.DataFrame(bus_stops)
        bus_stops.set_index(['station_id'], inplace=True)
        
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


def plot_bus_stops(network, save_dir_images):

    '''
    create figures with the bus stops present in the location
    '''

    stops_folder = os.path.join(save_dir_images, 'bus_stops')

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