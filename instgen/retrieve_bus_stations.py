import matplotlib.pyplot as plt
from multiprocessing import cpu_count
import os
import osmnx as ox
import pandas as pd
import ray
import warnings


def filter_bus_stations(network, shortest_path_drive, save_dir, output_file_base):

    '''
    this function deletes useless bus stops, i.e., they are not reacheable from any node 
    '''

    save_dir_csv = os.path.join(save_dir, 'csv')
    path_bus_stations = os.path.join(save_dir_csv, output_file_base+'.stations.csv')
    print('number of bus stops before cleaning: ', len(network.bus_stations))

    useless_bus_station = True
    while useless_bus_station:
        useless_bus_station = False
        for index1, stop1 in network.bus_stations.iterrows():
            unreachable_nodes = 0
            for index2, stop2 in network.bus_stations.iterrows():
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
                
            if unreachable_nodes == len(network.bus_stations) - 1:
                network.bus_stations = network.bus_stations.drop(index1)
                useless_bus_station = True
            
    indexes_to_drop = []
    for index1, stop1 in network.bus_stations.iterrows():
        for index2, stop2 in network.bus_stations.iterrows():
            if ((index1 != index2) and (index2 > index1)):
                osmid_origin_stop = int(stop1['osmid_drive'])
                osmid_destination_stop = str(int(stop2['osmid_drive']))
                #if (stop1['type'] != stop2['type']):
                try:
                    if (int(shortest_path_drive.loc[osmid_origin_stop, osmid_destination_stop]) == 0):
                        print("here")
                        if (int(stop1['type']) == 0):
                            if (index1 not in indexes_to_drop):
                                indexes_to_drop.append(index1)
                        elif (int(stop2['type']) == 0):
                            if (index2 not in indexes_to_drop):
                                indexes_to_drop.append(index2)
                except KeyError:
                    pass
                    

    for index_to_drop in indexes_to_drop:
        network.bus_stations = network.bus_stations.drop(index_to_drop)

    print('number of bus stops after removal: ', len(network.bus_stations))

    #stations_ids = range(0, len(network.bus_stations))
    #bus_stations.index = stations_ids

    network.bus_stations.to_csv(path_bus_stations)

@ray.remote
def get_bus_station(G_walk, G_drive, index, poi):

    if poi['highway'] == 'bus_stop':
        bus_station_point = (poi.geometry.centroid.y, poi.geometry.centroid.x)
        
        u, v, key = ox.get_nearest_edge(G_walk, bus_station_point)
        bus_station_node_walk = min((u, v), key=lambda n: ox.distance.great_circle_vec(poi.geometry.centroid.y, poi.geometry.centroid.x, G_walk.nodes[n]['y'], G_walk.nodes[n]['x']))
        
        u, v, key = ox.get_nearest_edge(G_drive, bus_station_point)
        bus_station_node_drive = min((u, v), key=lambda n: ox.distance.great_circle_vec(poi.geometry.centroid.y, poi.geometry.centroid.x, G_drive.nodes[n]['y'], G_drive.nodes[n]['x']))
        
        d = {
            #'station_id': index,
            'osmid_walk': bus_station_node_walk,
            'osmid_drive': bus_station_node_drive,
            'lat': poi.geometry.centroid.y,
            'lon': poi.geometry.centroid.x,
            'type': 0,
        }

        return d

def get_bus_stations_matrix_csv(G_walk, G_drive, place_name, save_dir, output_folder_base):

    warnings.filterwarnings(action="ignore")
    '''
    retrieve the bus stops from the location
    '''

    ray.shutdown()
    ray.init(num_cpus=cpu_count())

    save_dir_csv = os.path.join(save_dir, 'csv')

    if not os.path.isdir(save_dir_csv):
        os.mkdir(save_dir_csv)

    path_bus_stations = os.path.join(save_dir_csv, output_folder_base+'.stations.csv')

    if os.path.isfile(path_bus_stations):
        print('is file bus stations')
        bus_stations = pd.read_csv(path_bus_stations)
        
    else:
        print('creating file bus stations')

        #retrieve bus stops
        tags = {
            'highway':'bus_stop',
        }
        #poi_bus_stations = ox.geometries_from_polygon(polygon_drive, tags=tags)
        poi_bus_stations = ox.geometries_from_place(place_name, tags=tags)

        print('number pois: ', len(poi_bus_stations))
        G_walk_id = ray.put(G_walk)
        G_drive_id = ray.put(G_drive)
        bus_stations = ray.get([get_bus_station.remote(G_walk_id, G_drive_id, index, poi) for index, poi in poi_bus_stations.iterrows()]) 
        
        ray.shutdown()

        bus_stations = pd.DataFrame(bus_stations)
        
        #drop repeated occurences of bus stops
        drop_index_list = []
        for index1, stop1 in bus_stations.iterrows():
            if index1 not in drop_index_list:
                for index2, stop2 in bus_stations.iterrows():
                    if index2 not in drop_index_list:
                        if index1 != index2:
                            #if (stop1['osmid_drive'] == stop2['osmid_drive']) and (stop1['osmid_walk'] == stop2['osmid_walk']):
                            if (stop1['osmid_drive'] == stop2['osmid_drive']):
                                drop_index_list.append(index2)

        for index_to_drop in drop_index_list:
            bus_stations = bus_stations.drop(index_to_drop)

        '''

        it = 0
        for index, stop in bus_stations.iterrows():
            bus_stations.loc[int(index), 'station_id'] = it
            it += 1

        if len(bus_stations) > 0:
            bus_stations.set_index(['station_id'], inplace=True)
        '''

    return bus_stations


def plot_bus_stations(network, save_dir_images):

    '''
    create figures with the bus stops present in the location
    '''

    stops_folder = os.path.join(save_dir_images, 'bus_stations')

    if not os.path.isdir(stops_folder):
        os.mkdir(stops_folder)

    bus_station_list_nodes = []
    for index, stop in network.bus_stations.iterrows():
        bus_station_list_nodes.append(stop['osmid_walk'])

    nc = ['r' if (node in bus_station_list_nodes) else '#336699' for node in network.G_walk.nodes()]
    ns = [12 if (node in bus_station_list_nodes) else 6 for node in network.G_walk.nodes()]
    fig, ax = ox.plot_graph(network.G_walk, node_size=ns, show=False, node_color=nc, node_zorder=2, save=True, filepath=stops_folder+'/pre_existing_stops_walk.png')
    plt.close(fig)

    bus_station_list_nodes = []
    for index, stop in network.bus_stations.iterrows():
        bus_station_list_nodes.append(stop['osmid_drive'])

    nc = ['r' if (node in bus_station_list_nodes) else '#336699' for node in network.G_drive.nodes()]
    ns = [12 if (node in bus_station_list_nodes) else 6 for node in network.G_drive.nodes()]
    fig, ax = ox.plot_graph(network.G_drive, node_size=ns, show=False, node_color=nc, node_zorder=2, save=True, filepath=stops_folder+'/pre_existing_stops_drive.png')
    plt.close(fig)