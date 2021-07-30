import gc
import math
from multiprocessing import cpu_count
import networkx as nx
import os
import pandas as pd
import ray

@ray.remote
def shortest_path_nx(G, u, v):

    try:
        shortest_path_length = nx.dijkstra_path_length(G, u, v, weight='length')
        return shortest_path_length
    except nx.NetworkXNoPath:
        return -1

#ss -> single source to all nodes
@ray.remote
def shortest_path_nx_ss(G, u, weight):

    shortest_path_length_u = {}
    shortest_path_length_u = nx.single_source_dijkstra_path_length(G, u, weight=weight)
    return shortest_path_length_u

def _update_distance_matrix_walk(G_walk, bus_stops_fr, save_dir, output_file_base):
    
    ray.shutdown()
    ray.init(num_cpus=cpu_count())

    save_dir_csv = os.path.join(save_dir, 'csv')
    path_dist_csv_file_walk = os.path.join(save_dir_csv, output_file_base+'.dist.walk.csv')

    if os.path.isfile(path_dist_csv_file_walk):
        print('is file dist walk')
        shortest_path_walk = pd.read_csv(path_dist_csv_file_walk)

        #test_bus_stops_ids = bus_stops['osmid_walk'].tolist()
        test_bus_stops_ids = bus_stops_fr

        osmid_origins = shortest_path_walk['osmid_origin'].tolist()


        #remove duplicates from list
        bus_stops_ids2 = [] 
        [bus_stops_ids2.append(int(x)) for x in test_bus_stops_ids if x not in bus_stops_ids2] 

        bus_stops_ids = [] 
        [bus_stops_ids.append(int(x)) for x in bus_stops_ids2 if x not in osmid_origins] 

        
        G_walk_id = ray.put(G_walk)

        #calculate shortest path between nodes in the walking network to the bus stops
        #shortest_path_length_walk = []
        results = ray.get([shortest_path_nx_ss.remote(G_walk_id, u, weight="length") for u in bus_stops_ids])

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
            #print(d)
            shortest_path_walk = shortest_path_walk.append(d, ignore_index=True)

            j+=1
            del d

        #shortest_path_walk = pd.DataFrame(shortest_path_length_walk)
        #del shortest_path_length_walk
        del results
        gc.collect()

        shortest_path_walk.to_csv(path_dist_csv_file_walk)
        shortest_path_walk.set_index(['osmid_origin'], inplace=True)

        return shortest_path_walk


def _get_distance_matrix(G_walk, G_drive, bus_stops, save_dir, output_file_base):
    shortest_path_walk = []
    shortest_path_drive = []
    shortest_dist_drive = []
    
    ray.shutdown()
    ray.init(num_cpus=cpu_count())
   
    save_dir_csv = os.path.join(save_dir, 'csv')
    if not os.path.isdir(save_dir_csv):
        os.mkdir(save_dir_csv)
    
    path_dist_csv_file_walk = os.path.join(save_dir_csv, output_file_base+'.dist.walk.csv')
    path_dist_csv_file_drive = os.path.join(save_dir_csv, output_file_base+'.dist.drive.csv')
    path_tt_csv_file_drive = os.path.join(save_dir_csv, output_file_base+'.tt.drive.csv')
    
    shortest_path_drive = pd.DataFrame()
    shortest_path_walk = pd.DataFrame()
    shortest_dist_drive = pd.DataFrame()

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
        results = ray.get([shortest_path_nx_ss.remote(G_walk_id, u, weight="length") for u in bus_stops_ids])

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
    
    unreachable_nodes = []

    if os.path.isfile(path_dist_csv_file_drive):
        print('is file dist drive')
        shortest_path_drive = pd.read_csv(path_tt_csv_file_drive)
        shortest_path_drive.set_index(['osmid_origin'], inplace=True)

        shortest_dist_drive = pd.read_csv(path_dist_csv_file_drive)
        shortest_dist_drive.set_index(['osmid_origin'], inplace=True)
    else:

        print('calculating shortest paths drive network')

        '''
        calculate shortest path using travel time considering max speed allowed on roads
        '''

        
        list_nodes = list(G_drive.nodes)
        G_drive_id = ray.put(G_drive)
        #start = time.process_time()

        #traveltime
        shortest_path_length_drive = []
        results = ray.get([shortest_path_nx_ss.remote(G_drive_id, u, weight="travel_time") for u in list_nodes])

        j=0
        for u in list_nodes:
            d = {}
            d['osmid_origin'] = u
            count = 0
            for v in G_drive.nodes():
                
                dist_uv = -1
                try:
                    dist_uv = int(results[j][v])
                except KeyError:
                    pass
                if dist_uv != -1:
                    sv = str(v)
                    d[sv] = dist_uv
                    count += 1
            shortest_path_length_drive.append(d)

            if count == 1:
                unreachable_nodes.append(u)

            j+=1
            del d

        shortest_path_drive = pd.DataFrame(shortest_path_length_drive)
        del shortest_path_length_drive
        del results
        gc.collect()

        shortest_path_drive.to_csv(path_tt_csv_file_drive)
        shortest_path_drive.set_index(['osmid_origin'], inplace=True)

        #distance
        shortest_path_length_drive = []
        results = ray.get([shortest_path_nx_ss.remote(G_drive_id, u, weight="length") for u in list_nodes])

        j=0
        for u in list_nodes:
            d = {}
            d['osmid_origin'] = u
            count = 0
            for v in G_drive.nodes():
                
                dist_uv = -1
                try:
                    dist_uv = int(results[j][v])
                except KeyError:
                    pass
                if dist_uv != -1:
                    sv = str(v)
                    d[sv] = dist_uv
                    count += 1
            shortest_path_length_drive.append(d)

            if count == 1:
                unreachable_nodes.append(u)

            j+=1
            del d

        shortest_dist_drive = pd.DataFrame(shortest_path_length_drive)
        del shortest_path_length_drive
        del results
        gc.collect()

        shortest_dist_drive.to_csv(path_dist_csv_file_drive)
        shortest_dist_drive.set_index(['osmid_origin'], inplace=True)

    return shortest_path_walk, shortest_path_drive, shortest_dist_drive, unreachable_nodes