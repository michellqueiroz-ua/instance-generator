import math
import networkx as nx
import numpy as np
import os
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from .logger_utils import get_logger


class DummyRay:
    """Stand-in for ray so the unguarded ray calls below become no-ops.

    Only installed when ray is genuinely missing. It is not a parallel
    executor: ``put``/``get`` are the identity and ``remote`` runs the
    function inline, so every caller must still take the sequential branch
    for the actual work.
    """
    @staticmethod
    def remote(func):
        class RemoteWrapper:
            def remote(*args, **kwargs):
                return func(*args, **kwargs)
        return RemoteWrapper
    @staticmethod
    def shutdown():
        pass
    @staticmethod
    def init(*args, **kwargs):
        pass
    @staticmethod
    def put(obj):
        return obj
    @staticmethod
    def get(objs):
        return objs


try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = DummyRay()
    print("Warning: Ray not available. Using sequential processing (slower but works on Windows).")


def _ray_init():
    """Restart a local ray instance, sized by ray from the host machine.

    A no-op when ray is missing, so call sites can call it unconditionally.

    The previous call sites hardcoded ``num_cpus=8`` and a 14 GB object store
    (and one of them disagreed with the other three), which fails outright on
    any machine with less RAM or fewer cores. Passing nothing lets ray read
    the actual core count and available memory, so there is one helper and no
    fixed sizes to outgrow.
    """
    if not RAY_AVAILABLE:
        return
    ray.shutdown()
    ray.init()


def divide_chunks(l, n):
      
    # looping till length l
    for i in range(0, len(l), n): 
        yield l[i:i + n]

# Non-ray version for sequential processing
def shortest_path_nx_sequential(G, u, v, weight='length'):
    try:
        shortest_path_length = nx.dijkstra_path_length(G, u, v, weight=weight)
        return shortest_path_length
    except nx.NetworkXNoPath:
        return -1

# Non-ray version for single source to all nodes
def shortest_path_nx_ss_sequential(G, u, weight):
    shortest_path_length_u = {}
    shortest_path_length_u = nx.single_source_dijkstra_path_length(G, u, weight=weight)
    return shortest_path_length_u


def _distance_matrix_rows(G, origins, weight, value_type=float, chunksize=100):
    nodes = list(G.nodes)
    node_to_index = {node: index for index, node in enumerate(nodes)}
    edge_weights = {}
    for u, v, attributes in G.edges(data=True):
        edge = (node_to_index[u], node_to_index[v])
        edge_weight = float(attributes.get(weight, 1))
        if edge not in edge_weights or edge_weight < edge_weights[edge]:
            edge_weights[edge] = edge_weight

    edges = list(edge_weights)
    graph = csr_matrix(
        (
            [edge_weights[edge] for edge in edges],
            ([edge[0] for edge in edges], [edge[1] for edge in edges]),
        ),
        shape=(len(nodes), len(nodes)),
    )
    rows = []
    reachable_counts = []
    origins = list(origins)
    for chunk in divide_chunks(origins, chunksize):
        source_indices = [node_to_index[origin] for origin in chunk]
        distances = dijkstra(
            graph, directed=G.is_directed(), indices=source_indices
        )
        for origin, source_distances in zip(chunk, distances):
            reachable = np.flatnonzero(np.isfinite(source_distances))
            row = {"osmid_origin": origin}
            for target_index in reachable:
                row[str(nodes[target_index])] = value_type(
                    source_distances[target_index]
                )
            rows.append(row)
            reachable_counts.append(len(reachable))
    return rows, reachable_counts


if RAY_AVAILABLE:
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
    save_dir_csv = os.path.join(save_dir, 'csv')
    path_dist_csv_file_walk = os.path.join(save_dir_csv, output_file_base+'.dist.walk.csv')

    if os.path.isfile(path_dist_csv_file_walk):
        print('is file dist walk')
        shortest_path_walk = pd.read_csv(path_dist_csv_file_walk)

        test_bus_stops_ids = bus_stops_fr

        osmid_origins = shortest_path_walk['osmid_origin'].tolist()

        #remove duplicates from list
        bus_stops_ids2 = [] 
        [bus_stops_ids2.append(int(x)) for x in test_bus_stops_ids if x not in bus_stops_ids2] 

        bus_stops_ids = [] 
        [bus_stops_ids.append(int(x)) for x in bus_stops_ids2 if x not in osmid_origins] 

        
        rows, _ = _distance_matrix_rows(G_walk, bus_stops_ids, "length")
        if rows:
            shortest_path_walk = pd.concat(
                [shortest_path_walk, pd.DataFrame(rows)], ignore_index=True
            )

        shortest_path_walk.to_csv(path_dist_csv_file_walk)
        shortest_path_walk.set_index(['osmid_origin'], inplace=True)

        return shortest_path_walk


def _get_distance_matrix(G_walk, G_drive, bus_stops, save_dir, output_file_base):
    shortest_path_walk = []
    shortest_path_drive = []
    shortest_dist_drive = []
    
    save_dir_csv = os.path.join(save_dir, 'csv')
    if not os.path.isdir(save_dir_csv):
        os.mkdir(save_dir_csv)
    
    path_dist_csv_file_walk = os.path.join(save_dir_csv, output_file_base+'.dist.walk.csv')
    path_dist_csv_file_drive = os.path.join(save_dir_csv, output_file_base+'.dist.drive.csv')
    path_tt_csv_file_drive = os.path.join(save_dir_csv, output_file_base+'.tt.drive.csv')
    
    shortest_path_drive = pd.DataFrame()
    shortest_path_walk = pd.DataFrame()
    shortest_dist_drive = pd.DataFrame()

    #calculates the shortest paths between all nodes walk

    logger = get_logger()
    
    if os.path.isfile(path_dist_csv_file_walk):
        if logger:
            logger.info('Loading existing walk distance matrix...')
        shortest_path_walk = pd.read_csv(path_dist_csv_file_walk)
        shortest_path_walk.set_index(['osmid_origin'], inplace=True)
    else:
        if logger:
            logger.subsection('Computing Walk Distance Matrix')
        else:
            print('Computing walk distance matrix...')
        
        test_bus_stops_ids = bus_stops['osmid_walk'].tolist()
        #remove duplicates from list
        bus_stops_ids = [] 
        [bus_stops_ids.append(x) for x in test_bus_stops_ids if x not in bus_stops_ids] 
        walk_rows, _ = _distance_matrix_rows(
            G_walk, bus_stops_ids, "length"
        )
        shortest_path_walk = pd.DataFrame(walk_rows)

        shortest_path_walk.to_csv(path_dist_csv_file_walk)
        shortest_path_walk.set_index(['osmid_origin'], inplace=True)
    
    unreachable_nodes = []

    # Calculate travel time matrix for drive network
    if os.path.isfile(path_tt_csv_file_drive):
        if logger:
            logger.info('Loading existing drive travel time matrix...')
        shortest_path_drive = pd.read_csv(path_tt_csv_file_drive)
        shortest_path_drive.set_index(['osmid_origin'], inplace=True)

    else:
        if logger:
            logger.subsection('Computing Drive Travel Time Matrix')
            logger.info('This uses actual road speeds from the network')
        else:
            print('Computing drive travel time matrix...')
        
        travel_time_rows, reachable_counts = _distance_matrix_rows(
            G_drive, G_drive.nodes, "travel_time", int
        )
        shortest_path_drive = pd.DataFrame(travel_time_rows)
        unreachable_nodes.extend(
            origin
            for origin, count in zip(G_drive.nodes, reachable_counts)
            if count == 1
        )

        shortest_path_drive.to_csv(path_tt_csv_file_drive)
        shortest_path_drive.set_index(['osmid_origin'], inplace=True)


    if os.path.isfile(path_dist_csv_file_drive):
        if logger:
            logger.info('Loading existing drive distance matrix...')
        else:
            print('Loading drive distance matrix...')

        shortest_dist_drive = pd.read_csv(path_dist_csv_file_drive)
        shortest_dist_drive.set_index(['osmid_origin'], inplace=True)

    else:
        if logger:
            logger.subsection('Computing Drive Distance Matrix')
        else:
            print('Computing drive distance matrix...')

        '''
        calculate shortest path using travel time considering max speed allowed on roads
        '''

        distance_rows, reachable_counts = _distance_matrix_rows(
            G_drive, G_drive.nodes, "length"
        )
        shortest_dist_drive = pd.DataFrame(distance_rows)
        unreachable_nodes.extend(
            origin
            for origin, count in zip(G_drive.nodes, reachable_counts)
            if count == 1
        )

        shortest_dist_drive.to_csv(path_dist_csv_file_drive)
        shortest_dist_drive.set_index(['osmid_origin'], inplace=True)


    return shortest_path_walk, shortest_path_drive, shortest_dist_drive, unreachable_nodes