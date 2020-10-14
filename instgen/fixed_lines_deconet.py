import os
import networkx as nx
import ray

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