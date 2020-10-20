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

def get_fixed_lines_osm(param, G_walk, G_drive, polygon):

    '''
    get fixed lines from OpenStreetMaps
    '''
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

