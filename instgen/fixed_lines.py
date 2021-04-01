import math
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
import numpy as np
import networkx as nx
import os
import osmapi as osm
import osmnx as ox
import pandas as pd
import ray

def _evaluate_best_fixed_route(network, fixed_lines, origin_node_drive, destination_node_drive, max_walking):
    #evaluate the expected gain of the passenger to take the fixed line

    best_fixed_route = []
    best_evaluation_route = -math.inf
    eta_bus_full = network.shortest_path_drive.loc[origin_node_drive, str(destination_node_drive)]
    #walking_time_bus_full = max_walking * 2

    for line in fixed_lines:
        
        eta_by_fixed_line = line['eta']
        walking_time_fl = 0

        u_drive = int(network.deconet_network_nodes.loc[int(line['u']), 'osmid_drive'])
        v_drive = int(network.deconet_network_nodes.loc[int(line['v']), 'osmid_drive'])

        if line['option'] == 1:
            #walking_time_fl += line['walking_time_u'] + line['walking_time_v']
            eta_bus_partial = 0
            #walking_time_bus_partial = 0

        if line['option'] == 2:
            #walking_time_fl += line['walking_time_v']
            eta_bus_partial = network.shortest_path_drive.loc[origin_node_drive, str(u_drive)]
            #walking_time_bus_partial = max_walking * 2

        if line['option'] == 3:
            #walking_time_fl += line['walking_time_u']
            eta_bus_partial = network.shortest_path_drive.loc[v_drive, str(destination_node_drive)]
            #walking_time_bus_partial = max_walking * 2

        if line['option'] == 4:
            eta_bus_partial = network.shortest_path_drive.loc[origin_node_drive, str(u_drive)]
            eta_bus_partial += network.shortest_path_drive.loc[v_drive, str(destination_node_drive)]
            #walking_time_bus_partial = max_walking * 4

        #evaluation_route = (eta_bus_full + walking_time_bus_full) - (walking_time_fl + eta_bus_partial + walking_time_bus_partial)
        evaluation_route = (eta_bus_full) - (eta_by_fixed_line + eta_bus_partial)

        if evaluation_route > best_evaluation_route:
            best_evaluation_route = evaluation_route
            best_fixed_route = line

    return best_fixed_route 
    


def _retrieve_new_bus_stations(network, node_walk, max_walking, request_walk_speed, to_node_walk):

    #return bus stops that within walking threshold from node_walk

    #to_node_walk (true) => walking times are calculated to the node_walk to the stops
    #to_node_walk (false) => walking times are calculated from the stops to the node_walk

    stops = []
    stops_walking_time = []

    for index in network.bus_stations_ids:

        osmid_possible_stop = int(network.bus_stations.loc[index, 'osmid_walk'])
        if to_node_walk:
            eta_walk = network.get_eta_walk(osmid_possible_stop, node_walk, request_walk_speed)
        else:
            eta_walk = network.get_eta_walk(osmid_possible_stop, node_walk, request_walk_speed)

        if eta_walk >= 0 and eta_walk <= max_walking:
            stops.append(index)
            stops_walking_time.append(eta_walk)

    return stops, stops_walking_time

def _return_kappa_nearest_stations(network, kappa, origin_node_drive, destination_node_drive):

    stations_origin = []

    stations_destination = []
    
    for u in network.nodes_covered_fixed_lines:

        u_drive = int(network.deconet_network_nodes.loc[int(u), 'osmid_drive'])
        dist_ou_drive = network.shortest_path_drive.loc[origin_node_drive, str(u_drive)]
        dist_du_drive = network.shortest_path_drive.loc[destination_node_drive, str(u_drive)]

        og = (u, dist_ou_drive)
        stations_origin.append(og)

        dn = (u, dist_du_drive)
        stations_destination.append(dn)

    stations_origin.sort(key = lambda x: x[1])

    stations_destination.sort(key = lambda x: x[1])

    kappa_stations_origin = []
    kappa_stations_destination = []

    for i in range(kappa):
        kappa_stations_origin.append(stations_origin[i][0])

        kappa_stations_destination.append(stations_destination[i][0])

    return kappa_stations_origin, kappa_stations_destination

def _return_stations_within_radius(network, node_drive, radius):
    
    '''
    return stations covered by fixed lines that are within a given radius
    '''

    stations = []

    for u in network.nodes_covered_fixed_lines:

        u_drive = int(network.deconet_network_nodes.loc[int(u), 'osmid_drive'])
        
        try:
            dist_ou_drive = nx.dijkstra_path_length(network.G_drive, node_drive, u_drive, weight='length')

            if dist_ou_drive <= radius:
                stations.append(u)

        except nx.NetworkXNoPath:
            pass
        
    return stations 

def _check_subway_routes_serve_passenger(network, origin_node_walk, destination_node_walk, origin_node_drive, destination_node_drive, max_walking, request_walk_speed):
    
    '''
    check which subway lines can serve the passenger request
    '''
    subway_routes = []

    #distance between origin and destinations nodes
    dist_od = network.shortest_path_drive.loc[origin_node_drive, str(destination_node_drive)]

    #fl_stations_walk = [] 
    #fl_stations_drive = []

    #check KAPPA stations which are closer to origin and destination of the passenger
    #stations_origin, stations_destination = _return_kappa_nearest_stations(network, 3, origin_node_drive, destination_node_drive)

    #radius of 2km
    radius = 2000 
    stations_origin = _return_stations_within_radius(network, origin_node_drive, radius)
    stations_destination = _return_stations_within_radius(network, destination_node_drive, radius)

    if (stations_origin) and (stations_destination):
        for lid in network.subway_lines:
            #u,v are part of the subset of stations
            d = {
                'u': np.nan,
                'v': np.nan,
                'line_id': lid,
                'eta': math.inf,
                'dist_ou_drive': math.inf,
                'dist_vd_drive': math.inf,
                'option': np.nan,
            }

            for u in stations_origin:
                for v in stations_destination:

                    if u != v:
                    
                        try:

                            eta = nx.dijkstra_path_length(network.subway_lines[lid]['route_graph'], u, v, weight='duration_avg')

                            u_drive = int(network.deconet_network_nodes.loc[int(u), 'osmid_drive'])
                            v_drive = int(network.deconet_network_nodes.loc[int(v), 'osmid_drive'])

                            u_walk = int(network.deconet_network_nodes.loc[int(u), 'osmid_walk'])
                            v_walk = int(network.deconet_network_nodes.loc[int(v), 'osmid_walk'])
                            
                            #distance from origin to node u and 
                            dist_ou_drive = network.shortest_path_drive.loc[origin_node_drive, str(u_drive)]
                            #distance from v to destination
                            dist_vd_drive = network.shortest_path_drive.loc[v_drive, str(destination_node_drive)]
                            
                            eta_ou_walk = network.get_eta_walk(origin_node_walk, u_walk, request_walk_speed)
                            eta_vd_walk = network.get_eta_walk(v_walk, destination_node_walk, request_walk_speed)

                            
                            if not math.isnan(dist_vd_drive):
                                
                                #check if (u,v) gets the passenger closer to the destination
                                if (dist_vd_drive >= 0) and (dist_vd_drive < d['dist_vd_drive']):

                                    d['u'] = u
                                    d['v'] = v
                                    d['eta'] = eta
                                    if not math.isnan(dist_ou_drive):
                                        d['dist_ou_drive'] = dist_ou_drive
                                    d['dist_vd_drive'] = dist_vd_drive

                                    #FIXED LINE ONLY
                                    if (eta_ou_walk <= max_walking) and (eta_vd_walk <= max_walking):
                                        d['option'] = 1
                                        d['walking_time_u'] = eta_ou_walk
                                        d['walking_time_v'] = eta_vd_walk

                                    # ON DEMAND + FIXED LINE
                                    if (eta_ou_walk > max_walking) and (eta_vd_walk <= max_walking):
                                        d['option'] = 2
                                        d['walking_time_v'] = eta_vd_walk

                                    # FIXED LINE + ON DEMAND
                                    if (eta_ou_walk <= max_walking) and (eta_vd_walk > max_walking):
                                        d['option'] = 3
                                        d['walking_time_u'] = eta_ou_walk
                                       
                                    # ON DEMAND + FIXED LINE + ON DEMAND
                                    if (eta_ou_walk > max_walking) and (eta_vd_walk > max_walking):
                                        d['option'] = 4
                                    

                                else:
                                    #otherwise, check if distance from origin point to station is smaller than before
                                    if not math.isnan(dist_ou_drive):
                                        if (dist_vd_drive == d['dist_vd_drive']) and (dist_ou_drive < d['dist_ou_drive']):

                                            d['u'] = u
                                            d['v'] = v
                                            d['eta'] = eta    
                                            d['dist_ou_drive'] = dist_ou_drive
                                            d['dist_vd_drive'] = dist_vd_drive

                                            #FIXED LINE ONLY
                                            if (eta_ou_walk <= max_walking) and (eta_vd_walk <= max_walking):
                                                d['option'] = 1
                                                d['walking_time_u'] = eta_ou_walk
                                                d['walking_time_v'] = eta_vd_walk

                                            # ON DEMAND + FIXED LINE
                                            if (eta_ou_walk > max_walking) and (eta_vd_walk <= max_walking):
                                                d['option'] = 2
                                                d['walking_time_v'] = eta_vd_walk

                                            # FIXED LINE + ON DEMAND
                                            if (eta_ou_walk <= max_walking) and (eta_vd_walk > max_walking):
                                                d['option'] = 3
                                                d['walking_time_u'] = eta_ou_walk
                                                
                                            # ON DEMAND + FIXED LINE + ON DEMAND
                                            if (eta_ou_walk > max_walking) and (eta_vd_walk > max_walking):
                                                d['option'] = 4
                                                

                        except (nx.NetworkXNoPath, KeyError, nx.NodeNotFound):

                            pass

            if not math.isnan(d['u']):

                u_walk = int(network.deconet_network_nodes.loc[int(d['u']), 'osmid_walk'])
                v_walk = int(network.deconet_network_nodes.loc[int(d['v']), 'osmid_walk'])

                if d['option'] == 2:
                    #get new drop off stops (around node u)
                    stops_u, walking_time_u = _retrieve_new_bus_stations(network, u_walk, max_walking, request_walk_speed, to_node_walk=True)
                    d['stops_u'] = stops_u
                    d['walking_time_u'] = walking_time_u
                    
                if d['option'] == 3:
                    #get new pick up stops (around node v)
                    stops_v, walking_time_v = _retrieve_new_bus_stations(network, v_walk, max_walking, request_walk_speed, to_node_walk=False)
                    d['stops_v'] = stops_v
                    d['walking_time_v'] = walking_time_v

                if d['option'] == 4:
                    #get new drop off stops (around node u)
                    stops_u, walking_time_u = _retrieve_new_bus_stations(network, u_walk, max_walking, request_walk_speed, to_node_walk=True)
                    d['stops_u'] = stops_u
                    d['walking_time_u'] = walking_time_u

                    #get new pick up stations (around node v)
                    stops_v, walking_time_v = _retrieve_new_bus_stations(network, v_walk, max_walking, request_walk_speed, to_node_walk=False)
                    d['stops_v'] = stops_v
                    d['walking_time_v'] = walking_time_v

                subway_routes.append(d)
    
    return subway_routes

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

def get_all_shortest_paths_fix_lines(fixed_lines, network_nodes):
    
    ray.shutdown()
    ray.init(num_cpus=cpu_count())

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

def break_lines_in_pieces(subway_lines):

    connecting_nodes = []
    transfer_nodes = []

    for id1 in subway_lines:
        for id2 in subway_lines:

            if (id1 != id2) and (id1 > id2):

                nodes1 = list(subway_lines[id1]['route_graph'].nodes)
                nodes2 = list(subway_lines[id2]['route_graph'].nodes)

                for n in nodes1:
                    if n in nodes2:
                        if (n not in connecting_nodes):
                            connecting_nodes.append(n)

                        if (n not in transfer_nodes):
                            transfer_nodes.append(n)

    linepieces = []
    direct_lines = []

    for ids in subway_lines:

        begin_route = []
        end_route = []

        begin_route = [node for node in subway_lines[ids]['route_graph'].nodes if subway_lines[ids]['route_graph'].in_degree(node) == 0]
        end_route = [node for node in subway_lines[ids]['route_graph'].nodes if subway_lines[ids]['route_graph'].out_degree(node) == 0]

        if (len(begin_route) > 0):
            subway_lines[ids]['begin_route'] = begin_route[0]
        #else:
            #circular it doesnot matter the begin or end

        if (len(end_route) > 0):
            subway_lines[ids]['end_route'] = end_route[0]
        #else:
            #circular it doesnot matter the begin or end

        if (begin_route[0] not in connecting_nodes):
            connecting_nodes.append(begin_route[0])

        if (end_route[0] not in connecting_nodes):
            connecting_nodes.append(end_route[0])

        nodes_path = nx.dijkstra_path(subway_lines[ids]['route_graph'], begin_route[0], end_route[0], weight='duration_avg')
        

        ix = 0
        jx = 1

        i = nodes_path[ix]
        j = nodes_path[jx]

        if (len(nodes_path) == 2):
       
            #i to j is a line piece
            linepieces.append(nodes_path)


        dl = []
        while j != end_route[0]:

            lp = [] 

            while (j not in transfer_nodes) or (j != end_route[0]):
                jx += 1
                j = nodes_path[jx]

            #i to j is a line piece
            for k in range(ix,jx+1):
                lp.append(nodes_path[k])

            linepieces.append(lp)

            ix = jx
            jx = ix+1

            if (ix < len(nodes_path)):
                i = nodes_path[ix]

            if (jx < len(nodes_path)):
                j = nodes_path[jx]

            if (i == begin_route[0]):
                dl.append(i)
                dl.append(j)
            else:
                dl.append(j)

        direct_lines.append(dl)


    return linepieces, connecting_nodes, direct_lines, transfer_nodes

def get_fixed_lines_deconet(network, folder_path, save_dir, output_folder_base):

    #num_of_cpu = cpu_count()
    nodes_covered_fixed_lines = []
    ray.shutdown()
    ray.init(num_cpus=cpu_count())

    save_dir_csv = os.path.join(save_dir, 'csv')

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

                '''
                if len(rts) > 1:
                    if int(row['from_stop_I']) not in connecting_nodes:
                        connecting_nodes.append(int(row['from_stop_I']))

                    if int(row['to_stop_I']) not in connecting_nodes:
                        connecting_nodes.append(int(row['to_stop_I']))
                '''

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

                    if int(row['from_stop_I']) not in nodes_covered_fixed_lines:
                        nodes_covered_fixed_lines.append(int(row['from_stop_I']))

                    if int(row['to_stop_I']) not in nodes_covered_fixed_lines:
                        nodes_covered_fixed_lines.append(int(row['to_stop_I']))

                    dict_subway_lines[route_id]['route_graph'].add_edge(row['from_stop_I'], row['to_stop_I'], duration_avg=float(row['duration_avg']))

              
        #add network nodes e shortest_path_subway para network file
       
        #network.shortest_path_subway = shortest_path_subway
        
        network.deconet_network_nodes = deconet_network_nodes
        network.nodes_covered_fixed_lines = nodes_covered_fixed_lines
        network.subway_lines = dict_subway_lines

        linepieces, connecting_nodes, direct_lines, transfer_nodes = break_lines_in_pieces(dict_subway_lines)

        network.linepieces = linepieces
        network.connecting_nodes = connecting_nodes
        network.direct_lines = direct_lines
        network.transfer_nodes = transfer_nodes

        plot_fixed_lines(network, save_dir)

        tram_lines_filename = folder_path+'/network_tram.csv'
        if os.path.isfile(tram_lines_filename):
            tram_lines = pd.read_csv(tram_lines_filename, delimiter=";")

        bus_lines_filename = folder_path+'/network_bus.csv'
        if os.path.isfile(bus_lines_filename):
            bus_lines = pd.read_csv(bus_lines_filename, delimiter=";")

def plot_pt_fixed_lines(G, pt_fixed_lines, save_dir):

    save_dir_images = os.path.join(save_dir, 'images')
    pt_lines_folder = os.path.join(save_dir_images, 'pt_fixed_lines')

    if not os.path.isdir(pt_lines_folder):
        os.mkdir(pt_lines_folder)

    for index, lines in pt_fixed_lines.iterrows():
        #bus_stop_list_nodes.append(stop['osmid_walk'])

        nc = ['r' if (str(node) in lines['osm_nodes']) else '#336699' for node in G.nodes()]
        ns = [12 if (str(node) in lines['osm_nodes']) else 6 for node in G.nodes()]
        fig, ax = ox.plot_graph(G, node_size=ns, show=False, node_color=nc, node_zorder=2, save=True, filepath=pt_lines_folder+'/'+str(index)+'_'+str(lines['name'])+'.pt_fixed_lines.png')
        plt.close(fig)

def get_fixed_lines_osm(G_walk, G_drive, polygon, save_dir, output_file_base):

    '''
    get fixed lines from OpenStreetMaps
    '''
    api_osm = osm.OsmApi()
    pt_fixed_lines = []

    save_dir_csv = os.path.join(save_dir, 'csv')

    if not os.path.isdir(save_dir_csv):
        os.mkdir(save_dir_csv)

    #pt = public transport
    path_pt_lines_csv_file = os.path.join(save_dir_csv, output_file_base+'.pt.lines.csv')

    if os.path.isfile(path_pt_lines_csv_file):
        print('is file pt routes')
        pt_fixed_lines = pd.read_csv(path_pt_lines_csv_file)

    else:
        index_line = 0
        print('creating file pt routes')

        tags = {
            'route':'bus',
            'route':'subway',
            'route':'tram',
        }
        
        routes = ox.geometries_from_polygon(polygon, tags=tags)

        #print('number of routes', len(routes))

        for index, poi in routes.iterrows():
            
            try:

                keys = poi.keys()
                print(poi)
                    
                if str(poi['nodes']) != 'nan':

                    print(poi)
                    
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

def plot_fixed_lines(network, save_dir):

    #plot all nodes in the network that have a fixed line passing by
    fl_stations_walk = [] 
    fl_stations_drive = []
    
    save_dir_images = os.path.join(save_dir, 'images')

    for node in network.nodes_covered_fixed_lines:

        fl_station_walk = network.deconet_network_nodes.loc[int(node), 'osmid_walk']
        fl_station_drive = network.deconet_network_nodes.loc[int(node), 'osmid_drive']
        
        fl_stations_walk.append(fl_station_walk)
        fl_stations_drive.append(fl_station_drive)

    stops_folder = os.path.join(save_dir_images, 'bus_stops')
    nc = ['r' if (node in fl_stations_drive) else '#336699' for node in network.G_drive.nodes()]
    ns = [12 if (node in fl_stations_drive) else 6 for node in network.G_drive.nodes()]
    fig, ax = ox.plot_graph(network.G_drive, node_size=ns, show=False, node_color=nc, node_zorder=2, save=True, filepath=stops_folder+'/fixed_lines_nodes_drive.png')
    plt.close(fig)

    nc = ['r' if (node in fl_stations_walk) else '#336699' for node in network.G_walk.nodes()]
    ns = [12 if (node in fl_stations_walk) else 6 for node in network.G_walk.nodes()]
    fig, ax = ox.plot_graph(network.G_walk, node_size=ns, show=False, node_color=nc, node_zorder=2, save=True, filepath=stops_folder+'/fixed_lines_nodes_walk.png')
    plt.close(fig)

