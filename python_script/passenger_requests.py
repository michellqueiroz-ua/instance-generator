import json
import os
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from shapely.geometry import Point

import osmnx as ox

def get_bus_stops(param, network, node_walk, to_node_walk):

    #return bus stops that within walking threshold from node_walk

    #to_node_walk (true) => walking times are calculated to the node_walk to the stops
    #to_node_walk (false) => walking times are calculated from the stops to the node_walk

    stops = []
    stops_walking_time = []

    for index in network.list_bus_stops:

        osmid_possible_stop = int(network.bus_stops.loc[index, 'osmid_walk'])
        if to_node_walk:
            eta_walk = network.get_eta_walk(osmid_possible_stop, node_walk)
        else:
            eta_walk = network.get_eta_walk(node_walk, osmid_possible_stop)

        if eta_walk >= 0 and eta_walk <= param.max_walking:
            stops.append(index)
            stops_walking_time.append(eta_walk)

    return stops, stops_walking_time

def get_subway_routes(param, network, origin_node_walk, destination_node_walk, origin_node_drive, destination_node_drive):
    
    subway_routes = []

    #distance between origin and destinations nodes
    dist_od = network.shortest_path_drive.loc[origin_node_drive, str(destination_node_drive)]

    #fl_stations_walk = [] 
    #fl_stations_drive = []

    #check maybe KAPPA station here before this next loop

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
            #'dist_ou_walk': math.inf,
            #'dist_vd_walk': math.inf,
        }

        for u in network.nodes_covered_fixed_lines:
            for v in network.nodes_covered_fixed_lines:

                if u != v :
                
                    try:

                        eta = nx.dijkstra_path_length(network.subway_lines[lid]['route_graph'], u, v, weight='duration_avg')

                        u_drive = int(network.deconet_network_nodes.loc[int(u), 'osmid_drive'])
                        v_drive = int(network.deconet_network_nodes.loc[int(v), 'osmid_drive'])

                        u_walk = int(network.deconet_network_nodes.loc[int(u), 'osmid_walk'])
                        v_walk = int(network.deconet_network_nodes.loc[int(v), 'osmid_walk'])
                        
                        #distance from origin to node u and  #distance from v to destination
                        dist_ou_drive = network.shortest_path_drive.loc[origin_node_drive, str(u_drive)]
                        dist_vd_drive = network.shortest_path_drive.loc[v_drive, str(destination_node_drive)]

                        #dist_ou_walk = network.shortest_path_walk.loc[origin_node_walk, str(u_walk)]
                        #dist_vd_walk = network.shortest_path_walk.loc[v_walk, str(destination_node_walk)]
                        dist_ou_walk = network.get_eta_walk(origin_node_walk, u_walk)
                        dist_vd_walk = network.get_eta_walk(v_walk, destination_node_walk)

                        eta_vd_walk = math.inf
                        eta_ou_walk = math.inf

                        if not math.isnan(dist_vd_walk):
                            eta_vd_walk = int(math.ceil(dist_vd_walk/network.walk_speed))
                        if not math.isnan(dist_ou_walk):
                            eta_ou_walk = int(math.ceil(dist_ou_walk/network.walk_speed))

                       
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
                                if (eta_ou_walk <= param.max_walking) and (eta_vd_walk <= param.max_walking):
                                    d['option'] = 1
                                    d['walking_time_u'] = eta_ou_walk
                                    d['walking_time_v'] = eta_vd_walk

                                # ON DEMAND + FIXED LINE
                                if (eta_ou_walk > param.max_walking) and (eta_vd_walk <= param.max_walking):
                                    d['option'] = 2
                                    d['walking_time_v'] = eta_vd_walk

                                # FIXED LINE + ON DEMAND
                                if (eta_ou_walk <= param.max_walking) and (eta_vd_walk > param.max_walking):
                                    d['option'] = 3
                                    d['walking_time_u'] = eta_ou_walk
                                   
                                # ON DEMAND + FIXED LINE + ON DEMAND
                                if (eta_ou_walk > param.max_walking) and (eta_vd_walk > param.max_walking):
                                    d['option'] = 4
                                

                            else:
                                #otherwise, check if distance from origin to station is smaller than before
                                if not math.isnan(dist_ou_drive):
                                    if (dist_vd_drive == d['dist_vd_drive']) and (dist_ou_drive < d['dist_ou_drive']):

                                        d['u'] = u
                                        d['v'] = v
                                        d['eta'] = eta    
                                        d['dist_ou_drive'] = dist_ou_drive
                                        d['dist_vd_drive'] = dist_vd_drive

                                        #FIXED LINE ONLY
                                        if (eta_ou_walk <= param.max_walking) and (eta_vd_walk <= param.max_walking):
                                            d['option'] = 1
                                            d['walking_time_u'] = eta_ou_walk
                                            d['walking_time_v'] = eta_vd_walk

                                        # ON DEMAND + FIXED LINE
                                        if (eta_ou_walk > param.max_walking) and (eta_vd_walk <= param.max_walking):
                                            d['option'] = 2
                                            d['walking_time_v'] = eta_vd_walk

                                        # FIXED LINE + ON DEMAND
                                        if (eta_ou_walk <= param.max_walking) and (eta_vd_walk > param.max_walking):
                                            d['option'] = 3
                                            d['walking_time_u'] = eta_ou_walk
                                            
                                        # ON DEMAND + FIXED LINE + ON DEMAND
                                        if (eta_ou_walk > param.max_walking) and (eta_vd_walk > param.max_walking):
                                            d['option'] = 4
                                            

                    except (nx.NetworkXNoPath, KeyError, nx.NodeNotFound):

                        pass

        if not math.isnan(d['u']):

            u_walk = int(network.deconet_network_nodes.loc[int(d['u']), 'osmid_walk'])
            v_walk = int(network.deconet_network_nodes.loc[int(d['v']), 'osmid_walk'])

            if d['option'] == 2:
                #get new drop off stops (around node u)
                stops_u, walking_time_u = get_bus_stops(param, network, u_walk, to_node_walk=True)
                d['stops_u'] = stops_u
                d['walking_time_u'] = walking_time_u
                
            if d['option'] == 3:
                #get new pick up stops (around node v)
                stops_v, walking_time_v = get_bus_stops(param, network, v_walk, to_node_walk=False)
                d['stops_v'] = stops_v
                d['walking_time_v'] = walking_time_v

            if d['option'] == 4:
                #get new drop off stops (around node u)
                stops_u, walking_time_u = get_bus_stops(param, network, u_walk, to_node_walk=True)
                d['stops_u'] = stops_u
                d['walking_time_u'] = walking_time_u

                #get new pick up stations (around node v)
                stops_v, walking_time_v = get_bus_stops(param, network, v_walk, to_node_walk=False)
                d['stops_v'] = stops_v
                d['walking_time_v'] = walking_time_v

            subway_routes.append(d)
    
    return subway_routes

def generate_requests(param, network, replicate):

    output_file_json = os.path.join(param.save_dir_json, param.output_file_base + '_' + str(replicate) + '.json')
    instance_data = {}  

    print("Now generating " + " request_data")
    lines = []
    all_requests = {} 
    h = 0
    param.num_requests = 0

    #for each PDF
    for r in range(len(param.request_demand)):
        
        #fig, ax = ox.plot_graph(network.G_walk, show=False, close=False)

        #randomly generates the earliest departure times or latest arrival times
        param.request_demand[r].set_demand()

        #randomly generates the zones 
        
        #num_zones = len(network.zones)
        #print('tam zones', num_zones)
        
        if param.request_demand[r].is_random_origin_zones:
            param.request_demand[r].randomly_set_origin_zones(len(network.zones))

        if param.request_demand[r].is_random_destination_zones:
            param.request_demand[r].randomly_set_destination_zones(len(network.zones))     

        num_requests = param.request_demand[r].num_requests
        print(num_requests)

        for i in range(num_requests):
            
            dep_time = None 
            arr_time = None

            if param.request_demand[r].time_type == "EDT":
                dep_time = param.request_demand[r].demand[i]
                dep_time = int(dep_time)

                if (dep_time >= 0) and (dep_time >= param.min_early_departure) and (dep_time <= param.max_early_departure):
                    nok = True
                else:
                    nok = False
            else:
                if param.request_demand[r].time_type == "LAT":
                    arr_time = param.request_demand[r].demand[i]
                    arr_time = int(arr_time)

                    if (arr_time >= 0) and (arr_time >= param.min_early_departure) and (arr_time <= param.max_early_departure):
                        nok = True
                    else:
                        nok = False

            request_data = {}  #holds information about this request

            #randomly choosing coordinates and repeated for a passenger until he can go to at least 1 stop (for both origin and destination)
            
            num_attempts = 0 #limit the number of attempts to generate a request and avoid infinite loop
            while nok:
                num_attempts += 1
                nok = False
                
                route_length_drive = -1
                origin_point = []
                destination_point = []
                unfeasible_request = True
                
                #def generate_feasible_request
                while unfeasible_request:

                    #generate coordinates for origin and destination
                    if param.request_demand[r].num_origins == -1:
                        origin_point = network.get_random_coord(network.polygon_walk)
                        #ax.scatter(origin_point.x, origin_point.y, c='red')
                        origin_point = (origin_point.y, origin_point.x)
    
                    else:
                        random_zone = np.random.uniform(0, param.request_demand[r].num_origins, 1)
                        random_zone = int(random_zone)
                        #print('index origin: ', param.request_demand[r].origin_zones[random_zone])
                        random_zone_id = int(param.request_demand[r].origin_zones[random_zone])
                        polygon_zone = network.zones.loc[random_zone_id]['polygon']
                        #print(random_zone)
                        
                        origin_point = network.get_random_coord(polygon_zone)
                        #ax.scatter(origin_point.x, origin_point.y, c='red')
                        #print(origin_point.y, origin_point.x)
                        origin_point = (origin_point.y, origin_point.x)

                    if param.request_demand[r].num_destinations == -1:
                        
                        destination_point = network.get_random_coord(network.polygon_walk)
                        #ax.scatter(destination_point.x, destination_point.y, c='green')
                        destination_point = (destination_point.y, destination_point.x)

                    else:
                        random_zone = np.random.uniform(0, param.request_demand[r].num_destinations, 1)
                        random_zone = int(random_zone)
                        #print('index dest: ', param.request_demand[r].destination_zones[random_zone])
                        random_zone_id = int(param.request_demand[r].destination_zones[random_zone])
                        polygon_zone = network.zones.loc[random_zone_id]['polygon']
                        #print(random_zone)
                        destination_point = network.get_random_coord(polygon_zone)
                        #ax.scatter(destination_point.x, destination_point.y, c='green')

                        destination_point = (destination_point.y, destination_point.x)

                    #print('chegou aq')
                    origin_node_walk = ox.get_nearest_node(network.G_walk, origin_point)
                    destination_node_walk = ox.get_nearest_node(network.G_walk, destination_point)
                    time_walking = network.get_eta_walk(origin_node_walk, destination_node_walk)

                    #print(time_walking)
                    if time_walking > param.max_walking:
                        unfeasible_request = False
                
                origin_node_drive = ox.get_nearest_node(network.G_drive, origin_point)
                destination_node_drive = ox.get_nearest_node(network.G_drive, destination_point)
                
                stops_origin = []
                stops_destination = []

                #REMOVE THIS
                #for distance matrix c++. regular indexing (0, 1, 2...)
                stops_origin_id = []
                stops_destination_id = []

                stops_origin_walking_distance = []
                stops_destination_walking_distance = []

                if time_walking > param.max_walking: #if distance between origin and destination is too small the person just walks
                    #add the request
                    #calculates the stations which are close enough to the origin and destination of the request
                    for index in network.bus_stops_ids:
                    #for index, node in network.bus_stops.iterrows():

                        #osmid_possible_stop = int(stop_node['osmid_walk'])
                        osmid_possible_stop = int(network.bus_stops.loc[index, 'osmid_walk'])

                        eta_walk_origin = network.get_eta_walk(origin_node_walk, osmid_possible_stop)
                        if eta_walk_origin >= 0 and eta_walk_origin <= param.max_walking:
                            stops_origin.append(index)
                            #stops_origin_id.append(int(stop_node['itid']))
                            stops_origin_id.append(index)
                            stops_origin_walking_distance.append(eta_walk_origin)

                        #eta_walk_destination = network.get_eta_walk(destination_node_walk, osmid_possible_stop) 
                        eta_walk_destination = network.get_eta_walk(osmid_possible_stop, destination_node_walk)       
                        if eta_walk_destination >= 0 and eta_walk_destination <= param.max_walking:
                            stops_destination.append(index)
                            #stops_destination_id.append(int(stop_node['itid']))
                            stops_destination_id.append(index)
                            stops_destination_walking_distance.append(eta_walk_destination)

                #print('aqui')
                #print(time_walking)
                # Check whether each passenger can walk to stops (origin + destination)
                if len(stops_origin) > 0 and len(stops_destination) > 0:
                    if not (set(stops_origin) & set(stops_destination)):
                        #time window for the arrival time
                        
                        if arr_time is None:
                            hour = int(math.floor(dep_time/(60*60)))
                            hour = 0

                            #print('dep time min, dep time hour: ', dep_time, hour)
                            max_eta_bus, min_eta_bus = network.return_estimated_arrival_bus_osmnx(stops_origin, stops_destination, hour)
                            if min_eta_bus >= 0:
                                arr_time = (dep_time) + (param.bus_factor * max_eta_bus) + (param.max_walking * 2)
                            else:
                                unfeasible_request = True
                        else:

                            if dep_time is None:
                                hour = int(arr_time/(60*60)) - 1
                                hour = 0
                                
                                #print('dep time min, dep time hour: ', dep_time, hour)
                                max_eta_bus, min_eta_bus = network.return_estimated_arrival_bus_osmnx(stops_origin, stops_destination, hour)
                                if min_eta_bus >= 0:
                                    dep_time = (arr_time) - (param.bus_factor * max_eta_bus) - (param.max_walking * 2)
                                else:
                                    unfeasible_request = True

                    else:
                        unfeasible_request = True
                else:
                    unfeasible_request = True

                if  not unfeasible_request:   
                    #prints in the json file if the request is 'viable'
                    
                    request_data.update({'originx': origin_point[1]})
                    request_data.update({'originy': origin_point[0]})

                    request_data.update({'destinationx': destination_point[1]})
                    request_data.update({'destinationy': destination_point[0]})

                    '''
                    #used for testing
                    if param.num_requests == 23:
                        origin_node = network.distance_matrix.loc[(stops_origin[0],stops_destination[0]), 'origin_osmid_drive']
                        destination_node = network.distance_matrix.loc[(stops_origin[0],stops_destination[0]), 'destination_osmid_drive']
                        route = nx.shortest_path(network.G_drive, origin_node, destination_node, weight='length')
                        fig, ax = ox.plot_graph_route(network.G_drive, route, origin_point=origin_point, destination_point=destination_point)
                    '''

                    #maybe add osmid also
                    request_data.update({'num_stops_origin': len(stops_origin_id)})
                    request_data.update({'stops_origin': stops_origin_id})
                    request_data.update({'walking_time_origin_to_stops': stops_origin_walking_distance})

                    request_data.update({'num_stops_destination': len(stops_destination_id)})
                    request_data.update({'stops_destination': stops_destination_id})
                    request_data.update({'walking_time_stops_to_destination': stops_destination_walking_distance})

                    #departure time
                    request_data.update({'dep_time': int(dep_time)})
                    #arrival time
                    request_data.update({'arr_time': int(arr_time)})

                    # add request_data to instance_data container
                    all_requests.update({param.num_requests: request_data})
                    
                    
                    subway_routes = get_subway_routes(param, network, origin_node_walk, destination_node_walk, origin_node_drive, destination_node_drive)
                    
                    request_data.update({'num_subway_routes': len(subway_routes)})

                    subway_line_ids = []
                    for route in subway_routes:
                        subway_line_ids.append(route['line_id'])

                    request_data.update({'subway_line_ids': subway_line_ids})


                    for route in subway_routes:

                        line_id = str(route['line_id'])
                        #add line_id como acrescimo no fim, pq senao a tag fica repetida
                        d = {'line_id': line_id}
                        d['option'+line_id] = route['option']

                        if route['option'] == 1:
                            d['eta_in_vehicle'+line_id] = route['eta']
                            d['walking_time_to_pick_up'+line_id] = route['walking_time_u']
                            d['walking_time_from_drop_off'+line_id] = route['walking_time_v']

                        if route['option'] == 2:
                            #request_data.update({'line_id': route['line_id']})
                            d['eta_in_vehicle'+line_id] = route['eta']
                            d['num_stops_nearby_pick_up'+line_id] = len(route['stops_u'])
                            d['stops_nearby_pick_up'+line_id] = route['stops_u']
                            d['walking_time_to_pick_up'+line_id] = route['walking_time_u']
                            d['walking_time_from_drop_off'+line_id] = route['walking_time_v']

                        if route['option'] == 3:
                            #request_data.update({'line_id': route['line_id']})
                            d['eta_in_vehicle'+line_id] = route['eta']
                            d['walking_time_to_pick_up'+line_id] = route['walking_time_u']
                            d['num_stops_nearby_drop_off'+line_id] = len(route['stops_v'])
                            d['stops_nearby_drop_off'+line_id] = route['stops_v']
                            d['walking_time_from_drop_off'+line_id] =  route['walking_time_v']

                        if route['option'] == 4:
                            #request_data.update({'line_id': route['line_id']})
                            d['eta_in_vehicle'+line_id] = route['eta']
                            d['num_stops_nearby_pick_up'+line_id] = len(route['stops_u'])
                            d['stops_nearby_pick_up'+line_id] = route['stops_u']
                            d['walking_time_to_pick_up'+line_id] = route['walking_time_u']
                            d['num_stops_nearby_drop_off'+line_id] = len(route['stops_v'])
                            d['stops_nearby_drop_off'+line_id] = route['stops_v']
                            d['walking_time_from_drop_off'+line_id] = route['walking_time_v']
                            
                        request_data.update(d)

                    #increases the number of requests
                    param.num_requests += 1
                    print('#:', param.num_requests)

                else:
                    nok = True
                    if num_attempts > 20:
                        nok = False
        
        #plt.show()
        #plt.savefig('images/foo.png')
        #plt.close(fig) 

    network.all_requests = all_requests

    #travel_time_json = network.get_travel_time_matrix_osmnx("bus", 0)
    #travel_time_json = travel_time_json.tolist()
    travel_time_json = []
    instance_data.update({'requests': all_requests})
    #instance_data.update({'requests': all_requests,
    #                      'num_stations': network.num_stations,
    #                      'distance_matrix': travel_time_json})

    with open(output_file_json, 'w') as file:
        json.dump(instance_data, file, indent=4)
        file.close()