import json
import os
import osmnx as ox
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
import ray
from shapely.geometry import Point


def get_bus_stops(param, network, node_walk, max_walking, min_walk_speed, max_walk_speed, to_node_walk):

    #return bus stops that within walking threshold from node_walk

    #to_node_walk (true) => walking times are calculated to the node_walk to the stops
    #to_node_walk (false) => walking times are calculated from the stops to the node_walk

    stops = []
    stops_walking_time = []

    for index in network.list_bus_stops:

        osmid_possible_stop = int(network.bus_stops.loc[index, 'osmid_walk'])
        if to_node_walk:
            eta_walk = network.get_eta_walk(osmid_possible_stop, node_walk, max_walk_speed)
        else:
            eta_walk = network.get_eta_walk(node_walk, osmid_possible_stop, max_walk_speed)

        if eta_walk >= 0 and eta_walk <= max_walking:
            stops.append(index)
            stops_walking_time.append(eta_walk)

    return stops, stops_walking_time

def get_subway_routes(network, origin_node_walk, destination_node_walk, origin_node_drive, destination_node_drive, max_walking):
    
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
                stops_u, walking_time_u = get_bus_stops(param, network, u_walk, max_walking, min_walk_speed, max_walk_speed, to_node_walk=True)
                d['stops_u'] = stops_u
                d['walking_time_u'] = walking_time_u
                
            if d['option'] == 3:
                #get new pick up stops (around node v)
                stops_v, walking_time_v = get_bus_stops(param, network, v_walk, max_walking, min_walk_speed, max_walk_speed, to_node_walk=False)
                d['stops_v'] = stops_v
                d['walking_time_v'] = walking_time_v

            if d['option'] == 4:
                #get new drop off stops (around node u)
                stops_u, walking_time_u = get_bus_stops(param, network, u_walk, max_walking, min_walk_speed, max_walk_speed, to_node_walk=True)
                d['stops_u'] = stops_u
                d['walking_time_u'] = walking_time_u

                #get new pick up stations (around node v)
                stops_v, walking_time_v = get_bus_stops(param, network, v_walk, max_walking, min_walk_speed, max_walk_speed, to_node_walk=False)
                d['stops_v'] = stops_v
                d['walking_time_v'] = walking_time_v

            subway_routes.append(d)
    
    return subway_routes

def generate_requests_ODBRPFL( 
    network, 
    request_demand,
    min_early_departure,
    max_early_departure,
    min_walk_speed,
    max_walk_speed,
    max_walking,
    bus_factor,
    replicate_num,
    save_dir_json,
    output_folder_base
):
    
    '''
    generate requests for the on demand bus routing problem
    '''

    output_file_json = os.path.join(save_dir_json, output_folder_base + '_' + str(replicate_num) + '.json')
    instance_data = {}  

    print("Now generating " + " request_data")
    lines = []
    all_requests = {} 
    h = 0
    num_requests = 0

    #for each PDF
    for r in range(len(request_demand)):
        
        #randomly generates the earliest departure times or latest arrival times
        request_demand[r].set_demand()

        #randomly generates the zones 
        if request_demand[r].is_random_origin_zones:
            request_demand[r].randomly_set_origin_zones(len(network.zones))

        if request_demand[r].is_random_destination_zones:
            request_demand[r].randomly_set_destination_zones(len(network.zones))     

        num_requests = request_demand[r].num_requests
        print(num_requests)

        for i in range(num_requests):
            
            dep_time = None 
            arr_time = None

            #gets the arrival and departure time
            if request_demand[r].time_type == "EDT":
                dep_time = request_demand[r].demand[i]
                dep_time = int(dep_time)

                if (dep_time >= 0) and (dep_time >= min_early_departure) and (dep_time <= max_early_departure):
                    nok = True
                else:
                    nok = False
            else:
                if request_demand[r].time_type == "LAT":
                    arr_time = request_demand[r].demand[i]
                    arr_time = int(arr_time)

                    if (arr_time >= 0) and (arr_time >= min_early_departure) and (arr_time <= max_early_departure):
                        nok = True
                    else:
                        nok = False

            request_data = {}  #holds information about this request

            #randomly choosing coordinates and repeated for a passenger until he can go to at least 1 stop (for both origin and destination)
            
            num_attempts = 0 #limit the number of attempts to generate a request and avoid infinite loop
            while nok:
                num_attempts += 1
                nok = False
                
                
                origin_point = []
                destination_point = []
                unfeasible_request = True
                
                #def generate_feasible_request
                while unfeasible_request:

                    #generate coordinates for origin
                    if request_demand[r].num_origins == -1:

                        origin_point = network.get_random_coord(network.polygon_walk)
                        origin_point = (origin_point.y, origin_point.x)
    
                    else:

                        random_zone = np.random.uniform(0, request_demand[r].num_origins, 1)
                        random_zone = int(random_zone)

                        random_zone_id = int(request_demand[r].origin_zones[random_zone])
                        polygon_zone = network.zones.loc[random_zone_id]['polygon']
                        
                        #generate coordinates within the given zone
                        origin_point = network.get_random_coord(polygon_zone)
                        #ax.scatter(origin_point.x, origin_point.y, c='red')
                        origin_point = (origin_point.y, origin_point.x)

                    #generate coordinates for destination
                    if request_demand[r].num_destinations == -1:
                        
                        destination_point = network.get_random_coord(network.polygon_walk)
                        #ax.scatter(destination_point.x, destination_point.y, c='green')
                        destination_point = (destination_point.y, destination_point.x)

                    else:
                        
                        random_zone = np.random.uniform(0, request_demand[r].num_destinations, 1)
                        random_zone = int(random_zone)
                        
                        random_zone_id = int(request_demand[r].destination_zones[random_zone])
                        polygon_zone = network.zones.loc[random_zone_id]['polygon']
            
                        #generate coordinates within the given zone
                        destination_point = network.get_random_coord(polygon_zone)
                        #ax.scatter(destination_point.x, destination_point.y, c='green')
                        destination_point = (destination_point.y, destination_point.x)

                    origin_node_walk = ox.get_nearest_node(network.G_walk, origin_point)
                    destination_node_walk = ox.get_nearest_node(network.G_walk, destination_point)
                    time_walking = network.get_eta_walk(origin_node_walk, destination_node_walk, max_walk_speed)

                    #time walking from origin to destination must be higher of max_walking by the user
                    if time_walking > max_walking:
                        unfeasible_request = False
                
                origin_node_drive = ox.get_nearest_node(network.G_drive, origin_point)
                destination_node_drive = ox.get_nearest_node(network.G_drive, destination_point)
                
                stops_origin = []
                stops_destination = []

                stops_origin_walking_distance = []
                stops_destination_walking_distance = []

                if time_walking > max_walking: #if distance between origin and destination is too small the person just walks
                    #add the request
                    #calculates the stations which are close enough to the origin and destination of the request
                    for index in network.bus_stops_ids:
                    #for index, node in network.bus_stops.iterrows():

                        #osmid_possible_stop = int(stop_node['osmid_walk'])
                        osmid_possible_stop = int(network.bus_stops.loc[index, 'osmid_walk'])

                        eta_walk_origin = network.get_eta_walk(origin_node_walk, osmid_possible_stop, max_walk_speed)
                        if eta_walk_origin >= 0 and eta_walk_origin <= max_walking:
                            stops_origin.append(index)
                            stops_origin_walking_distance.append(eta_walk_origin)

                        #eta_walk_destination = network.get_eta_walk(destination_node_walk, osmid_possible_stop) 
                        eta_walk_destination = network.get_eta_walk(osmid_possible_stop, destination_node_walk, max_walk_speed)       
                        if eta_walk_destination >= 0 and eta_walk_destination <= max_walking:
                            stops_destination.append(index)
                            stops_destination_walking_distance.append(eta_walk_destination)

                # Check whether each passenger can walk to stops (origin + destination)
                if len(stops_origin) > 0 and len(stops_destination) > 0:
                    if not (set(stops_origin) & set(stops_destination)):
                        #time window for the arrival time
                        
                        if arr_time is None:
                            hour = int(math.floor(dep_time/(60*60)))
                            hour = 0

                            #print('dep time min, dep time hour: ', dep_time, hour)
                            max_eta_bus, min_eta_bus = network.return_estimated_travel_time_bus(stops_origin, stops_destination, hour)
                            if min_eta_bus >= 0:
                                arr_time = (dep_time) + (bus_factor * max_eta_bus) + (max_walking * 2)
                            else:
                                unfeasible_request = True
                        else:

                            if dep_time is None:
                                hour = int(arr_time/(60*60)) - 1
                                hour = 0
                                
                                #print('dep time min, dep time hour: ', dep_time, hour)
                                max_eta_bus, min_eta_bus = network.return_estimated_travel_time_bus(stops_origin, stops_destination, hour)
                                if min_eta_bus >= 0:
                                    dep_time = (arr_time) - (bus_factor * max_eta_bus) - (max_walking * 2)
                                else:
                                    unfeasible_request = True

                    else:
                        unfeasible_request = True
                else:
                    unfeasible_request = True

                if  not unfeasible_request:   
                    #prints in the json file if the request is feasible

                    #coordinate origin
                    request_data.update({'originx': origin_point[1]})
                    request_data.update({'originy': origin_point[0]})

                    #coordinate destination
                    request_data.update({'destinationx': destination_point[1]})
                    request_data.update({'destinationy': destination_point[0]})

                    request_data.update({'num_stops_origin': len(stops_origin_id)})
                    request_data.update({'stops_origin': stops_origin_id})
                    request_data.update({'walking_time_origin_to_stops': stops_origin_walking_distance})

                    request_data.update({'num_stops_destination': len(stops_destination_id)})
                    request_data.update({'stops_destination': stops_destination_id})
                    request_data.update({'walking_time_stops_to_destination': stops_destination_walking_distance})

                    #timestamp -> time the request was made
                    request_time_stamp = random.randint(0, dep_time)
                    request_data.update({'time_stamp': int(request_time_stamp)})
                    
                    #departure time
                    request_data.update({'dep_time': int(dep_time)})

                    #arrival time
                    request_data.update({'arr_time': int(arr_time)})

                    #when generating the requests, consider algo getting the fixed lines
                    if get_fixed_lines == 'deconet':
                        
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


                    # add request_data to instance_data container
                    all_requests.update({num_requests: request_data})

                    #increases the number of requests
                    num_requests += 1
                    print('#:', num_requests)

                else:
                    nok = True
                    if num_attempts > 20:
                        nok = False
        
        #plt.show()
        #plt.savefig('images/foo.png')
        #plt.close(fig) 

    #network.all_requests = all_requests

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

def generate_requests_DARP( 
    network, 
    request_demand,
    min_early_departure,
    max_early_departure,
    vehicle_factor,
    replicate_num,
    save_dir_json,
    output_folder_base
):
    
    '''
    generate requests for the dial-a-ride problem
    '''

    output_file_json = os.path.join(save_dir_json, output_folder_base + '_' + str(replicate_num) + '.json')
    instance_data = {}  

    print("Now generating " + " request_data")
    lines = []
    all_requests = {} 
    h = 0
    num_requests = 0

    #for each PDF
    for r in range(len(request_demand)):
        
        #fig, ax = ox.plot_graph(network.G_walk, show=False, close=False)

        #randomly generates the earliest departure times or latest arrival times
        request_demand[r].set_demand()

        #randomly generates the zones 
        if request_demand[r].is_random_origin_zones:
            request_demand[r].randomly_set_origin_zones(len(network.zones))

        if request_demand[r].is_random_destination_zones:
            request_demand[r].randomly_set_destination_zones(len(network.zones))     

        num_requests = request_demand[r].num_requests
        print(num_requests)

        for i in range(num_requests):
            
            dep_time = None 
            arr_time = None

            if request_demand[r].time_type == "EDT":
                dep_time = request_demand[r].demand[i]
                dep_time = int(dep_time)

                if (dep_time >= 0) and (dep_time >= min_early_departure) and (dep_time <= max_early_departure):
                    nok = True
                else:
                    nok = False
            else:
                if request_demand[r].time_type == "LAT":
                    arr_time = request_demand[r].demand[i]
                    arr_time = int(arr_time)

                    if (arr_time >= 0) and (arr_time >= min_early_departure) and (arr_time <= max_early_departure):
                        nok = True
                    else:
                        nok = False

            request_data = {}  #holds information about this request

            #randomly choosing coordinates and repeated for a passenger until he can go to at least 1 stop (for both origin and destination)
            
            num_attempts = 0 #limit the number of attempts to generate a request and avoid infinite loop
            while nok:
                num_attempts += 1
                nok = False
                
                origin_point = []
                destination_point = []
                unfeasible_request = True
                
                while unfeasible_request:

                    #generate coordinates for origin
                    if request_demand[r].num_origins == -1:

                        origin_point = network.get_random_coord(network.polygon_walk)
                        origin_point = (origin_point.y, origin_point.x)
    
                    else:

                        random_zone = np.random.uniform(0, request_demand[r].num_origins, 1)
                        random_zone = int(random_zone)
                        random_zone_id = int(request_demand[r].origin_zones[random_zone])
                        polygon_zone = network.zones.loc[random_zone_id]['polygon']
            
                        origin_point = network.get_random_coord(polygon_zone)
                        origin_point = (origin_point.y, origin_point.x)

                    #generate coordinates for destination
                    if request_demand[r].num_destinations == -1:
                        
                        destination_point = network.get_random_coord(network.polygon_walk)
                        destination_point = (destination_point.y, destination_point.x)

                    else:

                        random_zone = np.random.uniform(0, request_demand[r].num_destinations, 1)
                        random_zone = int(random_zone)
                        random_zone_id = int(request_demand[r].destination_zones[random_zone])
                        polygon_zone = network.zones.loc[random_zone_id]['polygon']

                        destination_point = network.get_random_coord(polygon_zone)
                        destination_point = (destination_point.y, destination_point.x)

                #origin_node_walk = ox.get_nearest_node(network.G_walk, origin_point)
                #destination_node_walk = ox.get_nearest_node(network.G_walk, destination_point)
                    
                
                origin_node_drive = ox.get_nearest_node(network.G_drive, origin_point)
                destination_node_drive = ox.get_nearest_node(network.G_drive, destination_point)
                
                if arr_time is None:

                    hour = int(math.floor(dep_time/(60*60)))
                    hour = 0

                    estimated_travel_time = network.return_estimated_travel_time(origin_node_drive, destination_node_drive)
                    if estimated_travel_time >= 0:
                        arr_time = (dep_time) + (vehicle_factor * estimated_travel_time) 
                    else:
                        unfeasible_request = True

                else:

                    if dep_time is None:

                        hour = int(arr_time/(60*60)) - 1
                        hour = 0
                        
                        estimated_travel_time = network.return_estimated_travel_time(origin_node_drive, destination_node_drive)
                        if estimated_travel_time >= 0:
                            dep_time = (arr_time) - (vehicle_factor * estimated_travel_time) 
                        else:
                            unfeasible_request = True

                    
                

                if  not unfeasible_request:   
                    #prints in the json file if the request is 'viable'
                    
                    request_data.update({'originx': origin_point[1]})
                    request_data.update({'originy': origin_point[0]})

                    request_data.update({'destinationx': destination_point[1]})
                    request_data.update({'destinationy': destination_point[0]})

                    
                    #timestamp -> time the request was made
                    request_time_stamp = random.randint(0, dep_time)
                    request_data.update({'time_stamp': int(request_time_stamp)})
                    
                    #departure time
                    request_data.update({'dep_time': int(dep_time)})

                    #arrival time
                    request_data.update({'arr_time': int(arr_time)})

                    # add request_data to instance_data container
                    all_requests.update({num_requests: request_data})

                    #increases the number of requests
                    num_requests += 1
                    print('#:', num_requests)

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

def generate_requests_ODBRP( 
    network, 
    request_demand,
    min_early_departure,
    max_early_departure,
    min_walk_speed,
    max_walk_speed,
    max_walking,
    bus_factor,
    replicate_num,
    save_dir_json,
    output_folder_base
):
    
    '''
    generate requests for the on demand bus routing problem
    '''

    output_file_json = os.path.join(save_dir_json, output_folder_base + '_' + str(replicate_num) + '.json')
    instance_data = {}  

    print("Now generating " + " request_data")
    lines = []
    all_requests = {} 
    h = 0
    num_requests = 0

    #for each PDF
    for r in range(len(request_demand)):
        
        #randomly generates the earliest departure times or latest arrival times
        request_demand[r].set_demand()

        #randomly generates the zones 
        if request_demand[r].is_random_origin_zones:
            request_demand[r].randomly_set_origin_zones(len(network.zones))

        if request_demand[r].is_random_destination_zones:
            request_demand[r].randomly_set_destination_zones(len(network.zones))     

        num_requests = request_demand[r].num_requests
        print(num_requests)

        for i in range(num_requests):
            
            dep_time = None 
            arr_time = None

            #gets the arrival and departure time
            if request_demand[r].time_type == "EDT":
                dep_time = request_demand[r].demand[i]
                dep_time = int(dep_time)

                if (dep_time >= 0) and (dep_time >= min_early_departure) and (dep_time <= max_early_departure):
                    nok = True
                else:
                    nok = False
            else:
                if request_demand[r].time_type == "LAT":
                    arr_time = request_demand[r].demand[i]
                    arr_time = int(arr_time)

                    if (arr_time >= 0) and (arr_time >= min_early_departure) and (arr_time <= max_early_departure):
                        nok = True
                    else:
                        nok = False

            #holds information about this request
            request_data = {}  

            #randomly choosing coordinates and repeated for a passenger until he can go to at least 1 stop (for both origin and destination)
            
            num_attempts = 0 #limit the number of attempts to generate a request and avoid infinite loop
            while nok:
                num_attempts += 1
                nok = False
                
                
                origin_point = []
                destination_point = []
                unfeasible_request = True
                
                #def generate_feasible_request
                while unfeasible_request:

                    #generate coordinates for origin
                    if request_demand[r].num_origins == -1:

                        origin_point = network.get_random_coord(network.polygon_walk)
                        origin_point = (origin_point.y, origin_point.x)
    
                    else:

                        random_zone = np.random.uniform(0, request_demand[r].num_origins, 1)
                        random_zone = int(random_zone)

                        random_zone_id = int(request_demand[r].origin_zones[random_zone])
                        polygon_zone = network.zones.loc[random_zone_id]['polygon']
                        
                        #generate coordinates within the given zone
                        origin_point = network.get_random_coord(polygon_zone)
                        #ax.scatter(origin_point.x, origin_point.y, c='red')
                        origin_point = (origin_point.y, origin_point.x)

                    #generate coordinates for destination
                    if request_demand[r].num_destinations == -1:
                        
                        destination_point = network.get_random_coord(network.polygon_walk)
                        #ax.scatter(destination_point.x, destination_point.y, c='green')
                        destination_point = (destination_point.y, destination_point.x)

                    else:
                        
                        random_zone = np.random.uniform(0, request_demand[r].num_destinations, 1)
                        random_zone = int(random_zone)
                        
                        random_zone_id = int(request_demand[r].destination_zones[random_zone])
                        polygon_zone = network.zones.loc[random_zone_id]['polygon']
            
                        #generate coordinates within the given zone
                        destination_point = network.get_random_coord(polygon_zone)
                        #ax.scatter(destination_point.x, destination_point.y, c='green')
                        destination_point = (destination_point.y, destination_point.x)

                    origin_node_walk = ox.get_nearest_node(network.G_walk, origin_point)
                    destination_node_walk = ox.get_nearest_node(network.G_walk, destination_point)
                    time_walking = network.get_eta_walk(origin_node_walk, destination_node_walk, max_walk_speed)

                    #time walking from origin to destination must be higher of max_walking by the user
                    if time_walking > max_walking:
                        unfeasible_request = False
                
                origin_node_drive = ox.get_nearest_node(network.G_drive, origin_point)
                destination_node_drive = ox.get_nearest_node(network.G_drive, destination_point)
                
                stops_origin = []
                stops_destination = []

                stops_origin_walking_distance = []
                stops_destination_walking_distance = []

                if time_walking > max_walking: #if distance between origin and destination is too small the person just walks
                    

                    #calculates the stations which are close enough to the origin and destination of the request
                    for index in network.bus_stops_ids:
                    
                        osmid_possible_stop = int(network.bus_stops.loc[index, 'osmid_walk'])

                        eta_walk_origin = network.get_eta_walk(origin_node_walk, osmid_possible_stop, max_walk_speed)
                        if eta_walk_origin >= 0 and eta_walk_origin <= max_walking:
                            stops_origin.append(index)
                            stops_origin_walking_distance.append(eta_walk_origin)

                        #eta_walk_destination = network.get_eta_walk(destination_node_walk, osmid_possible_stop) 
                        eta_walk_destination = network.get_eta_walk(osmid_possible_stop, destination_node_walk, max_walk_speed)       
                        if eta_walk_destination >= 0 and eta_walk_destination <= max_walking:
                            stops_destination.append(index)
                            stops_destination_walking_distance.append(eta_walk_destination)

                # Check whether each passenger can walk to stops (origin + destination)
                if len(stops_origin) > 0 and len(stops_destination) > 0:
                    if not (set(stops_origin) & set(stops_destination)):
                        #time window for the arrival time
                        
                        if arr_time is None:
                            hour = int(math.floor(dep_time/(60*60)))
                            hour = 0

                            #print('dep time min, dep time hour: ', dep_time, hour)
                            max_eta_bus, min_eta_bus = network.return_estimated_travel_time_bus(stops_origin, stops_destination, hour)
                            if min_eta_bus >= 0:
                                arr_time = (dep_time) + (bus_factor * max_eta_bus) + (max_walking * 2)
                            else:
                                unfeasible_request = True
                        else:

                            if dep_time is None:
                                hour = int(arr_time/(60*60)) - 1
                                hour = 0
                                
                                #print('dep time min, dep time hour: ', dep_time, hour)
                                max_eta_bus, min_eta_bus = network.return_estimated_travel_time_bus(stops_origin, stops_destination, hour)
                                if min_eta_bus >= 0:
                                    dep_time = (arr_time) - (bus_factor * max_eta_bus) - (max_walking * 2)
                                else:
                                    unfeasible_request = True

                    else:
                        unfeasible_request = True
                else:
                    unfeasible_request = True

                if  not unfeasible_request:   
                    #prints in the json file if the request is feasible

                    #coordinate origin
                    request_data.update({'originx': origin_point[1]})
                    request_data.update({'originy': origin_point[0]})

                    #coordinate destination
                    request_data.update({'destinationx': destination_point[1]})
                    request_data.update({'destinationy': destination_point[0]})

                    request_data.update({'num_stops_origin': len(stops_origin_id)})
                    request_data.update({'stops_origin': stops_origin_id})
                    request_data.update({'walking_time_origin_to_stops': stops_origin_walking_distance})

                    request_data.update({'num_stops_destination': len(stops_destination_id)})
                    request_data.update({'stops_destination': stops_destination_id})
                    request_data.update({'walking_time_stops_to_destination': stops_destination_walking_distance})

                    #timestamp -> time the request was made
                    request_time_stamp = random.randint(0, dep_time)
                    request_data.update({'time_stamp': int(request_time_stamp)})
                    
                    #departure time
                    request_data.update({'dep_time': int(dep_time)})

                    #arrival time
                    request_data.update({'arr_time': int(arr_time)})

                    # add request_data to instance_data container
                    all_requests.update({num_requests: request_data})

                    #increases the number of requests
                    num_requests += 1
                    print('#:', num_requests)

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

def generate_requests_SBRP( 
    network, 
    school,
    request_demand,
    min_early_departure,
    max_early_departure,
    min_walk_speed,
    max_walk_speed,
    max_walking,
    bus_factor,
    replicate_num,
    save_dir_json,
    output_folder_base
):
    
    output_file_json = os.path.join(save_dir_json, output_folder_base + '_' + str(replicate_num) + '.json')
    instance_data = {}  

    print("Now generating " + " request_data")
    lines = []
    all_requests = {} 
    h = 0
    num_requests = 0

    #for each PDF
    for r in range(len(request_demand)):
         
        if request_demand[r].is_random_origin_zones:
            request_demand[r].randomly_set_origin_zones(len(network.zones))

        #destination is the school
          
        num_requests = request_demand[r].num_requests
        print(num_requests)

        for i in range(num_requests):
            
            nok = True
            request_data = {}  #holds information about this request

            #randomly choosing coordinates and repeated for a passenger until he can go to at least 1 stop (for both origin and destination)
            
            num_attempts = 0 #limit the number of attempts to generate a request and avoid infinite loop
            while nok:
                num_attempts += 1
                nok = False
                
                origin_point = []
                destination_point = []
                unfeasible_request = True
                
                #def generate_feasible_request
                while unfeasible_request:

                    #generate coordinates for origin
                    if request_demand[r].num_origins == -1:
                        origin_point = network.get_random_coord(network.polygon_walk)
                        origin_point = (origin_point.y, origin_point.x)
    
                    else:
                        random_zone = np.random.uniform(0, request_demand[r].num_origins, 1)
                        random_zone = int(random_zone)
                        random_zone_id = int(request_demand[r].origin_zones[random_zone])
                        polygon_zone = network.zones.loc[random_zone_id]['polygon']
                        
                        origin_point = network.get_random_coord(polygon_zone)
                        origin_point = (origin_point.y, origin_point.x)

                    #destination is the school
                    destination_point = (school['y'], school['x'])

                    origin_node_walk = ox.get_nearest_node(network.G_walk, origin_point)
                    destination_node_walk = ox.get_nearest_node(network.G_walk, destination_point)
                    time_walking = network.get_eta_walk(origin_node_walk, destination_node_walk)

                    #print(time_walking)
                    if time_walking > max_walking:
                        unfeasible_request = False
                
                origin_node_drive = ox.get_nearest_node(network.G_drive, origin_point)
                destination_node_drive = ox.get_nearest_node(network.G_drive, destination_point)
                
                stops_origin = []
                stops_origin_walking_distance = []
                
                #if distance between origin and destination is too small the person just walks
                if time_walking > max_walking: 
                    
                    #add the request
                    
                    #calculates the stations which are close enough to the origin and destination of the request
                    for index in network.bus_stops_ids:
                    
                        #osmid_possible_stop = int(stop_node['osmid_walk'])
                        osmid_possible_stop = int(network.bus_stops.loc[index, 'osmid_walk'])

                        eta_walk_origin = network.get_eta_walk(origin_node_walk, osmid_possible_stop)
                        if eta_walk_origin >= 0 and eta_walk_origin <= max_walking:
                            stops_origin.append(index)
                            stops_origin_id.append(index)
                            stops_origin_walking_distance.append(eta_walk_origin)
  
                else:

                    unfeasible_request = True

                if  not unfeasible_request:   
                    #outputs in the json file if the request is 'feasible'
                    
                    request_data.update({'originx': origin_point[1]})
                    request_data.update({'originy': origin_point[0]})

                    request_data.update({'destinationx': destination_point[1]})
                    request_data.update({'destinationy': destination_point[0]})

                    request_data.update({'num_stops_origin': len(stops_origin_id)})
                    request_data.update({'stops_origin': stops_origin_id})
                    request_data.update({'walking_time_origin_to_stops': stops_origin_walking_distance})

                    # add request_data to instance_data container
                    all_requests.update({num_requests: request_data})

                    #increases the number of requests
                    num_requests += 1
                    print('#:', num_requests)

                else:
                    nok = True
                    if num_attempts > 20:
                        nok = False
        
        #plt.show()
        #plt.savefig('images/foo.png')
        #plt.close(fig) 

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


