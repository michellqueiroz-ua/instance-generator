from fixed_lines import retrieve_new_bus_stations
from fixed_lines import check_subway_routes_serve_passenger
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

def generate_requests_ODBRPFL( 
    network, 
    request_demand,
    min_early_departure,
    max_early_departure,
    min_walk_speed,
    max_walk_speed,
    max_walking,
    vehicle_factor,
    inbound_outbound_factor,
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

                        origin_point = network.get_random_coord(network.polygon)
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
                        
                        destination_point = network.get_random_coord(network.polygon)
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
                    time_walking = network.get_eta_walk(origin_node_walk, destination_node_walk, min_walk_speed, max_walk_speed)

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

                        eta_walk_origin = network.get_eta_walk(origin_node_walk, osmid_possible_stop, min_walk_speed, max_walk_speed)
                        if eta_walk_origin >= 0 and eta_walk_origin <= max_walking:
                            stops_origin.append(index)
                            stops_origin_walking_distance.append(eta_walk_origin)

                        #eta_walk_destination = network.get_eta_walk(destination_node_walk, osmid_possible_stop) 
                        eta_walk_destination = network.get_eta_walk(osmid_possible_stop, destination_node_walk, min_walk_speed, max_walk_speed)       
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
                                arr_time = (dep_time) + (vehicle_factor * max_eta_bus) + (max_walking * 2)
                            else:
                                unfeasible_request = True
                        else:

                            if dep_time is None:
                                hour = int(arr_time/(60*60)) - 1
                                hour = 0
                                
                                #print('dep time min, dep time hour: ', dep_time, hour)
                                max_eta_bus, min_eta_bus = network.return_estimated_travel_time_bus(stops_origin, stops_destination, hour)
                                if min_eta_bus >= 0:
                                    dep_time = (arr_time) - (vehicle_factor * max_eta_bus) - (max_walking * 2)
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

                    request_data.update({'num_stops_origin': len(stops_origin)})
                    request_data.update({'stops_origin': stops_origin})
                    request_data.update({'walking_time_origin_to_stops': stops_origin_walking_distance})

                    request_data.update({'num_stops_destination': len(stops_destination)})
                    request_data.update({'stops_destination': stops_destination})
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
                        
                        subway_routes = check_subway_routes_serve_passenger(param, network, origin_node_walk, destination_node_walk, origin_node_drive, destination_node_drive)
                        
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
                    all_requests.update({i: request_data})

                    #increases the number of requests
                    print('#:', i)

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
    inbound_outbound_factor,
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
        i=0
        while i < num_requests:

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
                unfeasible_request = False
                
                #generate coordinates for origin
                if request_demand[r].num_origins == -1:

                    origin_point = network.get_random_coord(network.polygon)
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
                    
                    destination_point = network.get_random_coord(network.polygon)
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

                    estimated_travel_time = network.return_estimated_travel_time_drive(origin_node_drive, destination_node_drive)
                    if estimated_travel_time >= 0:
                        arr_time = (dep_time) + (vehicle_factor * estimated_travel_time) 
                    else:
                        unfeasible_request = True

                else:

                    if dep_time is None:

                        hour = int(arr_time/(60*60)) - 1
                        hour = 0
                        
                        estimated_travel_time = network.return_estimated_travel_time_drive(origin_node_drive, destination_node_drive)
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
                    all_requests.update({i: request_data})

                    #increases the number of requests
                    i += 1
                    print('#:', i)

                    
                    create_return_request = random.randint(0, 100)

                    if create_return_request <= inbound_outbound_factor*100 and arr_time <= max_early_departure and i < num_requests:

                        request_data_return = {}

                        request_data_return.update({'destinationx': origin_point[1]})
                        request_data_return.update({'destinationy': origin_point[0]})

                        request_data_return.update({'originx': destination_point[1]})
                        request_data_return.update({'originy': destination_point[0]})

                        #timestamp -> time the request was made
                        request_data.update({'time_stamp': int(request_time_stamp)})
                        
                        #departure time for the return
                        dep_time_return = random.randint(arr_time, max_early_departure)
                        request_data_return.update({'dep_time': int(dep_time_return)})

                        #arrival time
                        estimated_travel_time = network.return_estimated_travel_time_drive(destination_node_drive, origin_node_drive)
                        arr_time_return = (dep_time_return) + (vehicle_factor * estimated_travel_time)
                        request_data_return.update({'arr_time': int(arr_time_return)})

                        all_requests.update({num_requests: request_data_return})

                        #increases the number of requests
                        i += 1
                        print('#r:', i)
                else:
                    nok = True
                    if num_attempts > 20:
                        nok = False
            i += 1
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
    vehicle_factor,
    inbound_outbound_factor,
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
        i = 0
        while i < num_requests:
            
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

                        origin_point = network.get_random_coord(network.polygon)
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
                        
                        destination_point = network.get_random_coord(network.polygon)
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
                    time_walking = network.get_eta_walk(origin_node_walk, destination_node_walk, min_walk_speed, max_walk_speed)

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

                        eta_walk_origin = network.get_eta_walk(origin_node_walk, osmid_possible_stop, min_walk_speed, max_walk_speed)
                        if eta_walk_origin >= 0 and eta_walk_origin <= max_walking:
                            stops_origin.append(index)
                            stops_origin_walking_distance.append(eta_walk_origin)

                        #eta_walk_destination = network.get_eta_walk(destination_node_walk, osmid_possible_stop) 
                        eta_walk_destination = network.get_eta_walk(osmid_possible_stop, destination_node_walk, min_walk_speed, max_walk_speed)       
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
                                arr_time = (dep_time) + (vehicle_factor * max_eta_bus) + (max_walking * 2)
                            else:
                                unfeasible_request = True
                        else:

                            if dep_time is None:
                                hour = int(arr_time/(60*60)) - 1
                                hour = 0
                                
                                #print('dep time min, dep time hour: ', dep_time, hour)
                                max_eta_bus, min_eta_bus = network.return_estimated_travel_time_bus(stops_origin, stops_destination, hour)
                                if min_eta_bus >= 0:
                                    dep_time = (arr_time) - (vehicle_factor * max_eta_bus) - (max_walking * 2)
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

                    request_data.update({'num_stops_origin': len(stops_origin)})
                    request_data.update({'stops_origin': stops_origin})
                    request_data.update({'walking_time_origin_to_stops': stops_origin_walking_distance})

                    request_data.update({'num_stops_destination': len(stops_destination)})
                    request_data.update({'stops_destination': stops_destination})
                    request_data.update({'walking_time_stops_to_destination': stops_destination_walking_distance})

                    #timestamp -> time the request was made
                    request_time_stamp = random.randint(0, dep_time)
                    request_data.update({'time_stamp': int(request_time_stamp)})
                    
                    #departure time
                    request_data.update({'dep_time': int(dep_time)})

                    #arrival time
                    request_data.update({'arr_time': int(arr_time)})

                    #increases the number of requests\

                    print('#:', i)

                    # add request_data to instance_data container
                    all_requests.update({i: request_data})

                    create_return_request = random.randint(0, 100)

                    if create_return_request <= inbound_outbound_factor*100 and arr_time <= max_early_departure and i < num_requests:
                        #create "copy of the request as a return"
                        request_data_return = {}

                        #coordinate origin
                        request_data_return.update({'originx': destination_point[1]})
                        request_data_return.update({'originy': destination_point[0]})

                        #coordinate destination
                        request_data_return.update({'destinationx': origin_point[1]})
                        request_data_return.update({'destinationy': origin_point[0]})

                        request_data_return.update({'num_stops_origin': len(stops_destination)})
                        request_data_return.update({'stops_origin': stops_destination})
                        request_data_return.update({'walking_time_origin_to_stops': stops_destination_walking_distance})

                        request_data_return.update({'num_stops_destination': len(stops_origin)})
                        request_data_return.update({'stops_destination': stops_origin})
                        request_data_return.update({'walking_time_stops_to_destination': stops_origin_walking_distance})

                        request_data_return.update({'time_stamp': int(request_time_stamp)})
                    
                        #departure time for the return
                        dep_time_return = random.randint(int(arr_time), int(max_early_departure))
                        request_data_return.update({'dep_time': int(dep_time_return)})

                        #arrival time
                        max_eta_bus, min_eta_bus = network.return_estimated_travel_time_bus(stops_origin, stops_destination, hour)
                        arr_time_return = (dep_time_return) + (vehicle_factor * max_eta_bus) + (max_walking * 2) 
                        request_data_return.update({'arr_time': int(arr_time_return)})

                        i += 1
                        print('#r:', i)
                        all_requests.update({i: request_data_return})

                        #increases the number of requests
                        
                    
                        
                else:
                    nok = True
                    if num_attempts > 20:
                        nok = False
            i += 1
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
    school_id,
    request_demand,
    min_early_departure,
    max_early_departure,
    min_walk_speed,
    max_walk_speed,
    max_walking,
    vehicle_factor,
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
        i=0
        while i < num_requests:
            
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
                
                #generate coordinates for origin
                while unfeasible_request:
                    
                    if request_demand[r].num_origins == -1:
                        origin_point = network.get_random_coord(network.polygon)
                        origin_point = (origin_point.y, origin_point.x)

                    else:
                        random_zone = np.random.uniform(0, request_demand[r].num_origins, 1)
                        random_zone = int(random_zone)
                        random_zone_id = int(request_demand[r].origin_zones[random_zone])
                        polygon_zone = network.zones.loc[random_zone_id]['polygon']
                        
                        origin_point = network.get_random_coord(polygon_zone)
                        origin_point = (origin_point.y, origin_point.x)

                    #destination is the school
                    destination_point = (network.schools.loc[school_id, 'lat'], network.schools.loc[school_id, 'lon'])

                    origin_node_walk = ox.get_nearest_node(network.G_walk, origin_point)
                    destination_node_walk = ox.get_nearest_node(network.G_walk, destination_point)
                    time_walking = network.get_eta_walk(origin_node_walk, destination_node_walk, min_walk_speed, max_walk_speed)

                    
                    #if time_walking > max_walking:
                    #    unfeasible_request = False
                    
                    origin_node_drive = ox.get_nearest_node(network.G_drive, origin_point)
                    destination_node_drive = ox.get_nearest_node(network.G_drive, destination_point)
                    
                    stops_origin = []
                    stops_origin_walking_distance = []
                    
                    #if distance between origin and destination is too small the person just walks
                    #add the request
                        
                    #calculates the stations which are close enough to the origin and destination of the request
                    for index in network.bus_stops_ids:
                    
                        #osmid_possible_stop = int(stop_node['osmid_walk'])
                        osmid_possible_stop = int(network.bus_stops.loc[index, 'osmid_walk'])

                        eta_walk_origin = network.get_eta_walk(origin_node_walk, osmid_possible_stop, min_walk_speed, max_walk_speed)
                        if eta_walk_origin >= 0 and eta_walk_origin <= max_walking:
                            stops_origin.append(index)
                            stops_origin_walking_distance.append(eta_walk_origin)
                    
                    #print(len(stops_origin))
                    if len(stops_origin) > 0:
                        unfeasible_request = False
                
                
                if not unfeasible_request:   
                    #outputs in the json file if the request is 'feasible'
                    
                    request_data.update({'originx': origin_point[1]})
                    request_data.update({'originy': origin_point[0]})

                    request_data.update({'destinationx': destination_point[1]})
                    request_data.update({'destinationy': destination_point[0]})

                    request_data.update({'num_stops_origin': len(stops_origin)})
                    request_data.update({'stops_origin': stops_origin})
                    request_data.update({'walking_time_origin_to_stops': stops_origin_walking_distance})

                    # add request_data to instance_data container
                    all_requests.update({i: request_data})

                    #increases the number of requests
                    print('#:', i)

                    #not necessary to create return request because the student always returns home
                    #return time is dependent of the school's time window

                else:
                    nok = True
                    if num_attempts > 20:
                        nok = False
                i += 1
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


