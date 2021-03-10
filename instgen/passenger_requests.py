from fixed_lines import _check_subway_routes_serve_passenger
from fixed_lines import _evaluate_best_fixed_route
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

def plot_requests(network, save_dir_images, origin_points, destination_points):


    fig, ax = ox.plot_graph(network.G_drive, show=False, close=False, node_color='#336699', node_size=6)

    requests_folder = os.path.join(save_dir_images, 'requests')

    if not os.path.isdir(requests_folder):
        os.mkdir(requests_folder)


    #plot origin locations
    for origin in origin_points:

        ax.scatter(origin[1], origin[0], c='red', s=8)

    plt.savefig(requests_folder+'/requests_origin_locations')
    plt.close(fig) 


    #plot destination locations
    fig, ax = ox.plot_graph(network.G_drive, show=False, close=False, node_color='#336699', node_size=6)

    for destination in destination_points:

        ax.scatter(destination[1], destination[0], c='green', s=8)

    plt.savefig(requests_folder+'/requests_destination_locations')
    plt.close(fig)

def _generate_requests_ODBRPFL( 
    inst,
    replicate_num
):
    
    '''
    generate requests for the on demand bus routing problem with fixed lines
    '''

    inst.output_file_json = os.path.join(inst.save_dir_json, inst.output_folder_base + '_' + str(replicate_num) + '.json')
    instance_data = {}  

    print("Now generating " + " request_data")
    origin_points=[]
    destination_points=[]
    lines = []
    all_requests = {} 
    h = 0
    num_requests = 0
    request_id = 0

    #for each PDF
    for r in range(len(inst.request_demand)):
        
        #randomly generates the earliest departure times or latest arrival times
        inst.request_demand[r].sample_times()


        weightssd = (inst.spatial_distribution[0].prob, inst.spatial_distribution[1].prob, inst.spatial_distribution[2].prob, inst.spatial_distribution[3].prob)
        
        for sd in inst.spatial_distribution:

            #randomly generates the zones 
            if sd.is_random_origin_zones:
                sd.randomly_sample_origin_zones(len(inst.network.zones))

            if sd.is_random_destination_zones:
                sd.randomly_sample_destination_zones(len(inst.network.zones))     

        num_requests = inst.request_demand[r].num_requests
        print(num_requests)


        i=0
        while i < num_requests:
            
            #timestamp -> time the request was received by the system
            dep_time = inst.request_demand[r].demand[i]
            
            request_lead_time = random.randint(inst.min_lead_time, inst.max_lead_time)

            request_time_stamp = int(dep_time - request_lead_time)

            if (dep_time >= 0) and (dep_time >= inst.min_early_departure) and (dep_time <= inst.max_early_departure):
                nok = True
            else:
                nok = False
            
            arr_time = None

            request_walk_speed = random.randint(int(inst.min_walk_speed), int(inst.max_walk_speed))

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

                    sdlist = [0,1,2,3]
                    sdid = random.choices(sdlist, weights=weightssd, k=1)
                    sd = inst.spatial_distribution[sdid[0]]

                    #generate coordinates for origin
                    if sd.num_origins == -1:

                        origin_point = inst.network._get_random_coord(inst.network.polygon)
                        origin_point = (origin_point.y, origin_point.x)
    
                    else:

                        random_zone = np.random.uniform(0, sd.num_origins, 1)
                        random_zone = int(random_zone)

                        random_zone_id = int(sd.origin_zones[random_zone])
                        polygon_zone = inst.network.zones.loc[random_zone_id]['polygon']
                        
                        #generate coordinates within the given zone
                        origin_point = inst.network._get_random_coord(polygon_zone)
                        #ax.scatter(origin_point.x, origin_point.y, c='red')
                        origin_point = (origin_point.y, origin_point.x)

                    #generate coordinates for destination
                    if sd.num_destinations == -1:
                        
                        destination_point = inst.network._get_random_coord(inst.network.polygon)
                        #ax.scatter(destination_point.x, destination_point.y, c='green')
                        destination_point = (destination_point.y, destination_point.x)

                    else:
                        
                        random_zone = np.random.uniform(0, sd.num_destinations, 1)
                        random_zone = int(random_zone)
                        
                        random_zone_id = int(sd.destination_zones[random_zone])
                        polygon_zone = inst.network.zones.loc[random_zone_id]['polygon']
            
                        #generate coordinates within the given zone
                        destination_point = inst.network._get_random_coord(polygon_zone)
                        #ax.scatter(destination_point.x, destination_point.y, c='green')
                        destination_point = (destination_point.y, destination_point.x)

                    origin_node_walk = ox.get_nearest_node(inst.network.G_walk, origin_point)
                    destination_node_walk = ox.get_nearest_node(inst.network.G_walk, destination_point)
                    time_walking = inst.network.get_eta_walk(origin_node_walk, destination_node_walk, request_walk_speed)

                    max_walking_user = random.randint(int(inst.lb_max_walking), int(inst.ub_max_walking))
                    
                    #time walking from origin to destination must be higher of maximum walking by the user
                    if time_walking > max_walking_user:
                        unfeasible_request = False
                
                origin_node_drive = ox.get_nearest_node(inst.network.G_drive, origin_point)
                destination_node_drive = ox.get_nearest_node(inst.network.G_drive, destination_point)
                
                stops_origin = []
                stops_destination = []

                stops_origin_walking_distance = []
                stops_destination_walking_distance = []


                if time_walking > max_walking_user: #if distance between origin and destination is too small the person just walks
                    #add the request
                    #calculates the stations which are close enough to the origin and destination of the request
                    for index in inst.network.bus_stations_ids:
                    #for index, node in inst.network.bus_stations.iterrows():

                        #osmid_possible_stop = int(stop_node['osmid_walk'])
                        osmid_possible_stop = int(inst.network.bus_stations.loc[index, 'osmid_walk'])

                        eta_walk_origin = inst.network.get_eta_walk(origin_node_walk, osmid_possible_stop, request_walk_speed)
                        if eta_walk_origin >= 0 and eta_walk_origin <= max_walking_user:
                            stops_origin.append(index)
                            stops_origin_walking_distance.append(eta_walk_origin)

                        #eta_walk_destination = inst.network.get_eta_walk(destination_node_walk, osmid_possible_stop) 
                        eta_walk_destination = inst.network.get_eta_walk(osmid_possible_stop, destination_node_walk, request_walk_speed)       
                        if eta_walk_destination >= 0 and eta_walk_destination <= max_walking_user:
                            stops_destination.append(index)
                            stops_destination_walking_distance.append(eta_walk_destination)

                    fl_stations_origin = []
                    fl_stations_destination = []

                    fl_stations_origin_walking_distance = []
                    fl_stations_destination_walking_distance = []

                    #calculates the FIXED LINE stations which are close enough - i.e. walking distance- to the origin and destination of the request
                    for node in inst.network.nodes_covered_fixed_lines:

                        osmid_possible_fixed_station = inst.network.deconet_network_nodes.loc[int(node), 'osmid_walk']
                        eta_walk_origin = inst.network.get_eta_walk(origin_node_walk, osmid_possible_fixed_station, request_walk_speed)
                        if eta_walk_origin >= 0 and eta_walk_origin <= max_walking_user:
                            fl_stations_origin.append(node)
                            fl_stations_origin_walking_distance.append(eta_walk_origin)

                        eta_walk_destination = inst.network.get_eta_walk(osmid_possible_fixed_station, destination_node_walk, request_walk_speed) 
                        if eta_walk_destination >= 0 and eta_walk_destination <= max_walking_user:
                            fl_stations_destination.append(node)
                            fl_stations_destination_walking_distance.append(eta_walk_destination)

                # Check whether each passenger can walk to stops (origin + destination)
                if len(stops_origin) > 0 and len(stops_destination) > 0:
                    if not (set(stops_origin) & set(stops_destination)):
                        
                        #compute arrival time
                        max_eta_bus, min_eta_bus = inst.network.return_estimated_travel_time_bus(stops_origin, stops_destination)
                        if min_eta_bus >= 0:
                            #arr_time = (dep_time) + (inst.delay_vehicle_factor * max_eta_bus) + (max_walking_user * inst.delay_walk_factor)
                            flex_time = inst.delay_vehicle_factor * max_eta_bus
                            arr_time = dep_time + max_eta_bus + flex_time
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

                    origin_points.append(origin_point)

                    #coordinate destination
                    request_data.update({'destinationx': destination_point[1]})
                    request_data.update({'destinationy': destination_point[0]})

                    destination_points.append(destination_point)

                    request_data.update({'num_stops_origin': len(stops_origin)})
                    request_data.update({'stops_origin': stops_origin})
                    request_data.update({'walking_time_origin_to_stops': stops_origin_walking_distance})

                    request_data.update({'num_stops_destination': len(stops_destination)})
                    request_data.update({'stops_destination': stops_destination})
                    request_data.update({'walking_time_stops_to_destination': stops_destination_walking_distance})

                    request_data.update({'num_stations_fl_origin': len(fl_stations_origin)})
                    
                    request_data.update({'stations_fl_origin': fl_stations_origin})
                    request_data.update({'walking_time_origin_to_stations_fl': fl_stations_origin_walking_distance})

                    request_data.update({'num_stations_fl_destination': len(fl_stations_destination)})
                   
                    request_data.update({'stations_fl_destination': fl_stations_destination})
                    request_data.update({'walking_time_stations_fl_to_destination': fl_stations_destination_walking_distance})


                    #timestamp -> time the request was made
                    request_data.update({'time_stamp': int(request_time_stamp)})
                    
                    #departure time
                    request_data.update({'dep_time': int(dep_time)})

                    #arrival time
                    request_data.update({'arr_time': int(arr_time)})

                    #when generating the requests, consider also getting the fixed lines
                     
                    # add request_data to instance_data container
                    all_requests.update({request_id: request_data})
                    request_id+=1

                    create_return_request = random.randint(0, 100)

                    if create_return_request <= inst.return_factor*100 and arr_time <= inst.max_early_departure and i < num_requests-1:
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

                        request_data_return.update({'num_stations_fl_origin': len(fl_stations_destination)})
                        
                        request_data_return.update({'stations_fl_origin': fl_stations_destination})
                        request_data_return.update({'walking_time_origin_to_stations_fl': fl_stations_destination_walking_distance})

                        request_data_return.update({'num_stations_fl_destination': len(fl_stations_origin)})
                        
                        request_data_return.update({'stations_fl_destination': fl_stations_origin})
                        request_data_return.update({'walking_time_stations_fl_to_destination': fl_stations_origin_walking_distance})

                        request_data_return.update({'time_stamp': int(request_time_stamp)})
                    
                        #departure time for the return
                        dep_time_return = random.randint(int(arr_time), int(inst.max_early_departure))
                        request_data_return.update({'dep_time': int(dep_time_return)})

                        #arrival time for the return
                        max_eta_bus, min_eta_bus = inst.network.return_estimated_travel_time_bus(stops_origin, stops_destination)
                        #arr_time_return = (dep_time_return) + (inst.delay_vehicle_factor * max_eta_bus) + (max_walking_user * inst.delay_walk_factor) 
                        flex_time = inst.delay_vehicle_factor * max_eta_bus
                        arr_time_return = dep_time_return + max_eta_bus + flex_time
                        request_data_return.update({'arr_time': int(arr_time_return)})

                        #increases the number of requests
                        i += 1
                        
                        all_requests.update({request_id: request_data_return})
                        request_id+=1

                        
                else:
                    nok = True
                    if num_attempts > 20:
                        nok = False
            i += 1
        
        #plt.show()
        #plt.savefig('images/foo.png')
        #plt.close(fig) 

    #inst.network.all_requests = all_requests

    #travel time between bus stations
    travel_time_bus_json = inst.network._get_travel_time_matrix("bus")

    #how the subway stations connect with each other
    travel_time_subway_json = inst.network._get_travel_time_matrix("subway")

    #how the bus stations and fixed line stations are connected with each other by walking times
    travel_time_hybrid_json = inst.network._get_travel_time_matrix("hybrid", inst=inst)
    
    instance_data.update({'num_requests:': len(all_requests),
                          'requests': all_requests,
                          'num_stations': inst.network.num_stations,
                          'travel_time_matrix': travel_time_bus_json,
                          'travel_time_matrix_subway': travel_time_subway_json,
                          'travel_time_matrix_hybrid': travel_time_hybrid_json,
                          })

    save_dir = os.getcwd()+'/'+inst.output_folder_base
    save_dir_images = os.path.join(save_dir, 'images')
    plot_requests(inst.network, save_dir_images, origin_points, destination_points)

    with open(inst.output_file_json, 'w') as file:
        json.dump(instance_data, file, indent=4)
        file.close()

def _generate_requests_DARP( 
    inst,
    replicate_num,
):
    
    '''
    generate requests for the dial-a-ride problem
    '''

    inst.output_file_json = os.path.join(inst.save_dir_json, inst.output_folder_base + '_' + str(replicate_num) + '.json')
    instance_data = {}  

    print("Now generating " + " request_data")
    origin_points=[]
    destination_points=[]
    lines = []
    all_requests = {} 
    h = 0
    num_requests = 0
    request_id = 0

    depot_point = inst.network._get_random_coord(inst.network.polygon)
    depot_point = (depot_point.y, depot_point.x)
    depot_node_drive = ox.get_nearest_node(inst.network.G_drive, depot_point)
    node_list = []

    node_list.append(depot_node_drive)
    #for each PDF
    for r in range(len(inst.request_demand)):
        
        #fig, ax = ox.plot_graph(inst.network.G_walk, show=False, close=False)

        #randomly generates the time stamp of the requests
        inst.request_demand[r].sample_times()

        weightssd = (inst.spatial_distribution[0].prob, inst.spatial_distribution[1].prob, inst.spatial_distribution[2].prob, inst.spatial_distribution[3].prob)

        #randomly generates the zones 
        for sd in inst.spatial_distribution:

            #randomly generates the zones 
            if sd.is_random_origin_zones:
                sd.randomly_sample_origin_zones(len(inst.network.zones))

            if sd.is_random_destination_zones:
                sd.randomly_sample_destination_zones(len(inst.network.zones))      

        num_requests = inst.request_demand[r].num_requests
        print(num_requests)
        i=0
        while i < num_requests:

            #timestamp -> time the request was received by the system
            dep_time = inst.request_demand[r].demand[i]
            
            request_lead_time = random.randint(inst.min_lead_time, inst.max_lead_time)

            request_time_stamp = int(dep_time - request_lead_time)

            if (dep_time >= 0) and (dep_time >= inst.min_early_departure) and (dep_time <= inst.max_early_departure):
                nok = True
            else:
                nok = False
            
            arr_time = None

            request_data = {}  #holds information about this request

            #randomly choosing coordinates and repeated for a passenger until he can go to at least 1 stop (for both origin and destination)
            
            num_attempts = 0 #limit the number of attempts to generate a request and avoid infinite loop
            while nok:
                num_attempts += 1
                nok = False
                
                origin_point = []
                destination_point = []
                unfeasible_request = False

                sdlist = [0,1,2,3]
                sdid = random.choices(sdlist, weights=weightssd, k=1)
                sd = inst.spatial_distribution[sdid[0]]
                
                #generate coordinates for origin
                if sd.num_origins == -1:

                    origin_point = inst.network._get_random_coord(inst.network.polygon)
                    origin_point = (origin_point.y, origin_point.x)

                else:

                    random_zone = np.random.uniform(0, sd.num_origins, 1)
                    random_zone = int(random_zone)
                    random_zone_id = int(sd.origin_zones[random_zone])
                    polygon_zone = inst.network.zones.loc[random_zone_id]['polygon']
        
                    origin_point = inst.network._get_random_coord(polygon_zone)
                    origin_point = (origin_point.y, origin_point.x)

                #generate coordinates for destination
                if sd.num_destinations == -1:
                    
                    destination_point = inst.network._get_random_coord(inst.network.polygon)
                    destination_point = (destination_point.y, destination_point.x)

                else:

                    random_zone = np.random.uniform(0, sd.num_destinations, 1)
                    random_zone = int(random_zone)
                    random_zone_id = int(sd.destination_zones[random_zone])
                    polygon_zone = inst.network.zones.loc[random_zone_id]['polygon']

                    destination_point = inst.network._get_random_coord(polygon_zone)
                    destination_point = (destination_point.y, destination_point.x)
    
                origin_node_drive = ox.get_nearest_node(inst.network.G_drive, origin_point)
                destination_node_drive = ox.get_nearest_node(inst.network.G_drive, destination_point)
                
                if origin_node_drive not in node_list:
                    node_list.append(origin_node_drive)

                if destination_node_drive not in node_list:
                    node_list.append(destination_node_drive)
                
                #compute estimated arrival time
                estimated_travel_time = inst.network._return_estimated_travel_time_drive(origin_node_drive, destination_node_drive)
                if estimated_travel_time >= 0:
                    #arr_time = (dep_time) + (inst.delay_vehicle_factor * estimated_travel_time) 
                    flex_time = inst.delay_vehicle_factor * estimated_travel_time
                    arr_time = dep_time + estimated_travel_time + flex_time
                else:
                    unfeasible_request = True

                if  not unfeasible_request:   
                    #prints in the json file if the request is 'viable'
                    
                    request_data.update({'originx': origin_point[1]})
                    request_data.update({'originy': origin_point[0]})
                    request_data.update({'origin_node': int(origin_node_drive)})

                    origin_points.append(origin_point)

                    request_data.update({'destinationx': destination_point[1]})
                    request_data.update({'destinationy': destination_point[0]})
                    request_data.update({'destination_node': int(destination_node_drive)})

                    destination_points.append(destination_point)

                    request_data.update({'time_stamp': int(request_time_stamp)})
                    
                    #departure time
                    request_data.update({'dep_time': int(dep_time)})

                    #arrival time
                    request_data.update({'arr_time': int(arr_time)})

                    #compute if request need some vehicle requirement, e.g., wheelchair, ambulatory passenger
                    #0-> no requirement
                    #1-> wheelchair
                    #2-> ambulatory
                    min_req = 0
                    max_req = 0
                    if inst.wheelchair and inst.ambulatory:
                        max_req = 2
                        vehicle_requirement = random.randint(min_req, max_req)
                    else:
                        if inst.wheelchair:
                            max_req = 1
                            vehicle_requirement = random.randint(min_req, max_req)
                        elif inst.ambulatory:
                            max_req = 1
                            vehicle_requirement = random.randint(min_req, max_req)
                            if vehicle_requirement > 0:
                                vehicle_requirement = 2
                    request_data.update({'vehicle_requirement': int(vehicle_requirement)})

                    # add request_data to instance_data container
                    all_requests.update({request_id: request_data})
                    request_id+=1

                    create_return_request = random.randint(0, 100)

                    if create_return_request <= inst.return_factor*100 and arr_time <= inst.max_early_departure and i < num_requests-1:

                        request_data_return = {}

                        #coordinate origin
                        request_data_return.update({'originx': destination_point[1]})
                        request_data_return.update({'originy': destination_point[0]})
                        request_data_return.update({'origin_node': int(destination_node_drive)})

                        #coordinate destination
                        request_data_return.update({'destinationx': origin_point[1]})
                        request_data_return.update({'destinationy': origin_point[0]})
                        request_data_return.update({'destination_node': int(origin_node_drive)})

                        #timestamp -> time the request was made
                        request_data_return.update({'time_stamp': int(request_time_stamp)})
                        
                        #departure time for the return
                        dep_time_return = random.randint(arr_time, inst.max_early_departure)
                        request_data_return.update({'dep_time': int(dep_time_return)})

                        #arrival time
                        estimated_travel_time = inst.network._return_estimated_travel_time_drive(destination_node_drive, origin_node_drive)
                        #arr_time_return = (dep_time_return) + (inst.delay_vehicle_factor * estimated_travel_time)
                        flex_time = inst.delay_vehicle_factor * estimated_travel_time
                        arr_time_return = dep_time_return + estimated_travel_time + flex_time
                        request_data_return.update({'arr_time': int(arr_time_return)})
                        request_data_return.update({'vehicle_requirement': int(vehicle_requirement)})

                        all_requests.update({request_id: request_data_return})
                        request_id+=1

                        #increases the number of requests
                        i += 1
                        
                else:
                    nok = True
                    if num_attempts > 20:
                        nok = False
            
            #increases the number of requests
            i += 1

        #plt.show()
        #plt.savefig('images/foo.png')
        #plt.close(fig) 

    inst.network.node_list_darp = node_list

    travel_time_json = inst.network._get_travel_time_matrix("list", node_list=node_list)
    instance_data.update({'num_requests:': len(all_requests),
                          'requests': all_requests,
                          'depot': str(depot_node_drive),
                          'num_nodes': len(node_list),
                          'travel_time_matrix': travel_time_json
                          })

    save_dir = os.getcwd()+'/'+inst.output_folder_base
    save_dir_images = os.path.join(save_dir, 'images')
    plot_requests(inst.network, save_dir_images, origin_points, destination_points)

    with open(inst.output_file_json, 'w') as file:
        json.dump(instance_data, file, indent=4)
        file.close()

def _generate_requests_ODBRP( 
    inst,
    replicate_num,
):
    
    '''
    generate requests for the on demand bus routing problem
    '''

    inst.output_file_json = os.path.join(inst.save_dir_json, inst.output_folder_base + '_' + str(replicate_num) + '.json')
    instance_data = {}  

    print("Now generating " + " request_data")
    origin_points=[]
    destination_points=[]
    lines = []
    all_requests = {} 
    h = 0
    num_requests = 0
    request_id = 0

    #for each PDF
    for r in range(len(inst.request_demand)):
        
        #randomly samples the earliest departure times or latest arrival times
        inst.request_demand[r].sample_times()

        weightssd = (inst.spatial_distribution[0].prob, inst.spatial_distribution[1].prob, inst.spatial_distribution[2].prob, inst.spatial_distribution[3].prob)

        #randomly generates the zones 
        for sd in inst.spatial_distribution:

            #randomly generates the zones 
            if sd.is_random_origin_zones:
                sd.randomly_sample_origin_zones(len(inst.network.zones))

            if sd.is_random_destination_zones:
                sd.randomly_sample_destination_zones(len(inst.network.zones))      

        num_requests = inst.request_demand[r].num_requests
        print(num_requests)
        i = 0
        while i < num_requests:
            
            #timestamp -> time the request was received by the system
            dep_time = inst.request_demand[r].demand[i]
            
            request_lead_time = random.randint(inst.min_lead_time, inst.max_lead_time)

            request_time_stamp = int(dep_time - request_lead_time)

            if (dep_time >= 0) and (dep_time >= inst.min_early_departure) and (dep_time <= inst.max_early_departure):
                nok = True
            else:
                nok = False
            
            arr_time = None

            request_walk_speed = random.randint(int(inst.min_walk_speed), int(inst.max_walk_speed))

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

                    sdlist = [0,1,2,3]
                    sdid = random.choices(sdlist, weights=weightssd, k=1)
                    sd = inst.spatial_distribution[sdid[0]]

                    #generate coordinates for origin
                    if sd.num_origins == -1:

                        origin_point = inst.network._get_random_coord(inst.network.polygon)
                        origin_point = (origin_point.y, origin_point.x)
    
                    else:

                        random_zone = np.random.uniform(0, sd.num_origins, 1)
                        random_zone = int(random_zone)

                        random_zone_id = int(sd.origin_zones[random_zone])
                        polygon_zone = inst.network.zones.loc[random_zone_id]['polygon']
                        
                        #generate coordinates within the given zone
                        origin_point = inst.network._get_random_coord(polygon_zone)
                        #ax.scatter(origin_point.x, origin_point.y, c='red')
                        origin_point = (origin_point.y, origin_point.x)

                    #generate coordinates for destination
                    if sd.num_destinations == -1:
                        
                        destination_point = inst.network._get_random_coord(inst.network.polygon)
                        #ax.scatter(destination_point.x, destination_point.y, c='green')
                        destination_point = (destination_point.y, destination_point.x)

                    else:
                        
                        random_zone = np.random.uniform(0, sd.num_destinations, 1)
                        random_zone = int(random_zone)
                        
                        random_zone_id = int(sd.destination_zones[random_zone])
                        polygon_zone = inst.network.zones.loc[random_zone_id]['polygon']
            
                        #generate coordinates within the given zone
                        destination_point = inst.network._get_random_coord(polygon_zone)
                        #ax.scatter(destination_point.x, destination_point.y, c='green')
                        destination_point = (destination_point.y, destination_point.x)

                    origin_node_walk = ox.get_nearest_node(inst.network.G_walk, origin_point)
                    destination_node_walk = ox.get_nearest_node(inst.network.G_walk, destination_point)
                    time_walking = inst.network.get_eta_walk(origin_node_walk, destination_node_walk, request_walk_speed)

                    max_walking_user = random.randint(int(inst.lb_max_walking), int(inst.ub_max_walking))

                    #time walking from origin to destination must be higher of max_walking by the user
                    if time_walking > max_walking_user:
                        unfeasible_request = False
                
                origin_node_drive = ox.get_nearest_node(inst.network.G_drive, origin_point)
                destination_node_drive = ox.get_nearest_node(inst.network.G_drive, destination_point)
                
                stops_origin = []
                stops_destination = []

                stops_origin_walking_distance = []
                stops_destination_walking_distance = []

                if time_walking > max_walking_user: #if distance between origin and destination is too small the person just walks
                    
                    #calculates the stations which are close enough to the origin and destination of the request
                    for index in inst.network.bus_stations_ids:
                    
                        osmid_possible_stop = int(inst.network.bus_stations.loc[index, 'osmid_walk'])

                        eta_walk_origin = inst.network.get_eta_walk(origin_node_walk, osmid_possible_stop, request_walk_speed)
                        if eta_walk_origin >= 0 and eta_walk_origin <= max_walking_user:
                            stops_origin.append(index)
                            stops_origin_walking_distance.append(eta_walk_origin)

                        #eta_walk_destination = inst.network.get_eta_walk(destination_node_walk, osmid_possible_stop) 
                        eta_walk_destination = inst.network.get_eta_walk(osmid_possible_stop, destination_node_walk, request_walk_speed)       
                        if eta_walk_destination >= 0 and eta_walk_destination <= max_walking_user:
                            stops_destination.append(index)
                            stops_destination_walking_distance.append(eta_walk_destination)

                # Check whether each passenger can walk to stops (origin + destination)
                if len(stops_origin) > 0 and len(stops_destination) > 0:
                    #check if stops_origin and stops_destination dont have the same bus station
                    if not (set(stops_origin) & set(stops_destination)):
                        
                        #compute arrival time
                        max_eta_bus, min_eta_bus = inst.network.return_estimated_travel_time_bus(stops_origin, stops_destination)
                        if min_eta_bus >= 0:
                            #arr_time = (dep_time) + (inst.delay_vehicle_factor * max_eta_bus) + (max_walking_user * inst.delay_walk_factor)
                            flex_time = inst.delay_vehicle_factor * max_eta_bus
                            arr_time = dep_time + max_eta_bus + flex_time
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

                    origin_points.append(origin_point)

                    #coordinate destination
                    request_data.update({'destinationx': destination_point[1]})
                    request_data.update({'destinationy': destination_point[0]})

                    destination_points.append(destination_point)

                    request_data.update({'num_stops_origin': len(stops_origin)})
                    request_data.update({'stops_origin': stops_origin})
                    request_data.update({'walking_time_origin_to_stops': stops_origin_walking_distance})

                    request_data.update({'num_stops_destination': len(stops_destination)})
                    request_data.update({'stops_destination': stops_destination})
                    request_data.update({'walking_time_stops_to_destination': stops_destination_walking_distance})

                    #timestamp -> time the request was made
                    request_data.update({'time_stamp': int(request_time_stamp)})
                    
                    #departure time
                    request_data.update({'dep_time': int(dep_time)})

                    #arrival time
                    request_data.update({'arr_time': int(arr_time)})

                    #increases the number of requests\

                    # add request_data to instance_data container
                    all_requests.update({request_id: request_data})
                    request_id+=1

                    create_return_request = random.randint(0, 100)

                    if create_return_request <= inst.return_factor*100 and arr_time <= inst.max_early_departure and i < num_requests-1:
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
                        dep_time_return = random.randint(int(arr_time), int(inst.max_early_departure))
                        request_data_return.update({'dep_time': int(dep_time_return)})

                        #arrival time for the return
                        max_eta_bus, min_eta_bus = inst.network.return_estimated_travel_time_bus(stops_origin, stops_destination)
                        #arr_time_return = (dep_time_return) + (inst.delay_vehicle_factor * max_eta_bus) + (max_walking_user * inst.delay_walk_factor) 
                        flex_time = inst.delay_vehicle_factor * max_eta_bus
                        arr_time_return = dep_time_return + max_eta_bus + flex_time
                        request_data_return.update({'arr_time': int(arr_time_return)})

                        i += 1
                        
                        all_requests.update({request_id: request_data_return})
                        request_id+=1

                        #increases the number of requests
                else:
                    nok = True
                    if num_attempts > 20:
                        nok = False
            i += 1
        #plt.show()
        #plt.savefig('images/foo.png')
        #plt.close(fig) 

    inst.network.all_requests = all_requests

    travel_time_json = inst.network._get_travel_time_matrix("bus")
    instance_data.update({'num_requests:': len(all_requests),
                          'requests': all_requests,
                          'num_stations': inst.network.num_stations,
                          'travel_time_matrix': travel_time_json})

    save_dir = os.getcwd()+'/'+inst.output_folder_base
    save_dir_images = os.path.join(save_dir, 'images')
    plot_requests(inst.network, save_dir_images, origin_points, destination_points)

    with open(inst.output_file_json, 'w') as file:
        json.dump(instance_data, file, indent=4)
        file.close()

def _generate_requests_SBRP( 
    inst,
    replicate_num,
):
    #file that will store the information of this instance in json format
    inst.output_file_json = os.path.join(inst.save_dir_json, inst.output_folder_base + '_' + str(replicate_num) + '.json')
    instance_data = {}  

    print("Now generating " + " request_data")
    origin_points=[]
    destination_points=[]
    lines = []
    all_requests = {} 
    h = 0
    num_requests = 0
    request_id = 0

    #for each PDF
    for r in range(len(inst.request_demand)):

        weightssd = (inst.spatial_distribution[0].prob, inst.spatial_distribution[1].prob, inst.spatial_distribution[2].prob, inst.spatial_distribution[3].prob)
         
        for sd in inst.spatial_distribution:

            #randomly generates the zones 
            if sd.is_random_origin_zones:
                sd.randomly_sample_origin_zones(len(inst.network.zones))

        #destination is the school
          
        num_requests = inst.request_demand[r].num_requests
        print(num_requests)
        i=0
        while i < num_requests:
            
            nok = True
            request_data = {}  #holds information about this request

            #randomly choosing coordinates and repeated for a passenger until he can go to at least 1 stop (for both origin and destination)
            request_walk_speed = random.randint(int(inst.min_walk_speed), int(inst.max_walk_speed))

            num_attempts = 0 #limit the number of attempts to generate a request and avoid infinite loop
            while nok:
                num_attempts += 1
                nok = False
                
                origin_point = []
                destination_point = []
                unfeasible_request = True
                
                #generate coordinates for origin
                while unfeasible_request:

                    sdlist = [0,1,2,3]
                    sdid = random.choices(sdlist, weights=weightssd, k=1)
                    sd = inst.spatial_distribution[sdid[0]]
                    
                    if sd.num_origins == -1:
                        origin_point = inst.network._get_random_coord(inst.network.polygon)
                        origin_point = (origin_point.y, origin_point.x)

                    else:
                        random_zone = np.random.uniform(0, sd.num_origins, 1)
                        random_zone = int(random_zone)
                        random_zone_id = int(sd.origin_zones[random_zone])
                        polygon_zone = inst.network.zones.loc[random_zone_id]['polygon']
                        
                        origin_point = inst.network._get_random_coord(polygon_zone)
                        origin_point = (origin_point.y, origin_point.x)

                    #destination is the school
                    destination_point = (inst.network.schools.loc[inst.school_id, 'lat'], inst.network.schools.loc[inst.school_id, 'lon'])

                    origin_node_walk = ox.get_nearest_node(inst.network.G_walk, origin_point)
                    destination_node_walk = ox.get_nearest_node(inst.network.G_walk, destination_point)
                    time_walking = inst.network.get_eta_walk(origin_node_walk, destination_node_walk, request_walk_speed)

                    #if time_walking > max_walking:
                    #    unfeasible_request = False
                    
                    origin_node_drive = ox.get_nearest_node(inst.network.G_drive, origin_point)
                    destination_node_drive = ox.get_nearest_node(inst.network.G_drive, destination_point)
                    
                    stops_origin = []
                    stops_origin_walking_distance = []
                    
                    #if distance between origin and destination is too small the person just walks
                    #add the request
                        
                    #calculates the stations which are close enough to the origin and destination of the request
                    for index in inst.network.bus_stations_ids:
                    
                        osmid_possible_stop = int(inst.network.bus_stations.loc[index, 'osmid_walk'])

                        eta_walk_origin = inst.network.get_eta_walk(origin_node_walk, osmid_possible_stop, request_walk_speed)
                        
                        max_walking_user = random.randint(int(inst.lb_max_walking), int(inst.ub_max_walking))
                        if eta_walk_origin >= 0 and eta_walk_origin <= max_walking_user:
                            stops_origin.append(index)
                            stops_origin_walking_distance.append(eta_walk_origin)
                    
                    if len(stops_origin) > 0:
                        unfeasible_request = False
                
                if not unfeasible_request:   
                    #outputs in the json file if the request is 'feasible'

                    request_data.update({'originx': origin_point[1]})
                    request_data.update({'originy': origin_point[0]})
                    request_data.update({'origin_node': int(origin_node_drive)})

                    origin_points.append(origin_point)

                    request_data.update({'destinationx': destination_point[1]})
                    request_data.update({'destinationy': destination_point[0]})
                    request_data.update({'destination_node': int(destination_node_drive)})
                
                    request_data.update({'num_stops_origin': len(stops_origin)})
                    request_data.update({'stops_origin': stops_origin})
                    request_data.update({'walking_time_origin_to_stops': stops_origin_walking_distance})

                    # add request_data to instance_data container
                    all_requests.update({request_id: request_data})
                    request_id+=1

                    #increases the number of requests
                    
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

    travel_time_json = inst.network._get_travel_time_matrix("bus")
    travel_time_to_school = inst.network._get_travel_time_from_stops_to_school(inst.school_id)
    
    instance_data.update({'num_requests:': len(all_requests),
                          'requests': all_requests,
                          'num_stations': inst.network.num_stations,
                          'travel_time_matrix': travel_time_json,
                          'travel_time_to_school': travel_time_to_school})

    save_dir = os.getcwd()+'/'+inst.output_folder_base
    save_dir_images = os.path.join(save_dir, 'images')
    plot_requests(inst.network, save_dir_images, origin_points, destination_points)

    with open(inst.output_file_json, 'w') as file:
        json.dump(instance_data, file, indent=4)
        file.close()        


