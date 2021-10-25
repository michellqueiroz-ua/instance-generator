from fixed_lines import _check_subway_routes_serve_passenger
from fixed_lines import _evaluate_best_fixed_route
import json
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import osmnx as ox
import pandas as pd
import pickle
import random
import ray
import re
from shapely.geometry import Point
from scipy.stats import cauchy
from scipy.stats import expon
from scipy.stats import gamma
from scipy.stats import gilbrat
from scipy.stats import lognorm
from scipy.stats import powerlaw
from scipy.stats import wald
from multiprocessing import cpu_count
import gc
from datetime import datetime

def eval_expression(input_string):
    
    allowed_names = {"len": len, 
                    "set": set,
                    "abs": abs,
                    "float": float,
                    "int": int,
                    "max": max,
                    "min": min,
                    "pow": pow,
                    "round": round,
                    "str": str,
                    "isinstance": isinstance
                    }
    
    code = compile(input_string, "<string>", "eval")
    
    for name in code.co_names:
        if name not in allowed_names:
            raise NameError(f"Use of "+name+" is not allowed OR it is not in the parameters/attributes")
    return eval(code, {"__builtins__": {}}, allowed_names)

def plot_requests(network, save_dir_images, origin_points, destination_points):


    fig, ax = ox.plot_graph(network.G_drive, show=False, close=False, node_color='#000000', node_size=6, bgcolor="#ffffff", edge_color="#999999")

    requests_folder = os.path.join(save_dir_images, 'requests')

    if not os.path.isdir(requests_folder):
        os.mkdir(requests_folder)


    #plot origin locations
    for origin in origin_points:

        ax.scatter(origin[1], origin[0], c='red', s=8, marker=",")

    plt.savefig(requests_folder+'/requests_origin_locations')
    plt.close(fig) 


    #plot destination locations
    fig, ax = ox.plot_graph(network.G_drive, show=False, close=False, node_color='#000000', node_size=6, bgcolor="#ffffff", edge_color="#999999")

    for destination in destination_points:

        ax.scatter(destination[1], destination[0], c='green', s=8, marker="o")

    plt.savefig(requests_folder+'/requests_destination_locations')
    plt.close(fig)

def _generate_requests_ODBRPFL( 
    inst,
    replicate_num
):
    
    '''
    generate requests for the on demand bus routing problem with fixed lines
    '''

    random.seed(inst.seed)
    np.random.seed(inst.seed)

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
                
                max_walking_user = random.randint(int(inst.lb_max_walking), int(inst.ub_max_walking))

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

                    #origin_node_walk = ox.get_nearest_node(inst.network.G_walk, origin_point)
                    #destination_node_walk = ox.get_nearest_node(inst.network.G_walk, destination_point)
                    origin_node_walk = ox.nearest_nodes(inst.network.G_walk, origin_point[1], origin_point[0])
                    destination_node_walk = ox.nearest_nodes(inst.network.G_walk, destination_point[1], destination_point[0])
                    
                    time_walking = inst.network.get_eta_walk(origin_node_walk, destination_node_walk, request_walk_speed)

                    
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

                    '''
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
                    '''

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

                    #request_data.update({'num_stations_fl_origin': len(fl_stations_origin)})
                    
                    #request_data.update({'stations_fl_origin': fl_stations_origin})
                    #request_data.update({'walking_time_origin_to_stations_fl': fl_stations_origin_walking_distance})

                    #request_data.update({'num_stations_fl_destination': len(fl_stations_destination)})
                   
                    #request_data.update({'stations_fl_destination': fl_stations_destination})
                    #request_data.update({'walking_time_stations_fl_to_destination': fl_stations_destination_walking_distance})


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

                        #request_data_return.update({'num_stations_fl_origin': len(fl_stations_destination)})
                        
                        #request_data_return.update({'stations_fl_origin': fl_stations_destination})
                        #request_data_return.update({'walking_time_origin_to_stations_fl': fl_stations_destination_walking_distance})

                        #request_data_return.update({'num_stations_fl_destination': len(fl_stations_origin)})
                        
                        #request_data_return.update({'stations_fl_destination': fl_stations_origin})
                        #request_data_return.update({'walking_time_stations_fl_to_destination': fl_stations_origin_walking_distance})

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
    travel_time_bus = inst.network._get_travel_time_matrix("bus")

    travel_time_bus_json = travel_time_bus.tolist()

    gtt = nx.DiGraph() 

    for index in inst.network.bus_stations.index:

        gtt.add_node(index, type="busstation")

    for u in inst.network.bus_stations.index:
        for v in inst.network.bus_stations.index:

            gtt.add_edge(u, v, travel_time=travel_time_json[u][v])

    output_name_graphml = os.path.join(inst.save_dir_graphml, inst.output_folder_base + '_' + str(replicate_num) + '.graphml')
    #output_name_graphml = output_name_graphml.replace(" ", "")

    nx.write_graphml(gtt, output_name_graphml)

    #how the subway stations connect with each other
    #travel_time_subway_json = inst.network._get_travel_time_matrix("subway")

    #how the bus stations and fixed line stations are connected with each other by walking times
    #travel_time_hybrid_json = inst.network._get_travel_time_matrix("hybrid", inst=inst)
    
    instance_data.update({'num_requests:': len(all_requests),
                          'requests': all_requests,
                          'num_stations': inst.network.num_stations,
                          'travel_time_matrix': travel_time_bus_json,
                          #'travel_time_matrix_subway': travel_time_subway_json,
                          #'travel_time_matrix_hybrid': travel_time_hybrid_json,
                          })

    save_dir = os.getcwd()+'/'+inst.output_folder_base
    save_dir_images = os.path.join(save_dir, 'images')
    plot_requests(inst.network, save_dir_images, origin_points, destination_points)

    with open(inst.output_file_json, 'w') as file:
        json.dump(instance_data, file, indent=4)
        file.close()

@ray.remote
def _generate_single_data(GA, network, sorted_attributes, parameters, reqid, method_poi, distancex, pu, do, pdfunc, loc, scale, aux):
    
    #print(reqid)
    #print("here")
    attributes = {}
    feasible_data = False
    first_time = True
    while not feasible_data:
        
        attributes = {}
        feasible_data = True
        if method_poi:
            heads_or_tails = random.randint(0,1)
            if heads_or_tails == 1:
                
                #sorted_attributes[0] = pu
                #sorted_attributes[1] = do   
                #else:

                for sa in range(len(sorted_attributes)):
                    if sorted_attributes[sa] == pu:
                        sorted_attributes[sa] = do
                    else:
                        if sorted_attributes[sa] == do:
                            sorted_attributes[sa] = pu

        #print(sorted_attributes)

        for att in sorted_attributes:

            not_feasible_attribute = True
            exhaustion_iterations = 0

            while (not_feasible_attribute) and (exhaustion_iterations < 100):

                if GA.nodes[att]['type'] == 'location':

                    random_zone_id = -1

                    if 'subset_zones' in GA.nodes[att]:

                        zone = GA.nodes[att]['subset_zones']

                        if zone is False:

                            if method_poi:

                                type_coord = att
                                #print(type_coord)
                                zones = network.zones.index.tolist()
                                
                                if heads_or_tails == 0:

                                    if type_coord == do:
                                        
                                        '''
                                        probabilities = network.zones['density_pois'].tolist()
                                        random_zone_id = np.random.choice(a=zones, size=1, p=probabilities)
                                        
                                        random_zone_id = random_zone_id[0]
                                        polygon_zone = network.zones.loc[random_zone_id]['polygon']
                                        point = network._get_random_coord(polygon_zone)
                                        '''

                                        if not first_time:
                                            random.seed(datetime.now())
                                            rseed = np.random.uniform(0,1)
                                            rseed = rseed * 1000
                                            #print(rseed)
                                            rseed = int(rseed)

                                            if pdfunc == 'cauchy':
                                                cauchy.random_state = np.random.RandomState(seed=rseed)
                                                radius = cauchy.rvs(loc=loc, scale=scale, size=1)

                                            if pdfunc == 'expon':
                                                expon.random_state = np.random.RandomState(seed=rseed)
                                                radius = expon.rvs(loc=loc, scale=scale, size=1)

                                            if pdfunc == 'gamma':
                                                gamma.random_state = np.random.RandomState(seed=rseed)
                                                radius = gamma.rvs(a=aux, loc=loc, scale=scale, size=1)

                                            if pdfunc == 'gilbrat':
                                                gilbrat.random_state = np.random.RandomState(seed=rseed)
                                                radius = gilbrat.rvs(loc=loc, scale=scale, size=1)

                                            if pdfunc == 'lognorm':
                                                lognorm.random_state = np.random.RandomState(seed=rseed)
                                                radius = lognorm.rvs(loc=loc, scale=scale, size=1)

                                            if pdfunc == 'normal':
                                                normal.random_state = np.random.RandomState(seed=rseed)
                                                radius = normal.rvs(loc=loc, scale=scale, size=1)

                                            if pdfunc == 'powerlaw':
                                                powerlaw.random_state = np.random.RandomState(seed=rseed)
                                                radius = powerlaw.rvs(a=aux, loc=loc, scale=scale, size=1)

                                            if pdfunc == 'uniform':
                                                uniform.random_state = np.random.RandomState(seed=rseed)
                                                radius = uniform.rvs(loc=loc, scale=scale, size=1)

                                            if pdfunc == 'wald':
                                                wald.random_state = np.random.RandomState(seed=rseed)
                                                radius = wald.rvs(loc=loc, scale=scale, size=1)
                                            
                                            #wald.random_state = np.random.RandomState(seed=rseed)
                                            #radius = wald.rvs(loc=34.028995255744775, scale=7108.993523334802, size=1)

                                        else:
                                            radius = distancex
                                            first_time = False
                                        #if radius > 20000:
                                        #    print(radius)
                                        #print(radius)
                                        att_start = pu
                                        lat = attributes[att_start+'y']
                                        lon = attributes[att_start+'x']
                                        point = network._get_random_coord_radius(lat, lon, radius, network.polygon)
                                        #print(point)

                                    elif type_coord == pu:
                                        
                                        #att_start = GA.nodes[att]['start_point']
                                        #zone_start = int(attributes[att_start+'zone'])
                                        #probabilities = network.zone_probabilities[zone_start].tolist()
                                        #print(inst.network.zone_probabilities[zone_start].head())
                                        #print(probabilities)
                                        probabilities = network.zones['density_pois'].tolist()
                                        random_zone_id = np.random.choice(a=zones, size=1, p=probabilities)

                                        random_zone_id = random_zone_id[0]
                                        polygon_zone = network.zones.loc[random_zone_id]['polygon']
                                        point = network._get_random_coord(polygon_zone)
                                        #print(point)
                                    
                                else:

                                    if type_coord == pu:
                                        
                                        '''
                                        probabilities = network.zones['density_pois'].tolist()
                                        random_zone_id = np.random.choice(a=zones, size=1, p=probabilities)
                                        
                                        random_zone_id = random_zone_id[0]
                                        polygon_zone = network.zones.loc[random_zone_id]['polygon']
                                        point = network._get_random_coord(polygon_zone)
                                        '''
                                        if not first_time:
                                            random.seed(datetime.now())
                                            rseed = np.random.uniform(0,1)
                                            rseed = rseed * 1000
                                            #print(rseed)
                                            rseed = int(rseed)

                                            if pdfunc == 'cauchy':
                                                cauchy.random_state = np.random.RandomState(seed=rseed)
                                                radius = cauchy.rvs(loc=loc, scale=scale, size=1)

                                            if pdfunc == 'expon':
                                                expon.random_state = np.random.RandomState(seed=rseed)
                                                radius = expon.rvs(loc=loc, scale=scale, size=1)

                                            if pdfunc == 'gamma':
                                                gamma.random_state = np.random.RandomState(seed=rseed)
                                                radius = gamma.rvs(a=aux, loc=loc, scale=scale, size=1)

                                            if pdfunc == 'gilbrat':
                                                gilbrat.random_state = np.random.RandomState(seed=rseed)
                                                radius = gilbrat.rvs(loc=loc, scale=scale, size=1)

                                            if pdfunc == 'lognorm':
                                                lognorm.random_state = np.random.RandomState(seed=rseed)
                                                radius = lognorm.rvs(loc=loc, scale=scale, size=1)

                                            if pdfunc == 'normal':
                                                normal.random_state = np.random.RandomState(seed=rseed)
                                                radius = normal.rvs(loc=loc, scale=scale, size=1)

                                            if pdfunc == 'powerlaw':
                                                powerlaw.random_state = np.random.RandomState(seed=rseed)
                                                radius = powerlaw.rvs(a=aux, loc=loc, scale=scale, size=1)

                                            if pdfunc == 'uniform':
                                                uniform.random_state = np.random.RandomState(seed=rseed)
                                                radius = uniform.rvs(loc=loc, scale=scale, size=1)

                                            if pdfunc == 'wald':
                                                wald.random_state = np.random.RandomState(seed=rseed)
                                                radius = wald.rvs(loc=loc, scale=scale, size=1)
                                            
                                        else:
                                            radius = distancex
                                            first_time = False

                                        #if radius > 20000:
                                        #    print(radius)
                                        #print(radius)
                                        att_start = do
                                        lat = attributes[att_start+'y']
                                        lon = attributes[att_start+'x']
                                        point = network._get_random_coord_radius(lat, lon, radius, network.polygon)
                                        #print(point)

                                    elif type_coord == do:
                                        
                                        #att_start = GA.nodes[att]['start_point']
                                        #zone_start = int(attributes[att_start+'zone'])
                                        #probabilities = network.zone_probabilities[zone_start].tolist()
                                        #print(inst.network.zone_probabilities[zone_start].head())
                                        #print(probabilities)
                                        probabilities = network.zones['density_pois'].tolist()
                                        random_zone_id = np.random.choice(a=zones, size=1, p=probabilities)

                                        random_zone_id = random_zone_id[0]
                                        polygon_zone = network.zones.loc[random_zone_id]['polygon']
                                        point = network._get_random_coord(polygon_zone)
                                        #print(point)

                            else:

                                point = network._get_random_coord(network.polygon)
                            
                            #print(point)
                            point = (point.y, point.x)

                        else:

                            zones = parameters[zone]['zones'+str(replicate_num)]

                            if 'weights' in GA.nodes[att]:

                                random_zone_id = random.choices(zones, weights=GA.nodes[att]['weights'], k=1)
                                random_zone_id = random_zone_id[0]
                                polygon_zone = network.zones.loc[random_zone_id]['polygon']

                            else:

                                random_zone_id = random.choice(zones)
                                polygon_zone = network.zones.loc[random_zone_id]['polygon']
                            
                            try:

                                if math.isnan(polygon_zone):
                                    R = network.zones.loc[random_zone_id]['radius']
                                    clat = network.zones.loc[random_zone_id]['center_y']
                                    clon = network.zones.loc[random_zone_id]['center_x']
                                    point = network._get_random_coord_circle(R, clat, clon)
                                    
                            except TypeError:
                            
                                point = network._get_random_coord(polygon_zone)                                

                            point = (point.y, point.x)
                    else:

                        if 'subset_locations' in GA.nodes[att]:

                            loc = GA.nodes[att]['subset_locations']
                            
                            if 'weights' in GA.nodes[att]:

                                if parameters[loc]['locs'] == 'schools':
                                    idxs = random.choices(parameters[loc]['list_ids'+str(replicate_num)], weights=GA.nodes[att]['weights'], k=1)
                                    point = (network.schools.loc[idxs[0], 'lat'], network.schools.loc[idxs[0], 'lon'])
                            else:

                                if parameters[loc]['locs'] == 'schools':
                                    idxs = random.choice(parameters[loc]['list_ids'+str(replicate_num)])
                                    point = (network.schools.loc[idxs, 'lat'], network.schools.loc[idxs, 'lon'])


                    attributes[att] = point
                    attributes[att+'x'] = point[1]
                    attributes[att+'y'] = point[0]
                    
                    if point[1] != -1:
                        not_feasible_attribute = False
                        #node_drive = ox.get_nearest_node(network.G_drive, point)
                        #node_walk = ox.get_nearest_node(network.G_walk, point)

                        node_drive = ox.nearest_nodes(network.G_drive, point[1], point[0])
                        #node_walk = ox.nearest_nodes(network.G_walk, point[1], point[0])

                        attributes[att+'node_drive'] = int(node_drive)
                        #attributes[att+'node_walk'] = int(node_walk)

                        attributes[att+'zone'] = int(random_zone_id)

                    else:
                        #print('minus 1')
                        not_feasible_attribute = True
                        

                if att == pu:         
                    if not_feasible_attribute:
                
                        feasible_data = False
                        break 

                if att == do:         
                    if not_feasible_attribute:
                
                        feasible_data = False
                        break 

                if 'pdf' in GA.nodes[att]:

                    if GA.nodes[att]['pdf'][0]['type'] == 'normal':

                        attributes[att] = np.random.normal(GA.nodes[att]['pdf'][0]['loc'], GA.nodes[att]['pdf'][0]['scale'])
                        
                        if (GA.nodes[att]['type'] == 'time') or (GA.nodes[att]['type'] == 'integer'):
                            attributes[att] = int(attributes[att])

                    
                    if GA.nodes[att]['pdf'][0]['type'] == 'uniform':


                        if 'weights' in GA.nodes[att]:

                            attributes[att] = random.choices(GA.nodes[att]['all_values'], weights=GA.nodes[att]['weights'], k=1)
                            attributes[att] = attributes[att][0]
                            #print(attributes[att])

                        else:

                            if (GA.nodes[att]['type'] == 'time') or (GA.nodes[att]['type'] == 'integer'):
                                attributes[att] = np.random.randint(GA.nodes[att]['pdf'][0]['loc'], GA.nodes[att]['pdf'][0]['scale'])
                            else:
                                attributes[att] = np.random.uniform(GA.nodes[att]['pdf'][0]['loc'], GA.nodes[att]['pdf'][0]['scale'])

                            #print(inst.GA.nodes[att]['pdf'][0]['min'], inst.GA.nodes[att]['pdf'][0]['max'])
                            #if att == 'ambulatory':
                                #print(attributes[att])

                    if GA.nodes[att]['pdf'][0]['type'] == 'cauchy':

                        attributes[att] = cauchy.rvs(loc=GA.nodes[att]['pdf'][0]['loc'], scale=GA.nodes[att]['pdf'][0]['scale'], size=1)

                        if (GA.nodes[att]['type'] == 'time') or (GA.nodes[att]['type'] == 'integer'):
                            attributes[att] = int(attributes[att])

                    if GA.nodes[att]['pdf'][0]['type'] == 'expon':

                        attributes[att] = expon.rvs(loc=GA.nodes[att]['pdf'][0]['loc'], scale=GA.nodes[att]['pdf'][0]['scale'], size=1)

                        if (GA.nodes[att]['type'] == 'time') or (GA.nodes[att]['type'] == 'integer'):
                            attributes[att] = int(attributes[att])

                    if GA.nodes[att]['pdf'][0]['type'] == 'gamma':

                        attributes[att] = gamma.rvs(a=GA.nodes[att]['pdf'][0]['aux'], loc=GA.nodes[att]['pdf'][0]['loc'], scale=GA.nodes[att]['pdf'][0]['scale'], size=1)

                        if (GA.nodes[att]['type'] == 'time') or (GA.nodes[att]['type'] == 'integer'):
                            attributes[att] = int(attributes[att])

                    if GA.nodes[att]['pdf'][0]['type'] == 'gilbrat':

                        attributes[att] = gilbrat.rvs(loc=GA.nodes[att]['pdf'][0]['loc'], scale=GA.nodes[att]['pdf'][0]['scale'], size=1)

                        if (GA.nodes[att]['type'] == 'time') or (GA.nodes[att]['type'] == 'integer'):
                            attributes[att] = int(attributes[att])

                    if GA.nodes[att]['pdf'][0]['type'] == 'lognorm':

                        attributes[att] = lognorm.rvs(loc=GA.nodes[att]['pdf'][0]['loc'], scale=GA.nodes[att]['pdf'][0]['scale'], size=1)

                        if (GA.nodes[att]['type'] == 'time') or (GA.nodes[att]['type'] == 'integer'):
                            attributes[att] = int(attributes[att])

                    if GA.nodes[att]['pdf'][0]['type'] == 'powerlaw':

                        attributes[att] = powerlaw.rvs(a=GA.nodes[att]['pdf'][0]['aux'], loc=GA.nodes[att]['pdf'][0]['loc'], scale=GA.nodes[att]['pdf'][0]['scale'], size=1)

                        if (GA.nodes[att]['type'] == 'time') or (GA.nodes[att]['type'] == 'integer'):
                            attributes[att] = int(attributes[att])

                    if GA.nodes[att]['pdf'][0]['type'] == 'wald':

                        attributes[att] = wald.rvs(loc=GA.nodes[att]['pdf'][0]['loc'], scale=GA.nodes[att]['pdf'][0]['scale'], size=1)

                        if (GA.nodes[att]['type'] == 'time') or (GA.nodes[att]['type'] == 'integer'):
                            attributes[att] = int(attributes[att])

                    '''
                    if GA.nodes[att]['pdf'][0]['type'] == 'poisson':

                        attributes[att] = np.random.poisson(GA.nodes[att]['pdf'][0]['lam'])
                        
                        if (GA.nodes[att]['type'] == 'time') or (GA.nodes[att]['type'] == 'integer'):
                            attributes[att] = int(attributes[att])
                    '''



                elif 'expression' in GA.nodes[att]:

                    expression = GA.nodes[att]['expression']

                    if att == 'time_stamp':
                        static = np.random.uniform(0, 1)
                        #print(static)
                        if static < GA.nodes[att]['static_probability']:
                            expression = '0'
                    
                    for attx in sorted_attributes:

                        if attx in attributes:
                            expression = re.sub(attx, str(attributes[attx]), expression)                        
                    
                    #print(expression)

                    try:
                        
                        #attributes[att] = eval(expression)
                        #print(expression)
                        attributes[att] = eval_expression(expression)
                        #print(attributes[att])

                        if (GA.nodes[att]['type'] == 'time') or (GA.nodes[att]['type'] == 'integer'):
                            attributes[att] = int(attributes[att])
                        #if att == 'time_stamp':
                        #   print(attributes[att])
                    
                    except (SyntaxError, NameError, ValueError, TypeError):

                        #print(inst.GA.nodes[att]['expression'])
                        
                        expression = re.split(r"[(,) ]", GA.nodes[att]['expression'])
                        
                        if expression[0] == 'dtt':

                            if (expression[1] in attributes) and (expression[2] in attributes):
                                node_drive1 = attributes[expression[1]+'node_drive']
                                node_drive2 = attributes[expression[2]+'node_drive']
                            else:
                                raise ValueError('expression '+GA.nodes[att]['expression']+' not possible to evaluate. check the parameters')

                            attributes[att] = network._return_estimated_travel_time_drive(node_drive1, node_drive2)

                        elif expression[0] == 'stops':

                            stops = []
                            stops_walking_distance = []

                            if (expression[1] in attributes):
                                node_walk = attributes[expression[1]+'node_walk']
                            else:
                                raise ValueError('expression '+GA.nodes[att]['expression']+' not possible to evaluate. check the parameters')
                            

                            for index in network.bus_stations_ids:
                    
                                osmid_possible_stop = int(network.bus_stations.loc[index, 'osmid_walk'])

                                eta_walk = network.get_eta_walk(node_walk, osmid_possible_stop, attributes['walk_speed'])
                                if (eta_walk >= 0) and (eta_walk <= attributes['max_walking']):
                                    #print(eta_walk)
                                    stops.append(index)
                                    stops_walking_distance.append(eta_walk)

                            '''
                            fig, ax = ox.plot_graph(inst.network.G_drive, show=False, close=False, node_color='#000000', node_size=6, bgcolor="#ffffff", edge_color="#999999")

                            node_id = attributes[expression[1]+'node_drive']
                            x = inst.network.G_drive.nodes[node_id]['x']
                            y = inst.network.G_drive.nodes[node_id]['y']
                            ax.scatter(x, y, c='red', s=8, marker=",")

                            for stop in stops:
                                node_id = inst.network.bus_stations.loc[int(stop), 'osmid_drive']
                                x = inst.network.G_drive.nodes[node_id]['x']
                                y = inst.network.G_drive.nodes[node_id]['y']
                                ax.scatter(x, y, c='green', s=8, marker=",")

                            plt.show()
                            plt.close(fig)
                            '''

                            attributes[att] = stops
                            attributes[att+'_walking_distance'] = stops_walking_distance

                        elif expression[0] == 'walk':

                            if (expression[1] in attributes) and (expression[2] in attributes):
                                node_walk1 = attributes[expression[1]+'node_walk']
                                node_walk2 = attributes[expression[2]+'node_walk']
                            else:
                                raise ValueError('expression '+GA.nodes[att]['expression']+' not possible to evaluate. check the parameters')
                            
                            attributes[att] = network.get_eta_walk(node_walk1, node_walk2, attributes['walk_speed'])

                        elif expression[0] == 'dist_drive':

                            
                            if (expression[1] in attributes) and (expression[2] in attributes):
                                node_drive1 = attributes[expression[1]+'node_drive']
                                node_drive2 = attributes[expression[2]+'node_drive']
                            else:
                                raise ValueError('expression '+GA.nodes[att]['expression']+' not possible to evaluate. check the parameters')

                            attributes[att] = network._return_estimated_distance_drive(node_drive1, node_drive2)

                            #print(att)
                            #print(attributes[att])
                            '''
                            if att == 'direct_distance':

                                not_feasible_attribute = True

                                if ((attributes[att] <= (distances[i] + 500)) and (attributes[att] >= (distances[i] - 500))):
                                    #print('here')
                                    not_feasible_attribute = False
                            '''

                        else:
                            raise SyntaxError('expression '+GA.nodes[att]['expression']+' is not supported')
                '''
                if att == 'direct_distance':         
                    if not_feasible_attribute:
                
                        feasible_data = False
                        break 
                ''' 

                #check constraints

                if 'constraints' in GA.nodes[att]:

                    not_feasible_attribute = False

                    for constraint in GA.nodes[att]['constraints']:

                        #print(constraint)

                        for attx in sorted_attributes:
                            if attx in attributes:
                                constraint = re.sub(attx, str(attributes[attx]), constraint) 

                        for paramx in parameters:
                            if 'value' in parameters[paramx]:
                                constraint = re.sub(paramx, str(parameters[paramx]['value']), constraint)                       
                        
                        #print(constraint)

                        #check_constraint = eval(constraint)
                        check_constraint = eval_expression(constraint)
                        #print(check_constraint)
                        
                        if not check_constraint:
                            not_feasible_attribute = True
                            exhaustion_iterations += 1
                            if 'expression' in GA.nodes[att]:
                                #this means that another attribute should be remaked because of this, therefore everything is discarded
                                exhaustion_iterations = 9999

                else:

                    not_feasible_attribute = False
           
            if not_feasible_attribute:
            
                feasible_data = False
                break

        if feasible_data:
            #print("here")
            return attributes

    return attributes
     
def _generate_requests( 
    inst,
    replicate_num,
):
    
    print("Now generating " + " request_data")

    origin_points=[]
    destination_points=[]
    h = 0
    num_requests = 0

    node_list = []
    node_list_seq = []
    inst.depot_nodes_seq = []

    if replicate_num == 0:
        inst.network.zones['density_pois'] = inst.network.zones['density_pois']/100

    distances = []
    #distancesx = wald.rvs(loc=65.081, scale=7035.153, size=inst.parameters['records']['value']+10000)

    #distancesx = wald.rvs(loc=116.81, scale=6578.84, size=inst.parameters['records']['value']+20000)

    method_poi = False
    pu = None 
    do = None 
    pdfunc = None
    aux = None 
    loc = None 
    scale = None 

    num_requests = inst.parameters['records']['value']
    if 'method_pois' in inst.parameters:

        method_poi = True 
        pu = inst.parameters['method_pois']['value']['locations'][0]
        do = inst.parameters['method_pois']['value']['locations'][1]
        #print(pu, do)
        pdfunc = inst.parameters['method_pois']['value']['pdf']['type']

        aux = -1
        loc = inst.parameters['method_pois']['value']['pdf']['loc']
        scale = inst.parameters['method_pois']['value']['pdf']['scale']
        #print('here')
        if inst.parameters['method_pois']['value']['pdf']['type'] == 'cauchy':
            distancesx = cauchy.rvs(loc=inst.parameters['method_pois']['value']['pdf']['loc'], scale=inst.parameters['method_pois']['value']['pdf']['scale'], size=inst.parameters['records']['value']+20000)

        if inst.parameters['method_pois']['value']['pdf']['type'] == 'expon':
            distancesx = expon.rvs(loc=inst.parameters['method_pois']['value']['pdf']['loc'], scale=inst.parameters['method_pois']['value']['pdf']['scale'], size=inst.parameters['records']['value']+20000)

        if inst.parameters['method_pois']['value']['pdf']['type'] == 'gamma':
            distancesx = gamma.rvs(a=inst.parameters['method_pois']['value']['pdf']['aux'], loc=inst.parameters['method_pois']['value']['pdf']['loc'], scale=inst.parameters['method_pois']['value']['pdf']['scale'], size=inst.parameters['records']['value']+20000)
            aux = inst.parameters['method_pois']['value']['pdf']['aux']

        if inst.parameters['method_pois']['value']['pdf']['type'] == 'gilbrat':
            distancesx = gilbrat.rvs(loc=inst.parameters['method_pois']['value']['pdf']['loc'], scale=inst.parameters['method_pois']['value']['pdf']['scale'], size=inst.parameters['records']['value']+20000)

        if inst.parameters['method_pois']['value']['pdf']['type'] == 'lognorm':
            distancesx = lognorm.rvs(loc=inst.parameters['method_pois']['value']['pdf']['loc'], scale=inst.parameters['method_pois']['value']['pdf']['scale'], size=inst.parameters['records']['value']+20000)

        if inst.parameters['method_pois']['value']['pdf']['type'] == 'normal':
            distancesx = normal.rvs(loc=inst.parameters['method_pois']['value']['pdf']['loc'], scale=inst.parameters['method_pois']['value']['pdf']['scale'], size=inst.parameters['records']['value']+20000)

        if inst.parameters['method_pois']['value']['pdf']['type'] == 'powerlaw':
            distancesx = powerlaw.rvs(a=inst.parameters['method_pois']['value']['pdf']['aux'], loc=inst.parameters['method_pois']['value']['pdf']['loc'], scale=inst.parameters['method_pois']['value']['pdf']['scale'], size=inst.parameters['records']['value']+20000)
            aux = inst.parameters['method_pois']['value']['pdf']['aux']

        if inst.parameters['method_pois']['value']['pdf']['type'] == 'uniform':
            distancesx = uniform.rvs(loc=inst.parameters['method_pois']['value']['pdf']['loc'], scale=inst.parameters['method_pois']['value']['pdf']['scale'], size=inst.parameters['records']['value']+20000)

        if inst.parameters['method_pois']['value']['pdf']['type'] == 'wald':
            distancesx = wald.rvs(loc=inst.parameters['method_pois']['value']['pdf']['loc'], scale=inst.parameters['method_pois']['value']['pdf']['scale'], size=inst.parameters['records']['value']+20000)


        distances = [x for x in distancesx if x >= 500]
        distances = [x for x in distancesx if x < 32000]
        #print(distances)
        print('len distances: ', len(distances))
    else:

        distances = [0] * num_requests

    #print(inst.network.zones['density_pois'].sum())

    #for idx, zone in inst.network.zone_ranks.iterrows():

        #normsum = inst.network.zone_ranks.loc[idx].sum()

        #inst.network.zone_ranks.loc[idx] = inst.network.zone_ranks.loc[idx]/normsum

        #print(inst.network.zone_ranks.loc[idx].sum())

    for param in inst.parameters:

        if inst.parameters[param]['type'] == 'array_locations':

            if inst.parameters[param]['locs'] == 'random':

                inst.parameters[param]['list'+str(replicate_num)] = []
                for elem in inst.parameters[param]['list']:
                    inst.parameters[param]['list'+str(replicate_num)].append(elem)

                inst.parameters[param]['list_node_drive'+str(replicate_num)] = []    
                for elem in inst.parameters[param]['list_node_drive']:
                    inst.parameters[param]['list_node_drive'+str(replicate_num)].append(elem)

                while len(inst.parameters[param]['list'+str(replicate_num)]) < inst.parameters[param]['size']:

                    point = inst.network._get_random_coord(inst.network.polygon)
                    point = (point.y, point.x)
                    #node_drive = ox.get_nearest_node(inst.network.G_drive, point)
                    node_drive = ox.nearest_nodes(inst.network.G_drive, point[1], point[0])
                    if node_drive not in inst.parameters[param]['list_node_drive'+str(replicate_num)]:
                        inst.parameters[param]['list'+str(replicate_num)].append("random_loc"+len(inst.parameters[param]['list'+str(replicate_num)]))
                        inst.parameters[param]['list_node_drive'+str(replicate_num)].append(node_drive)

                #print(inst.parameters[param]['list_node_drive'+str(replicate_num)])

            if inst.parameters[param]['locs'] == 'schools':

                inst.parameters[param]['list_ids'+str(replicate_num)] = []
                for elem in inst.parameters[param]['list_ids']:
                    inst.parameters[param]['list_ids'+str(replicate_num)].append(elem)

                inst.parameters[param]['list_node_drive'+str(replicate_num)] = []    
                for elem in inst.parameters[param]['list_node_drive']:
                    inst.parameters[param]['list_node_drive'+str(replicate_num)].append(elem)

                while len(inst.parameters[param]['list_ids'+str(replicate_num)]) < inst.parameters[param]['size']:

                    random_school_id = np.random.randint(0, len(inst.network.schools))
                    random_school_id = int(random_school_id)

                    if random_school_id not in inst.parameters[param]['list_ids'+str(replicate_num)]:
                        inst.parameters[param]['list_ids'+str(replicate_num)].append(random_school_id)
                        node_drive = inst.network.schools.loc[int(random_school_id), 'osmid_drive']
                        inst.parameters[param]['list_node_drive'+str(replicate_num)].append(node_drive)

                #print(inst.parameters[param]['list_ids'+str(replicate_num)])

        if inst.parameters[param]['type'] == 'array_zones':

            inst.parameters[param]['zones'+str(replicate_num)] = []
            for elem in inst.parameters[param]['zones']:
                inst.parameters[param]['zones'+str(replicate_num)].append(elem)

            while len(inst.parameters[param]['zones'+str(replicate_num)]) < inst.parameters[param]['size']: 
            
                random_zone_id = np.random.randint(0, len(inst.network.zones))
                random_zone_id = int(random_zone_id)

                if random_zone_id not in inst.parameters[param]['zones'+str(replicate_num)]:
                    inst.parameters[param]['zones'+str(replicate_num)].append(random_zone_id)

            #print(inst.parameters[param]['zones'+str(replicate_num)])               

    
    print(num_requests)
    instance_data = {}
    all_instance_data = {}  

    gc.collect()
    ray.shutdown()
    ray.init(num_cpus=cpu_count(), object_store_memory=14000000000)

    GA_id = ray.put(inst.GA)
    #del inst.GA
    #gc.collect()
    #print('here')
    #radius = wald.rvs(loc=65.081, scale=7035.153, size=40790)
    ##distances = [x for x in radius if x > 20000]
    #print('teste', len(distances))
    #print(distances)

    #gc.collect()
    #print("heerree")
    network_id = ray.put(inst.network)
    #del inst.network
    #gc.collect()
    sorted_attributes_id = ray.put(inst.sorted_attributes)
    parameters_id = ray.put(inst.parameters)
    print("HERE")
    all_reqs = ray.get([_generate_single_data.remote(GA_id, network_id, sorted_attributes_id, parameters_id, i, method_poi, distances[i], pu, do, pdfunc, loc, scale, aux) for i in range(num_requests)]) 
    print("OUT HERE")
    #inst.GA = GA_id
    del GA_id
    gc.collect()

    #inst.network = network_id
    del network_id
    gc.collect()

    del sorted_attributes_id
    del parameters_id
    gc.collect()

    i = 0
    for req in all_reqs:

        instance_data.update({i: req})
        i += 1    
              
    del all_reqs
    gc.collect()

    #count, bins, ignored = plt.hist(ps, 30, density=True)
    #plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')    
    #plt.show()
    all_instance_data.update({'num_data:': len(instance_data),
                          'requests': instance_data
                          })


    final_filename = ''
    #print(inst.instance_filename)
    for p in inst.instance_filename:

        if p in inst.parameters:
            if 'value' in inst.parameters[p]:
                strv = str(inst.parameters[p]['value'])
                strv = strv.replace(" ", "")

                if len(final_filename) > 0:
                    final_filename = final_filename + '_' + strv
                else: final_filename = strv

    #print(final_filename)

    if 'travel_time_matrix' in inst.parameters:
        if inst.parameters['travel_time_matrix']['value'] is True:
            node_list = []
            node_list_seq = []
            lvid = 0

            for location in inst.parameters['travel_time_matrix']['locations']:

                if location == 'bus_stations':

                    for index, row in inst.network.bus_stations.iterrows():

                        node_list.append(row['osmid_drive'])
                        node_list_seq.append(index)
                        lvid = index

                if location in inst.parameters:

                    inst.parameters[location]['list_seq_id'+str(replicate_num)] = []
                    #print(inst.parameters[location]['list_node_drive'+str(replicate_num)])
                    for d in inst.parameters[location]['list_node_drive'+str(replicate_num)]:

                        node_list.append(d)
                        node_list_seq.append(lvid)
                        inst.parameters[location]['list_seq_id'+str(replicate_num)].append(lvid)
                        lvid += 1

                if location in inst.sorted_attributes:

                    for d in instance_data:

                        node_list.append(instance_data[d][location+'node_drive'])
                        instance_data[d][location+'id'] = lvid
                        node_list_seq.append(lvid)
                        lvid += 1

            if replicate_num < 0:
                print('ttm')
                travel_time = inst.network._get_travel_time_matrix("list", node_list=node_list)
                travel_time_json = travel_time.tolist()

                all_instance_data.update({'travel_time_matrix': travel_time_json
                                })
                
                print('leave ttm')          
                if 'graphml' in inst.parameters:
                    if inst.parameters['graphml']['value'] is True:
                        #creates a graph that will serve as the travel time matrix for the given set of requests
                        gtt = nx.DiGraph() 

                        if 'bus_stations' in inst.parameters['travel_time_matrix']['locations']:

                            for index, row in inst.network.bus_stations.iterrows():

                                gtt.add_node(index, type='bus_stations')

                        for param in inst.parameters:

                            if param in inst.parameters['travel_time_matrix']['locations']:

                                for d in inst.parameters[param]['list_seq_id'+str(replicate_num)]:

                                    node = d
                                    gtt.add_node(node, type=param)

                        for att in inst.sorted_attributes:
                            
                            if att in inst.parameters['travel_time_matrix']['locations']:

                                for d in instance_data:

                                    node = instance_data[d][att+'id']
                                    gtt.add_node(node, type=att)

                        for u in node_list_seq:
                            for v in node_list_seq:

                                gtt.add_edge(u, v, travel_time=travel_time_json[u][v])

                        output_name_graphml = os.path.join(inst.save_dir_graphml, inst.output_folder_base + '_' + str(replicate_num) + '.graphml')

                        nx.write_graphml(gtt, output_name_graphml)


                output_file_ttm_csv = os.path.join(inst.save_dir_ttm, 'travel_time_matrix_' + final_filename + '_' + str(replicate_num) + '.csv')
            
                ttml = []
                for u in node_list_seq:
                    d = {}
                    d['osmid_origin'] = u
                    for v in node_list_seq:

                        dist_uv = int(travel_time_json[u][v])
                        sv = str(v)
                        d[sv] = dist_uv
                    ttml.append(d)
                    del d

                ttmpd = pd.DataFrame(ttml)

                ttmpd.to_csv(output_file_ttm_csv)
                ttmpd.set_index(['osmid_origin'], inplace=True)


    save_dir = os.getcwd()+'/'+inst.output_folder_base
    save_dir_images = os.path.join(save_dir, 'images')
    #plot_requests(inst.network, save_dir_images, origin_points, destination_points)

    inst.output_file_json = os.path.join(inst.save_dir_json, final_filename + '_' + str(replicate_num) + '.json')

    with open(inst.output_file_json, 'w') as file:
        json.dump(all_instance_data, file, indent=4)
        file.close() 

    del all_instance_data
    gc.collect()

def _generate_requests_regular( 
    inst,
    replicate_num,
):
    
    print("Now generating " + " request_data")


    origin_points=[]
    destination_points=[]
    h = 0
    num_requests = 0

    node_list = []
    node_list_seq = []
    inst.depot_nodes_seq = []

    if replicate_num == 0:
        inst.network.zones['density_pois'] = inst.network.zones['density_pois']/100
    #print(inst.network.zones['density_pois'].sum())

    #for idx, zone in inst.network.zone_ranks.iterrows():

        #normsum = inst.network.zone_ranks.loc[idx].sum()

        #inst.network.zone_ranks.loc[idx] = inst.network.zone_ranks.loc[idx]/normsum

        #print(inst.network.zone_ranks.loc[idx].sum())

    for param in inst.parameters:

        if inst.parameters[param]['type'] == 'array_locations':

            if inst.parameters[param]['locs'] == 'random':

                inst.parameters[param]['list'+str(replicate_num)] = []
                for elem in inst.parameters[param]['list']:
                    inst.parameters[param]['list'+str(replicate_num)].append(elem)

                inst.parameters[param]['list_node_drive'+str(replicate_num)] = []    
                for elem in inst.parameters[param]['list_node_drive']:
                    inst.parameters[param]['list_node_drive'+str(replicate_num)].append(elem)

                while len(inst.parameters[param]['list'+str(replicate_num)]) < inst.parameters[param]['size']:

                    point = inst.network._get_random_coord(inst.network.polygon)
                    point = (point.y, point.x)
                    node_drive = ox.get_nearest_node(inst.network.G_drive, point)
                    if node_drive not in inst.parameters[param]['list_node_drive'+str(replicate_num)]:
                        inst.parameters[param]['list'+str(replicate_num)].append("random_loc"+len(inst.parameters[param]['list'+str(replicate_num)]))
                        inst.parameters[param]['list_node_drive'+str(replicate_num)].append(node_drive)

                #print(inst.parameters[param]['list_node_drive'+str(replicate_num)])

            if inst.parameters[param]['locs'] == 'schools':

                inst.parameters[param]['list_ids'+str(replicate_num)] = []
                for elem in inst.parameters[param]['list_ids']:
                    inst.parameters[param]['list_ids'+str(replicate_num)].append(elem)

                inst.parameters[param]['list_node_drive'+str(replicate_num)] = []    
                for elem in inst.parameters[param]['list_node_drive']:
                    inst.parameters[param]['list_node_drive'+str(replicate_num)].append(elem)

                while len(inst.parameters[param]['list_ids'+str(replicate_num)]) < inst.parameters[param]['size']:

                    random_school_id = np.random.randint(0, len(inst.network.schools))
                    random_school_id = int(random_school_id)

                    if random_school_id not in inst.parameters[param]['list_ids'+str(replicate_num)]:
                        inst.parameters[param]['list_ids'+str(replicate_num)].append(random_school_id)
                        node_drive = inst.network.schools.loc[int(random_school_id), 'osmid_drive']
                        inst.parameters[param]['list_node_drive'+str(replicate_num)].append(node_drive)

                #print(inst.parameters[param]['list_ids'+str(replicate_num)])

        if inst.parameters[param]['type'] == 'array_zones':

            inst.parameters[param]['zones'+str(replicate_num)] = []
            for elem in inst.parameters[param]['zones']:
                inst.parameters[param]['zones'+str(replicate_num)].append(elem)

            while len(inst.parameters[param]['zones'+str(replicate_num)]) < inst.parameters[param]['size']: 
            
                random_zone_id = np.random.randint(0, len(inst.network.zones))
                random_zone_id = int(random_zone_id)

                if random_zone_id not in inst.parameters[param]['zones'+str(replicate_num)]:
                    inst.parameters[param]['zones'+str(replicate_num)].append(random_zone_id)

            #print(inst.parameters[param]['zones'+str(replicate_num)])               

    num_requests = inst.parameters['records']['value']
    print(num_requests)
    instance_data = {}
    all_instance_data = {}  
    i=0
    #ps = []
    while i < num_requests:

        print(i)
        attributes = {}
       
        feasible_data = True
        for att in inst.sorted_attributes:

            not_feasible_attribute = True
            exhaustion_iterations = 0

            while (not_feasible_attribute) and (exhaustion_iterations < 100):

                if inst.GA.nodes[att]['type'] == 'location':

                    random_zone_id = -1

                    if 'subset_zones' in inst.GA.nodes[att]:

                        zone = inst.GA.nodes[att]['subset_zones']

                        if zone is False:

                            if 'rank_model' in inst.GA.nodes[att]:

                                type_coord = inst.GA.nodes[att]['rank_model']
                                zones = inst.network.zones.index.tolist()
                                

                                if type_coord == 'destination':
                                    
                                    probabilities = inst.network.zones['density_pois'].tolist()
                                    random_zone_id = np.random.choice(a=zones, size=1, p=probabilities)
                                    
                                    random_zone_id = random_zone_id[0]
                                    polygon_zone = inst.network.zones.loc[random_zone_id]['polygon']
                                    point = inst.network._get_random_coord(polygon_zone)

                                elif type_coord == 'origin':
                                    
                                    att_start = inst.GA.nodes[att]['start_point']
                                    zone_start = int(attributes[att_start+'zone'])
                                    probabilities = inst.network.zone_probabilities[zone_start].tolist()
                                    #print(inst.network.zone_probabilities[zone_start].head())
                                    #print(probabilities)
                                    random_zone_id = np.random.choice(a=zones, size=1, p=probabilities)

                                    random_zone_id = random_zone_id[0]
                                    polygon_zone = inst.network.zones.loc[random_zone_id]['polygon']
                                    point = inst.network._get_random_coord(polygon_zone)

                            else:

                                point = inst.network._get_random_coord(inst.network.polygon)
                            
                            #print(point)
                            point = (point.y, point.x)

                        else:

                            zones = inst.parameters[zone]['zones'+str(replicate_num)]

                            if 'weights' in inst.GA.nodes[att]:

                                random_zone_id = random.choices(zones, weights=inst.GA.nodes[att]['weights'], k=1)
                                random_zone_id = random_zone_id[0]
                                polygon_zone = inst.network.zones.loc[random_zone_id]['polygon']

                            else:

                                random_zone_id = random.choice(zones)
                                polygon_zone = inst.network.zones.loc[random_zone_id]['polygon']
                            
                            try:

                                if math.isnan(polygon_zone):
                                    R = inst.network.zones.loc[random_zone_id]['radius']
                                    clat = inst.network.zones.loc[random_zone_id]['center_y']
                                    clon = inst.network.zones.loc[random_zone_id]['center_x']
                                    point = inst.network._get_random_coord_circle(R, clat, clon)
                                    
                            except TypeError:
                            
                                point = inst.network._get_random_coord(polygon_zone)                                

                            point = (point.y, point.x)
                    else:

                        if 'subset_locations' in inst.GA.nodes[att]:

                            loc = inst.GA.nodes[att]['subset_locations']
                            
                            if 'weights' in inst.GA.nodes[att]:

                                if inst.parameters[loc]['locs'] == 'schools':
                                    idxs = random.choices(inst.parameters[loc]['list_ids'+str(replicate_num)], weights=inst.GA.nodes[att]['weights'], k=1)
                                    point = (inst.network.schools.loc[idxs[0], 'lat'], inst.network.schools.loc[idxs[0], 'lon'])
                            else:

                                if inst.parameters[loc]['locs'] == 'schools':
                                    idxs = random.choice(inst.parameters[loc]['list_ids'+str(replicate_num)])
                                    point = (inst.network.schools.loc[idxs, 'lat'], inst.network.schools.loc[idxs, 'lon'])


                    attributes[att] = point
                    attributes[att+'x'] = point[1]
                    attributes[att+'y'] = point[0]
                    
                    node_drive = ox.get_nearest_node(inst.network.G_drive, point)
                    node_walk = ox.get_nearest_node(inst.network.G_walk, point)

                    attributes[att+'node_drive'] = int(node_drive)
                    attributes[att+'node_walk'] = int(node_walk)

                    attributes[att+'zone'] = int(random_zone_id)

                
                if 'pdf' in inst.GA.nodes[att]:

                    if inst.GA.nodes[att]['pdf'][0]['type'] == 'normal':

                        attributes[att] = np.random.normal(inst.GA.nodes[att]['pdf'][0]['mean'], inst.GA.nodes[att]['pdf'][0]['std'])
                        
                        if (inst.GA.nodes[att]['type'] == 'time') or (inst.GA.nodes[att]['type'] == 'integer'):
                            attributes[att] = int(attributes[att])


                    if inst.GA.nodes[att]['pdf'][0]['type'] == 'poisson':

                        attributes[att] = np.random.poisson(inst.GA.nodes[att]['pdf'][0]['lam'])
                        
                        if (inst.GA.nodes[att]['type'] == 'time') or (inst.GA.nodes[att]['type'] == 'integer'):
                            attributes[att] = int(attributes[att])

                        
                        #print(attributes[att])

                    if inst.GA.nodes[att]['pdf'][0]['type'] == 'uniform':


                        if 'weights' in inst.GA.nodes[att]:

                            attributes[att] = random.choices(inst.GA.nodes[att]['all_values'], weights=inst.GA.nodes[att]['weights'], k=1)
                            attributes[att] = attributes[att][0]
                            #print(attributes[att])

                        else:
                            if (inst.GA.nodes[att]['type'] == 'time') or (inst.GA.nodes[att]['type'] == 'integer'):
                                attributes[att] = np.random.randint(inst.GA.nodes[att]['pdf'][0]['min'], inst.GA.nodes[att]['pdf'][0]['max'])
                            else:
                                attributes[att] = np.random.uniform(inst.GA.nodes[att]['pdf'][0]['min'], inst.GA.nodes[att]['pdf'][0]['max'])

                            #print(inst.GA.nodes[att]['pdf'][0]['min'], inst.GA.nodes[att]['pdf'][0]['max'])
                            #if att == 'ambulatory':
                                #print(attributes[att])

                elif 'expression' in inst.GA.nodes[att]:

                    expression = inst.GA.nodes[att]['expression']

                    if att == 'time_stamp':
                        static = np.random.uniform(0, 1)
                        #print(static)
                        if static < inst.GA.nodes[att]['static_probability']:
                            expression = '0'
                    
                    for attx in inst.sorted_attributes:

                        if attx in attributes:
                            expression = re.sub(attx, str(attributes[attx]), expression)                        
                    
                    #print(expression)

                    try:
                        
                        #attributes[att] = eval(expression)
                        #print(expression)
                        attributes[att] = eval_expression(expression)
                        #print(attributes[att])

                        if (inst.GA.nodes[att]['type'] == 'time') or (inst.GA.nodes[att]['type'] == 'integer'):
                            attributes[att] = int(attributes[att])
                        #if att == 'time_stamp':
                        #   print(attributes[att])
                    
                    except (SyntaxError, NameError, ValueError, TypeError):

                        #print(inst.GA.nodes[att]['expression'])
                        
                        expression = re.split(r"[(,) ]", inst.GA.nodes[att]['expression'])
                        
                        if expression[0] == 'dtt':

                            if (expression[1] in attributes) and (expression[2] in attributes):
                                node_drive1 = attributes[expression[1]+'node_drive']
                                node_drive2 = attributes[expression[2]+'node_drive']
                            else:
                                raise ValueError('expression '+inst.GA.nodes[att]['expression']+' not possible to evaluate. check the parameters')

                            attributes[att] = inst.network._return_estimated_travel_time_drive(node_drive1, node_drive2)

                        elif expression[0] == 'stops':

                            stops = []
                            stops_walking_distance = []

                            if (expression[1] in attributes):
                                node_walk = attributes[expression[1]+'node_walk']
                            else:
                                raise ValueError('expression '+inst.GA.nodes[att]['expression']+' not possible to evaluate. check the parameters')
                            

                            for index in inst.network.bus_stations_ids:
                    
                                osmid_possible_stop = int(inst.network.bus_stations.loc[index, 'osmid_walk'])

                                eta_walk = inst.network.get_eta_walk(node_walk, osmid_possible_stop, attributes['walk_speed'])
                                if (eta_walk >= 0) and (eta_walk <= attributes['max_walking']):
                                    #print(eta_walk)
                                    stops.append(index)
                                    stops_walking_distance.append(eta_walk)

                            '''
                            fig, ax = ox.plot_graph(inst.network.G_drive, show=False, close=False, node_color='#000000', node_size=6, bgcolor="#ffffff", edge_color="#999999")

                            node_id = attributes[expression[1]+'node_drive']
                            x = inst.network.G_drive.nodes[node_id]['x']
                            y = inst.network.G_drive.nodes[node_id]['y']
                            ax.scatter(x, y, c='red', s=8, marker=",")

                            for stop in stops:
                                node_id = inst.network.bus_stations.loc[int(stop), 'osmid_drive']
                                x = inst.network.G_drive.nodes[node_id]['x']
                                y = inst.network.G_drive.nodes[node_id]['y']
                                ax.scatter(x, y, c='green', s=8, marker=",")

                            plt.show()
                            plt.close(fig)
                            '''

                            attributes[att] = stops
                            attributes[att+'_walking_distance'] = stops_walking_distance

                        elif expression[0] == 'walk':

                            if (expression[1] in attributes) and (expression[2] in attributes):
                                node_walk1 = attributes[expression[1]+'node_walk']
                                node_walk2 = attributes[expression[2]+'node_walk']
                            else:
                                raise ValueError('expression '+inst.GA.nodes[att]['expression']+' not possible to evaluate. check the parameters')
                            
                            attributes[att] = inst.network.get_eta_walk(node_walk1, node_walk2, attributes['walk_speed'])

                        elif expression[0] == 'dist_drive':

                            
                            if (expression[1] in attributes) and (expression[2] in attributes):
                                node_drive1 = attributes[expression[1]+'node_drive']
                                node_drive2 = attributes[expression[2]+'node_drive']
                            else:
                                raise ValueError('expression '+inst.GA.nodes[att]['expression']+' not possible to evaluate. check the parameters')

                            attributes[att] = inst.network._return_estimated_distance_drive(node_drive1, node_drive2)

                            #print(att)
                            #print(attributes[att])

                        else:
                            raise SyntaxError('expression '+inst.GA.nodes[att]['expression']+' is not supported')
                            
                #check constraints

                if 'constraints' in inst.GA.nodes[att]:

                    not_feasible_attribute = False

                    for constraint in inst.GA.nodes[att]['constraints']:

                        #print(constraint)

                        for attx in inst.sorted_attributes:
                            if attx in attributes:
                                constraint = re.sub(attx, str(attributes[attx]), constraint) 

                        for paramx in inst.parameters:
                            if 'value' in inst.parameters[paramx]:
                                constraint = re.sub(paramx, str(inst.parameters[paramx]['value']), constraint)                       
                        
                        #print(constraint)

                        #check_constraint = eval(constraint)
                        check_constraint = eval_expression(constraint)
                        #print(check_constraint)
                        
                        if not check_constraint:
                            not_feasible_attribute = True
                            exhaustion_iterations += 1
                            if 'expression' in inst.GA.nodes[att]:
                                #this means that another attribute should be remaked because of this, therefore everything is discarded
                                exhaustion_iterations = 9999

                else:

                    not_feasible_attribute = False

                #if att == 'direct_travel_time':
                #    print(attributes[att])
                
                ##if att == 'earliest_departure':
                #    print(attributes[att])

                #if att == 'inbound_outbound':
                #    print(attributes[att])
           
            if not_feasible_attribute:
            
                feasible_data = False
                break

        #print(attributes)
        if feasible_data:
            #print(attributes)

            
            #ps.append(attributes['earliest_departure'])
            
            instance_data.update({i: attributes})

            for att in attributes:

                if att == 'originx':

                    origin_points.append((attributes['originy'], attributes[att]))

                else:

                    if att == 'destinationx':

                        destination_points.append((attributes['destinationy'], attributes[att]))  
            #print(i)
            i += 1     
              

    #count, bins, ignored = plt.hist(ps, 30, density=True)
    #plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')    
    #plt.show()
    all_instance_data.update({'num_data:': len(instance_data),
                          'requests': instance_data
                          })


    final_filename = ''
    #print(inst.instance_filename)
    for p in inst.instance_filename:

        if p in inst.parameters:
            if 'value' in inst.parameters[p]:
                strv = str(inst.parameters[p]['value'])
                strv = strv.replace(" ", "")

                if len(final_filename) > 0:
                    final_filename = final_filename + '_' + strv
                else: final_filename = strv

    #print(final_filename)

    if 'travel_time_matrix' in inst.parameters:
        if inst.parameters['travel_time_matrix']['value'] is True:
            node_list = []
            node_list_seq = []
            lvid = 0

            for location in inst.parameters['travel_time_matrix']['locations']:

                if location == 'bus_stations':

                    for index, row in inst.network.bus_stations.iterrows():

                        node_list.append(row['osmid_drive'])
                        node_list_seq.append(index)
                        lvid = index

                if location in inst.parameters:

                    inst.parameters[location]['list_seq_id'+str(replicate_num)] = []
                    #print(inst.parameters[location]['list_node_drive'+str(replicate_num)])
                    for d in inst.parameters[location]['list_node_drive'+str(replicate_num)]:

                        node_list.append(d)
                        node_list_seq.append(lvid)
                        inst.parameters[location]['list_seq_id'+str(replicate_num)].append(lvid)
                        lvid += 1

                if location in inst.sorted_attributes:

                    for d in instance_data:

                        node_list.append(instance_data[d][location+'node_drive'])
                        instance_data[d][location+'id'] = lvid
                        node_list_seq.append(lvid)
                        lvid += 1

            if replicate_num < 0:
                print('ttm')
                travel_time = inst.network._get_travel_time_matrix("list", node_list=node_list)
                travel_time_json = travel_time.tolist()

                all_instance_data.update({'travel_time_matrix': travel_time_json
                                })
                
                print('leave ttm')          
                if 'graphml' in inst.parameters:
                    if inst.parameters['graphml']['value'] is True:
                        #creates a graph that will serve as the travel time matrix for the given set of requests
                        gtt = nx.DiGraph() 

                        if 'bus_stations' in inst.parameters['travel_time_matrix']['locations']:

                            for index, row in inst.network.bus_stations.iterrows():

                                gtt.add_node(index, type='bus_stations')

                        for param in inst.parameters:

                            if param in inst.parameters['travel_time_matrix']['locations']:

                                for d in inst.parameters[param]['list_seq_id'+str(replicate_num)]:

                                    node = d
                                    gtt.add_node(node, type=param)

                        for att in inst.sorted_attributes:
                            
                            if att in inst.parameters['travel_time_matrix']['locations']:

                                for d in instance_data:

                                    node = instance_data[d][att+'id']
                                    gtt.add_node(node, type=att)

                        for u in node_list_seq:
                            for v in node_list_seq:

                                gtt.add_edge(u, v, travel_time=travel_time_json[u][v])

                        output_name_graphml = os.path.join(inst.save_dir_graphml, inst.output_folder_base + '_' + str(replicate_num) + '.graphml')

                        nx.write_graphml(gtt, output_name_graphml)


                output_file_ttm_csv = os.path.join(inst.save_dir_ttm, 'travel_time_matrix_' + final_filename + '_' + str(replicate_num) + '.csv')
            
                ttml = []
                for u in node_list_seq:
                    d = {}
                    d['osmid_origin'] = u
                    for v in node_list_seq:

                        dist_uv = int(travel_time_json[u][v])
                        sv = str(v)
                        d[sv] = dist_uv
                    ttml.append(d)
                    del d

                ttmpd = pd.DataFrame(ttml)

                ttmpd.to_csv(output_file_ttm_csv)
                ttmpd.set_index(['osmid_origin'], inplace=True)


    save_dir = os.getcwd()+'/'+inst.output_folder_base
    save_dir_images = os.path.join(save_dir, 'images')
    plot_requests(inst.network, save_dir_images, origin_points, destination_points)

    inst.output_file_json = os.path.join(inst.save_dir_json, final_filename + '_' + str(replicate_num) + '.json')

    with open(inst.output_file_json, 'w') as file:
        json.dump(all_instance_data, file, indent=4)
        file.close()     

def _generate_requests_DARP( 
    inst,
    replicate_num,
):
    
    '''
    generate requests for the dial-a-ride problem
    '''

    #random.seed(inst.seed)
    #np.random.seed(inst.seed)

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

    node_list = []
    node_list_seq = []
    inst.depot_nodes_seq = []

    if (len(inst.depot_nodes_drive) > 0):

        for i in range(0, len(inst.depot_nodes_drive)):

            depot_node_drive = inst.depot_nodes_drive[i]
            node_list.append(depot_node_drive)
            node_list_seq.append(i)
            inst.depot_nodes_seq.append(i)

        inst.num_depots = len(inst.depot_nodes_drive)
        
    else:

        for i in range(0, inst.num_depots):

            depot_point = inst.network._get_random_coord(inst.network.polygon)
            depot_point = (depot_point.y, depot_point.x)
            depot_node_drive = ox.get_nearest_node(inst.network.G_drive, depot_point)
            inst.depot_nodes_drive.append(depot_node_drive)

            node_list.append(depot_node_drive)
            node_list_seq.append(i)
            inst.depot_nodes_seq.append(i)

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

                    #print(sd.origin_zones)
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
                

                #compute estimated arrival time
                estimated_travel_time = inst.network._return_estimated_travel_time_drive(origin_node_drive, destination_node_drive)
                if estimated_travel_time >= 0:
                    #arr_time = (dep_time) + (inst.delay_vehicle_factor * estimated_travel_time) 
                    flex_time = inst.delay_vehicle_factor * estimated_travel_time
                    arr_time = dep_time + estimated_travel_time + flex_time
                else:
                    unfeasible_request = True


                origin_node_seq = 0
                destination_node_seq = 0

                if origin_node_drive not in node_list:

                    node_list.append(origin_node_drive)
                    node_list_seq.append(len(node_list)-1)
                    origin_node_seq = len(node_list)-1

                else:

                    for si in range(len(node_list)):
                        if origin_node_drive == node_list[si]:
                            origin_node_seq = si
                            break

                if destination_node_drive not in node_list:

                    node_list.append(destination_node_drive)
                    node_list_seq.append(len(node_list)-1)
                    destination_node_seq = len(node_list)-1

                else:

                    for si in range(len(node_list)):
                        if destination_node_drive == node_list[si]:
                            destination_node_seq = si
                            break
                
                if  not unfeasible_request:   
                    #prints in the json file if the request is 'viable'
                    
                    request_data.update({'originx': origin_point[1]})
                    request_data.update({'originy': origin_point[0]})
                    request_data.update({'origin_node': int(origin_node_seq)})

                    origin_points.append(origin_point)

                    request_data.update({'destinationx': destination_point[1]})
                    request_data.update({'destinationy': destination_point[0]})
                    request_data.update({'destination_node': int(destination_node_seq)})

                    destination_points.append(destination_point)

                    request_data.update({'time_stamp': int(request_time_stamp)})
                    
                    #earliest departure time
                    request_data.update({'dep_time': int(dep_time)})

                    #latest departure time
                    l_dep_time = random.randint(int(dep_time), int(dep_time) + inst.g)
                    request_data.update({'lat_dep_time': int(l_dep_time)})
                    
                    #earliest arrival time
                    e_arr_time = random.randint(int(arr_time) - inst.g, int(arr_time))
                    request_data.update({'ear_arr_time': int(e_arr_time)})

                    #arrival time
                    request_data.update({'arr_time': int(arr_time)})

                    #compute if request need some vehicle requirement, e.g., wheelchair, ambulatory passenger
                    #0-> no requirement
                    #1-> wheelchair
                    #2-> ambulatory
                    min_req = 0
                    max_req = 0
                    vehicle_requirement = -1
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
                        request_data_return.update({'origin_node': int(destination_node_seq)})

                        #coordinate destination
                        request_data_return.update({'destinationx': origin_point[1]})
                        request_data_return.update({'destinationy': origin_point[0]})
                        request_data_return.update({'destination_node': int(origin_node_seq)})

                        #timestamp -> time the request was made
                        request_data_return.update({'time_stamp': int(request_time_stamp)})
                        
                        #departure time for the return
                        dep_time_return = random.randint(int(arr_time), inst.max_early_departure)
                        request_data_return.update({'dep_time': int(dep_time_return)})

                        #latest departure time
                        l_dep_time_return = random.randint(int(dep_time_return), int(dep_time_return) + inst.g)
                        request_data_return.update({'lat_dep_time': int(l_dep_time)})
                        
                        #arrival time
                        estimated_travel_time = inst.network._return_estimated_travel_time_drive(destination_node_drive, origin_node_drive)
                        #arr_time_return = (dep_time_return) + (inst.delay_vehicle_factor * estimated_travel_time)
                        flex_time = inst.delay_vehicle_factor * estimated_travel_time
                        arr_time_return = dep_time_return + estimated_travel_time + flex_time

                        #earliest arrival time
                        e_arr_time_return = random.randint(int(arr_time_return) - inst.g, int(arr_time_return))
                        request_data_return.update({'ear_arr_time': int(e_arr_time_return)})

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
    inst.network.node_list_darp_seq = node_list_seq

    travel_time = inst.network._get_travel_time_matrix("list", node_list=node_list)

    travel_time_json = travel_time.tolist()
    #creates a graph that will serve as the travel time matrix for the given set of requests
    gtt = nx.DiGraph() 

    for depot in inst.depot_nodes_seq:

        gtt.add_node(depot, type="depot")

    for node in range(len(inst.depot_nodes_seq), len(node_list)):

        gtt.add_node(node, type="node")

    for u in node_list_seq:
        for v in node_list_seq:

            gtt.add_edge(u, v, travel_time=travel_time_json[u][v])

    output_name_graphml = os.path.join(inst.save_dir_graphml, inst.output_folder_base + '_' + str(replicate_num) + '.graphml')
    #output_name_graphml = output_name_graphml.replace(" ", "")

    nx.write_graphml(gtt, output_name_graphml)

    instance_data.update({'num_requests:': len(all_requests),
                          'requests': all_requests,
                          'num_depots': inst.num_depots,
                          'depots': inst.depot_nodes_seq,
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

    random.seed(inst.seed)
    np.random.seed(inst.seed)

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

            print(i)
            
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
                
                max_walking_user = random.randint(int(inst.lb_max_walking), int(inst.ub_max_walking))

                #def generate_feasible_request
                while unfeasible_request:

                    sdlist = [0,1,2,3]
                    sdid = random.choices(sdlist, weights=weightssd, k=1)
                    #print(sdid)
                    sd = inst.spatial_distribution[sdid[0]]

                    #generate coordinates for origin
                    if sd.num_origins == -1:

                        origin_point = inst.network._get_random_coord(inst.network.polygon)
                        origin_point = (origin_point.y, origin_point.x)
    
                    else:

                        #print(sd.origin_zones)
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
                        #print(max_eta_bus)
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
                    request_data.update({'stops_orgn': stops_origin})
                    request_data.update({'stops_orgn_walking_distance': stops_origin_walking_distance})

                    request_data.update({'num_stops_destination': len(stops_destination)})
                    request_data.update({'stops_dest': stops_destination})
                    request_data.update({'stops_dest_walking_distance': stops_destination_walking_distance})

                    #timestamp -> time the request was made
                    request_data.update({'time_stamp': int(request_time_stamp)})
                    
                    #departure time
                    request_data.update({'earliest_departure': int(dep_time)})

                    #arrival time
                    request_data.update({'latest_arrival': int(arr_time)})

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
                        request_data_return.update({'stops_orgn': stops_destination})
                        request_data_return.update({'stops_orgn_walking_distance': stops_destination_walking_distance})

                        request_data_return.update({'num_stops_destination': len(stops_origin)})
                        request_data_return.update({'stops_dest': stops_origin})
                        request_data_return.update({'stops_dest_walking_distance': stops_origin_walking_distance})

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

    travel_time = inst.network._get_travel_time_matrix("bus")

    travel_time_json = travel_time.tolist()

    gtt = nx.DiGraph() 

    for index in inst.network.bus_stations.index:

        gtt.add_node(index, type="busstation")

    for u in inst.network.bus_stations.index:
        for v in inst.network.bus_stations.index:

            gtt.add_edge(u, v, travel_time=travel_time_json[u][v])

    output_name_graphml = os.path.join(inst.save_dir_graphml, inst.output_folder_base + '_' + str(replicate_num) + '.graphml')
    #output_name_graphml = output_name_graphml.replace(" ", "")

    nx.write_graphml(gtt, output_name_graphml)

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

    random.seed(inst.seed)
    np.random.seed(inst.seed)
    
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

    node_list = []
    node_list_seq = []
    lid = inst.network.bus_stations.last_valid_index() + 1
    inst.school_ids_seq = []
    inst.school_ids_osm = []

    if len(inst.school_ids) == 0:

        while len(inst.school_ids) < inst.num_schools:

            random_school_id = np.random.uniform(0, len(inst.network.schools), 1)
            random_school_id = int(random_school_id)

            if random_school_id not in inst.school_ids:
                inst.school_ids.append(random_school_id)

    for i in range(inst.num_schools):

        inst.school_ids_seq.append(int(lid))
        inst.school_ids_osm.append(inst.network.schools.loc[inst.school_ids[i], 'osmid_drive'])
        lid += 1

    node_list = inst.network.bus_stations['osmid_drive'].tolist() + inst.school_ids_osm
    node_list_seq = inst.network.bus_stations.index.values.tolist() + inst.school_ids_seq

    gtt = nx.DiGraph() 

    for index in inst.network.bus_stations.index:

        gtt.add_node(index, type="busstation")

    for school in inst.school_ids_seq:

        gtt.add_node(school, type="school")

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
                
                max_walking_user = random.randint(int(inst.lb_max_walking), int(inst.ub_max_walking))

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

                    #destination is the school (randomly chosen between the number of schools)
                    random_school_id = np.random.randint(0, inst.num_schools, 1)
                    
                    random_school_id = int(random_school_id)
                    #print(random_school_id)
                    destination_point = (inst.network.schools.loc[inst.school_ids[random_school_id], 'lat'], inst.network.schools.loc[inst.school_ids[random_school_id], 'lon'])

                    origin_node_walk = ox.get_nearest_node(inst.network.G_walk, origin_point)
                    destination_node_walk = inst.network.schools.loc[inst.school_ids[random_school_id], 'osmid_walk']
                    time_walking = inst.network.get_eta_walk(int(origin_node_walk), int(destination_node_walk), request_walk_speed)

                    #if random_school_id == 2:
                        #print(time_walking)
                    if time_walking > max_walking_user:
                        unfeasible_request = False
                    
                if not unfeasible_request:

                    origin_node_drive = ox.get_nearest_node(inst.network.G_drive, origin_point)
                    destination_node_drive = inst.network.schools.loc[inst.school_ids[random_school_id], 'osmid_drive']

                    origin_node_seq = 0
                    destination_node_seq = inst.school_ids_seq[random_school_id]

                    if origin_node_drive not in node_list:

                        node_list.append(origin_node_drive)
                        origin_node_seq = lid
                        node_list_seq.append(origin_node_seq)
                        gtt.add_node(origin_node_seq, type="home")
                        lid += 1

                    else:

                        for si in range(len(node_list)):
                            if origin_node_drive == node_list[si]:
                                origin_node_seq = si
                                break
                    
                    stops_origin = []
                    stops_origin_walking_distance = []
                        
                    #calculates the stations which are close enough to the origin and destination of the request
                    for index in inst.network.bus_stations_ids:
                    
                        osmid_possible_stop = int(inst.network.bus_stations.loc[index, 'osmid_walk'])

                        eta_walk_origin = inst.network.get_eta_walk(origin_node_walk, osmid_possible_stop, request_walk_speed)
                        
                        if eta_walk_origin >= 0 and eta_walk_origin <= max_walking_user:
                            stops_origin.append(index)
                            stops_origin_walking_distance.append(eta_walk_origin)
                    
                    #if len(stops_origin) > 0:
                    #    unfeasible_request = False
                   
                    #outputs in the json file if the request is 'feasible'

                    request_data.update({'originx': origin_point[1]})
                    request_data.update({'originy': origin_point[0]})
                    request_data.update({'origin_node': int(origin_node_seq)})

                    origin_points.append(origin_point)

                    request_data.update({'destinationx': destination_point[1]})
                    request_data.update({'destinationy': destination_point[0]})
                    request_data.update({'destination_node': int(destination_node_seq)})
                
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


    
    inst.network.node_list_seq_school = node_list_seq
    travel_time = inst.network._get_travel_time_matrix("list", node_list=node_list)
    #travel_time_to_school = inst.network._get_travel_time_from_stops_to_school(inst.school_id)

    travel_time_json = travel_time.tolist()

    for u in node_list_seq:
        for v in node_list_seq:

            gtt.add_edge(u, v, travel_time=travel_time_json[u][v])

    output_name_graphml = os.path.join(inst.save_dir_graphml, inst.output_folder_base + '_' + str(replicate_num) + '.graphml')
    #output_name_graphml = output_name_graphml.replace(" ", "")

    nx.write_graphml(gtt, output_name_graphml)

    instance_data.update({'num_requests:': len(all_requests),
                          'requests': all_requests,
                          'num_schools': int(inst.num_schools),
                          'schools': inst.school_ids_seq,
                          'num_nodes': int(len(node_list_seq)),
                          'travel_time_matrix': travel_time_json,
                          #'travel_time_to_school': travel_time_to_school
                         }
                          )

    save_dir = os.getcwd()+'/'+inst.output_folder_base
    save_dir_images = os.path.join(save_dir, 'images')
    plot_requests(inst.network, save_dir_images, origin_points, destination_points)

    with open(inst.output_file_json, 'w') as file:
        json.dump(instance_data, file, indent=4)
        file.close()        
