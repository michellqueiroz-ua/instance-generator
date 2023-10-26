from fixed_lines import _check_subway_routes_serve_passenger
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
import shapely
from shapely.geometry import Point
from scipy.stats import cauchy
from scipy.stats import expon
from scipy.stats import gamma
from scipy.stats import gilbrat
from scipy.stats import lognorm
from scipy.stats import norm
from scipy.stats import powerlaw
from scipy.stats import poisson
from scipy.stats import uniform
from scipy.stats import wald
from multiprocessing import cpu_count
import gc
from datetime import datetime
from dynamism import dynamism

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


@ray.remote
def _generate_single_data(GA, network, sorted_attributes, parameters, reqid, method_poi, distancex, pu, do, pdfunc, loc, scale, aux, time_stamps, replicate_num, reaction_times, num_requests, zonescsv):
    
    attributes = {}
    feasible_data = False
    first_time = True
    print(reqid)
    loc_attempts = 0
    att_attempts = 0
    while not feasible_data:
        
        attributes = {}
        feasible_data = True
        if method_poi:
            heads_or_tails = random.randint(0,1)
            if heads_or_tails == 1:
                
                for sa in range(len(sorted_attributes)):
                    if sorted_attributes[sa] == pu:
                        sorted_attributes[sa] = do
                    else:
                        if sorted_attributes[sa] == do:
                            sorted_attributes[sa] = pu

        
        for att in sorted_attributes:

            not_feasible_attribute = True
            exhaustion_iterations = 0

            while (not_feasible_attribute) and (exhaustion_iterations < 100):

                seed_attribute = ((reqid+1)*111*(replicate_num+1))+att_attempts+num_requests
                att_attempts += 1
                if GA.nodes[att]['type'] == 'location':

                    seed_location = ((reqid+1)*1111*(replicate_num+1))+loc_attempts+num_requests
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
                                        
                                        if not first_time:
                                            random.seed(seed_location)
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
                                                mux = (loc + scale) / 2
                                                a_scale = mux / aux
                                                radius = gamma.rvs(a=aux, scale=a_scale, size=1)

                                            if pdfunc == 'gilbrat':
                                                gilbrat.random_state = np.random.RandomState(seed=rseed)
                                                radius = gilbrat.rvs(loc=loc, scale=scale, size=1)

                                            if pdfunc == 'lognorm':
                                                lognorm.random_state = np.random.RandomState(seed=rseed)
                                                radius = lognorm.rvs(loc=loc, scale=scale, size=1)

                                            if pdfunc == 'normal':
                                                norm.random_state = np.random.RandomState(seed=rseed)
                                                radius = norm.rvs(loc=loc, scale=scale, size=1)

                                            if pdfunc == 'powerlaw':
                                                powerlaw.random_state = np.random.RandomState(seed=rseed)
                                                radius = powerlaw.rvs(a=aux, loc=loc, scale=scale, size=1)

                                            if pdfunc == 'uniform':
                                                uniform.random_state = np.random.RandomState(seed=rseed)
                                                radius = uniform.rvs(loc=loc, scale=scale, size=1)

                                            if pdfunc == 'wald':
                                                wald.random_state = np.random.RandomState(seed=rseed)
                                                radius = wald.rvs(loc=loc, scale=scale, size=1)
                                            
                                            if pdfunc == 'poisson':
                                                poisson.random_state = np.random.RandomState(seed=rseed)
                                                radius = poisson.rvs(a=aux, size=1)

                                        else:
                                            radius = distancex
                                            first_time = False
                                        
                                        att_start = pu
                                        try:
                                            lat = attributes[att_start+'y']
                                            lon = attributes[att_start+'x']
                                            point = network._get_random_coord_radius(lat, lon, radius, network.polygon, seed_location)
                                        except KeyError:
                                            point = Point(-1, -1)

                                    elif type_coord == pu:
                                        
                                        probabilities = network.zones['density_pois'].tolist()
                                        random_zone_id = np.random.choice(a=zones, size=1, p=probabilities)

                                        random_zone_id = random_zone_id[0]
                                        polygon_zone = network.zones.loc[random_zone_id]['polygon']
                                        point = network._get_random_coord(polygon_zone, seed_location)
                                    
                                else:

                                    if type_coord == pu:
                                        
                                        if not first_time:
                                            random.seed(seed_location)
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
                                                mux = (loc + scale) / 2
                                                a_scale = mux / aux
                                                radius = gamma.rvs(a=aux, scale=a_scale, size=1)

                                            if pdfunc == 'gilbrat':
                                                gilbrat.random_state = np.random.RandomState(seed=rseed)
                                                radius = gilbrat.rvs(loc=loc, scale=scale, size=1)

                                            if pdfunc == 'lognorm':
                                                lognorm.random_state = np.random.RandomState(seed=rseed)
                                                radius = lognorm.rvs(loc=loc, scale=scale, size=1)

                                            if pdfunc == 'normal':
                                                norm.random_state = np.random.RandomState(seed=rseed)
                                                radius = norm.rvs(loc=loc, scale=scale, size=1)

                                            if pdfunc == 'powerlaw':
                                                powerlaw.random_state = np.random.RandomState(seed=rseed)
                                                radius = powerlaw.rvs(a=aux, loc=loc, scale=scale, size=1)

                                            if pdfunc == 'uniform':
                                                uniform.random_state = np.random.RandomState(seed=rseed)
                                                radius = uniform.rvs(loc=loc, scale=scale, size=1)

                                            if pdfunc == 'wald':
                                                wald.random_state = np.random.RandomState(seed=rseed)
                                                radius = wald.rvs(loc=loc, scale=scale, size=1)

                                            if pdfunc == 'poisson':
                                                poisson.random_state = np.random.RandomState(seed=rseed)
                                                radius = poisson.rvs(a=aux, size=1)
                                            
                                        else:
                                            radius = distancex
                                            first_time = False

                                        att_start = do
                                        try:
                                            lat = attributes[att_start+'y']
                                            lon = attributes[att_start+'x']
                                            point = network._get_random_coord_radius(lat, lon, radius, network.polygon, seed_location)
                                        except KeyError:
                                            point = Point(-1, -1)

                                    elif type_coord == do:
                                        
                                        probabilities = network.zones['density_pois'].tolist()
                                        random_zone_id = np.random.choice(a=zones, size=1, p=probabilities)

                                        random_zone_id = random_zone_id[0]
                                        polygon_zone = network.zones.loc[random_zone_id]['polygon']
                                        point = network._get_random_coord(polygon_zone, seed_location)

                            else:

                                if (att != 'destination') and (att != 'origin') and (parameters['set_geographic_dispersion']['value'] == True):
                                    raise ValueError('Location attributes must be named origin and destination when setting Geographic Dispersion')
                                    
                                if (att == 'destination') and (parameters['set_geographic_dispersion']['value'] == True):
                                    
                                    random.seed(seed_location)
                                    rseed = np.random.uniform(0,1)
                                    rseed = rseed * 1000
                                    rseed = int(rseed)
                                    uniform.random_state = np.random.RandomState(seed=rseed)

                                    if network.vehicle_speed is not None:

                                        if 'min_dtt' in parameters:

                                            loc1 = parameters['min_dtt']['value']*network.vehicle_speed

                                        else:
                                            raise ValueError('Parameter min_dtt must be specified for Geographic dispersion')

                                        if 'max_dtt' in parameters:

                                            scale1 = parameters['max_dtt']['value']*network.vehicle_speed - loc1

                                        else:
                                            raise ValueError('Parameter max_dtt must be specified for Geographic dispersion')

                                        print(loc1, scale1)
                                        radius = uniform.rvs(loc=loc1, scale=scale1, size=1)

                                    else:
                                        raise ValueError('Geographic dispersion is only supported with fixed speed values. Set with set_fixed_speed')

                                    
                                    point = network._get_random_coord_radius(attributes['originy'], attributes['originx'], radius, network.polygon, seed_location)

                                else:
                                    point = network._get_random_coord(network.polygon, seed_location)
                            
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
                                    point = network._get_random_coord_circle(R, clat, clon, seed_location)
                                    
                            except TypeError:
                            
                                point = network._get_random_coord(polygon_zone, seed_location)                                

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
                    

                    #point = locations[(reqid*num_excl_loc)+loc_attempts]
                    #print(reqid, (reqid*num_excl_loc)+loc_attempts)

                    attributes[att] = point
                    attributes[att+'x'] = point[1]
                    attributes[att+'y'] = point[0]
                    
                    if point[1] != -1:
                        not_feasible_attribute = False
                        
                        node_drive = ox.nearest_nodes(network.G_drive, point[1], point[0])
                        node_walk = ox.nearest_nodes(network.G_walk, point[1], point[0])

                        attributes[att+'node_drive'] = int(node_drive)
                        attributes[att+'node_walk'] = int(node_walk)

                        attributes[att+'zone'] = int(random_zone_id)

                    else:
                        not_feasible_attribute = True
                    
                    loc_attempts += 1
                
                if att == pu:         
                    if not_feasible_attribute:
                
                        feasible_data = False
                        break 

                if att == do:         
                    if not_feasible_attribute:
                
                        feasible_data = False
                        break 

                #remove this later
                
                if att == 'destination':         
                    if not_feasible_attribute:
                
                        feasible_data = False
                        break 
                
                
                if 'pdf' in GA.nodes[att]:

                    if att != 'reaction_time':

                        random.seed(seed_attribute)
                        rseed = np.random.uniform(0,1)
                        rseed = rseed * 1000
                        rseed = int(rseed)

                        if GA.nodes[att]['pdf'][0]['type'] == 'normal':

                            if ((att == 'time_stamp') and (len(time_stamps) > 0)):

                                    attributes[att] = int(time_stamps[reqid])

                            else:
                                norm.random_state = np.random.RandomState(seed=rseed)
                                attributes[att] = norm.rvs(loc=GA.nodes[att]['pdf'][0]['loc'], scale=GA.nodes[att]['pdf'][0]['scale'], size=1)
                                
                                if (GA.nodes[att]['type'] == 'time') or (GA.nodes[att]['type'] == 'integer'):
                                    attributes[att] = int(attributes[att])

                        
                        if GA.nodes[att]['pdf'][0]['type'] == 'uniform':


                            if 'weights' in GA.nodes[att]:

                                attributes[att] = random.choices(GA.nodes[att]['all_values'], weights=GA.nodes[att]['weights'], k=1)
                                attributes[att] = attributes[att][0]

                            else:

                                if ((att == 'time_stamp') and (len(time_stamps) > 0)):

                                    attributes[att] = int(time_stamps[reqid])

                                else:
                                    if (GA.nodes[att]['type'] == 'time') or (GA.nodes[att]['type'] == 'integer'):
                                        attributes[att] = np.random.randint(GA.nodes[att]['pdf'][0]['loc'], GA.nodes[att]['pdf'][0]['scale'] + GA.nodes[att]['pdf'][0]['loc'] + 1)

                                    else:
                                        uniform.random_state = np.random.RandomState(seed=rseed)
                                        attributes[att] = uniform.rvs(loc=GA.nodes[att]['pdf'][0]['loc'], scale=GA.nodes[att]['pdf'][0]['scale'], size=1)
                                        attributes[att] = attributes[att][0]

                        if GA.nodes[att]['pdf'][0]['type'] == 'cauchy':

                            cauchy.random_state = np.random.RandomState(seed=rseed)
                            attributes[att] = cauchy.rvs(loc=GA.nodes[att]['pdf'][0]['loc'], scale=GA.nodes[att]['pdf'][0]['scale'], size=1)

                            if (GA.nodes[att]['type'] == 'time') or (GA.nodes[att]['type'] == 'integer'):
                                attributes[att] = int(attributes[att])

                        if GA.nodes[att]['pdf'][0]['type'] == 'expon':

                            expon.random_state = np.random.RandomState(seed=rseed)
                            attributes[att] = expon.rvs(loc=GA.nodes[att]['pdf'][0]['loc'], scale=GA.nodes[att]['pdf'][0]['scale'], size=1)

                            if (GA.nodes[att]['type'] == 'time') or (GA.nodes[att]['type'] == 'integer'):
                                attributes[att] = int(attributes[att])

                        if GA.nodes[att]['pdf'][0]['type'] == 'gamma':

                            gamma.random_state = np.random.RandomState(seed=rseed)
                            mux = (GA.nodes[att]['pdf'][0]['loc'] + GA.nodes[att]['pdf'][0]['scale']) / 2
                            a_scale = mux / GA.nodes[att]['pdf'][0]['aux']
                            attributes[att] = gamma.rvs(a=GA.nodes[att]['pdf'][0]['aux'], scale=a_scale, size=1)

                            if (GA.nodes[att]['type'] == 'time') or (GA.nodes[att]['type'] == 'integer'):
                                attributes[att] = int(attributes[att])

                        if GA.nodes[att]['pdf'][0]['type'] == 'gilbrat':

                            gilbrat.random_state = np.random.RandomState(seed=rseed)
                            attributes[att] = gilbrat.rvs(loc=GA.nodes[att]['pdf'][0]['loc'], scale=GA.nodes[att]['pdf'][0]['scale'], size=1)

                            if (GA.nodes[att]['type'] == 'time') or (GA.nodes[att]['type'] == 'integer'):
                                attributes[att] = int(attributes[att])

                        if GA.nodes[att]['pdf'][0]['type'] == 'lognorm':

                            lognorm.random_state = np.random.RandomState(seed=rseed)
                            attributes[att] = lognorm.rvs(loc=GA.nodes[att]['pdf'][0]['loc'], scale=GA.nodes[att]['pdf'][0]['scale'], size=1)

                            if (GA.nodes[att]['type'] == 'time') or (GA.nodes[att]['type'] == 'integer'):
                                attributes[att] = int(attributes[att])

                        if GA.nodes[att]['pdf'][0]['type'] == 'powerlaw':

                            powerlaw.random_state = np.random.RandomState(seed=rseed)
                            attributes[att] = powerlaw.rvs(a=GA.nodes[att]['pdf'][0]['aux'], loc=GA.nodes[att]['pdf'][0]['loc'], scale=GA.nodes[att]['pdf'][0]['scale'], size=1)

                            if (GA.nodes[att]['type'] == 'time') or (GA.nodes[att]['type'] == 'integer'):
                                attributes[att] = int(attributes[att])

                        if GA.nodes[att]['pdf'][0]['type'] == 'wald':

                            wald.random_state = np.random.RandomState(seed=rseed)
                            attributes[att] = wald.rvs(loc=GA.nodes[att]['pdf'][0]['loc'], scale=GA.nodes[att]['pdf'][0]['scale'], size=1)

                            if (GA.nodes[att]['type'] == 'time') or (GA.nodes[att]['type'] == 'integer'):
                                attributes[att] = int(attributes[att])

                        if GA.nodes[att]['pdf'][0]['type'] == 'poisson':

                            poisson.random_state = np.random.RandomState(seed=rseed)
                            attributes[att] = poisson.rvs(mu=GA.nodes[att]['pdf'][0]['a'], size=1)
                            
                            if (GA.nodes[att]['type'] == 'time') or (GA.nodes[att]['type'] == 'integer'):
                                attributes[att] = int(attributes[att])
                    else:

                        attributes[att] = reaction_times[reqid]

                    
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
                    
                    try:
                        
                        attributes[att] = eval_expression(expression)
                        
                        if (GA.nodes[att]['type'] == 'time') or (GA.nodes[att]['type'] == 'integer'):
                            attributes[att] = int(attributes[att])
                        
                    except (SyntaxError, NameError, ValueError, TypeError):

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
                            
                            if att == 'stops_orgn':
                                    pnt = Point(attributes['originx'], attributes['originy'])
                            else:
                                if att == 'stops_dest':
                                    pnt = Point(attributes['destinationx'], attributes['destinationy'])
                            index_inside_zone = -1
                            for indexz, zone in zonescsv.iterrows():

                                polygonz = shapely.wkt.loads(zone['polygon'])
                                if polygonz.contains(pnt):
                                    index_inside_zone = indexz
                                    break

                            all_zones = []
                            all_zones.append(index_inside_zone)
                            
                            all_zones.append(index_inside_zone-1)
                            all_zones.append(index_inside_zone+1)
                            for r in range(5):
                                all_zones.append(index_inside_zone-20+r)
                            for r in range(5):
                                all_zones.append(index_inside_zone-20-r)
                            for r in range(5):
                                all_zones.append(index_inside_zone+20-r)
                            for r in range(5):
                                all_zones.append(index_inside_zone+20+r)
                            
                            print('zones')
                            print(all_zones)
                            for z in all_zones:

                                try:
                                    stationsz = zonescsv.loc[z]['stations']
                                    stationsz = json.loads(stationsz)
                                    print('stations')
                                    print(stationsz) 

                                    for index in stationsz:

                                        osmid_possible_stop = int(network.bus_stations.loc[index, 'osmid_walk'])

                                        eta_walk = network.get_eta_walk(node_walk, osmid_possible_stop, attributes['walk_speed'])
                                        if (eta_walk >= 0) and (eta_walk <= attributes['max_walking']):
                                            stops.append(index)
                                            stops_walking_distance.append(eta_walk)
                                except KeyError:
                                    pass
                            
                            
                            '''
                            for index in network.bus_stations_ids:
                    
                                osmid_possible_stop = int(network.bus_stations.loc[index, 'osmid_walk'])

                                eta_walk = network.get_eta_walk(node_walk, osmid_possible_stop, attributes['walk_speed'])
                                if (eta_walk >= 0) and (eta_walk <= attributes['max_walking']):
                                    stops.append(index)
                                    stops_walking_distance.append(eta_walk)
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

                        else:
                            raise SyntaxError('expression '+GA.nodes[att]['expression']+' is not supported')
                

                #check constraints
                if 'constraints' in GA.nodes[att]:

                    not_feasible_attribute = False

                    for constraint in GA.nodes[att]['constraints']:

                        for attx in sorted_attributes:
                            if attx in attributes:
                                constraint = re.sub(attx, str(attributes[attx]), constraint) 

                        for paramx in parameters:
                            if 'value' in parameters[paramx]:
                                constraint = re.sub(paramx, str(parameters[paramx]['value']), constraint)                       
                        
                        check_constraint = eval_expression(constraint)
                        
                        if not check_constraint:
                            print('hier')
                            print(constraint)
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
            attributes['seed_location'] = seed_location
            attributes['reqid'] = reqid
            return attributes

    
    return attributes

def _generate_requests( 
    inst,
    replicate_num,
    base_save_folder_name,
    inst_directory,
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

    #print('suuum1 ', inst.network.zones['density_pois'].sum())

    #if replicate_num == 0:
    #    inst.network.zones['density_pop'] = inst.network.zones['density_pop']/100

    #print('suuum2 ', inst.network.zones['density_pop'].sum())

    distances = []
    
    method_poi = False
    pu = None 
    do = None 
    pdfunc = None
    aux = None 
    loc = None 
    scale = None 
    np.random.seed(inst.seed*(replicate_num+1))
    random.seed(inst.seed*(replicate_num+1))

    zonescsv = pd.read_csv(inst.save_dir+'/csv/'+inst.output_folder_base+'.zones.csv')

    num_requests = inst.parameters['requests']['value']

    if 'method_pois' in inst.parameters:

        method_poi = True 
        pu = inst.parameters['method_pois']['value']['locations'][0]
        do = inst.parameters['method_pois']['value']['locations'][1]
        pdfunc = inst.parameters['method_pois']['value']['pdf']['type']

        aux = -1
        loc = inst.parameters['method_pois']['value']['pdf']['loc']
        scale = inst.parameters['method_pois']['value']['pdf']['scale']

        if inst.parameters['method_pois']['value']['pdf']['type'] == 'cauchy':
            distancesx = cauchy.rvs(loc=inst.parameters['method_pois']['value']['pdf']['loc'], scale=inst.parameters['method_pois']['value']['pdf']['scale'], size=inst.parameters['requests']['value']+20000)

        if inst.parameters['method_pois']['value']['pdf']['type'] == 'expon':
            distancesx = expon.rvs(loc=inst.parameters['method_pois']['value']['pdf']['loc'], scale=inst.parameters['method_pois']['value']['pdf']['scale'], size=inst.parameters['requests']['value']+20000)

        if inst.parameters['method_pois']['value']['pdf']['type'] == 'gamma':
            mux = (inst.parameters['method_pois']['value']['pdf']['loc'] + inst.parameters['method_pois']['value']['pdf']['scale']) / 2
            a_scale = mux / inst.parameters['method_pois']['value']['pdf']['aux']
            distancesx = gamma.rvs(a=inst.parameters['method_pois']['value']['pdf']['aux'], scale=a_scale, size=inst.parameters['requests']['value']+20000)
            aux = inst.parameters['method_pois']['value']['pdf']['aux']

        if inst.parameters['method_pois']['value']['pdf']['type'] == 'gilbrat':
            distancesx = gilbrat.rvs(loc=inst.parameters['method_pois']['value']['pdf']['loc'], scale=inst.parameters['method_pois']['value']['pdf']['scale'], size=inst.parameters['requests']['value']+20000)

        if inst.parameters['method_pois']['value']['pdf']['type'] == 'lognorm':
            distancesx = lognorm.rvs(loc=inst.parameters['method_pois']['value']['pdf']['loc'], scale=inst.parameters['method_pois']['value']['pdf']['scale'], size=inst.parameters['requests']['value']+20000)

        if inst.parameters['method_pois']['value']['pdf']['type'] == 'normal':
            distancesx = norm.rvs(loc=inst.parameters['method_pois']['value']['pdf']['loc'], scale=inst.parameters['method_pois']['value']['pdf']['scale'], size=inst.parameters['requests']['value']+20000)

        if inst.parameters['method_pois']['value']['pdf']['type'] == 'powerlaw':
            distancesx = powerlaw.rvs(a=inst.parameters['method_pois']['value']['pdf']['aux'], loc=inst.parameters['method_pois']['value']['pdf']['loc'], scale=inst.parameters['method_pois']['value']['pdf']['scale'], size=inst.parameters['requests']['value']+20000)
            aux = inst.parameters['method_pois']['value']['pdf']['aux']

        if inst.parameters['method_pois']['value']['pdf']['type'] == 'uniform':
            distancesx = uniform.rvs(loc=inst.parameters['method_pois']['value']['pdf']['loc'], scale=inst.parameters['method_pois']['value']['pdf']['scale'], size=inst.parameters['requests']['value']+20000)

        if inst.parameters['method_pois']['value']['pdf']['type'] == 'wald':
            distancesx = wald.rvs(loc=inst.parameters['method_pois']['value']['pdf']['loc'], scale=inst.parameters['method_pois']['value']['pdf']['scale'], size=inst.parameters['requests']['value']+20000)

        if inst.parameters['method_pois']['value']['pdf']['type'] == 'poisson':
            aux = inst.parameters['method_pois']['value']['pdf']['a']
            distancesx = poisson.rvs(a=inst.parameters['method_pois']['value']['pdf']['aux'], size=inst.parameters['requests']['value']+20000)

        distances = [x for x in distancesx if x >= 500]
        distances = [x for x in distancesx if x < 32000]
        
        #print('len distances: ', len(distances))
    else:

        distances = [0] * num_requests
    param_num = 1
    for param in inst.parameters:

        
        #generate random locations according to given input
        if inst.parameters[param]['type'] == 'array_locations':

            if inst.parameters[param]['locs'] == 'random':

                inst.parameters[param]['list'+str(replicate_num)] = []
                for elem in inst.parameters[param]['list']:
                    inst.parameters[param]['list'+str(replicate_num)].append(elem)

                inst.parameters[param]['list_node_drive'+str(replicate_num)] = []    
                for elem in inst.parameters[param]['list_node_drive']:
                    inst.parameters[param]['list_node_drive'+str(replicate_num)].append(elem)

                while len(inst.parameters[param]['list'+str(replicate_num)]) < inst.parameters[param]['size']:

                    seed_location = (inst.seed*222)+param_num
                    param_num += 1
                    point = inst.network._get_random_coord(inst.network.polygon, seed_location)
                    point = (point.y, point.x)
                    node_drive = ox.nearest_nodes(inst.network.G_drive, point[1], point[0])
                    if node_drive not in inst.parameters[param]['list_node_drive'+str(replicate_num)]:
                        inst.parameters[param]['list'+str(replicate_num)].append("random_loc"+len(inst.parameters[param]['list'+str(replicate_num)]))
                        inst.parameters[param]['list_node_drive'+str(replicate_num)].append(node_drive)


            #generate random schools according to given input
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

        #generate random zones according to given input                
        if inst.parameters[param]['type'] == 'array_zones':

            inst.parameters[param]['zones'+str(replicate_num)] = []
            for elem in inst.parameters[param]['zones']:
                inst.parameters[param]['zones'+str(replicate_num)].append(elem)

            while len(inst.parameters[param]['zones'+str(replicate_num)]) < inst.parameters[param]['size']: 
            
                random_zone_id = np.random.randint(0, len(inst.network.zones))
                random_zone_id = int(random_zone_id)

                if random_zone_id not in inst.parameters[param]['zones'+str(replicate_num)]:
                    inst.parameters[param]['zones'+str(replicate_num)].append(random_zone_id)               


    #to generate a specific level of dynamism
    time_stamps = []
    if 'time_stamp' in inst.GA.nodes:
        if 'dynamism' in inst.GA.nodes['time_stamp']:

            #starts from 100% and iteratevely decreases so it reaches the desired level
            #time_stamps = list(range(inst.GA.nodes['time_stamp']['pdf'][0]['loc'], inst.GA.nodes['time_stamp']['pdf'][0]['scale'] + inst.GA.nodes['time_stamp']['pdf'][0]['loc']))
            time_stamps = list(np.linspace(inst.GA.nodes['time_stamp']['pdf'][0]['loc'], inst.GA.nodes['time_stamp']['pdf'][0]['scale'] + inst.GA.nodes['time_stamp']['pdf'][0]['loc'], num_requests, endpoint=False))
            dynamismlvl = dynamism(time_stamps, inst.GA.nodes['time_stamp']['pdf'][0]['loc'], inst.GA.nodes['time_stamp']['pdf'][0]['scale'] + inst.GA.nodes['time_stamp']['pdf'][0]['loc'])
            if (dynamismlvl < inst.GA.nodes['time_stamp']['dynamism']):
                not_reached_dynamism = False
            else:
                not_reached_dynamism = True

            if inst.GA.nodes['time_stamp']['dynamism'] == 0:
                for ts in range(len(time_stamps)):
                    time_stamps[ts] = inst.GA.nodes['time_stamp']['pdf'][0]['loc']
                dynamismlvl = dynamism(time_stamps, inst.GA.nodes['time_stamp']['pdf'][0]['loc'], inst.GA.nodes['time_stamp']['pdf'][0]['scale'] + inst.GA.nodes['time_stamp']['pdf'][0]['loc'])
                print('dynamism zero')
                print(dynamismlvl)
                not_reached_dynamism = False

            for ts in range(len(time_stamps)):
                time_stamps[ts] = int(time_stamps[ts])

            while (not_reached_dynamism):

                for ts in range(len(time_stamps)):

                    dynamismlvl = dynamism(time_stamps, inst.GA.nodes['time_stamp']['pdf'][0]['loc'], inst.GA.nodes['time_stamp']['pdf'][0]['scale'] + inst.GA.nodes['time_stamp']['pdf'][0]['loc'])
                    dynamismlvl *= 100
                    dynamismlvl = int(dynamismlvl)
                    print(dynamismlvl, inst.GA.nodes['time_stamp']['dynamism'])
                    if not ((dynamismlvl >= inst.GA.nodes['time_stamp']['dynamism'] - 1) and (dynamismlvl <= inst.GA.nodes['time_stamp']['dynamism'])):
                        #time_stamps[ts] = int(np.random.randint(low=inst.GA.nodes['time_stamp']['pdf'][0]['loc'], high=(inst.GA.nodes['time_stamp']['pdf'][0]['scale'] + inst.GA.nodes['time_stamp']['pdf'][0]['loc']), size=1))
                        time_stamps[ts] = int(uniform.rvs(loc=inst.GA.nodes['time_stamp']['pdf'][0]['loc'], scale=inst.GA.nodes['time_stamp']['pdf'][0]['scale'], size=1))
                    else:
                        not_reached_dynamism = False
                        break
    
    reaction_times = []
    if 'reaction_time' in inst.GA.nodes: 

        if inst.GA.nodes['reaction_time']['pdf'][0]['type'] == 'normal':
            reaction_times = norm.rvs(loc=inst.GA.nodes['reaction_time']['pdf'][0]['loc'], scale=inst.GA.nodes['reaction_time']['pdf'][0]['scale'], size=num_requests)              
            

        if inst.GA.nodes['reaction_time']['pdf'][0]['type'] == 'uniform':
            reaction_times = uniform.rvs(loc=inst.GA.nodes['reaction_time']['pdf'][0]['loc'], scale=inst.GA.nodes['reaction_time']['pdf'][0]['scale'], size=num_requests)
           
        
        for r in range(num_requests):
            reaction_times[r] = int(reaction_times[r])
            while (reaction_times[r] < 0):
                if inst.GA.nodes['reaction_time']['pdf'][0]['type'] == 'normal':
                    reaction_times[r] = int(norm.rvs(loc=inst.GA.nodes['reaction_time']['pdf'][0]['loc'], scale=inst.GA.nodes['reaction_time']['pdf'][0]['scale'], size=1))

                if inst.GA.nodes['reaction_time']['pdf'][0]['type'] == 'uniform':
                    reaction_times[r] = int(uniform.rvs(loc=inst.GA.nodes['reaction_time']['pdf'][0]['loc'], scale=inst.GA.nodes['reaction_time']['pdf'][0]['scale'], size=1))
        
        
        not_reached_urgency = True
        while(not_reached_urgency):

            for r in range(num_requests):

                mean = sum(reaction_times) / len(reaction_times)
                variance = sum([((x - mean) ** 2) for x in reaction_times]) / len(reaction_times)
                stdv = variance ** 0.5
                #print(mean, stdv)
                #and (int(stdv) == inst.GA.nodes['reaction_time']['pdf'][0]['scale'])
                if (int(mean) <= inst.GA.nodes['reaction_time']['pdf'][0]['loc'] + inst.GA.nodes['reaction_time']['pdf'][0]['loc']*0.01):
                    not_reached_urgency = False
                    break
                
                else:
                    while(True):
                        if (reaction_times[r] > inst.GA.nodes['reaction_time']['pdf'][0]['loc']) or (reaction_times[r] < 0):
                            if inst.GA.nodes['reaction_time']['pdf'][0]['type'] == 'normal':
                                reaction_times[r] = int(norm.rvs(loc=inst.GA.nodes['reaction_time']['pdf'][0]['loc'], scale=inst.GA.nodes['reaction_time']['pdf'][0]['scale'], size=1))

                            if inst.GA.nodes['reaction_time']['pdf'][0]['type'] == 'uniform':
                                reaction_times[r] = int(uniform.rvs(loc=inst.GA.nodes['reaction_time']['pdf'][0]['loc'], scale=inst.GA.nodes['reaction_time']['pdf'][0]['scale'], size=1))
                        
                        if (reaction_times[r] >= 0) and (reaction_times[r] <= inst.GA.nodes['reaction_time']['pdf'][0]['loc']):    
                            break

        print(reaction_times)

    print(num_requests)
    instance_data = {}
    all_instance_data = {}  

    gc.collect()
    ray.shutdown()
    ray.init(num_cpus=12, object_store_memory=14000000000)

    GA_id = ray.put(inst.GA)
    network_id = ray.put(inst.network)
    zonescsvid = ray.put(zonescsv)
    sorted_attributes_id = ray.put(inst.sorted_attributes)
    parameters_id = ray.put(inst.parameters)
    print("HERE")
    print(inst.sorted_attributes)
    all_reqs = ray.get([_generate_single_data.remote(GA_id, network_id, sorted_attributes_id, parameters_id, i, method_poi, distances[i], pu, do, pdfunc, loc, scale, aux, time_stamps, replicate_num, reaction_times, num_requests, zonescsvid) for i in range(num_requests)]) 
    print("OUT HERE")
    ray.shutdown()

    plt.hist(all_reqs["time_stamp"], bins=30, density=True, alpha=0.6, color="g")
    plt.title("Gamma Distribution")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    # Save the plot as an image (e.g., PNG)
    plt.savefig("gamma_distribution_plot.png")

    del GA_id
    del network_id
    del sorted_attributes_id
    del parameters_id
    gc.collect()

    i = 0
    for req in all_reqs:

        instance_data.update({i: req})
        i += 1    
              
    del all_reqs
    gc.collect()

    all_instance_data.update({'num_data:': len(instance_data),
                          'requests': instance_data
                          })

    final_filename = ''
    

    for p in inst.instance_filename:

        if p in inst.parameters:
            if 'value' in inst.parameters[p]:
                strv = str(inst.parameters[p]['value'])
                strv = strv.replace(" ", "")

                if len(final_filename) > 0:
                    final_filename = final_filename + '_' + strv
                else: final_filename = strv
    
        
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
                        #print(index)
                        lvid = index+1

                if location in inst.parameters:

                    inst.parameters[location]['list_seq_id'+str(replicate_num)] = []
                    #print(inst.parameters[location]['list_node_drive'+str(replicate_num)])
                    for d in inst.parameters[location]['list_node_drive'+str(replicate_num)]:

                        node_list.append(d)
                        node_list_seq.append(lvid)
                        print(lvid)
                        inst.parameters[location]['list_seq_id'+str(replicate_num)].append(lvid)
                        lvid += 1

                if location in inst.sorted_attributes:

                    for d in instance_data:

                        node_list.append(instance_data[d][location+'node_drive'])
                        instance_data[d][location+'id'] = lvid
                        node_list_seq.append(lvid)
                        print(lvid)
                        lvid += 1

            if replicate_num == 0:
                print('ttm')
                travel_time = inst.network._get_travel_time_matrix("list", node_list=node_list)
                travel_time_json = travel_time.tolist()

                all_instance_data.update({'travel_time_matrix': travel_time_json
                                })
                
                print('leave ttm')
                print(len(node_list), len(node_list_seq))          
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
            
                '''
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
                ''' 
                ttmpd = pd.DataFrame(travel_time)

                ttmpd.to_csv(output_file_ttm_csv)
                print('leave ttm file')    
                #ttmpd.set_index(['osmid_origin'], inplace=True)


    save_dir = os.getcwd()+'/'+inst.output_folder_base
    save_dir_images = os.path.join(save_dir, 'images')
    plot_requests(inst.network, save_dir_images, origin_points, destination_points)

    
    final_filename = inst.filename_json.replace(inst_directory, "")
    final_filename = final_filename.replace(".json", "")
    
    inst.output_file_json = os.path.join(inst.save_dir_json+'/'+base_save_folder_name, final_filename + '_' + str(replicate_num+1) + '.json')

    with open(inst.output_file_json, 'w') as file:
        json.dump(all_instance_data, file, indent=4)
        file.close() 

    del all_instance_data
    gc.collect()
