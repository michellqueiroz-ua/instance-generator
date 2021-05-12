import json
import math
import networkx as nx
import numpy as np
import random
import re
import os
import osmnx as ox

from instance_class import Instance
from output_files import JsonConverter
from output_files import output_fixed_route_network
from pathlib import Path
from retrieve_network import download_network_information
from shapely.geometry import Point
from shapely.geometry import Polygon
from streamlit import caching

def get_multiplier_time_unit(time_unit):

    mult = 1
    if time_unit == 'min':
        mult = 60

    elif time_unit == 'h':
        mult = 3600
    
    return mult

def get_multiplier_length_unit(length_unit):

    mult = 1
    if length_unit == 'km':
        mult = 1000

    elif length_unit == 'mi':
        mult = 1609.34
    
    return mult

def get_multiplier_speed_unit(speed_unit):

    mult = 1
    if speed_unit == "kmh":
        mult = float(1/3.6)

    elif speed_unit == "mph":
        mult = float(1/2.237)

    return mult

def input_json(filename_json):

    f = open(filename_json,)

    data = json.load(f)

    if 'seed' in data:
        
        for j in data['seed']:
            
            if 'value' in j:
                value = j['value']
            else: raise ValueError('value parameter for seed is mandatory')

            if 'increment' in j:
                increment = j['increment']
            else:
                increment = 1

    else: 

        value = 1
        increment = 1

    random.seed(value)
    np.random.seed(value)

    if 'network' in data:

        place_name=data['network']
        
        save_dir = os.getcwd()+'/'+place_name
        pickle_dir = os.path.join(save_dir, 'pickle')
        network_class_file = pickle_dir+'/'+place_name+'.network.class.pkl'

        if Path(network_class_file).is_file():

            inst = Instance(folder_to_network=place_name)

        else:

            if 'get_fixed_lines' in data:
                get_fixed_lines = data['get_fixed_lines']
            else:
                get_fixed_lines = None

            network = download_network_information(place_name=place_name, max_speed_factor=0.5, get_fixed_lines=get_fixed_lines)

            save_dir_fr = os.path.join(save_dir, 'fr_network')
            if not os.path.isdir(save_dir_fr):
                os.mkdir(save_dir_fr)

            output_name_fr = place_name+'.frn'

            if get_fixed_lines is not None:
                output_fixed_route_network(output_file_name=os.path.join(save_dir_fr, output_name_fr), network=network)

            inst = Instance(folder_to_network=place_name)

    else: raise ValueError('network parameter is mandatory')

    if 'problem' in data:

        inst.set_problem_type(problem_type=data['problem'])

    inst.set_seed(seed=value, increment_seed=increment)

    if 'request_demand_uniform' in data:

        for j in data['request_demand_uniform']:

            if 'min_time' in j:
                min_time = j['min_time']
            else: raise ValueError('min_time parameter for request_demand_uniform is mandatory')

            if 'max_time' in j:
                max_time = j['max_time']
            else: raise ValueError('max_time parameter for request_demand_uniform is mandatory')

            if 'number_of_requests' in j:
                number_of_requests = j['number_of_requests']
            else: raise ValueError('number_of_requests parameter for request_demand_uniform is mandatory')

            if 'time_unit' in j:
                time_unit = j['time_unit']
            else: raise ValueError('time_unit parameter for request_demand_uniform is mandatory')

        inst.add_request_demand_uniform(min_time=min_time, max_time=max_time, number_of_requests=number_of_requests, time_unit=time_unit)

    if 'spatial_distribution1' in data:

        for j in data['spatial_distribution1']:

            if 'num_origins' in j:
                num_origins = j['num_origins']
            else: raise ValueError('num_origins parameter for request_demand_uniform is mandatory')

            if 'num_destinations' in j:
                num_destinations = j['num_destinations']
            else: raise ValueError('num_destinations parameter for request_demand_uniform is mandatory')

            if 'prob' in j:
                prob = j['prob']
            else: raise ValueError('number_of_requests parameter for request_demand_uniform is mandatory')

            if 'is_random_origin_zones' in j:
                is_random_origin_zones = j['is_random_origin_zones']
            else: raise ValueError('is_random_origin_zones parameter for request_demand_uniform is mandatory')

            if 'is_random_destination_zones' in j:
                is_random_destination_zones = j['is_random_destination_zones']
            else: raise ValueError('is_random_destination_zones parameter for request_demand_uniform is mandatory')

        inst.add_spatial_distribution(num_origins=num_origins, num_destinations=num_destinations, prob=prob, is_random_origin_zones=is_random_origin_zones, is_random_destination_zones=is_random_destination_zones)

    if 'spatial_distribution2' in data:

        for j in data['spatial_distribution2']:

            if 'num_origins' in j:
                num_origins = j['num_origins']
            else: raise ValueError('num_origins parameter for request_demand_uniform is mandatory')

            if 'num_destinations' in j:
                num_destinations = j['num_destinations']
            else: raise ValueError('num_destinations parameter for request_demand_uniform is mandatory')

            if 'prob' in j:
                prob = j['prob']
            else: raise ValueError('number_of_requests parameter for request_demand_uniform is mandatory')

            if 'is_random_origin_zones' in j:
                is_random_origin_zones = j['is_random_origin_zones']
            else: raise ValueError('is_random_origin_zones parameter for request_demand_uniform is mandatory')

            if 'is_random_destination_zones' in j:
                is_random_destination_zones = j['is_random_destination_zones']
            else: raise ValueError('is_random_destination_zones parameter for request_demand_uniform is mandatory')

        inst.add_spatial_distribution(num_origins=num_origins, num_destinations=num_destinations, prob=prob, is_random_origin_zones=is_random_origin_zones, is_random_destination_zones=is_random_destination_zones)

    inst.add_spatial_distribution(num_origins=-1, num_destinations=-1, prob=0)
    inst.add_spatial_distribution(num_origins=-1, num_destinations=-1, prob=0)

    if 'planning_horizon' in data:

        for j in data['planning_horizon']:

            if 'min_early_departure' in j:
                min_early_departure = j['min_early_departure']
                #inst.parameters['min_early_departure'] = j['min_early_departure']*3600
            else: raise ValueError('min_early_departure parameter for planning_horizon is mandatory')

            if 'max_early_departure' in j:
                max_early_departure = j['max_early_departure']
                #inst.parameters['max_early_departure'] = j['max_early_departure']*3600
            else: raise ValueError('max_early_departure parameter for planning_horizon is mandatory')

            if 'time_unit' in j:
                time_unit = j['time_unit']
                #inst.parameters['time_unit'] = j['time_unit']
            else: raise ValueError('time_unit parameter for planning_horizon is mandatory')

            inst.set_time_window(min_early_departure=min_early_departure, max_early_departure=max_early_departure, time_unit=time_unit)

    if 'locations' in data:
        location_names = []
        for j in data['locations']:

            if 'name' in j:
                if not (isinstance(j['name'], (str))): 
                    raise TypeError('name for a location must be a string')
                namelocation = j['name']
            else: raise ValueError('name parameter for locations is mandatory')
            location_names.append(j['name'])

            if ('centroid' in j) or (('lat' in j) and ('lon' in j)):
                   
                if ('lon' in j) and ('lat' in j):
                    lon = j['lon']
                    lat = j['lat']
                else:
                    if j['centroid'] is True:

                        pt = inst.network.polygon.centroid
                        lon = pt.x
                        lat = pt.y

                    else: raise ValueError('lon/lat or centroid parameter for locations is mandatory')

            else: raise ValueError('lon/lat or centroid parameter for locations is mandatory')

            if not inst.network.polygon.contains(Point(lon,lat)):
                raise ValueError('location for '+namelocation+' is not within the boundaries of network')

            if 'type' in j: 
                typelocs = ['school', 'coordinate']

                if j['type'] not in typelocs:
                    raise ValueError('location '+str(j['type'])+' is not supported')

                if j['type'] == 'school':
                    inst.network.add_new_school(name=namelocation, x=lon, y=lat)

        for x in range(len(data['locations'])):

            point = (data['locations'][x]['lat'], data['locations'][x]['lon'])
            data['locations'][x]['node_drive'] = ox.get_nearest_node(inst.network.G_drive, point)
            data['locations'][x]['node_walk'] = ox.get_nearest_node(inst.network.G_walk, point)

    if 'zones' in data:

        for j in data['zones']:

            if 'name' in j:
                if not (isinstance(j['name'], (str))): 
                    raise TypeError('name for a zone must be a string')
                nameszone  = j['name']
            else: raise ValueError('name parameter for zone is mandatory')

            if 'lon' in j:
                lon = j['lon']
            
                if 'lat' in j:
                    lat = j['lat']
                else: raise ValueError('lat parameter for zone is mandatory when lon is set')

            else:
                if 'centroid' in j:
                    if j['centroid'] is True:
                        pt = inst.network.polygon.centroid
                        lon = pt.x
                        lat = pt.y

                    else: raise ValueError('either lon/lat must be given or centroid must be set to true')

                else: raise ValueError('either lon/lat or centroid parameters must be set')

            if not inst.network.polygon.contains(Point(lon,lat)):

                raise ValueError('location for '+nameszone+' is not within the boundaries of network')

            if 'length_unit' in j:
                lunit = j['length_unit']
                if (lunit != 'km') and (lunit != 'm') and (lunit != 'mi'):
                    raise ValueError('length_unit must be m, km or mi')

                length_unit  = j['length_unit']
            else:
                length_unit = "m"

            mult = get_multiplier_length_unit(length_unit)

            if 'length_lon' in j:
                if not (isinstance(j['length_lon'], (int, float))): 
                    raise TypeError('length_lon for type length must be a number (integer, float)')
                if j['length_lon'] < 0:
                    raise TypeError('negative number is not allowed for type length')
                length_lon  = j['length_lon']*mult
            else:
                length_lon = 0

            if 'length_lat' in j:
                if not (isinstance(j['length_lat'], (int, float))): 
                    raise TypeError('length_lat for type length must be a number (integer, float)')
                if j['length_lat'] < 0:
                    raise TypeError('negative number is not allowed for type length')
                length_lat  = j['length_lat']*mult
            else:
                length_lat = 0

            if 'radius' in j:
                if not (isinstance(j['radius'], (int, float))): 
                    raise TypeError('radius for type length must be a number (integer, float)')
                if j['radius'] < 0:
                    raise TypeError('negative number is not allowed for type length')
                radius  = j['radius']
            else:
                radius = 0

            inst.network.add_new_zone(name=nameszone, center_x=lon, center_y=lat, length_x=length_lon, length_y=length_lat, radius=radius)

    if 'parameters' in data:

        inst.parameters['all_locations'] = {}
        inst.parameters['all_locations']['type'] = 'builtin'
        inst.parameters['all_locations']['list'] = []
        inst.parameters['all_locations']['list'].append('bus_stations')

        for j in data['parameters']:

            if 'name' in j:

                if not (isinstance(j['name'], (str))): 
                    raise TypeError('name for an attribute must be a string')

                inst.parameters[j['name']] = {}
                if 'value' in j:
                    
                    mult = 1
                    if 'time_unit' in j:
                        
                        tunit = j['time_unit']
                        if (tunit != 's') and (tunit != 'min') and (tunit != 'h'):
                            raise ValueError('time_unit must be s, min or h')

                        if not (isinstance(j['value'], (int, float))): 
                            raise TypeError('value for type time must be a number (integer, float)')
                        if j['value'] < 0:
                            raise TypeError('negative number is not allowed for type time')
                        mult = get_multiplier_time_unit(j['time_unit'])

                        inst.parameters[j['name']]['value'] = j['value']*mult

                    elif 'speed_unit' in j:
                        mult = get_multiplier_speed_unit(j['speed_unit'])

                        sunit = j['speed_unit']
                        if (sunit != 'kmh') and (sunit != 'mps') and (sunit != 'mph'):
                            raise ValueError('speed_unit must be mps, kmh or mph')

                        if not (isinstance(j['value'], (int, float))): 
                            raise TypeError('value for type speed must be a number (integer, float)')

                        if j['value'] < 0:
                            raise TypeError('negative number is not allowed for type speed')

                        inst.parameters[j['name']]['value'] = j['value']*mult

                    elif 'length_unit' in j:
                        mult = get_multiplier_length_unit(j['length_unit'])

                        lunit = j['length_unit']
                        if (lunit != 'km') and (lunit != 'm') and (lunit != 'mi'):
                            raise ValueError('length_unit must be m, km or mi')

                        if not (isinstance(j['value'], (int, float))): 
                            raise TypeError('value for type length must be a number (integer, float)')

                        if j['value'] < 0:
                            raise TypeError('negative number is not allowed for type length')
                    
                        inst.parameters[j['name']]['value'] = j['value']*mult

                    else:

                        inst.parameters[j['name']]['value'] = j['value']



                else: inst.parameters[j['name']]['value'] = np.nan

                if j['name'] == 'travel_time_matrix':

                    if 'locations' in j:
                        if (isinstance(j['locations'], (list))): 
                            inst.parameters[j['name']]['locations'] = j['locations']
                        else: raise TypeError('locations for travel_time_matrix must be a list')

                    else: raise ValueError('locations for travel_time_matrix is mandatory')

                if 'type' in j:

                    types = ['string', 'integer', 'time', 'speed', 'length', 'list_coordinates', 'list_zones', 'matrix', 'graphml']
                    if not (j['type'] in types):
                        raise ValueError('type ' +j['type']+' is not supported')

                    inst.parameters[j['name']]['type'] = j['type']

                    if j['type'] == 'integer':
                        if not (isinstance(j['value'], (int))): 
                            raise TypeError('value for '+j['name']+' must be integer')

                    if j['type'] == 'string':
                        if not (isinstance(j['value'], (str))): 
                            raise TypeError('value for '+j['name']+' must be string')

                    if j['type'] == 'float':
                        if not (isinstance(j['value'], (float))): 
                            raise TypeError('value for '+j['name']+' must be float')

                    if j['type'] == 'list_coordinates':
                        inst.parameters['all_locations']['list'].append(j['name'])


                        if 'size' in j:
                            
                            if (isinstance(j['size'], (int))): 
                                inst.parameters[j['name']]['size'] = j['size']
                            else: raise TypeError('size must be an integer number')

                            if j['size'] < 0:
                                raise TypeError('size must be a positive integer number')

                        if 'list' in j:
                                                        
                            if (isinstance(j['list'], (list))): 
                                inst.parameters[j['name']]['list'] = j['list']
                            else: raise TypeError('list parameter must be type list')

                            for elem in inst.parameters[j['name']]['list']:

                                if elem not in location_names:
                                    raise ValueError('locaton '+elem+' does not exist')

                        else:
                            inst.parameters[j['name']]['list'] = []

                        inst.parameters[j['name']]['list_node_drive'] = []
                        inst.parameters[j['name']]['list_node_walk'] = []

                        if 'locs' in j:
                            

                            if (isinstance(j['locs'], (str))): 
                                inst.parameters[j['name']]['locs'] = j['locs']
                            else:
                                raise TypeError('locs must be a string')

                            loctypes = ['anywhere', 'schools']
                            if not (inst.parameters[j['name']]['locs'] in loctypes):
                                raise ValueError('loc ' +inst.parameters[j['name']]['locs']+' is not supported')

                            if j['locs'] == 'schools':

                                inst.parameters[j['name']]['list_ids'] = []
                                for s in inst.parameters[j['name']]['list']:
                                    idxs = inst.network.schools.index[inst.network.schools['school_name'] == s].tolist()

                                    if len(idxs) > 0:
                                        index_school = idxs[0]
                                        inst.parameters[j['name']]['list_ids'].append(index_school)
                                        inst.parameters[j['name']]['list_node_drive'].append(inst.network.schools.loc[index_school, 'osmid_drive'])
                                        inst.parameters[j['name']]['list_node_walk'].append(inst.network.schools.loc[index_school, 'osmid_walk'])
                                    else:
                                        raise ValueError('no school named after '+s)
                                    #print(index_school)
                            else:

                                for x in data['locations']:

                                    if x['name'] in inst.parameters[j['name']]['list']:
                                        inst.parameters[j['name']]['list_node_drive'].append(x['node_drive'])
                                        inst.parameters[j['name']]['list_node_walk'].append(x['node_walk'])

                        else: raise ValueError('locs for a list_coordinates parameter is mandatory')

                    if j['type'] == 'list_zones':

                        inst.parameters[j['name']]['zones'] = []
                        if 'size' in j:
                            
                            if (isinstance(j['size'], (int))): 
                                inst.parameters[j['name']]['size'] = j['size']
                            else: raise TypeError('size parameter must be integer')

                            if j['size'] < 0:
                                raise TypeError('size must be a positive integer number')

                        if 'list' in j:

                            if (isinstance(j['list'], (list))): 
                                inst.parameters[j['name']]['list'] = j['list']
                            else: raise TypeError('list parameter must be type list')

                            for z in j['list']:

                                idxs = inst.network.zones.index[inst.network.zones['name'] == z].tolist()

                                if len(idxs) > 0:
                                    index_zone = idxs[0]
                                    inst.parameters[j['name']]['zones'].append(index_zone)
                                else:
                                    raise ValueError('no zone named after '+z)

                        else:
                            inst.parameters[j['name']]['list'] = []
                            
                        

            else: raise ValueError('name for a parameter is mandatory')

    if 'instance_filename' in data:

        inst.instance_filename = data['instance_filename']

        for x in data['instance_filename']:
            if x not in inst.parameters:
                raise ValueError(x+ ' is not a parameter, therefore not valid for instance_filename')

        #print(inst.instance_filename)

    if 'lead_time' in data:

        for j in data['lead_time']:

            if 'min_lead_time' in j:
                min_lead_time = j['min_lead_time']
            else: raise ValueError('min_lead_time parameter for lead_time is mandatory')

            if 'max_lead_time' in j:
                max_lead_time = j['max_lead_time']
            else: raise ValueError('max_lead_time parameter for lead_time is mandatory')

            if 'time_unit' in j:
                time_unit = j['time_unit']
            else: raise ValueError('time_unit parameter for lead_time is mandatory')

            inst.set_interval_lead_time(min_lead_time=min_lead_time, max_lead_time=max_lead_time, time_unit=time_unit)

    if 'walk_speed' in data:

        for j in data['walk_speed']:

            if 'min_walk_speed' in j:
                min_walk_speed = j['min_walk_speed']
            else: raise ValueError('min_walk_speed parameter for walk_speed is mandatory')

            if 'max_walk_speed' in j:
                max_walk_speed = j['max_walk_speed']
            else: raise ValueError('max_walk_speed parameter for walk_speed is mandatory')

            if 'speed_unit' in j:
                speed_unit = j['speed_unit']
            else: raise ValueError('speed_unit parameter for walk_speed is mandatory')

            inst.set_interval_walk_speed(min_walk_speed=min_walk_speed, max_walk_speed=max_walk_speed, speed_unit=speed_unit)

    if 'max_walking' in data:

        for j in data['max_walking']:

            if 'lb_max_walking' in j:
                lb_max_walking = j['lb_max_walking']
            else: raise ValueError('lb_max_walking parameter for max_walking is mandatory')

            if 'ub_max_walking' in j:
                ub_max_walking = j['ub_max_walking']
            else: raise ValueError('ub_max_walking parameter for max_walking is mandatory')

            if 'time_unit' in j:
                time_unit = j['time_unit']
            else: raise ValueError('time_unit parameter for max_walking is mandatory')

            inst.set_interval_max_walking(lb_max_walking=lb_max_walking, ub_max_walking=ub_max_walking, time_unit=time_unit)

    if 'number_replicas' in data:

        inst.set_number_replicas(number_replicas=data['number_replicas'])

    else: inst.set_number_replicas(number_replicas=1)

    if 'delay_vehicle_factor' in data:

        inst.set_delay_vehicle_factor(delay_vehicle_factor=data['delay_vehicle_factor'])

    if 'delay_walk_factor' in data:

        inst.set_delay_walk_factor(delay_walk_factor=data['delay_walk_factor'])

    if 'time_window_length' in data:

        for j in data['time_window_length']:

            if 'length' in j:
                length = j['length']
            else: raise ValueError('length parameter for time_window_length is mandatory')

            if 'time_unit' in j:
                time_unit = j['time_unit']
            else: raise ValueError('time_unit parameter for time_window_length is mandatory')

            inst.add_time_window_gap(g=length, time_unit=time_unit)


    inst.set_return_factor(return_factor=0.0)

    GA = nx.DiGraph()

    if 'attributes' in data:

        index=0
        for attribute in data['attributes']:

            if 'name' in attribute:

                if (isinstance(attribute['name'], (str))): 
                    name = attribute['name']
                else:
                    raise TypeError('name for an attribute must be a string')

                GA.add_node(name)
                
            else: raise ValueError('name parameter for attribute is mandatory')

            if 'type' in attribute:

                if (isinstance(attribute['type'], (str))): 
                    GA.nodes[name]['type'] = attribute['type']
                else:
                    raise TypeError('type for an attribute must be a string')

                types = ['time', 'speed', 'length', 'integer', 'float', 'string', 'list', 'coordinate']
                if not (GA.nodes[name]['type'] in types):
                    raise ValueError('type ' +GA.nodes[name]['type']+' is not supported')

                if attribute['type'] == 'coordinate':
                    inst.parameters['all_locations']['list'].append(attribute['name'])

                if attribute['type'] == 'time':

                    if 'time_unit' in attribute:

                        GA.nodes[name]['time_unit'] = attribute['time_unit']
                        
                    else: 

                        GA.nodes[name]['time_unit'] = 's'

                    if (attribute['time_unit'] != 's') and (attribute['time_unit'] != 'min') and (attribute['time_unit'] != 'h'):
                        raise ValueError('time_unit must be s, min or h')

                if attribute['type'] == 'speed':

                    if 'speed_unit' in attribute:

                        GA.nodes[name]['speed_unit'] = attribute['speed_unit']
                    else:

                        GA.nodes[name]['speed_unit'] = 'mps'

                    sunit = GA.nodes[name]['speed_unit']
                    if (sunit != 'mps') and (sunit != 'kmh') and (sunit != 'mph'):
                        raise ValueError('speed_unit must be mps, kmh or mph')

                if attribute['type'] == 'length':

                    if 'length_unit' in attribute:

                        GA.nodes[name]['length_unit'] = attribute['length_unit']
                    else:

                        GA.nodes[name]['length_unit'] = 'm'

                    lunit = GA.nodes[name]['length_unit']
                    if (lunit != 'm') and (lunit != 'km') and (lunit != 'mi'):
                        raise ValueError('length_unit must be m, km or mi')

            else: raise ValueError('type parameter for attribute is mandatory')

            if 'subset_zones' in attribute:

                if (isinstance(attribute['subset_zones'], (str))): 
                    GA.nodes[name]['subset_zones'] = attribute['subset_zones']
                else:
                    raise TypeError('subset_zones must be a string')

                if GA.nodes[name]['subset_zones'] not in inst.parameters:
                    raise ValueError('There is not parameter named '+GA.nodes[name]['subset_zones'])

            if 'subset_locations' in attribute:

                if (isinstance(attribute['subset_locations'], (str))): 
                    GA.nodes[name]['subset_locations'] = attribute['subset_locations']
                else:
                    raise TypeError('subset_locations must be a string')

                if GA.nodes[name]['subset_locations'] not in inst.parameters:
                    raise ValueError('There is not parameter named '+GA.nodes[name]['subset_locations'])


            if 'output_csv' in attribute:

                if (isinstance(attribute['output_csv'], (bool))):
                    GA.nodes[name]['output_csv'] = attribute['output_csv']
                else:
                    raise TypeError('output_csv value must be a boolean')

            else: GA.nodes[name]['output_csv'] = True
            
            if 'pdf' in attribute:

                GA.nodes[name]['pdf'] = attribute['pdf']

                mult = 1
                positiveV = False
                if 'time_unit' in GA.nodes[name]['pdf'][0]:
                    positiveV = True
                    tunit = GA.nodes[name]['pdf'][0]['time_unit']
                    if (tunit != 's') and (tunit != 'min') and (tunit != 'h'):
                        raise ValueError('time_unit must be s, min or h')

                    mult = get_multiplier_time_unit(GA.nodes[name]['pdf'][0]['time_unit'])

                elif 'speed_unit' in GA.nodes[name]['pdf'][0]:
                    positiveV = True
                    sunit = GA.nodes[name]['pdf'][0]['speed_unit']
                    if (sunit != 'mps') and (sunit != 'kmh') and (sunit != 'mph'):
                        raise ValueError('speed_unit must be mps, kmh or mph')

                    mult = get_multiplier_speed_unit(GA.nodes[name]['pdf'][0]['speed_unit'])

                elif 'length_unit' in GA.nodes[name]['pdf'][0]:
                    positiveV = True
                    lunit = GA.nodes[name]['pdf'][0]['length_unit']
                    if (lunit != 'm') and (lunit != 'km') and (lunit != 'mi'):
                        raise ValueError('length_unit must be m, km or mi')

                    mult = get_multiplier_length_unit(GA.nodes[name]['pdf'][0]['length_unit'])

                if GA.nodes[name]['pdf'][0]['type'] == 'normal':

                    if (isinstance(GA.nodes[name]['pdf'][0]['mean'], (int, float))):
                        GA.nodes[name]['pdf'][0]['mean'] = GA.nodes[name]['pdf'][0]['mean']*mult
                    else:
                        raise TypeError('mean value must be a number (integer, float)')

                    if (isinstance(GA.nodes[name]['pdf'][0]['std'], (int, float))):
                        GA.nodes[name]['pdf'][0]['std'] = GA.nodes[name]['pdf'][0]['std']*mult
                    else:
                        raise TypeError('std value must be a number (integer, float)')

                    if (positiveV) and (GA.nodes[name]['pdf'][0]['mean'] < 0):
                        raise TypeError('a negative "mean" number is not allowed for type time/speed/length')

                    if (positiveV) and (GA.nodes[name]['pdf'][0]['std'] < 0):
                        raise TypeError('a negative "std" number is not allowed for type time/speed/length')
                    
                elif GA.nodes[name]['pdf'][0]['type'] == 'uniform':

                    if (isinstance(GA.nodes[name]['pdf'][0]['max'], (int, float))):
                        GA.nodes[name]['pdf'][0]['max'] = GA.nodes[name]['pdf'][0]['max']*mult
                    else:
                        raise TypeError('max value must be a number (integer, float)')

                    if (isinstance(GA.nodes[name]['pdf'][0]['min'], (int, float))):
                        GA.nodes[name]['pdf'][0]['min'] = GA.nodes[name]['pdf'][0]['min']*mult
                    else:
                        raise TypeError('min value must be a number (integer, float)')

                    if (positiveV) and (GA.nodes[name]['pdf'][0]['max'] < 0):
                        raise TypeError('a negative "max" number is not allowed for type time/speed/length')

                    if (positiveV) and (GA.nodes[name]['pdf'][0]['min'] < 0):
                        raise TypeError('a negative "min" number is not allowed for type time/speed/length')

                    if (GA.nodes[name]['type'] == 'time') or (GA.nodes[name]['type'] == 'integer'):
                        GA.nodes[name]['pdf'][0]['max'] += 1

                elif GA.nodes[name]['pdf'][0]['type'] == 'poisson':

                    if (isinstance(GA.nodes[name]['pdf'][0]['lam'], (int, float))):
                        GA.nodes[name]['pdf'][0]['lam'] = GA.nodes[name]['pdf'][0]['lam']*mult
                    else:
                        raise TypeError('lam value must be a number (integer, float)')

                    if (positiveV) and (GA.nodes[name]['pdf'][0]['lam'] < 0):
                        raise TypeError('a negative "lam" number is not allowed for type time/speed/length')

                else:
                    raise TypeError('pdf must be normal, uniform or poisson')

            elif 'expression' in attribute:

                GA.nodes[name]['expression'] = attribute['expression']

            if 'constraints' in attribute:

                GA.nodes[name]['constraints'] = attribute['constraints']

            if 'weights' in attribute:

                if (isinstance(attribute['weights'], (list))): 
                    GA.nodes[name]['weights'] = attribute['weights']
                else:
                    raise TypeError('weights must be a list')

                size_all_values = 0

                if 'pdf' in attribute:
                    
                    if GA.nodes[name]['pdf'][0]['type'] == 'uniform':
                        GA.nodes[name]['all_values'] = list(range(math.ceil(GA.nodes[name]['pdf'][0]['min']), math.floor(GA.nodes[name]['pdf'][0]['max'])))
                        #print(GA.nodes[name]['all_values'])
                        size_all_values = len(GA.nodes[name]['all_values'])
                    else: raise ValueError('normal distribution and weights is not allowed')

                elif 'subset_zones' in attribute:

                    size_all_values = inst.parameters[attribute['subset_zones']]['size']

                if attribute['weights'][0] == 'randomized_weights':

                    GA.nodes[name]['weights'] = np.random.randint(0, 101, size_all_values)
                    #print(GA.nodes[name]['weights'])
                    sumall = 0
                    for w in GA.nodes[name]['weights']:
                        sumall += w

                    for w in range(len(GA.nodes[name]['weights'])):
                        GA.nodes[name]['weights'][w] = int((GA.nodes[name]['weights'][w]/sumall)*100)
                  
                    #print('randomized_weights')      
                    #print(GA.nodes[name]['weights'])

                    for w in GA.nodes[name]['weights']:
                        if not (isinstance(w, (np.integer))): 
                            raise TypeError('weights values must be numbers (integer, float)')

                else:
                    if len(attribute['weights']) < size_all_values:  
                        raise ValueError('size of weights list for '+att+' do not match')

            if name == 'time_stamp':
                
                if 'static_probability' in attribute:
                    if (isinstance(attribute['static_probability'], (float))):
                        if (attribute['static_probability'] >= 0) and (attribute['static_probability'] <= 1):
                            GA.nodes[name]['static_probability'] = float(attribute['static_probability'])
                        else:
                            raise ValueError('static_probability must be a float between [0,1]')
                    else:
                        raise TypeError('static_probability values must be a float number') 
                
                else:    
                    GA.nodes[name]['static_probability'] = 0




        for node in GA.nodes():

            if 'expression' in GA.nodes[node]:
                #print(GA.nodes[node]['expression'])
                expression = re.split(r"[(,) ]", GA.nodes[node]['expression'])
                
                for exp in expression:
                    if exp in GA:
                        GA.add_edge(exp, node)

                    #adds an specific dependency between max_walking_user and an attribute
                    if exp == 'stops':
                        GA.add_edge('max_walking', node)
                        GA.add_edge('walk_speed', node)

                    if exp == 'walk':
                        GA.add_edge('walk_speed', node)

            if 'constraints' in GA.nodes[node]:

                for constraint in GA.nodes[node]['constraints']:
                    constraint = re.split(r"[(,)><= ]", constraint)

                    for exp in constraint:
                        if exp in GA:
                            if exp != node:
                                GA.add_edge(exp, node)



        inst.sorted_attributes = list(nx.topological_sort(GA))
        inst.GA = GA
        print(inst.sorted_attributes)

        if 'travel_time_matrix' in inst.parameters:
            for loc in inst.parameters['travel_time_matrix']['locations']:
                if loc not in inst.parameters['all_locations']['list']:
                    raise ValueError(str(loc)+ ' is not recognized or does not exists for travel_time_matrix')

    else: raise ValueError('attributes for instance are mandatory')


    inst.generate_requests()

    caching.clear_cache()
        
    # convert instances from json to cpp and localsolver formats
    save_dir_cpp = os.path.join(inst.save_dir, 'cpp_format')
    if not os.path.isdir(save_dir_cpp):
        os.mkdir(save_dir_cpp)

    save_dir_csv = os.path.join(inst.save_dir, 'csv_format')
    if not os.path.isdir(save_dir_csv):
        os.mkdir(save_dir_csv)

    save_dir_localsolver = os.path.join(inst.save_dir, 'localsolver_format')
    if not os.path.isdir(save_dir_localsolver):
        os.mkdir(save_dir_localsolver)

    
    for instance in os.listdir(os.path.join(inst.save_dir, 'json_format')):
        
        if instance != ".DS_Store":
            input_name = os.path.join(inst.save_dir, 'json_format', instance)
            
            output_name_cpp = instance.split('.')[0] + '_cpp.pass'
            output_name_cpp = output_name_cpp.replace(" ", "")

            output_name_csv = instance.split('.')[0] + '.csv'
            output_name_csv = output_name_csv.replace(" ", "")
            
            output_name_ls = instance.split('.')[0] + '_ls.pass'

            converter = JsonConverter(file_name=input_name)
            converter.convert_normal(output_file_name=os.path.join(save_dir_cpp, output_name_cpp), inst=inst, problem_type=inst.problem_type, path_instance_csv_file=os.path.join(save_dir_csv, output_name_csv))
            #converter.convert_localsolver(output_file_name=os.path.join(save_dir_localsolver, output_name_ls))
    


    f.close()