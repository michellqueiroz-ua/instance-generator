import json
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
import re
import os
import osmnx as ox
import pickle

from instance_class import Instance
from parameters_class import Parameters
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

    #true = True
    #false = False

    data = json.load(f)

    if 'seed' in data:
        value = data['seed']
           
    else: 
        value = 1

    random.seed(value)
    np.random.seed(value)

    if 'network' in data:

        #if (len(data['network']) > 1):
        #    raise ValueError('only one network per instance is allowed')
      
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

            if 'max_speed_factor' in data:
                max_speed_factor = data['max_speed_factor']
            else:
                max_speed_factor = 0.5

            #print(max_speed_factor)
            network = download_network_information(place_name=place_name, max_speed_factor=max_speed_factor, get_fixed_lines=get_fixed_lines)

            save_dir_fr = os.path.join(save_dir, 'fr_network')
            if not os.path.isdir(save_dir_fr):
                os.mkdir(save_dir_fr)

            output_name_fr = place_name+'.frn'

            if get_fixed_lines is not None:
                output_fixed_route_network(output_file_name=os.path.join(save_dir_fr, output_name_fr), network=network)

            inst = Instance(folder_to_network=place_name)

        inst.parameters['network'] = {}
        inst.parameters['network']['value'] = data['network']
        inst.parameters['network']['type'] = 'network'

    else: raise ValueError('network parameter is mandatory')

    if 'problem' in data:

        inst.parameters['problem'] = {}
        inst.parameters['problem']['type'] = "string"
        inst.parameters['problem']['value'] = data['problem']

    if 'problem' in data:

        inst.parameters['problem'] = {}
        inst.parameters['problem']['type'] = "string"
        inst.parameters['problem']['value'] = data['problem']

    list_names = []
    if 'places' in data:

        location_names = []
        for j in data['places']:

            if j['type'] == 'location':
                
                if 'name' in j:
                    if not (isinstance(j['name'], (str))): 
                        raise TypeError('name for a location must be a string')
                    

                    word = j['name']
                    if not ((any(word in x for x in list_names)) or (any(x in word for x in list_names))):  
                        namelocation = j['name']
                        list_names.append(j['name'])
                    else:
                        raise ValueError('name '+j['name']+' is already a substring of another declared name. This is not allowed. Please change and try again.')
                
                else: raise ValueError('name parameter for places is mandatory')
                
                #print(j['name'])
                location_names.append(j['name'])
                #print(location_names)

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

                
                if 'class' in j: 
                    classlocs = ['school', 'coordinate']

                    if j['class'] not in classlocs:
                        raise ValueError('location '+str(j['type'])+' is not supported')

                    if j['class'] == 'school':
                        inst.network.add_new_school(name=namelocation, x=lon, y=lat)
                

            if j['type'] == 'zone':

                if 'name' in j:
                    if not (isinstance(j['name'], (str))): 
                        raise TypeError('name for a zone must be a string')

                    word = j['name']
                    if not ((any(word in x for x in list_names)) or (any(x in word for x in list_names))):  
                        nameszone  = j['name']
                        list_names.append(j['name'])
                    else:
                        raise ValueError('name '+j['name']+' is already a substring of another declared name. This is not allowed. Please change and try again.')
                    
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

        for x in range(len(data['places'])):

            if data['places'][x]['type'] == 'location': 
                point = (data['places'][x]['lat'], data['places'][x]['lon'])
                data['places'][x]['node_drive'] = ox.get_nearest_node(inst.network.G_drive, point)
                data['places'][x]['node_walk'] = ox.get_nearest_node(inst.network.G_walk, point)

    if 'parameters' in data:

        inst.parameters['all_locations'] = {}
        inst.parameters['all_locations']['type'] = 'builtin'
        inst.parameters['all_locations']['value'] = []
        inst.parameters['all_locations']['value'].append('bus_stations')

        for j in data['parameters']:

            if 'name' in j:

                if not (isinstance(j['name'], (str))): 
                    raise TypeError('name for an attribute must be a string')

                word = j['name']
                if not ((any(word in x for x in list_names)) or (any(x in word for x in list_names))):  
                    inst.parameters[j['name']] = {}
                    list_names.append(j['name'])
                else:
                    raise ValueError('name '+j['name']+' is already a substring of another declared name. This is not allowed. Please change and try again.')

                
                if 'value' in j:
                    
                    mult = 1
                    if 'time_unit' in j:
                        
                        tunit = j['time_unit']
                        if (tunit != 's') and (tunit != 'min') and (tunit != 'h'):
                            raise ValueError('time_unit must be s, min or h')

                        if not (isinstance(j['value'], (int, float))): 
                            raise TypeError('value be a number (integer, float)')
                        if j['value'] < 0:
                            raise TypeError('negative number is not allowed for time')
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

                if 'type' in j:

                    types = ['string', 'integer', 'speed', 'length', 'array_primitives', 'array_locations', 'array_zones', 'matrix', 'graphml']
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

                    if j['type'] == 'array_locations':
                        inst.parameters['all_locations']['value'].append(j['name'])


                        if 'size' in j:
                            
                            if (isinstance(j['size'], (int))): 
                                inst.parameters[j['name']]['size'] = j['size']
                            else: raise TypeError('size must be an integer number')

                            if j['size'] < 0:
                                raise TypeError('size must be a positive integer number')

                        if 'value' in j:
                                                        
                            if (isinstance(j['value'], (list))): 
                                inst.parameters[j['name']]['list'] = j['value']
                            else: raise TypeError('value parameter must be an array')

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

                            loctypes = ['random', 'schools']
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

                                for x in data['places']:

                                    if x['type'] == 'location':
                                        if x['name'] in inst.parameters[j['name']]['list']:
                                            inst.parameters[j['name']]['list_node_drive'].append(x['node_drive'])
                                            inst.parameters[j['name']]['list_node_walk'].append(x['node_walk'])

                        else: raise ValueError('locs for a array_locations parameter is mandatory')

                    if j['type'] == 'array_zones':

                        inst.parameters[j['name']]['zones'] = []
                        if 'size' in j:
                            
                            if (isinstance(j['size'], (int))): 
                                inst.parameters[j['name']]['size'] = j['size']
                            else: raise TypeError('size parameter must be integer')

                            if j['size'] < 0:
                                raise TypeError('size must be a positive integer number')

                        if 'value' in j:

                            if (isinstance(j['value'], (list))): 
                                inst.parameters[j['name']]['list'] = j['value']
                            else: raise TypeError('value parameter must be an array')

                            for z in j['value']:

                                idxs = inst.network.zones.index[inst.network.zones['name'] == z].tolist()

                                if len(idxs) > 0:
                                    index_zone = idxs[0]
                                    inst.parameters[j['name']]['zones'].append(index_zone)
                                else:
                                    raise ValueError('no zone named after '+z)

                        else:
                            inst.parameters[j['name']]['list'] = []
                            
                        

            else: raise ValueError('name for a parameter is mandatory')

    if 'replicas' in data:

        inst.set_number_replicas(number_replicas=data['replicas'])

    else: inst.set_number_replicas(number_replicas=1)

    if 'records' in data:

        inst.parameters['records'] = {}
        inst.parameters['records']['value'] = data['records']
        inst.parameters['records']['type'] = 'integer'

    if 'instance_filename' in data:

        inst.instance_filename = data['instance_filename']

        for x in data['instance_filename']:
            if x not in inst.parameters:
                raise ValueError(x+ ' is not a parameter, therefore not valid for instance_filename')

    
    GA = nx.DiGraph()

    if 'travel_time_matrix' in data:

        inst.parameters['travel_time_matrix'] = {}
        inst.parameters['travel_time_matrix']['type'] = 'matrix'
        inst.parameters['travel_time_matrix']['value'] = True
        
        if (isinstance(data['travel_time_matrix'], (list))): 
            inst.parameters['travel_time_matrix']['locations'] = data['travel_time_matrix']
        else: raise TypeError('locations for travel_time_matrix must be an array')

    if 'attributes' in data:

        index=0
        for attribute in data['attributes']:

            if 'name' in attribute:

                #print(attribute['name'])

                if (isinstance(attribute['name'], (str))): 
                
                    word = attribute['name']
                    if not ((any(word in x for x in list_names)) or (any(x in word for x in list_names))):  
                        name = attribute['name']
                        list_names.append(attribute['name'])
                    else:
                        raise ValueError('name '+attribute['name']+' is already a substring of another declared name. This is not allowed. Please change and try again.')

                else:
                    raise TypeError('name for an attribute must be a string')

                GA.add_node(name)
                
            else: raise ValueError('name parameter for attribute is mandatory')

            if 'type' in attribute:

                if (isinstance(attribute['type'], (str))): 
                    GA.nodes[name]['type'] = attribute['type']
                else:
                    raise TypeError('type for an attribute must be a string')

                types = ['integer', 'real', 'string', 'array_primitives', 'location']
                if not (GA.nodes[name]['type'] in types):
                    raise ValueError('type ' +GA.nodes[name]['type']+' is not supported')

                if attribute['type'] == 'location':
                    inst.parameters['all_locations']['value'].append(attribute['name'])

                
                if 'time_unit' in attribute:

                    GA.nodes[name]['time_unit'] = attribute['time_unit']
                    
                #else: 

                    #GA.nodes[name]['time_unit'] = 's'

                    if (attribute['time_unit'] != 's') and (attribute['time_unit'] != 'min') and (attribute['time_unit'] != 'h'):
                        raise ValueError('time_unit must be s, min or h')

                    

                
                if 'speed_unit' in attribute:

                    GA.nodes[name]['speed_unit'] = attribute['speed_unit']

                #else:

                    #GA.nodes[name]['speed_unit'] = 'mps'

                    sunit = GA.nodes[name]['speed_unit']
                    if (sunit != 'mps') and (sunit != 'kmh') and (sunit != 'mph'):
                        raise ValueError('speed_unit must be mps, kmh or mph')

                if 'length_unit' in attribute:

                    GA.nodes[name]['length_unit'] = attribute['length_unit']

                #else:

                    #GA.nodes[name]['length_unit'] = 'm'

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

            else:

                GA.nodes[name]['subset_zones'] = False


            if 'subset_locations' in attribute:

                if (isinstance(attribute['subset_locations'], (str))): 
                    GA.nodes[name]['subset_locations'] = attribute['subset_locations']
                else:
                    raise TypeError('subset_locations must be a string')

                if GA.nodes[name]['subset_locations'] not in inst.parameters:
                    raise ValueError('There is not parameter named '+GA.nodes[name]['subset_locations'])

            if 'subset_primitives' in attribute:

                if (isinstance(attribute['subset_primitives'], (str))): 
                    GA.nodes[name]['subset_primitives'] = attribute['subset_primitives']
                else:
                    raise TypeError('subset_primitives must be a string')

                if GA.nodes[name]['subset_primitives'] not in inst.parameters:
                    raise ValueError('There is not parameter named '+GA.nodes[name]['subset_primitives'])

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
                if 'time_unit' in attribute:
                    positiveV = True
                    tunit = attribute['time_unit']
                    #if (tunit != 's') and (tunit != 'min') and (tunit != 'h'):
                    #    raise ValueError('time_unit must be s, min or h')

                    mult = get_multiplier_time_unit(attribute['time_unit'])

                elif 'speed_unit' in attribute:
                    positiveV = True
                    sunit = attribute['speed_unit']
                    #if (sunit != 'mps') and (sunit != 'kmh') and (sunit != 'mph'):
                    #    raise ValueError('speed_unit must be mps, kmh or mph')

                    mult = get_multiplier_speed_unit(attribute['speed_unit'])

                elif 'length_unit' in attribute:
                    positiveV = True
                    lunit = attribute['length_unit']
                    #if (lunit != 'm') and (lunit != 'km') and (lunit != 'mi'):
                    #    raise ValueError('length_unit must be m, km or mi')

                    mult = get_multiplier_length_unit(attribute['length_unit'])

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

                    if (GA.nodes[name]['type'] == 'integer'):
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
                if loc not in inst.parameters['all_locations']['value']:
                    raise ValueError(str(loc)+ ' is not recognized or does not exists for travel_time_matrix')

    else: raise ValueError('attributes for instance are mandatory')

    
    #else: raise ValueError('locations for travel_time_matrix is mandatory')

    #caching.clear_cache()

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

    print(final_filename)

    #instance_class_file = os.path.join(inst.pickle_dir, final_filename + '.instance.class.pkl')
    pclassfile = Parameters()
    pclassfile.parameters = inst.parameters
    p_class_file = inst.pickle_dir+'/'+final_filename+'.param.class.pkl'
    output_p_class = open(p_class_file, 'wb')
    pickle.dump(pclassfile, output_p_class, pickle.HIGHEST_PROTOCOL)

    #plot network
    fig, ax = ox.plot_graph(inst.network.G_drive, show=False, close=False,  figsize=(8, 8), node_color='#000000', node_size=20, bgcolor="#ffffff", edge_color="#999999", edge_alpha=None, dpi=1440)

    save_dir = os.getcwd()+'/'+inst.output_folder_base
    save_dir_images = os.path.join(save_dir, 'images')
    network_folder = os.path.join(save_dir_images, 'network')

    if not os.path.isdir(network_folder):
        os.mkdir(network_folder)
    plt.savefig(network_folder+'/network_drive')

    #plot network
    fig, ax = ox.plot_graph(inst.network.G_walk, show=False, close=False,  figsize=(8, 8), node_color='#000000', node_size=12, bgcolor="#ffffff", edge_color="#999999", edge_alpha=None, dpi=1440)
    plt.savefig(network_folder+'/network_walk')

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
            print(output_name_cpp)

            output_name_csv = instance.split('.')[0] + '.csv'
            output_name_csv = output_name_csv.replace(" ", "")
            
            output_name_ls = instance.split('.')[0] + '_ls.pass'

            converter = JsonConverter(file_name=input_name)
            converter.convert_normal(output_file_name=os.path.join(save_dir_cpp, output_name_cpp), inst=inst, problem_type=inst.parameters['problem']['value'], path_instance_csv_file=os.path.join(save_dir_csv, output_name_csv))
            #converter.convert_localsolver(output_file_name=os.path.join(save_dir_localsolver, output_name_ls))
    


    f.close()