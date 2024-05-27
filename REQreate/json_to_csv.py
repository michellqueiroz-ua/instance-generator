import json
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
import re
import os
import osmnx as ox
import pandas as pd
import pickle

from instance_class import Instance
from parameters_class import Parameters
from output_files import JsonConverter
from output_files import output_fixed_route_network
from pathlib import Path
from retrieve_network import download_network_information
from compute_distance_matrix import _get_distance_matrix
from shapely.geometry import Point
from shapely.geometry import Polygon
#from streamlit import caching
import gc

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

    elif speed_unit == "miph":
        mult = float(1/2.237)

    return mult

if __name__ == '__main__':

    place_name = "Chicago, Illinois"
    base_save_folder_name = 'uneven_demand' #give a unique folder name to save the instances
    inst_directory = '../examples/uneven_demand/'
    directory = os.fsencode(inst_directory)
    inst = Instance(folder_to_network=place_name)

    save_dir_csv = os.path.join(inst.save_dir, 'csv_format')
    if not os.path.isdir(save_dir_csv):
        os.mkdir(save_dir_csv)

    if not os.path.isdir(save_dir_csv+'/'+base_save_folder_name):
        os.mkdir(save_dir_csv+'/'+base_save_folder_name)

    '''
    for instance in os.listdir(directory):

        instance_filename = os.fsdecode(instance)
        instance_filename = instance_filename.replace("._", "")
    '''

    '''
    final_filename = ''
    for p in inst.instance_filename:

        if p in inst.parameters:
            if 'value' in inst.parameters[p]:
                strv = str(inst.parameters[p]['value'])
                strv = strv.replace(" ", "")

                if len(final_filename) > 0:
                    if p == 'min_early_departure':
                        strv = inst.parameters[p]['value']/3600
                        strv = str(strv)

                    if p == 'max_early_departure':
                        strv = inst.parameters[p]['value']/3600
                        strv = str(strv)

                    final_filename = final_filename + '_' + strv
                else: final_filename = strv
    '''
    replicate_num = 1

    for instance in os.listdir(directory):

        if replicate_num == 1:
            instance_filename = os.fsdecode(instance)
            instance_filename = instance_filename.replace("._", "")

            filename_json = inst_directory+instance_filename
            #f = open(filename_json,)
            with open(filename_json, 'rb') as f:   # will close() when we leave this block
                data = json.load(f)
            
            if 'network' in data:

                place_name=data['network']
                
                save_dir = os.getcwd()+'/'+place_name
                pickle_dir = os.path.join(save_dir, 'pickle')
                network_class_file = pickle_dir+'/'+place_name+'.network.class.pkl'


                if Path(network_class_file).is_file():

                    inst = Instance(folder_to_network=place_name)
                    print('heeere')
                    
                    
                    if 'set_fixed_speed' in data:

                        vehicle_speed_data = "set"
                        if 'vehicle_speed_data_unit' in data['set_fixed_speed']:
                            mult = get_multiplier_speed_unit(data['set_fixed_speed']['vehicle_speed_data_unit'])

                            sunit = data['set_fixed_speed']['vehicle_speed_data_unit']
                            if (sunit != 'kmh') and (sunit != 'mps') and (sunit != 'miph'):
                                raise ValueError('vehicle_speed_data_unit must be mps, kmh or miph')

                            if not (isinstance(data['set_fixed_speed']['vehicle_speed_data'], (int, float))): 
                                raise TypeError('value for vehicle_speed_data speed must be a number (integer, float)')

                            if data['set_fixed_speed']['vehicle_speed_data'] < 0:
                                raise TypeError('negative number is not allowed for type speed')

                            vehicle_speed = data['set_fixed_speed']['vehicle_speed_data']
                            inst.network.vehicle_speed = vehicle_speed*mult
                            print('VEHICLE SPEED')
                            print(inst.network.vehicle_speed)
                    
                    
                else:

                    if 'get_fixed_lines' in data:
                        get_fixed_lines = data['get_fixed_lines']
                    else:
                        get_fixed_lines = None

                    if 'max_speed_factor' in data:
                        max_speed_factor = data['max_speed_factor']
                    else:
                        max_speed_factor = 0.5

                    vehicle_speed = 0
                    if 'set_fixed_speed' in data:

                        vehicle_speed_data = "set"
                        if 'vehicle_speed_data_unit' in data['set_fixed_speed']:
                            mult = get_multiplier_speed_unit(data['set_fixed_speed']['vehicle_speed_data_unit'])

                            sunit = data['set_fixed_speed']['vehicle_speed_data_unit']
                            if (sunit != 'kmh') and (sunit != 'mps') and (sunit != 'miph'):
                                raise ValueError('vehicle_speed_data_unit must be mps, kmh or miph')

                            if not (isinstance(data['set_fixed_speed']['vehicle_speed_data'], (int, float))): 
                                raise TypeError('value for vehicle_speed_data speed must be a number (integer, float)')

                            if data['set_fixed_speed']['vehicle_speed_data'] < 0:
                                raise TypeError('negative number is not allowed for type speed')

                            vehicle_speed = data['set_fixed_speed']['vehicle_speed_data']
                            vehicle_speed = vehicle_speed*mult
                        
                    else:
                        vehicle_speed_data = "max"
                        vehicle_speed = None

                    if 'point' in data:
                        graph_from_point = True 
                        lon = data['point']['lon']
                        lat = data['point']['lat']
                        dist = data['dist']
                    else:
                        graph_from_point = False
                        lon = -1
                        lat = -1
                        dist = -1

                    network = download_network_information(place_name=place_name, vehicle_speed_data=vehicle_speed_data, vehicle_speed=vehicle_speed, max_speed_factor=max_speed_factor, get_fixed_lines=get_fixed_lines, graph_from_point=graph_from_point, lon=lon, lat=lat, dist=dist)

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

            else: raise ValueError('network or point parameter is mandatory')

            '''
            print('hxxxxx')
            #remove this later
            inst.network.bus_stations = inst.network.bus_stations.iloc[0:0]
            '''

            if 'seed' in data:
                inst.seed = data['seed']
                value = data['seed']
                   
            else: 
                inst.seed = 1
                value = 1

            inst.filename_json = filename_json    
            random.seed(value)
            np.random.seed(value)
            
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

                        
                        if 'class' in j: 
                            classlocs = ['school', 'coordinate', 'bus_stop']

                            if j['class'] not in classlocs:
                                raise ValueError('location '+str(j['type'])+' is not supported')

                            if j['class'] == 'school':
                                inst.network.add_new_school(name=namelocation, x=lon, y=lat)

                            if j['class'] == 'bus_stop':
                                inst.network.add_new_stop(types=0, x=lon, y=lat)
                        

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
                        data['places'][x]['node_drive'] = ox.nearest_nodes(inst.network.G_drive, data['places'][x]['lon'], data['places'][x]['lat'])
                        data['places'][x]['node_walk'] = ox.nearest_nodes(inst.network.G_walk, data['places'][x]['lon'], data['places'][x]['lat'])

            
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
                                if (sunit != 'kmh') and (sunit != 'mps') and (sunit != 'miph'):
                                    raise ValueError('speed_unit must be mps, kmh or miph')

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

                            types = ['string', 'integer', 'float', 'speed', 'length', 'array_primitives', 'array_locations', 'array_zones', 'matrix', 'graphml']
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
                                inst.parameters[j['name']]['list_lon'] = []
                                inst.parameters[j['name']]['list_lat'] = []

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
                                                    inst.parameters[j['name']]['list_lon'].append(x['lon'])
                                                    inst.parameters[j['name']]['list_lat'].append(x['lat'])

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

            
            if ('min_dtt' in inst.parameters) and ('max_dtt' in inst.parameters):

                inst.parameters['set_geographic_dispersion'] = {}
                inst.parameters['set_geographic_dispersion']['type'] = 'builtin'
                inst.parameters['set_geographic_dispersion']['value'] = True 
                print('set geographic dispersion TRUE')
            else:
                inst.parameters['set_geographic_dispersion'] = {}
                inst.parameters['set_geographic_dispersion']['type'] = 'builtin'
                inst.parameters['set_geographic_dispersion']['value'] = False

            
            if 'replicas' in data:

                inst.set_number_replicas(number_replicas=data['replicas'])

            else: inst.set_number_replicas(number_replicas=1)

            if 'requests' in data:

                inst.parameters['requests'] = {}
                inst.parameters['requests']['value'] = data['requests']
                inst.parameters['requests']['type'] = 'integer'

            if 'instance_filename' in data:

                inst.instance_filename = data['instance_filename']

                for x in data['instance_filename']:
                    if ((x not in inst.parameters) and (x not in inst.properties)):
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
                            

                            if (attribute['time_unit'] != 's') and (attribute['time_unit'] != 'min') and (attribute['time_unit'] != 'h'):
                                raise ValueError('time_unit must be s, min or h')

                        if 'speed_unit' in attribute:

                            GA.nodes[name]['speed_unit'] = attribute['speed_unit']

                            sunit = GA.nodes[name]['speed_unit']
                            if (sunit != 'mps') and (sunit != 'kmh') and (sunit != 'miph'):
                                raise ValueError('speed_unit must be mps, kmh or miph')

                        if 'length_unit' in attribute:

                            GA.nodes[name]['length_unit'] = attribute['length_unit']

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

                            mult = get_multiplier_time_unit(attribute['time_unit'])

                        elif 'speed_unit' in attribute:
                            positiveV = True
                            sunit = attribute['speed_unit']

                            mult = get_multiplier_speed_unit(attribute['speed_unit'])

                        elif 'length_unit' in attribute:
                            positiveV = True
                            lunit = attribute['length_unit']

                            mult = get_multiplier_length_unit(attribute['length_unit'])

                        if GA.nodes[name]['pdf'][0]['type'] == 'normal':

                            if (isinstance(GA.nodes[name]['pdf'][0]['loc'], (int, float))):
                                GA.nodes[name]['pdf'][0]['loc'] = GA.nodes[name]['pdf'][0]['loc']*mult
                            else:
                                raise TypeError('loc value must be a number (integer, float)')

                            if (isinstance(GA.nodes[name]['pdf'][0]['scale'], (int, float))):
                                GA.nodes[name]['pdf'][0]['scale'] = GA.nodes[name]['pdf'][0]['scale']*mult
                            else:
                                raise TypeError('scale value must be a number (integer, float)')

                            if (positiveV) and (GA.nodes[name]['pdf'][0]['loc'] < 0):
                                raise TypeError('a negative "loc" number is not allowed for type time/speed/length')

                            if (positiveV) and (GA.nodes[name]['pdf'][0]['scale'] < 0):
                                raise TypeError('a negative "scale" number is not allowed for type time/speed/length')
                            
                        elif GA.nodes[name]['pdf'][0]['type'] == 'uniform':

                            if (isinstance(GA.nodes[name]['pdf'][0]['loc'], (int, float))):
                                GA.nodes[name]['pdf'][0]['loc'] = GA.nodes[name]['pdf'][0]['loc']*mult
                            else:
                                raise TypeError('loc value must be a number (integer, float)')

                            if (isinstance(GA.nodes[name]['pdf'][0]['scale'], (int, float))):
                                GA.nodes[name]['pdf'][0]['scale'] = GA.nodes[name]['pdf'][0]['scale']*mult
                            else:
                                raise TypeError('scale value must be a number (integer, float)')

                            if (positiveV) and (GA.nodes[name]['pdf'][0]['loc'] < 0):
                                raise TypeError('a negative "loc" number is not allowed for type time/speed/length')

                            if (positiveV) and (GA.nodes[name]['pdf'][0]['scale'] < 0):
                                raise TypeError('a negative "scale" number is not allowed for type time/speed/length')

                        elif GA.nodes[name]['pdf'][0]['type'] == 'cauchy':

                            if (isinstance(GA.nodes[name]['pdf'][0]['loc'], (int, float))):
                                GA.nodes[name]['pdf'][0]['loc'] = GA.nodes[name]['pdf'][0]['loc']
                            else:
                                raise TypeError('loc value must be a number (integer, float)')

                            if (isinstance(GA.nodes[name]['pdf'][0]['scale'], (int, float))):
                                GA.nodes[name]['pdf'][0]['scale'] = GA.nodes[name]['pdf'][0]['scale']
                            else:
                                raise TypeError('scale value must be a number (integer, float)')

                            if (positiveV) and (GA.nodes[name]['pdf'][0]['loc'] < 0):
                                raise TypeError('a negative "loc" number is not allowed for type time/speed/length')

                            if (positiveV) and (GA.nodes[name]['pdf'][0]['scale'] < 0):
                                raise TypeError('a negative "scale" number is not allowed for type time/speed/length')

                        elif GA.nodes[name]['pdf'][0]['type'] == 'expon':

                            if (isinstance(GA.nodes[name]['pdf'][0]['loc'], (int, float))):
                                GA.nodes[name]['pdf'][0]['loc'] = GA.nodes[name]['pdf'][0]['loc']
                            else:
                                raise TypeError('loc value must be a number (integer, float)')

                            if (isinstance(GA.nodes[name]['pdf'][0]['scale'], (int, float))):
                                GA.nodes[name]['pdf'][0]['scale'] = GA.nodes[name]['pdf'][0]['scale']
                            else:
                                raise TypeError('scale value must be a number (integer, float)')

                            if (positiveV) and (GA.nodes[name]['pdf'][0]['loc'] < 0):
                                raise TypeError('a negative "loc" number is not allowed for type time/speed/length')

                            if (positiveV) and (GA.nodes[name]['pdf'][0]['scale'] < 0):
                                raise TypeError('a negative "scale" number is not allowed for type time/speed/length')

                        elif GA.nodes[name]['pdf'][0]['type'] == 'gamma':

                            if (isinstance(GA.nodes[name]['pdf'][0]['loc'], (int, float))):
                                GA.nodes[name]['pdf'][0]['loc'] = GA.nodes[name]['pdf'][0]['loc']
                            else:
                                raise TypeError('loc value must be a number (integer, float)')

                            if (isinstance(GA.nodes[name]['pdf'][0]['scale'], (int, float))):
                                GA.nodes[name]['pdf'][0]['scale'] = GA.nodes[name]['pdf'][0]['scale']
                            else:
                                raise TypeError('scale value must be a number (integer, float)')

                            if (isinstance(GA.nodes[name]['pdf'][0]['aux'], (int, float))):
                                GA.nodes[name]['pdf'][0]['aux'] = GA.nodes[name]['pdf'][0]['aux']
                            else:
                                raise TypeError('a value must be a number (integer, float)')

                            if (positiveV) and (GA.nodes[name]['pdf'][0]['loc'] < 0):
                                raise TypeError('a negative "loc" number is not allowed for type time/speed/length')

                            if (positiveV) and (GA.nodes[name]['pdf'][0]['scale'] < 0):
                                raise TypeError('a negative "scale" number is not allowed for type time/speed/length')

                        elif GA.nodes[name]['pdf'][0]['type'] == 'gilbrat':

                            if (isinstance(GA.nodes[name]['pdf'][0]['loc'], (int, float))):
                                GA.nodes[name]['pdf'][0]['loc'] = GA.nodes[name]['pdf'][0]['loc']
                            else:
                                raise TypeError('loc value must be a number (integer, float)')

                            if (isinstance(GA.nodes[name]['pdf'][0]['scale'], (int, float))):
                                GA.nodes[name]['pdf'][0]['scale'] = GA.nodes[name]['pdf'][0]['scale']
                            else:
                                raise TypeError('scale value must be a number (integer, float)')

                            if (positiveV) and (GA.nodes[name]['pdf'][0]['loc'] < 0):
                                raise TypeError('a negative "loc" number is not allowed for type time/speed/length')

                            if (positiveV) and (GA.nodes[name]['pdf'][0]['scale'] < 0):
                                raise TypeError('a negative "scale" number is not allowed for type time/speed/length')

                        elif GA.nodes[name]['pdf'][0]['type'] == 'lognorm':

                            if (isinstance(GA.nodes[name]['pdf'][0]['loc'], (int, float))):
                                GA.nodes[name]['pdf'][0]['loc'] = GA.nodes[name]['pdf'][0]['loc']
                            else:
                                raise TypeError('loc value must be a number (integer, float)')

                            if (isinstance(GA.nodes[name]['pdf'][0]['scale'], (int, float))):
                                GA.nodes[name]['pdf'][0]['scale'] = GA.nodes[name]['pdf'][0]['scale']
                            else:
                                raise TypeError('scale value must be a number (integer, float)')

                            if (positiveV) and (GA.nodes[name]['pdf'][0]['loc'] < 0):
                                raise TypeError('a negative "loc" number is not allowed for type time/speed/length')

                            if (positiveV) and (GA.nodes[name]['pdf'][0]['scale'] < 0):
                                raise TypeError('a negative "scale" number is not allowed for type time/speed/length')

                        elif GA.nodes[name]['pdf'][0]['type'] == 'powerlaw':

                            if (isinstance(GA.nodes[name]['pdf'][0]['loc'], (int, float))):
                                GA.nodes[name]['pdf'][0]['loc'] = GA.nodes[name]['pdf'][0]['loc']
                            else:
                                raise TypeError('loc value must be a number (integer, float)')

                            if (isinstance(GA.nodes[name]['pdf'][0]['scale'], (int, float))):
                                GA.nodes[name]['pdf'][0]['scale'] = GA.nodes[name]['pdf'][0]['scale']
                            else:
                                raise TypeError('scale value must be a number (integer, float)')

                            if (isinstance(GA.nodes[name]['pdf'][0]['aux'], (int, float))):
                                GA.nodes[name]['pdf'][0]['aux'] = GA.nodes[name]['pdf'][0]['aux']
                            else:
                                raise TypeError('a value must be a number (integer, float)')

                            if (positiveV) and (GA.nodes[name]['pdf'][0]['loc'] < 0):
                                raise TypeError('a negative "loc" number is not allowed for type time/speed/length')

                            if (positiveV) and (GA.nodes[name]['pdf'][0]['scale'] < 0):
                                raise TypeError('a negative "scale" number is not allowed for type time/speed/length')

                        elif GA.nodes[name]['pdf'][0]['type'] == 'wald':

                            if (isinstance(GA.nodes[name]['pdf'][0]['loc'], (int, float))):
                                GA.nodes[name]['pdf'][0]['loc'] = GA.nodes[name]['pdf'][0]['loc']
                            else:
                                raise TypeError('loc value must be a number (integer, float)')

                            if (isinstance(GA.nodes[name]['pdf'][0]['scale'], (int, float))):
                                GA.nodes[name]['pdf'][0]['scale'] = GA.nodes[name]['pdf'][0]['scale']
                            else:
                                raise TypeError('scale value must be a number (integer, float)')

                            if (positiveV) and (GA.nodes[name]['pdf'][0]['loc'] < 0):
                                raise TypeError('a negative "loc" number is not allowed for type time/speed/length')

                            if (positiveV) and (GA.nodes[name]['pdf'][0]['scale'] < 0):
                                raise TypeError('a negative "scale" number is not allowed for type time/speed/length')

                        elif GA.nodes[name]['pdf'][0]['type'] == 'poisson':

                            if (isinstance(GA.nodes[name]['pdf'][0]['a'], (int, float))):
                                GA.nodes[name]['pdf'][0]['a'] = GA.nodes[name]['pdf'][0]['a']
                            else:
                                raise TypeError('"a" (rate of occurances in the time interval for poisson) value must be a number (integer, float)')

                            if (positiveV) and (GA.nodes[name]['pdf'][0]['a'] < 0):
                                raise TypeError('a negative "a" number is not allowed for type time/speed/length')

                        else:
                            raise TypeError('pdf must be cauchy, expon, gamma, gilbrat, lognorm, normal, poisson, powerlaw, uniform, or wald')

                    elif 'expression' in attribute:

                        GA.nodes[name]['expression'] = attribute['expression']

                    if 'constraints' in attribute:

                        GA.nodes[name]['constraints'] = attribute['constraints']

                    if 'dynamism' in attribute:

                        GA.nodes[name]['dynamism'] = attribute['dynamism']

                        #put here values

                    if 'weights' in attribute:

                        if (isinstance(attribute['weights'], (list))): 
                            GA.nodes[name]['weights'] = attribute['weights']
                        else:
                            raise TypeError('weights must be a list')

                        size_all_values = 0

                        if 'pdf' in attribute:
                            
                            if GA.nodes[name]['pdf'][0]['type'] == 'uniform':
                                GA.nodes[name]['all_values'] = list(range(math.ceil(GA.nodes[name]['pdf'][0]['loc']), math.floor(GA.nodes[name]['pdf'][0]['scale'])))
                                #print(GA.nodes[name]['all_values'])
                                size_all_values = len(GA.nodes[name]['all_values'])
                            else: raise ValueError('normal distribution and weights is not allowed')

                        elif 'subset_zones' in attribute:

                            size_all_values = inst.parameters[attribute['subset_zones']]['size']

                        if attribute['weights'][0] == 'randomized_weights':

                            GA.nodes[name]['weights'] = np.random.randint(0, 101, size_all_values)
                            sumall = 0
                            for w in GA.nodes[name]['weights']:
                                sumall += w

                            for w in range(len(GA.nodes[name]['weights'])):
                                GA.nodes[name]['weights'][w] = int((GA.nodes[name]['weights'][w]/sumall)*100)
                          
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


                if 'travel_time_matrix' in inst.parameters:
                    for loc in inst.parameters['travel_time_matrix']['locations']:
                        if loc not in inst.parameters['all_locations']['value']:
                            raise ValueError(str(loc)+ ' is not recognized or does not exists for travel_time_matrix')

            else: raise ValueError('attributes for instance are mandatory')

            if 'method_pois' in data:

                inst.parameters['method_pois'] = {}
                inst.parameters['method_pois']['value'] = data['method_pois'][0]
                inst.parameters['method_pois']['type'] = 'method' 

                print(inst.parameters['method_pois']['value']['locations'])
                if not (isinstance(inst.parameters['method_pois']['value']['locations'], (list))): 
                    raise TypeError('locations from method_pois must be an array')

                #adds an specific dependency between the two nodes
                GA.add_edge(inst.parameters['method_pois']['value']['locations'][0], inst.parameters['method_pois']['value']['locations'][1])

            
            inst.sorted_attributes = list(nx.topological_sort(GA))
            inst.GA = GA
            print(inst.sorted_attributes)

    
            for instance2 in os.listdir(os.path.join(inst.save_dir, 'json_format', base_save_folder_name)):

                '''
                filename_json = inst_directory+instance_filename
                base_filename = filename_json.replace(inst_directory, "")
                inst_base = instance.replace(inst.save_dir+'/json_format/'+base_save_folder_name+'/', "")
                inst_base = inst_base.replace('_1.json', "")
                inst_base = inst_base.replace('_2.json', "")
                inst_base = inst_base.replace('_3.json', "")
                inst_base = inst_base.replace('_4.json', "")
                inst_base = inst_base.replace('_5.json', "")
                inst_base = inst_base.replace('_6.json', "")
                inst_base = inst_base.replace('_7.json', "")
                inst_base = inst_base.replace('_8.json', "")
                inst_base = inst_base.replace('_9.json', "")
                inst_base = inst_base.replace('_10.json', "")
                inst_base = inst_base+'.json'
                '''

                if (instance2 != ".DS_Store"):

                    print(inst.save_dir)
                    print(base_save_folder_name)
                    print(instance2)
                    #instance2 = instance2.decode('utf-8')
                    #print(instance2)
                    input_name = os.path.join(inst.save_dir, 'json_format', base_save_folder_name, instance2)
                    
                    output_name_csv = instance2.split('.json')[0] + '.csv'
                    output_name_csv = output_name_csv.replace(" ", "")

                    converter = JsonConverter(file_name=input_name)
                    converter.convert_normal(inst=inst, problem_type="ODBRP", path_instance_csv_file=os.path.join(save_dir_csv, base_save_folder_name, output_name_csv))
                    
                    #inst1 = pd.read_csv(os.path.join(save_dir_csv, output_name_csv))

                    replicate_number = 1
                    sub_string_rep = "_"+str(replicate_number)+ ".json"

                    not_found = True
                    while not_found:
                        if sub_string_rep in instance2:
                            not_found = False
                        else:
                            replicate_number = replicate_number + 1
                            sub_string_rep = "_"+str(replicate_number)+ ".json"

                    full_final_filename = inst.filename_json.replace(inst_directory, "")
                    full_final_filename = full_final_filename.replace(".json", "")
                    full_final_filename = full_final_filename + '_' + str(replicate_number) + '.csv'


                    #print(instance2)
                    #print(output_name_csv)
                    #print(full_final_filename)

                    #os.rename(os.path.join(save_dir_csv, output_name_csv), os.path.join(save_dir_csv, full_final_filename))
                    replicate_num = replicate_num + 1
