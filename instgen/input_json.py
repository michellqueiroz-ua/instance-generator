import json
import networkx as nx
import numpy as np
import random
import re
import os

from instance_class import Instance
from output_files import JsonConverter
from output_files import output_fixed_route_network
from pathlib import Path
from retrieve_network import download_network_information
from streamlit import caching

def get_multiplier_time_unit(time_unit):

    mult = 1
    if time_unit == 'min':
        mult = 60

    elif time_unit == 'h':
        mult = 3600
    
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
                inst.parameters['min_early_departure'] = j['min_early_departure']*3600
            else: raise ValueError('min_early_departure parameter for planning_horizon is mandatory')

            if 'max_early_departure' in j:
                max_early_departure = j['max_early_departure']
                inst.parameters['max_early_departure'] = j['max_early_departure']*3600
            else: raise ValueError('max_early_departure parameter for planning_horizon is mandatory')

            if 'time_unit' in j:
                time_unit = j['time_unit']
                inst.parameters['time_unit'] = j['time_unit']
            else: raise ValueError('time_unit parameter for planning_horizon is mandatory')

            inst.set_time_window(min_early_departure=min_early_departure, max_early_departure=max_early_departure, time_unit=time_unit)

    if 'parameters' in data:

         for j in data['parameters']:

            if 'name' in j:

                if 'value' in j:
                    
                    mult = 1
                    if 'time_unit' in j:
                        mult = get_multiplier_time_unit(j['time_unit'])
                    
                    inst.parameters[j['name']] = j['value']*mult

                else: raise ValueError('value for a parameter is mandatory')

            else: raise ValueError('name for a parameter is mandatory')

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

                name = attribute['name']
                GA.add_node(name)
                
            else: raise ValueError('name parameter for attribute is mandatory')

            if 'type' in attribute:

                GA.nodes[name]['type'] = attribute['type']

                if attribute['type'] == 'time':

                    if 'time_unit' in attribute:

                        GA.nodes[name]['time_unit'] = attribute['time_unit']

                    else: 

                        GA.nodes[name]['time_unit'] = 's'

            else: raise ValueError('type parameter for attribute is mandatory')

            if 'subset_zones' in attribute:

                GA.nodes[name]['subset_zones'] = attribute['subset_zones']
                GA.nodes[name]['zones'] = []
                #pegar aqui specific zones

            if 'subset_schools' in attribute:

                GA.nodes[name]['subset_schools'] = attribute['subset_schools']
                inst.GA.nodes[att]['schools'] = []
                #pegar aqui specific schools

            if 'pdf' in attribute:

                GA.nodes[name]['pdf'] = attribute['pdf']

                mult = 1
                if GA.nodes[name]['pdf'][0]['time_unit'] == 'min':
                    mult = 60

                elif GA.nodes[name]['pdf'][0]['time_unit'] == 'h':
                    mult = 3600

                if GA.nodes[name]['pdf'][0]['type'] == 'normal':

                    GA.nodes[name]['pdf'][0]['mean'] = GA.nodes[name]['pdf'][0]['mean']*mult
                    GA.nodes[name]['pdf'][0]['std'] = GA.nodes[name]['pdf'][0]['std']*mult

                elif GA.nodes[name]['pdf'][0]['type'] == "uniform":

                    GA.nodes[name]['pdf'][0]['max'] = GA.nodes[name]['pdf'][0]['max']*mult
                    GA.nodes[name]['pdf'][0]['min'] = GA.nodes[name]['pdf'][0]['min']*mult

            elif 'expression' in attribute:

                GA.nodes[name]['expression'] = attribute['expression']

            if 'constraints' in attribute:

                GA.nodes[name]['constraints'] = attribute['constraints']

            if 'static_probability' in attribute:

                GA.nodes[name]['static_probability'] = attribute['static_probability']


        for node in GA.nodes():

            if 'expression' in GA.nodes[node]:
                expression = re.split(r"[(,) ]", GA.nodes[node]['expression'])

                for exp in expression:
                    if exp in GA:
                        GA.add_edge(exp, node)

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

    else: raise ValueError('attributes for instance are mandatory')


    inst.generate_requests()

    caching.clear_cache()
        
    # convert instances from json to cpp and localsolver formats
    save_dir_cpp = os.path.join(inst.save_dir, 'cpp_format')
    if not os.path.isdir(save_dir_cpp):
        os.mkdir(save_dir_cpp)

    save_dir_localsolver = os.path.join(inst.save_dir, 'localsolver_format')
    if not os.path.isdir(save_dir_localsolver):
        os.mkdir(save_dir_localsolver)

    for instance in os.listdir(os.path.join(inst.save_dir, 'json_format')):
        
        if instance != ".DS_Store":
            input_name = os.path.join(inst.save_dir, 'json_format', instance)
            
            output_name_cpp = instance.split('.')[0] + '_cpp.pass'
            output_name_cpp = output_name_cpp.replace(" ", "")
            
            output_name_ls = instance.split('.')[0] + '_ls.pass'

            converter = JsonConverter(file_name=input_name)
            converter.convert_normal(output_file_name=os.path.join(save_dir_cpp, output_name_cpp), network=inst.network, problem_type=inst.problem_type)
            #converter.convert_localsolver(output_file_name=os.path.join(save_dir_localsolver, output_name_ls))














                




    f.close()