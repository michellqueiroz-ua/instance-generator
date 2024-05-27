import os
import json
from instance_class import Instance
from output_files import JsonConverter
import networkx as nx
import pandas as pd

if __name__ == '__main__':

    place_name = "Chicago, Illinois"
    base_save_folder_name = 'uneven_demand' #give a unique folder name to save the instances
    inst_directory = '../examples/uneven_demand/'
    directory = os.fsencode(inst_directory)
    inst = Instance(folder_to_network=place_name)

    save_dir_csv = os.path.join(inst.save_dir, 'csv_format')
    if not os.path.isdir(save_dir_csv):
        os.mkdir(save_dir_csv)

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

        instance_filename = os.fsdecode(instance)
        instance_filename = instance_filename.replace("._", "")

        filename_json = inst_directory+instance_filename
        #f = open(filename_json,)
        with open(filename_json, 'rb') as f:   # will close() when we leave this block
            data = json.load(f)
        
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

    
    #for instance in os.listdir(os.path.join(inst.save_dir, 'json_format', base_save_folder_name)):

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

        if (instance != ".DS_Store"):

            input_name = os.path.join(inst.save_dir, 'json_format', base_save_folder_name, instance)
            
            output_name_csv = instance.split('.json')[0] + '.csv'
            output_name_csv = output_name_csv.replace(" ", "")

            converter = JsonConverter(file_name=input_name)
            converter.convert_normal(inst=inst, problem_type="ODBRP", path_instance_csv_file=os.path.join(save_dir_csv, base_save_folder_name, output_name_csv))
            
            inst1 = pd.read_csv(os.path.join(save_dir_csv, output_name_csv))


            full_final_filename = inst.filename_json.replace(inst_directory, "")
            full_final_filename = full_final_filename.replace(".json", "")
            full_final_filename = full_final_filename + '_' + str(replicate_num) + '.csv'


            print(instance)
            print(output_name_csv)
            print(full_final_filename)

            os.rename(os.path.join(save_dir_csv, output_name_csv), os.path.join(save_dir_csv, full_final_filename))
            replicate_num = replicate_num + 1
