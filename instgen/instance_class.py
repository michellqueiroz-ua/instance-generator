import math
import osmapi as osm
import os
import osmnx as ox
import networkx as nx
import numpy as np
import pickle
import random
from request_distribution_class import RequestDistributionTime
from spatial_distribution_class import SpatialDistribution
from shapely.geometry import Point
from passenger_requests import _generate_requests
from passenger_requests import _generate_requests_DARP
from passenger_requests import _generate_requests_ODBRP
from passenger_requests import _generate_requests_ODBRPFL
from passenger_requests import _generate_requests_SBRP
import ray
import gc
from multiprocessing import cpu_count

 
class Instance:

    def __init__(self, folder_to_network):

        self.folder_to_network = folder_to_network
        self.output_folder_base = self.folder_to_network

        self.save_dir = os.getcwd()+'/'+self.output_folder_base
        self.pickle_dir = os.path.join(self.save_dir, 'pickle')
        self.save_dir_json = os.path.join(self.save_dir, 'json_format')
        self.save_dir_graphml = os.path.join(self.save_dir, 'graphml_format')
        self.save_dir_ttm = os.path.join(self.save_dir, 'travel_time_matrix')
        
        self.network_class_file = self.pickle_dir+'/'+self.output_folder_base+'.network2.class.pkl'

        with open(self.network_class_file, 'rb') as self.network_class_file:
            self.network = pickle.load(self.network_class_file)

        #problem for which the instance is being created
        self.problem_type = None

        self.request_demand = []

        self.spatial_distribution = []

        #self.origin_weights = [0] * len(self.network.zones)

        #self.destination_weights = [0] * len(self.network.zones)

        '''
        self.num_origins = -1
        self.num_destinations = -1
        self.is_random_origin_zones = True
        self.is_random_destination_zones = True
        self.origin_zones = []
        self.destination_zones = []
        '''

        #time window
        self.min_early_departure = None
        self.max_early_departure = None

        #interval walk speed
        self.min_walk_speed = None
        self.max_walk_speed = None

        #interval of maximum walking threshold of the user
        self.lb_max_walking = None
        self.ub_max_walking = None

        #interval of lead time
        self.min_lead_time = None
        self.max_lead_time = None

        #number of replicas of the instance with randomized characteristics 
        self.number_replicas = None

        #num depots DARP
        self.num_depots = 1
        self.depot_nodes_drive = []
        self.depot_nodes_walk = []
        self.can_set_random_depot = True
        self.can_set_address_depot = True

        #school id in case of SBRP
        self.num_schools = 1
        self.school_ids = []
        self.can_set_random_school = True
        self.can_set_address_school = True
        #self.school_station = None

        #factor to compute delay travel time by the vehicle
        self.delay_vehicle_factor = None

        #probability of each request having a return
        self.return_factor = None

        #vehicle requirements for DARP problem
        self.wheelchair = False
        self.ambulatory = False

        self.seed = 0
        self.increment_seed = 1

        self.parameters = {}


    def add_request_demand_uniform(self, 
        min_time, 
        max_time, 
        number_of_requests, 
        time_unit
):

        '''
        add request demand that sample earliest departure time/latest arrival time using uniform distribution
        '''
        if time_unit == "s":
            min_time = int(min_time)
            max_time = int(max_time)
        
        elif time_unit == "h":
            min_time = int(min_time*3600)
            max_time = int(max_time*3600)

        elif time_unit == "min":
            min_time = int(min_time*60)
            max_time = int(max_time*60)

        else: raise ValueError('time_unit method argument must be either "h" or "min", or "s"')

        dnd = RequestDistributionTime(min_time, max_time, number_of_requests, "uniform")
        self.request_demand.append(dnd)

    def add_request_demand_normal(self, 
        mean, 
        std, 
        number_of_requests, 
        time_unit
):

        '''
        add request demand that sample earliest departure time/latest arrival time using normal distribution
        '''
        if time_unit == "s":
            mean = int(mean)
            std = int(std)
        
        elif time_unit == "h":
            mean = int(mean*3600)
            std = int(std*3600)

        elif time_unit == "min":
            mean = int(mean*60)
            std = int(std*60)

        else: raise ValueError('time_unit method argument must be either "h" or "min", or "s"')

        dnd = RequestDistributionTime(mean, std, number_of_requests, "normal")
        self.request_demand.append(dnd)

    def add_spatial_distribution(self, num_origins, num_destinations, prob, origin_zones=[], destination_zones=[], is_random_origin_zones=False, is_random_destination_zones=False):

        sd = SpatialDistribution(num_origins, num_destinations, prob, origin_zones, destination_zones, is_random_origin_zones, is_random_destination_zones)
        self.spatial_distribution.append(sd)

    '''
    def set_number_origins(self, num_origins):

        self.num_origins = int(num_origins)

    def set_number_destinations(self, num_destinations):

        self.num_destinations = int(num_destinations)

    def add_origin_zone(self, zone_id):

        self.is_random_origin_zones = False
        self.origin_zones.append(int(zone_id))

    def add_destination_zone(self, zone_id):

        self.is_random_destination_zones = False
        self.destination_zones.append(int(zone_id))
    
    def randomly_sample_origin_zones(self, num_zones):

        self.origin_zones = []

        if self.num_origins != -1:
            self.origin_zones = np.random.randint(0, num_zones, self.num_origins)

    def randomly_sample_destination_zones(self, num_zones):

        self.destination_zones = []

        if self.num_destinations != -1:
            self.destination_zones = np.random.randint(0, num_zones, self.num_destinations)
    '''

    def set_seed(self, seed, increment_seed):

        self.seed = seed
        self.increment_seed = increment_seed

    def set_time_window(self, min_early_departure, max_early_departure, time_unit):

        if time_unit == "s":
            self.min_early_departure = int(min_early_departure)
            self.max_early_departure = int(max_early_departure)

        elif time_unit == "h":
            self.min_early_departure = int(min_early_departure*3600)
            self.max_early_departure = int(max_early_departure*3600)

        elif time_unit == "min":
            self.min_early_departure = int(min_early_departure*60)
            self.max_early_departure = int(max_early_departure*60)

        else: raise ValueError('time_unit method argument must be either "h" or "min", or "s"')

    def set_interval_lead_time(self, min_lead_time, max_lead_time, time_unit):
    
        if time_unit == "s":
            self.min_lead_time = int(min_lead_time)
            self.max_lead_time = int(max_lead_time)

        elif time_unit == "h":
            self.min_lead_time = int(min_lead_time*3600)
            self.max_lead_time = int(max_lead_time*3600)

        elif time_unit == "min":
            self.min_lead_time = int(min_lead_time*60)
            self.max_lead_time = int(max_lead_time*60)

        else: raise ValueError('time_unit method argument must be either "h" or "min", or "s"')

    def set_interval_walk_speed(self, min_walk_speed, max_walk_speed, speed_unit):

        '''
        set the walking speed considering during computation of travel times
        value is randomized for each user
        '''


        if speed_unit == "mps":
            self.min_walk_speed = float(min_walk_speed)
            
        elif speed_unit == "kmh":
            self.min_walk_speed = float(min_walk_speed/3.6)

        elif speed_unit == "mph":
            self.min_walk_speed = float(min_walk_speed/2.237)

        else: raise ValueError('speed_unit method argument must be either "kmh", "mph" or "mps"')

        if speed_unit == "mps":
            self.max_walk_speed = float(max_walk_speed)
            
        elif speed_unit == "kmh":
            self.max_walk_speed = float(max_walk_speed/3.6)

        elif speed_unit == "mph":
            self.max_walk_speed = float(max_walk_speed/2.237)

        else: raise ValueError('speed_unit method argument must be either "kmh", "mph" or "mps"')

    def set_interval_max_walking(self, lb_max_walking, ub_max_walking, time_unit):

        '''
        interval for max desired walk by the user
        lb - lower bound
        ub - upper bound
        '''

        if time_unit == "s":
            self.lb_max_walking = int(lb_max_walking)

        elif time_unit == "min":
            self.lb_max_walking = int(lb_max_walking*60)

        elif time_unit == "h":
            self.lb_max_walking = int(lb_max_walking*3600)

        else: raise ValueError('time_unit method argument must be either "h" or "min", or "s"') 

        if time_unit == "s":
            self.ub_max_walking = int(ub_max_walking)

        elif time_unit == "min":
            self.ub_max_walking = int(ub_max_walking*60)

        elif time_unit == "h":
            self.ub_max_walking = int(ub_max_walking*3600)

        else: raise ValueError('time_unit method argument must be either "h" or "min", or "s"')

    def add_time_window_gap(self, g, time_unit):

        if time_unit == "s":
            self.g = int(g) 

        elif time_unit == "min":
            self.g = int(g*60)

        elif time_unit == "h":
            self.g = int(g*3600)

        else: raise ValueError('time_unit method argument must be either "h" or "min", or "s"')

    def set_problem_type(self, problem_type, school_id=None):

        if (problem_type == "ODBRP") or (problem_type == "ODBRPFL") or (problem_type == "SBRP") or (problem_type == "DARP"):
            self.problem_type = problem_type

        else: raise ValueError('problem_type method argument must be either "ODBRP",  "ODBRPFL", "DARP" or "SBRP"') 

        '''
        if problem_type == "SBRP":
            self.school_id = school_id
            
            if school_id is None:
                raise ValueError('problem SBRP requires school as parameter. please provide school ID')
        '''

    def set_num_schools(self, num_schools):

        if self.can_set_random_school:
            self.num_schools = num_schools
            self.can_set_address_school = False
        else: raise ValueError('this function can not be called after adding school from address')

    def set_num_depots(self, num_depots):

        if self.can_set_random_depot:
            self.num_depots = num_depots
            self.can_set_address_depot = False
        else: raise ValueError('this function can not be called after adding depot from address')

    def add_school_from_name(self, school_name):

        if self.can_set_address_school:
            school = self.network.schools[self.network.schools['school_name'] == school_name]

            if school is not None:
              
                self.school_ids.append(int(school.index.values))

                self.num_schools = len(self.school_ids)
                self.can_set_random_school = False

        else: raise ValueError('this function can not be called after setting the number of schools with set_num_schools')

    def add_school_from_address(self, address_school, school_name):

        #catch erro de não existir o lugar

        if self.can_set_address_school:
            school_point = ox.geocoder.geocode(query=address_school)
            school_node_drive = ox.get_nearest_node(self.network.G_drive, school_point)
            school_node_walk = ox.get_nearest_node(self.network.G_walk, school_point)

            lid = self.network.schools.last_valid_index()
            lid += 1

            self.network.schools.loc[lid] = [school_name, school_node_walk, school_node_drive, school_point[0], school_point[1]]
            self.school_ids.append(lid)

            self.can_set_random_school = False
            self.num_schools = len(self.school_ids)

        else: raise ValueError('this function can not be called after setting the number of schools with set_num_schools')

    def add_depot_from_address(self, depot_address):

        if self.can_set_address_depot:
            depot_point = ox.geocoder.geocode(query=depot_address)
            depot_node_drive = ox.get_nearest_node(self.network.G_drive, depot_point)
            depot_node_walk = ox.get_nearest_node(self.network.G_walk, depot_point)

            self.depot_nodes_drive.append(depot_node_drive)
            self.depot_nodes_walk.append(depot_node_walk)

            self.num_depots = len(self.depot_nodes_drive)
            self.can_set_random_depot = False

        else: raise ValueError('this function can not be called after setting the number of depots with set_num_depots')

    def set_number_replicas(self, number_replicas):

        self.number_replicas = int(number_replicas)

    def set_delay_vehicle_factor(self, delay_vehicle_factor):

        self.delay_vehicle_factor = float(delay_vehicle_factor)

    def set_delay_walk_factor(self, delay_walk_factor):

        self.delay_walk_factor = float(delay_walk_factor)

    def set_return_factor(self, return_factor):

        self.return_factor = float(return_factor)

    def add_vehicle_requirements(self, req):

        if req == "wheelchair":
            self.wheelchair = True

        if req == "ambulatory":
            self.ambulatory = True

    def generate_requests(self):

        for replicate_num in range(self.number_replicas):
            
            self.seed += self.increment_seed

            if self.problem_type == "DARP":
                _generate_requests(self, replicate_num)  

            elif self.problem_type == "ODBRP":
                _generate_requests_ODBRP(self, replicate_num) 

            elif self.problem_type == "ODBRPFL":
                _generate_requests_ODBRPFL(self, replicate_num) 

            elif self.problem_type == "SBRP":
                _generate_requests_SBRP(self, replicate_num)

            else: _generate_requests(self, replicate_num)

    def _assign_weights_subset_zones(self, attribute, num_zones, cum_weigth):

        id_zones = np.random.randint(0, len(self.zones), num_zones)

        sweights = np.random.uniform(0, 1, num_zones-1)
        sweights.append(0)
        sweights.append(1)
        sweights.sort()

        weights = []
        for w in range(len(sweights-1)):

            diff = sweights[w+1] - sweights[w]
            diff *= cum_weigth
            weights.append(diff)

        random.shuffle(weights)

        count = 0
        for idz in id_zones:

            if attribute == 'origin':
                self.network.zones.loc[int(idz), 'origin_weigth'] = weights[count]

            if attribute == 'destination':
                self.network.zones.loc[int(idz), 'destination_weigth'] = weights[count]

            count += 1

    def _assignt_weights_grid(self, attribute):

        tot_weigth = 0
        left_zones = 0
        for index, row in self.network.zones.iterrows():

            if attribute == 'origin':
                tot_weigth += row['origin_weigth']

                if (int(row['origin_weigth']) == 0):
                    left_zones += 1

            if attribute == 'destination':
                tot_weigth += row['destination_weigth']

                if (int(row['destination_weigth']) == 0):
                    left_zones += 1

        cum_weigth = 100 - tot_weigth

        num_zones = left_zones

        if (cum_weigth > 0):

            sweights = np.random.uniform(0, 1, num_zones-1)
            sweights.append(0)
            sweights.append(1)
            sweights.sort()

            weights = []
            for w in range(len(sweights-1)):

                diff = sweights[w+1] - sweights[w]
                diff *= cum_weigth
                weights.append(diff)

            random.shuffle(weights)

            count = 0
            for index, row in self.network.zones.iterrows():

                if attribute == 'origin':

                    if (int(row['origin_weigth']) == 0):
                       self.network.zones.loc[int(index), 'origin_weigth'] = weights[count]
                       count += 1

                if attribute == 'destination':

                    if (int(row['destination_weigth']) == 0):
                        self.network.zones.loc[int(index), 'destination_weigth'] = weights[count]
                        count += 1


        sumw = 0
        for index, row in self.network.zones.iterrows():

            if attribute == 'origin':

                sumw += row['origin_weigth']

            if attribute == 'destination':

                sumw += row['destination_weigth']


        print(sumw)
        sumw2 = 0
        if sumw > 100:

            for index, row in self.network.zones.iterrows():

                if attribute == 'origin':

                    new_weigth = (self.network.zones.loc[int(index), 'origin_weigth']/sumw)*100
                    self.network.zones.loc[int(index), 'origin_weigth'] = new_weigth
                    sumw2 += new_weigth

                if attribute == 'destination':

                    new_weigth = (self.network.zones.loc[int(index), 'destination_weigth']/sumw)*100
                    self.network.zones.loc[int(index), 'destination_weigth'] = new_weigth
                    sumw2 += new_weigth

            print(sumw2)

    '''
    sets the interval of walking time (in units of time), that the user is willing to walk to reach a pre defined location, such as bus stations
    value is randomized for each user
    '''

    @ray.remote
    def _generate_single_data(self):
    
        while True:
            
            attributes = {}
            feasible_data = True

            for att in self.sorted_attributes:

                not_feasible_attribute = True
                exhaustion_iterations = 0

                while (not_feasible_attribute) and (exhaustion_iterations < 100):

                    if self.GA.nodes[att]['type'] == 'location':

                        random_zone_id = -1

                        if 'subset_zones' in GA.nodes[att]:

                            zone = self.GA.nodes[att]['subset_zones']

                            if zone is False:

                                if 'rank_model' in self.GA.nodes[att]:

                                    type_coord = self.GA.nodes[att]['rank_model']
                                    zones = self.network.zones.index.tolist()
                                    

                                    if type_coord == 'destination':
                                        
                                        probabilities = self.network.zones['density_pois'].tolist()
                                        random_zone_id = np.random.choice(a=zones, size=1, p=probabilities)
                                        
                                        random_zone_id = random_zone_id[0]
                                        polygon_zone = self.network.zones.loc[random_zone_id]['polygon']
                                        point = self.network._get_random_coord(polygon_zone)

                                    elif type_coord == 'origin':
                                        
                                        att_start = self.GA.nodes[att]['start_point']
                                        zone_start = int(attributes[att_start+'zone'])
                                        probabilities = self.network.zone_probabilities[zone_start].tolist()
                                        #print(inst.network.zone_probabilities[zone_start].head())
                                        #print(probabilities)
                                        random_zone_id = np.random.choice(a=zones, size=1, p=probabilities)

                                        random_zone_id = random_zone_id[0]
                                        polygon_zone = self. network.zones.loc[random_zone_id]['polygon']
                                        point = self.network._get_random_coord(polygon_zone)

                                else:

                                    point = self.network._get_random_coord(network.polygon)
                                
                                #print(point)
                                point = (point.y, point.x)

                            else:

                                zones = self.parameters[zone]['zones'+str(replicate_num)]

                                if 'weights' in GA.nodes[att]:

                                    random_zone_id = random.choices(zones, weights=GA.nodes[att]['weights'], k=1)
                                    random_zone_id = random_zone_id[0]
                                    polygon_zone = self.network.zones.loc[random_zone_id]['polygon']

                                else:

                                    random_zone_id = random.choice(zones)
                                    polygon_zone = self.network.zones.loc[random_zone_id]['polygon']
                                
                                try:

                                    if math.isnan(polygon_zone):
                                        R = self.network.zones.loc[random_zone_id]['radius']
                                        clat = self.network.zones.loc[random_zone_id]['center_y']
                                        clon = self.network.zones.loc[random_zone_id]['center_x']
                                        point = self.network._get_random_coord_circle(R, clat, clon)
                                        
                                except TypeError:
                                
                                    point = self.network._get_random_coord(polygon_zone)                                

                                point = (point.y, point.x)
                        else:

                            if 'subset_locations' in self.GA.nodes[att]:

                                loc = self.GA.nodes[att]['subset_locations']
                                
                                if 'weights' in self.GA.nodes[att]:

                                    if self.parameters[loc]['locs'] == 'schools':
                                        idxs = random.choices(self.parameters[loc]['list_ids'+str(replicate_num)], weights=GA.nodes[att]['weights'], k=1)
                                        point = (self.network.schools.loc[idxs[0], 'lat'], self.network.schools.loc[idxs[0], 'lon'])
                                else:

                                    if self.parameters[loc]['locs'] == 'schools':
                                        idxs = random.choice(self.parameters[loc]['list_ids'+str(replicate_num)])
                                        point = (self.network.schools.loc[idxs, 'lat'], self.network.schools.loc[idxs, 'lon'])


                        attributes[att] = point
                        attributes[att+'x'] = point[1]
                        attributes[att+'y'] = point[0]
                        
                        node_drive = ox.get_nearest_node(self.network.G_drive, point)
                        node_walk = ox.get_nearest_node(self.network.G_walk, point)

                        attributes[att+'node_drive'] = int(node_drive)
                        attributes[att+'node_walk'] = int(node_walk)

                        attributes[att+'zone'] = int(random_zone_id)

                    
                    if 'pdf' in self.GA.nodes[att]:

                        if self.GA.nodes[att]['pdf'][0]['type'] == 'normal':

                            attributes[att] = np.random.normal(GA.nodes[att]['pdf'][0]['mean'], GA.nodes[att]['pdf'][0]['std'])
                            
                            if (self.GA.nodes[att]['type'] == 'time') or (self.GA.nodes[att]['type'] == 'integer'):
                                attributes[att] = int(attributes[att])


                        if self.GA.nodes[att]['pdf'][0]['type'] == 'poisson':

                            attributes[att] = np.random.poisson(GA.nodes[att]['pdf'][0]['lam'])
                            
                            if (GA.nodes[att]['type'] == 'time') or (GA.nodes[att]['type'] == 'integer'):
                                attributes[att] = int(attributes[att])

                            
                            #print(attributes[att])

                        if self.GA.nodes[att]['pdf'][0]['type'] == 'uniform':


                            if 'weights' in self.GA.nodes[att]:

                                attributes[att] = random.choices(GA.nodes[att]['all_values'], weights=GA.nodes[att]['weights'], k=1)
                                attributes[att] = attributes[att][0]
                                #print(attributes[att])

                            else:
                                if (self.GA.nodes[att]['type'] == 'time') or (self.GA.nodes[att]['type'] == 'integer'):
                                    attributes[att] = np.random.randint(self.GA.nodes[att]['pdf'][0]['min'], self.GA.nodes[att]['pdf'][0]['max'])
                                else:
                                    attributes[att] = np.random.uniform(self.GA.nodes[att]['pdf'][0]['min'], self.GA.nodes[att]['pdf'][0]['max'])

                                #print(inst.GA.nodes[att]['pdf'][0]['min'], inst.GA.nodes[att]['pdf'][0]['max'])
                                #if att == 'ambulatory':
                                    #print(attributes[att])

                    elif 'expression' in self.GA.nodes[att]:

                        expression = self.GA.nodes[att]['expression']

                        if att == 'time_stamp':
                            static = np.random.uniform(0, 1)
                            #print(static)
                            if static < self.GA.nodes[att]['static_probability']:
                                expression = '0'
                        
                        for attx in self.sorted_attributes:

                            if attx in attributes:
                                expression = re.sub(attx, str(attributes[attx]), expression)                        
                        
                        #print(expression)

                        try:
                            
                            #attributes[att] = eval(expression)
                            #print(expression)
                            attributes[att] = eval_expression(expression)
                            #print(attributes[att])

                            if (self.GA.nodes[att]['type'] == 'time') or (self.GA.nodes[att]['type'] == 'integer'):
                                attributes[att] = int(attributes[att])
                            #if att == 'time_stamp':
                            #   print(attributes[att])
                        
                        except (SyntaxError, NameError, ValueError, TypeError):

                            #print(inst.GA.nodes[att]['expression'])
                            
                            expression = re.split(r"[(,) ]", self.GA.nodes[att]['expression'])
                            
                            if expression[0] == 'dtt':

                                if (expression[1] in attributes) and (expression[2] in attributes):
                                    node_drive1 = attributes[expression[1]+'node_drive']
                                    node_drive2 = attributes[expression[2]+'node_drive']
                                else:
                                    raise ValueError('expression '+self.GA.nodes[att]['expression']+' not possible to evaluate. check the parameters')

                                attributes[att] = self.network._return_estimated_travel_time_drive(node_drive1, node_drive2)

                            elif expression[0] == 'stops':

                                stops = []
                                stops_walking_distance = []

                                if (expression[1] in attributes):
                                    node_walk = attributes[expression[1]+'node_walk']
                                else:
                                    raise ValueError('expression '+self.GA.nodes[att]['expression']+' not possible to evaluate. check the parameters')
                                

                                for index in self.network.bus_stations_ids:
                        
                                    osmid_possible_stop = int(self.network.bus_stations.loc[index, 'osmid_walk'])

                                    eta_walk = self.network.get_eta_walk(node_walk, osmid_possible_stop, attributes['walk_speed'])
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
                                    raise ValueError('expression '+self.GA.nodes[att]['expression']+' not possible to evaluate. check the parameters')
                                
                                attributes[att] = self.network.get_eta_walk(node_walk1, node_walk2, attributes['walk_speed'])

                            elif expression[0] == 'dist_drive':

                                
                                if (expression[1] in attributes) and (expression[2] in attributes):
                                    node_drive1 = attributes[expression[1]+'node_drive']
                                    node_drive2 = attributes[expression[2]+'node_drive']
                                else:
                                    raise ValueError('expression '+self.GA.nodes[att]['expression']+' not possible to evaluate. check the parameters')

                                attributes[att] = self.network._return_estimated_distance_drive(node_drive1, node_drive2)

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
                                raise SyntaxError('expression '+self.GA.nodes[att]['expression']+' is not supported')
                    '''
                    if att == 'direct_distance':         
                        if not_feasible_attribute:
                    
                            feasible_data = False
                            break 
                    ''' 

                    #check constraints

                    if 'constraints' in self.GA.nodes[att]:

                        not_feasible_attribute = False

                        for constraint in self.GA.nodes[att]['constraints']:

                            #print(constraint)

                            for attx in self.sorted_attributes:
                                if attx in attributes:
                                    constraint = re.sub(attx, str(attributes[attx]), constraint) 

                            for paramx in parameters:
                                if 'value' in self.parameters[paramx]:
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
                return attributes


    def _generate_full_requests( 
        self,
        replicate_num,
    ):
        
        print("Now generating " + " request_data X")


        origin_points=[]
        destination_points=[]
        h = 0
        num_requests = 0

        node_list = []
        node_list_seq = []
        self.depot_nodes_seq = []

        if replicate_num == 0:
            self.network.zones['density_pois'] = self.network.zones['density_pois']/100

        #distances = []
        #distancesx = gilbrat.rvs(loc=456.87, scale=4646.72, size=40000)

        #distances = [x for x in distancesx if x > 500]
        #print(distances)
        #print(len(distances))

        #print(self.network.zones['density_pois'].sum())

        #for idx, zone in self.network.zone_ranks.iterrows():

            #normsum = self.network.zone_ranks.loc[idx].sum()

            #self.network.zone_ranks.loc[idx] = self.network.zone_ranks.loc[idx]/normsum

            #print(self.network.zone_ranks.loc[idx].sum())

        for param in self.parameters:

            if self.parameters[param]['type'] == 'array_locations':

                if self.parameters[param]['locs'] == 'random':

                    self.parameters[param]['list'+str(replicate_num)] = []
                    for elem in self.parameters[param]['list']:
                        self.parameters[param]['list'+str(replicate_num)].append(elem)

                    self.parameters[param]['list_node_drive'+str(replicate_num)] = []    
                    for elem in self.parameters[param]['list_node_drive']:
                        self.parameters[param]['list_node_drive'+str(replicate_num)].append(elem)

                    while len(self.parameters[param]['list'+str(replicate_num)]) < self.parameters[param]['size']:

                        point = self.network._get_random_coord(self.network.polygon)
                        point = (point.y, point.x)
                        node_drive = ox.get_nearest_node(self.network.G_drive, point)
                        if node_drive not in self.parameters[param]['list_node_drive'+str(replicate_num)]:
                            self.parameters[param]['list'+str(replicate_num)].append("random_loc"+len(self.parameters[param]['list'+str(replicate_num)]))
                            self.parameters[param]['list_node_drive'+str(replicate_num)].append(node_drive)

                    #print(self.parameters[param]['list_node_drive'+str(replicate_num)])

                if self.parameters[param]['locs'] == 'schools':

                    self.parameters[param]['list_ids'+str(replicate_num)] = []
                    for elem in self.parameters[param]['list_ids']:
                        self.parameters[param]['list_ids'+str(replicate_num)].append(elem)

                    self.parameters[param]['list_node_drive'+str(replicate_num)] = []    
                    for elem in self.parameters[param]['list_node_drive']:
                        self.parameters[param]['list_node_drive'+str(replicate_num)].append(elem)

                    while len(self.parameters[param]['list_ids'+str(replicate_num)]) < self.parameters[param]['size']:

                        random_school_id = np.random.randint(0, len(self.network.schools))
                        random_school_id = int(random_school_id)

                        if random_school_id not in self.parameters[param]['list_ids'+str(replicate_num)]:
                            self.parameters[param]['list_ids'+str(replicate_num)].append(random_school_id)
                            node_drive = self.network.schools.loc[int(random_school_id), 'osmid_drive']
                            self.parameters[param]['list_node_drive'+str(replicate_num)].append(node_drive)

                    #print(self.parameters[param]['list_ids'+str(replicate_num)])

            if self.parameters[param]['type'] == 'array_zones':

                self.parameters[param]['zones'+str(replicate_num)] = []
                for elem in self.parameters[param]['zones']:
                    self.parameters[param]['zones'+str(replicate_num)].append(elem)

                while len(self.parameters[param]['zones'+str(replicate_num)]) < self.parameters[param]['size']: 
                
                    random_zone_id = np.random.randint(0, len(self.network.zones))
                    random_zone_id = int(random_zone_id)

                    if random_zone_id not in self.parameters[param]['zones'+str(replicate_num)]:
                        self.parameters[param]['zones'+str(replicate_num)].append(random_zone_id)

                #print(self.parameters[param]['zones'+str(replicate_num)])               

        num_requests = self.parameters['records']['value']
        print(num_requests)
        instance_data = {}
        all_instance_data = {}  

        gc.collect()
        ray.shutdown()
        ray.init(num_cpus=cpu_count())


        #GA_id = ray.put(self.GA)
        #network_id = ray.put(self.network)
        #sorted_attributes_id = ray.put(self.sorted_attributes)
        #parameters_id = ray.put(self.parameters)
        all_reqs = ray.get([self._generate_single_data.remote() for i in range(num_requests)]) 

        #del GA_id
        #del network_id
        #del sorted_attributes_id
        #del parameters_id
        #gc.collect()

        i = 0
        for req in all_reqs:

            instance_data.update({i: req})
            i += 1    
                  

        #count, bins, ignored = plt.hist(ps, 30, density=True)
        #plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')    
        #plt.show()
        all_instance_data.update({'num_data:': len(instance_data),
                              'requests': instance_data
                              })


        final_filename = ''
        #print(self.instance_filename)
        for p in self.instance_filename:

            if p in self.parameters:
                if 'value' in self.parameters[p]:
                    strv = str(self.parameters[p]['value'])
                    strv = strv.replace(" ", "")

                    if len(final_filename) > 0:
                        final_filename = final_filename + '_' + strv
                    else: final_filename = strv

        #print(final_filename)

        if 'travel_time_matrix' in self.parameters:
            if self.parameters['travel_time_matrix']['value'] is True:
                node_list = []
                node_list_seq = []
                lvid = 0

                for location in self.parameters['travel_time_matrix']['locations']:

                    if location == 'bus_stations':

                        for index, row in self.network.bus_stations.iterrows():

                            node_list.append(row['osmid_drive'])
                            node_list_seq.append(index)
                            lvid = index

                    if location in self.parameters:

                        self.parameters[location]['list_seq_id'+str(replicate_num)] = []
                        #print(self.parameters[location]['list_node_drive'+str(replicate_num)])
                        for d in self.parameters[location]['list_node_drive'+str(replicate_num)]:

                            node_list.append(d)
                            node_list_seq.append(lvid)
                            self.parameters[location]['list_seq_id'+str(replicate_num)].append(lvid)
                            lvid += 1

                    if location in self.sorted_attributes:

                        for d in instance_data:

                            node_list.append(instance_data[d][location+'node_drive'])
                            instance_data[d][location+'id'] = lvid
                            node_list_seq.append(lvid)
                            lvid += 1

                if replicate_num < 0:
                    print('ttm')
                    travel_time = self.network._get_travel_time_matrix("list", node_list=node_list)
                    travel_time_json = travel_time.tolist()

                    all_instance_data.update({'travel_time_matrix': travel_time_json
                                    })
                    
                    print('leave ttm')          
                    if 'graphml' in self.parameters:
                        if self.parameters['graphml']['value'] is True:
                            #creates a graph that will serve as the travel time matrix for the given set of requests
                            gtt = nx.DiGraph() 

                            if 'bus_stations' in self.parameters['travel_time_matrix']['locations']:

                                for index, row in self.network.bus_stations.iterrows():

                                    gtt.add_node(index, type='bus_stations')

                            for param in self.parameters:

                                if param in self.parameters['travel_time_matrix']['locations']:

                                    for d in self.parameters[param]['list_seq_id'+str(replicate_num)]:

                                        node = d
                                        gtt.add_node(node, type=param)

                            for att in self.sorted_attributes:
                                
                                if att in self.parameters['travel_time_matrix']['locations']:

                                    for d in instance_data:

                                        node = instance_data[d][att+'id']
                                        gtt.add_node(node, type=att)

                            for u in node_list_seq:
                                for v in node_list_seq:

                                    gtt.add_edge(u, v, travel_time=travel_time_json[u][v])

                            output_name_graphml = os.path.join(self.save_dir_graphml, self.output_folder_base + '_' + str(replicate_num) + '.graphml')

                            nx.write_graphml(gtt, output_name_graphml)


                    output_file_ttm_csv = os.path.join(self.save_dir_ttm, 'travel_time_matrix_' + final_filename + '_' + str(replicate_num) + '.csv')
                
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


        save_dir = os.getcwd()+'/'+self.output_folder_base
        save_dir_images = os.path.join(save_dir, 'images')
        #plot_requests(self.network, save_dir_images, origin_points, destination_points)

        self.output_file_json = os.path.join(self.save_dir_json, final_filename + '_' + str(replicate_num) + '.json')

        with open(self.output_file_json, 'w') as file:
            json.dump(all_instance_data, file, indent=4)
            file.close()     

