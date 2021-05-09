import json
import math
import networkx as nx
import pandas as pd

def output_fixed_route_network(output_file_name, network):

    with open(output_file_name, 'w') as file:

        fixed_lines_stations = []
        file.write(str(len(network.linepieces)))
        file.write('\n')

        i = 0
        for lp in network.linepieces:


            file.write(str(len(lp)))
            file.write(' ')

            for station in lp:
                if station not in fixed_lines_stations:
                    fixed_lines_stations.append(station)
                file.write(str(station))
                file.write(' ')

            file.write('\n')

            #file.write(str(len(network.linepieces_dist[i])))
            #file.write(' ')

            for distuv in network.linepieces_dist[i]:
                file.write(str(int(distuv)))
                file.write(' ')

            file.write('\n')

            i += 1

        file.write(str(len(network.connecting_nodes)))
        file.write('\n')

        for station in network.connecting_nodes:
            file.write(str(station))
            file.write(' ')
        file.write('\n')

        file.write(str(len(network.direct_lines)))
        file.write('\n')
        for dl in network.direct_lines:

            file.write(str(len(dl)))
            file.write(' ')

            for station in dl:
                file.write(str(station))
                file.write(' ')

            file.write('\n')

        file.write(str(len(network.transfer_nodes)))
        file.write('\n')

        for station in network.transfer_nodes:
            file.write(str(station))
            file.write(' ')
        file.write('\n')

        '''
        for ids in network.subway_lines:

            file.write(str(ids))
            file.write(' ')
            
            nodes_path = nx.dijkstra_path(network.subway_lines[ids]['route_graph'], network.subway_lines[ids]['begin_route'], network.subway_lines[ids]['end_route'], weight='duration_avg')

            file.write(str(len(nodes_path)))
            file.write(' ')

            for u in range(len(nodes_path)):
                file.write(str(int(network.deconet_network_nodes.loc[int(nodes_path[u]), 'bindex'])))
                file.write(' ')
            file.write('\n')
        '''

        for station in fixed_lines_stations:

            file.write(str(station))
            file.write(' ')

            file.write(str(network.bus_stations.loc[int(station), 'lat']))
            file.write(' ')

            file.write(str(network.bus_stations.loc[int(station), 'lon']))
            file.write('\n')


class JsonConverter(object):

    def __init__(self, file_name):

        with open(file_name, 'rb') as file:
            self.json_data = json.load(file)
            file.close()

    def convert_normal(self, output_file_name, inst, problem_type, path_instance_csv_file):

        instance = []
        requests = self.json_data.get('data')
        for request in requests.values():

            d = {}
            for att in inst.sorted_attributes:

                if request.get(att) is not None:
                    
                    if inst.GA.nodes[att]['output_csv'] is True:
                        d[att] = request.get(att)
                    
                    #if inst.GA.nodes[att]['type'] == 'coordinate':

                        #d[att+'x'] = request.get(att+'x')
                        #d[att+'y'] = request.get(att+'y')
                        
                    #else:

                       
            instance.append(d)

        instance = pd.DataFrame(instance)
        instance.to_csv(path_instance_csv_file)
        
        with open(output_file_name, 'w') as file:

            # first line: number of stations
            #file.write(str(self.json_data.get('num_stations')))
            #file.write('\n')

            # second line - nr station: distance matrix
            #dist_matrix = self.json_data.get('distance_matrix')
            #for row in dist_matrix:
            #    for distance in row:
            #        file.write(str(distance))
            #        file.write('\t')
            #    file.write('\n')

            
            if problem_type == "ODBRP" or problem_type == "ODBRPFL":
                #number of bus stations
                file.write(str(len(inst.network.bus_stations_ids)))
                file.write('\n')
                
                #id for each bus station
                for station in inst.network.bus_stations_ids:
                    #osmid_station = inst.network.bus_stations.loc[station, 'osmid_drive']
                    file.write(str(station))
                    file.write('\t')
                
                #print info of travel time bus
                travel_time_matrix_bus = self.json_data.get('travel_time_matrix')
                
                '''
                file.write(str(len(travel_time_matrix_bus)))
                file.write('\n')

                for pair in travel_time_matrix_bus:
                    for element in pair:
                        file.write(str(element))
                        file.write('\t')
                    file.write('\n')
                '''

                for row in travel_time_matrix_bus:
                    for element in row:
                        file.write(str(int(element)))
                        file.write('\t')
                    file.write('\n')

            if problem_type == "SBRP":

                #number of depots
                file.write(str(self.json_data.get('num_schools')))
                file.write('\n')

                #ids of depots
                school_ids = self.json_data.get('schools')
                for node in school_ids:
                    file.write(str(node))
                    file.write('\t')
                file.write('\n')

                #number of nodes
                file.write(str(self.json_data.get('num_nodes')))
                file.write('\n')

                for node in inst.network.node_list_seq_school:
                    file.write(str(node))
                    file.write('\t')
                file.write('\n')

                travel_time_matrix = self.json_data.get('travel_time_matrix')

                for row in travel_time_matrix:
                    for element in row:
                        file.write(str(int(element)))
                        file.write('\t')
                    file.write('\n')



            '''
            if problem_type == "ODBRPFL":
                #number of fixed line stations
                file.write(str(len(inst.network.nodes_covered_fixed_lines)))
                file.write('\n')

                #ids for stations fixed line
                for station in inst.network.nodes_covered_fixed_lines:
                    file.write(str(station))
                    file.write('\t')
                file.write('\n')

                #print info of travel time subway
                travel_time_matrix_subway = self.json_data.get('travel_time_matrix_subway')
                
                file.write(str(len(travel_time_matrix_subway)))
                file.write('\n')

                for pair in travel_time_matrix_subway:
                    for element in pair:
                        file.write(str(element))
                        file.write('\t')
                    file.write('\n')

                #print info of walking time between bus stations and fixed line stations

                travel_time_matrix_hybrid = self.json_data.get('travel_time_matrix_hybrid')
                file.write(str(len(travel_time_matrix_hybrid)))
                file.write('\n')

                for pair in travel_time_matrix_hybrid:
                    for element in pair:
                        file.write(str(element))
                        file.write('\t')
                    file.write('\n')
            '''

            if problem_type == "DARP":

                '''
                #number of depots
                file.write(str(self.json_data.get('num_depots')))
                file.write('\n')

                #ids of depots
                depot_ids = self.json_data.get('depots')
                for node in depot_ids:
                    file.write(str(node))
                    file.write('\t')
                file.write('\n')

                #number of nodes
                file.write(str(self.json_data.get('num_nodes')))
                file.write('\n')

                for node in inst.network.node_list_darp_seq:
                    file.write(str(node))
                    file.write('\t')
                file.write('\n')
                '''

                travel_time_matrix_darp = self.json_data.get('travel_time_matrix')

                '''
                file.write(str(len(travel_time_matrix_darp)))
                file.write('\n')

                for pair in travel_time_matrix_darp:
                    for element in pair:
                        file.write(str(element))
                        file.write('\t')
                    file.write('\n')
                '''

                for row in travel_time_matrix_darp:
                    for element in row:
                        file.write(str(int(element)))
                        file.write('\t')
                    file.write('\n')


            #request information
            requests = self.json_data.get('data')
            num_requests = len(requests)

            #first line: number of requests
            file.write(str(num_requests))
            file.write('\n')

            #foreach request
            for request in requests.values():

                #print(request)
                # origin coordinates
                if request.get('originx') is not None:
                    file.write(str(request.get('originx')) + '\t' + str(request.get('originy')))
                    file.write('\n')

                if request.get('originid') is not None:
                    file.write(str(request.get('originid')))
                    file.write('\n')

                # destination coordinates
                if request.get('destinationx') is not None:
                    file.write(str(request.get('destinationx')) + '\t' + str(request.get('destinationy')))
                    file.write('\n')

                if request.get('destinationid') is not None:
                    file.write(str(request.get('destinationid')))
                    file.write('\n')

                # num bus stations origin + bus stations origin
                if request.get('stops_orgn') is not None:

                    file.write(str(len(request.get('stops_orgn'))) + '\n')
                    
                    if len(request.get('stops_orgn')) > 0:
                        for stop in request.get('stops_orgn'):
                            file.write(str(int(stop)) + '\t')
                        file.write('\n')
                        for walking_distance in request.get('stops_orgn_walking_distance'):
                            file.write(str(walking_distance) + '\t')

                        file.write('\n')

                # num bus stations destination + bus stations destination
                if request.get('stops_dest') is not None:
                    
                    file.write(str(len(request.get('stops_dest'))) + '\n')

                    if len(request.get('stops_dest')) > 0:
                        for stop in request.get('stops_dest'):
                            file.write(str(int(stop)) + '\t')
                        file.write('\n')
                        for walking_distance in request.get('stops_dest_walking_distance'):
                            file.write(str(walking_distance) + '\t')

                        file.write('\n')

                # earliest departure time
                if request.get('earliest_departure') is not None:
                    file.write(str(request.get('earliest_departure')))
                
                    #latest departure time
                    #if problem_type == "DARP":
                    #    file.write(' ')
                    #    file.write(str(request.get('lat_dep_time')))
                        
                    file.write('\n')

                #earliest arrival time
                #if problem_type == "DARP":
                #    file.write(str(request.get('ear_arr_time')))
                #    file.write(' ')

                # latest arrival time
                if request.get('latest_arrival') is not None:
                    file.write(str(request.get('latest_arrival')))
                    file.write('\n')

            '''
            for request in requests.values():

                for att in inst.sorted_attributes:

                    if request.get(att) is not None:
                        
                        if inst.GA[att]['type'] == 'list':

                            for elem in request.get(att):
                                file.write(str(elem) + '\t')
                            file.write('\n')

                        elif: inst.GA[att]['type'] == 'coordinate':

                            file.write(str(request.get(att+'x')) + '\t' + str(request.get(att+'y')))
                            file.write('\n')

                        else:

                            file.write(str(request.get(att)))
            '''




    def convert_localsolver(self, output_file_name):

        with open(output_file_name, 'w') as file:

            # first line: number of stations
            file.write(str(self.json_data.get('num_stations')))
            file.write('\n')

            # second line - nr station: distance matrix
            #dist_matrix = self.json_data.get('distance_matrix')
            #for row in dist_matrix:
            #    for distance in row:
            #        file.write(str(distance))
            #        file.write('\t')
            #    file.write('\n')

            # start of request information
            requests = self.json_data.get('requests')
            num_requests = len(requests)

            # first line: number of requests
            file.write(str(num_requests))
            file.write('\n')

            # request information
            requests = self.json_data.get('requests')
            num_requests = len(requests)

            # line format: (index nb_stops stops demand=1 ear_dep_time lat_arr_time serve_time pick_ind deliv_ind)
            # first line: 0    0   0   0   1000    0   0   0
            # file.write('0\t0\t0\t0\t1000\t0\t0\t0')

            # foreach request - request split in pickup and delivery_pair
            index = 1
            for request in requests.values():

                index_pickup = index
                index_delivery = index_pickup + 1

                nb_stops_pickup = request.get('num_stops_origin')
                stops_pickup = request.get('stops_origin')

                nb_stops_delivery = request.get('num_stops_destination')
                stops_delivery = request.get('stops_destination')

                demand = 1
                serv_time = 0
                dep_time = request.get('dep_time')
                arr_time = request.get('arr_time')

                file.write('\n')

                # write pickup
                file.write(str(index_pickup) + '\t')
                file.write(str(nb_stops_pickup) + '\t')
                for stop in stops_pickup:
                    file.write(str(stop) + '\t')
                file.write(str(demand) + '\t')
                file.write(str(dep_time) + '\t')
                file.write(str(arr_time) + '\t')
                file.write(str(serv_time) + '\t')
                file.write('0' + '\t')  # pickup index always = 0 for pickups
                file.write(str(index_delivery) + '\t')

                file.write('\n')

                # write delivery
                demand = -1
                file.write(str(index_delivery) + '\t')
                file.write(str(nb_stops_delivery) + '\t')
                for stop in stops_delivery:
                    file.write(str(stop) + '\t')
                file.write(str(demand) + '\t')
                file.write(str(dep_time) + '\t')
                file.write(str(arr_time) + '\t')
                file.write(str(serv_time) + '\t')
                file.write(str(index_pickup) + '\t')
                file.write('0' + '\t')  # delivery index always 0 for deliveries

                index += 2