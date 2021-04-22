import json
import math
import networkx as nx

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

    def convert_normal(self, output_file_name, network, problem_type):

        
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
                file.write(str(len(network.bus_stations_ids)))
                file.write('\n')
                
                #id for each bus station
                for station in network.bus_stations_ids:
                    #osmid_station = network.bus_stations.loc[station, 'osmid_drive']
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

                for node in network.node_list_seq_school:
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
                file.write(str(len(network.nodes_covered_fixed_lines)))
                file.write('\n')

                #ids for stations fixed line
                for station in network.nodes_covered_fixed_lines:
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

                for node in network.node_list_darp_seq:
                    file.write(str(node))
                    file.write('\t')
                file.write('\n')

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
            requests = self.json_data.get('requests')
            num_requests = len(requests)

            #first line: number of requests
            file.write(str(num_requests))
            file.write('\n')

            #foreach request
            for request in requests.values():

                # origin coordinates
                if request.get('originx') is not None:
                    file.write(str(request.get('originx')) + '\t' + str(request.get('originy')))
                    file.write('\n')

                if request.get('origin_node') is not None:
                    file.write(str(request.get('origin_node')))
                    file.write('\n')

                # destination coordinates
                if request.get('destinationx') is not None:
                    file.write(str(request.get('destinationx')) + '\t' + str(request.get('destinationy')))
                    file.write('\n')

                if request.get('destination_node') is not None:
                    file.write(str(request.get('destination_node')))
                    file.write('\n')

                # num bus stations origin + bus stations origin
                if request.get('num_stops_origin') is not None:

                    file.write(str(request.get('num_stops_origin')) + '\n')
                    
                    if request.get('num_stops_origin') > 0:
                        for stop in request.get('stops_origin'):
                            #osmid_station = network.bus_stations.loc[stop, 'osmid_drive']
                            file.write(str(int(stop)) + '\t')
                        file.write('\n')
                        for walking_distance in request.get('walking_time_origin_to_stops'):
                            file.write(str(walking_distance) + '\t')

                        file.write('\n')

                # num bus stations destination + bus stations destination
                if request.get('num_stops_destination') is not None:
                    
                    file.write(str(request.get('num_stops_destination')) + '\n')

                    if request.get('num_stops_destination') > 0:
                        for stop in request.get('stops_destination'):
                            #osmid_station = network.bus_stations.loc[stop, 'osmid_drive']
                            file.write(str(int(stop)) + '\t')
                        file.write('\n')
                        for walking_distance in request.get('walking_time_stops_to_destination'):
                            file.write(str(walking_distance) + '\t')

                        file.write('\n')

                '''
                # num fixed line origin + stations fixed line origin
                if request.get('num_stations_fl_origin') is not None:
                    file.write(str(request.get('num_stations_fl_origin')) + '\n')
                    
                    if int(request.get('num_stations_fl_origin')) > 0:
                        for stop in request.get('stations_fl_origin'):
                            file.write(str(stop) + '\t')
                        file.write('\n')
                        for walking_distance in request.get('walking_time_origin_to_stations_fl'):
                            file.write(str(walking_distance) + '\t')

                        file.write('\n')

                # num fixed line destination + stations fixed line destination
                if request.get('num_stations_fl_destination') is not None:
                    file.write(str(request.get('num_stations_fl_destination')) + '\n')

                    if int(request.get('num_stations_fl_destination')) > 0:
                        for stop in request.get('stations_fl_destination'):
                            file.write(str(stop) + '\t')
                        file.write('\n')
                        for walking_distance in request.get('walking_time_stations_fl_to_destination'):
                            file.write(str(walking_distance) + '\t')

                        file.write('\n')
                '''

                # earliest departure time
                if request.get('dep_time') is not None:
                    file.write(str(request.get('dep_time')))
                
                    #latest departure time
                    if problem_type == "DARP":
                        file.write(' ')
                        file.write(str(request.get('lat_dep_time')))
                        
                    file.write('\n')

                #earliest arrival time
                if problem_type == "DARP":
                    file.write(str(request.get('ear_arr_time')))
                    file.write(' ')

                # latest arrival time
                if request.get('arr_time') is not None:
                    file.write(str(request.get('arr_time')))
                    file.write('\n')

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