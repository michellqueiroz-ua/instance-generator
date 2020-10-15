import json
import math

class JsonConverter(object):

    def __init__(self, file_name):

        with open(file_name, 'r') as file:
            self.json_data = json.load(file)
            file.close()

    def convert_normal(self, output_file_name, network):

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

            #number of stations
            file.write(str(len(network.bus_stops_ids)))
            file.write('\n')
            
            #id for each station
            for stop1 in network.bus_stops_ids:
                file.write(str(stop1))
                file.write('\n')

            #distance matrix
            for stop1 in network.bus_stops_ids:
                for stop2 in network.bus_stops_ids:
                    file.write(str(stop1))
                    file.write('\t')
                    file.write(str(stop2))
                    file.write('\t')
                    try:
                        
                        osmid_stop1 = network.bus_stops.loc[stop1, 'osmid_drive']
                        osmid_stop2 = network.bus_stops.loc[stop2, 'osmid_drive']
                        dists1s2 = network.shortest_path_drive.loc[osmid_stop1, str(osmid_stop2)]
                        
                        #it is not possible to reach the node
                        if math.isnan(dists1s2):
                            dists1s2 = -1
                            file.write(str(dists1s2))
                        else:
                            file.write(str(dists1s2))

                    except KeyError:
                        dists1s2 = -1
                        file.write(str(dists1s2))
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
                file.write(str(request.get('originx')) + '\t' + str(request.get('originy')))
                file.write('\n')

                # destination coordinates
                file.write(str(request.get('destinationx')) + '\t' + str(request.get('destinationy')))
                file.write('\n')

                # num stops origin + stops origin
                file.write(str(request.get('num_stops_origin')) + '\n')
                for stop in request.get('stops_origin'):
                    file.write(str(stop) + '\t')
                file.write('\n')
                for walking_distance in request.get('walking_time_origin_to_stops'):
                    file.write(str(walking_distance) + '\t')

                file.write('\n')

                # num stops destination + stops destination
                file.write(str(request.get('num_stops_destination')) + '\n')
                for stop in request.get('stops_destination'):
                    file.write(str(stop) + '\t')
                file.write('\n')
                for walking_distance in request.get('walking_time_stops_to_destination'):
                    file.write(str(walking_distance) + '\t')

                file.write('\n')

                # earliest departure time
                file.write(str(request.get('dep_time')))
                file.write('\n')

                # latest arrival time
                file.write(str(request.get('arr_time')))
                file.write('\n')

                #writing the fixed lines
                if request.get('subway_line_ids') is not None:
                    file.write(str(request.get('num_subway_routes')) + '\n')
                    for subway_line in request.get('subway_line_ids'):
                        line_id = str(subway_line)
                        option = request.get('option'+line_id)
                        
                        file.write(str(line_id) + '\n')
                        file.write(str(option) + '\n')
                        file.write(str(request.get('eta_in_vehicle'+line_id)) + '\n')

                        if option == 1:
                            file.write(str(request.get('walking_time_to_pick_up'+line_id)) + '\n')
                            file.write(str(request.get('walking_time_from_drop_off'+line_id)) + '\n')

                        if option == 2:
                            file.write(str(request.get('num_stops_nearby_pick_up'+line_id)) + '\n')
                            file.write(str(request.get('stops_nearby_pick_up'+line_id)) + '\n')
                            file.write(str(request.get('walking_time_to_pick_up'+line_id)) + '\n')
                            file.write(str(request.get('walking_time_from_drop_off'+line_id)) + '\n')

                        if option == 3:
                            file.write(str(request.get('walking_time_to_pick_up'+line_id)) + '\n')
                            file.write(str(request.get('num_stops_nearby_drop_off'+line_id)) + '\n')
                            file.write(str(request.get('stops_nearby_drop_off'+line_id)) + '\n')
                            file.write(str(request.get('walking_time_from_drop_off'+line_id)) + '\n')

                        if option == 4:
                            file.write(str(request.get('num_stops_nearby_pick_up'+line_id)) + '\n')
                            file.write(str(request.get('stops_nearby_pick_up'+line_id)) + '\n')
                            file.write(str(request.get('walking_time_to_pick_up'+line_id)) + '\n')
                            file.write(str(request.get('num_stops_nearby_drop_off'+line_id)) + '\n')
                            file.write(str(request.get('stops_nearby_drop_off'+line_id)) + '\n')
                            file.write(str(request.get('walking_time_from_drop_off'+line_id)) + '\n')


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