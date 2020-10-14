import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import json
from random import randint
import math

try:
    import tkinter as tk
    from tkinter import filedialog
except:
    pass

# USAGE INFORMATION
if len(sys.argv) != 3:
    print("Usage: python " + sys.argv[0] + " instance_file_name number_of_requests")
    exit()


def distance(x1, y1, x2, y2):
    return int(round(((x1 - x2) ** 2 + (y1 - y2) ** 2) ** .5))  # was vroeger int, maar gaf afrondingsproblemen


class JsonConverter(object):

    def __init__(self, file_name):

        with open(file_name, 'r') as file:
            self.json_data = json.load(file)
            file.close()

    def convert_normal(self, output_file_name):

        with open(output_file_name, 'w') as file:

            # first line: number of stations
            file.write(str(self.json_data.get('num_stations')))
            file.write('\n')

            # second line - nr station: distance matrix
            dist_matrix = self.json_data.get('distance_matrix')
            for row in dist_matrix:
                for distance in row:
                    file.write(str(distance))
                    file.write('\t')
                file.write('\n')

            # start of request information
            requests = self.json_data.get('requests')
            num_requests = len(requests)

            # first line: number of requests
            file.write(str(num_requests))
            file.write('\n')

            # foreach request
            for request in requests.values():

                # origin coordinates
                file.write(str(request.get('originx')) + '\t' + str(request.get('originy')))
                file.write('\n')

                # destination coordinates
                file.write(str(request.get('destinationx')) + '\t' + str(request.get('destinationy')))
                file.write('\n')

                # num stops origin + stops origin
                file.write(str(request.get('num_stops_origin')) + '\t')
                for stop in request.get('stops_origin'):
                    file.write(str(stop) + '\t')

                file.write('\n')

                # num stops destination + stops destination
                file.write(str(request.get('num_stops_destination')) + '\t')
                for stop in request.get('stops_destination'):
                    file.write(str(stop) + '\t')

                file.write('\n')

                # earliest departure time
                file.write(str(request.get('dep_time')))
                file.write('\n')

                # latest arrival time
                file.write(str(request.get('arr_time')))
                file.write('\n')

    def convert_localsolver(self, output_file_name):

        with open(output_file_name, 'w') as file:

            # first line: number of stations
            file.write(str(self.json_data.get('num_stations')))
            file.write('\n')

            # second line - nr station: distance matrix
            dist_matrix = self.json_data.get('distance_matrix')
            for row in dist_matrix:
                for distance in row:
                    file.write(str(distance))
                    file.write('\t')
                file.write('\n')

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


def generate_instances_json(nr_requests, num_replicates, save_dir, max_x=100, max_y=100, max_walking=10, bus_factor=2, max_early_departure=60,
                            nr_stations=121):

    save_dir_json = os.path.join(save_dir, 'json_format')
    output_file_base = sys.argv[1].split('.')[0]

    if not os.path.isdir(save_dir_json):
        os.mkdir(save_dir_json)

    stationsx = np.ndarray(nr_stations)
    stationsy = np.ndarray(nr_stations)

    for replicate in range(num_replicates):

        output_file_json = os.path.join(save_dir_json, output_file_base + '_' + str(replicate) + '.json')
        instance_data = {}  # holds information for each request

        i = 0
        a = 0
        b = 0

        while i < 121:
            if b <= 100:
                stationsx[i] = a
                stationsy[i] = b
                b = b + 10
                i = i + 1
            else:
                a = a + 10
                b = 0
                stationsx[i] = a
                stationsy[i] = b
                b = b + 10
                i = i + 1

        # create an array
        dist = np.ndarray((nr_stations, nr_stations), dtype=np.int)
        dist_json = [[None for i in range(nr_stations)] for j in range(nr_stations)]

        for i in range(nr_stations):
            for j in range(nr_stations):
                dist[i][j] = int(math.ceil(((stationsx[i] - stationsx[j]) ** 2 + (stationsy[i] - stationsy[j]) ** 2) ** .5))  # round was vroeger int #math.ceil was round
                dist_json[i][j] = int(round(dist[i][j]))

        print("Now generating " + str(nr_requests) + " instance_data")

        originx = np.ndarray(nr_requests).astype(int)
        originy = np.ndarray(nr_requests).astype(int)
        destinationx = np.ndarray(nr_requests).astype(int)
        destinationy = np.ndarray(nr_requests).astype(int)
        dep_time = np.ndarray(nr_requests).astype(int)
        arr_time = np.ndarray(nr_requests).astype(int)

        lines = []

        # Genereren van instance_data: origin, destination

        all_requests = {}  # holds all instance_data

        for i in range(nr_requests):

            nok = True
            request_data = {}  # holds information about this request
            # We herhalen het willekeurig kiezen van coordinaten voor een passagier tot hij minstens naar 1 stop kan
            # stappen (zowel voor origin als voor destination)

            while nok:

                nok = False
                # print ("Passenger " + str(i))

                originx[i] = randint(0, max_x + 1)
                originy[i] = randint(0, max_y + 1)

                destinationx[i] = randint(0, max_x + 1)
                destinationy[i] = randint(0, max_y + 1)

                stops_origin = []
                stops_destination = []

                for j in range(nr_stations):

                    if distance(originx[i], originy[i], stationsx[j], stationsy[j]) < max_walking:
                        stops_origin.append(j)

                    if distance(destinationx[i], destinationy[i], stationsx[j], stationsy[j]) < max_walking:
                        stops_destination.append(j)

                # print(len(stops_origin), stops_origin)
                # print(len(stops_destination), stops_destination)

                # Check whether (1) each passenger can walk to a stop (origin + destination), the intersection of the
                # origin and destination stop list is not empty (in which case they do not require a bus)

                if len(stops_origin) > 0 and len(stops_destination) > 0:
                    if not (set(stops_origin) & set(stops_destination)):

                        request_data.update({'originx': int(originx[i])})
                        request_data.update({'originy': int(originy[i])})

                        request_data.update({'destinationx': int(destinationx[i])})
                        request_data.update({'destinationy': int(destinationy[i])})

                        request_data.update({'num_stops_origin': len(stops_origin)})

                        request_data.update({'stops_origin': stops_origin})
                        request_data.update({'num_stops_destination': len(stops_destination)})

                        request_data.update({'stops_destination': stops_destination})

                        dep_time[i] = randint(0, max_early_departure)
                        # print(dep_time[i])

                        request_data.update({'dep_time': int(dep_time[i])})

                        arr_time[i] = dep_time[i] + bus_factor * distance(originx[i], originy[i], destinationx[i],
                                                                          destinationy[i]) + max_walking * 2
                        # print(arr_time[i])
                        request_data.update({'arr_time': int(arr_time[i])})

                        # add request_data to instance_data container
                        all_requests.update({i: request_data})
                    else:
                        # print ("Passenger cannot walk to a stop")
                        nok = True
                else:
                    # print ("Passenger cannot walk to a stop")
                    nok = True
            # if nok == False:

        instance_data.update({'requests': all_requests,
                              'num_stations': nr_stations,
                              'distance_matrix': dist_json})

        with open(output_file_json, 'w') as file:
            json.dump(instance_data, file, indent=4)
            file.close()


if __name__ == '__main__':

    # OPTIONS
    num_replicates = int(input('Give number of replicates: \n>> '))
    nr_requests = int(sys.argv[2])

    # OUTPUT DIRECTORY
    try:
        root = tk.Tk()
        save_dir = filedialog.askdirectory(initialdir='./')
        root.destroy()
    except:
        save_dir = os.getcwd()

    # generate instances in json output folder
    generate_instances_json(nr_requests=nr_requests, num_replicates=num_replicates, save_dir=save_dir)

    # convert instances from json to normal and localsolver format

    save_dir_cpp = os.path.join(save_dir, 'cpp_format')
    if not os.path.isdir(save_dir_cpp):
        os.mkdir(save_dir_cpp)

    save_dir_localsolver = os.path.join(save_dir, 'localsolver_format')
    if not os.path.isdir(save_dir_localsolver):
        os.mkdir(save_dir_localsolver)

    for instance in os.listdir(os.path.join(save_dir, 'json_format')):

        input_name = os.path.join(save_dir, 'json_format', instance)
        output_name_cpp = instance.split('.')[0] + '_cpp.pass'
        output_name_ls = instance.split('.')[0] + '_ls.pass'

        converter = JsonConverter(file_name=input_name)
        converter.convert_normal(output_file_name=os.path.join(save_dir_cpp, output_name_cpp))
        converter.convert_localsolver(output_file_name=os.path.join(save_dir_localsolver, output_name_ls))
