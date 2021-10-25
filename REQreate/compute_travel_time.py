import gc
import math
import numpy as np
from multiprocessing import cpu_count
import pandas as pd
import os
import ray
import time

@ray.remote
def calc_travel_time_od(origin, destination, vehicle_speed, shortest_path_drive, bus_stops):

    #curr_weight = 'travel_time_' + str(hour)
    #curr_weight = 'travel_time' 
    row = {}
    row['origin_id'] = origin
    row['destination_id'] = destination
    eta = -1
    
    try:

        origin = bus_stops.loc[origin, 'osmid_drive']
        sdestination = str(bus_stops.loc[destination, 'osmid_drive'])

        distance = shortest_path_drive.loc[origin, sdestination]
        
        if str(distance) != 'nan':
            distance = int(distance)
            eta = int(math.ceil(distance/vehicle_speed))

    except KeyError:
        pass

    if eta >= 0:
        row['eta'] = eta
        row['dist'] = distance
        #return eta
    else:
        row['eta'] = np.nan
        row['dist'] = np.nan

    return row


def _get_travel_time_matrix(vehicle_speed, bus_stops, shortest_path_drive, shortest_path_walk, save_dir, output_folder_base, filename=None): 
    
    ''' 
    compute a travel time matrix between bus stations
    it is not time dependent
    vehicle speed is determined beforehand.
    '''
    travel_time_matrix = []
    counter = 0
    save_dir_csv = os.path.join(save_dir, 'csv')

   
    ray.shutdown()
    ray.init(num_cpus=cpu_count())

    if not os.path.isdir(save_dir_csv):
        os.mkdir(save_dir_csv)

    if filename is None:
        path_csv_file = os.path.join(save_dir_csv, output_folder_base+'.travel.time.csv')
    else:
        path_csv_file = os.path.join(save_dir_csv, output_folder_base+filename)


    if os.path.isfile(path_csv_file):
        print('is file travel time')
        travel_time_matrix = pd.read_csv(path_csv_file)
        #print('rows travel time', len(travel_time_matrix))
    else:
        print('creating file estimated travel time')
        #start = time.process_time()
        #travel_time_matrix = pd.DataFrame()
        
        list_nodes = []
        for index, row in bus_stops.iterrows():
            list_nodes.append(index)

        #shortest_path_drive2 = pd2.DataFrame(shortest_path_drive)
        shortest_path_drive_id = ray.put(shortest_path_drive)

        #param_id = ray.put(param)

        #bus_stop2 = pd2.DataFrame(bus_stops)
        bus_stops_id = ray.put(bus_stops)
        
        #for origin in list_nodes:

        for origin in list_nodes:            
            

            #not parallel
            '''
            for destination in list_nodes:
                counter += 1
                row = {}
                row['stop_origin_osmid'] = bus_stops.loc[origin, 'osmid_drive']
                row['stop_destination_osmid'] = bus_stops.loc[destination, 'osmid_drive']
                row['eta'] = calc_travel_time_od(param, origin, destination, shortest_path_drive)
                travel_time_matrix = travel_time_matrix.append(row, ignore_index=True)
                del row
            '''

            #with multiprocessing
            '''
            pool = Pool(processes=num_of_cpu)
            results = pool.starmap(calc_travel_time_od, [(param, origin, destination, shortest_path_drive, 0) for destination in list_nodes])
            pool.close()
            pool.join()

            j=0
            for destination in list_nodes:
                counter += 1
                row = {}
                #row['stop_origin_id'] = bus_stops.loc[origin, 'itid']
                #row['stop_destination_id'] = bus_stops.loc[destination, 'itid']
                row['stop_origin_osmid'] = bus_stops.loc[origin, 'osmid_drive']
                row['stop_destination_osmid'] = bus_stops.loc[destination, 'osmid_drive']
                row['eta'] = results[j]
                travel_time_matrix = travel_time_matrix.append(row, ignore_index=True)
                j += 1
            '''

            #with ray
            #print("here")
            results = ray.get([calc_travel_time_od.remote(origin, destination, vehicle_speed, shortest_path_drive_id, bus_stops_id) for destination in list_nodes])
            for row in results:
                #travel_time_matrix = travel_time_matrix.append(row, ignore_index=True)
                travel_time_matrix.append(row)
                counter += 1

            del results
            #print("out")

            '''
            j=0
            for destination in list_nodes:
                counter += 1
                row = {}
                row['origin_osmid'] = bus_stops.loc[origin, 'osmid_drive']
                row['destination_osmid'] = bus_stops.loc[destination, 'osmid_drive']
                row['eta'] = results[j]
                travel_time_matrix = travel_time_matrix.append(row, ignore_index=True)
                j += 1
                del row
            '''

            #print('paths so far', counter)
            #print("total time so far", time.process_time() - start)
            #del results
            gc.collect()    
                               
        travel_time_matrix = pd.DataFrame(travel_time_matrix)
        travel_time_matrix.to_csv(path_csv_file)
        #print("total time", time.process_time() - start)

    travel_time_matrix.set_index(['origin_id', 'destination_id'], inplace=True)
    return travel_time_matrix
       
