import math
import matplotlib.pyplot as plt
import os
import osmnx as ox
import pandas as pd
import shapely
from shapely.geometry import Polygon
from shapely.geometry import Point
from pathlib import Path
from instance_class import Instance


def assign_bus_stops_to_zones():
    pass

if __name__ == '__main__':

    place_name = "Chicago, Illinois"
    save_dir = os.getcwd()+'/'+place_name
    pickle_dir = os.path.join(save_dir, 'pickle')
    network_class_file = pickle_dir+'/'+place_name+'.network2.class.pkl'

    network_directory = os.getcwd()+'/'+place_name

    if Path(network_class_file).is_file():
        inst = Instance(folder_to_network=place_name)

    save_dir_csv = os.path.join(inst.save_dir, 'csv')

    zones = pd.read_csv(save_dir_csv+'/'+place_name+'.zones.csv')
    stations = pd.read_csv(save_dir_csv+'/'+place_name+'.stations.csv')

    '''
    all_stations = []
    for indexz, zone in zones.iterrows():
        stationslis = []
        for indexs, station in stations.iterrows():

            polygon = shapely.wkt.loads(zone['polygon'])
            print(type(polygon))
            pnt = Point(station['lon'], station['lat'])
            if polygon.contains(pnt):
                stationslis.append(indexs)

        all_stations.append(stationslis)

    zones['stations'] = all_stations

    zones = pd.DataFrame(zones)
    zones.to_csv(save_dir_csv+'/'+place_name+'.zones.csv')
    '''
    path_tt_file = os.path.join(save_dir_csv, place_name+'.tt.matrix.stations.csv')
    pd_travel_time_matrix = pd.DataFrame()
    travel_time_matrix = []
    for indexs1, station1 in stations.iterrows():
        print(indexs1)
        d = {}
        d['station_index'] = indexs1
        for indexs2, station2 in stations.iterrows():
            sv = str(indexs2)
            if indexs2 != indexs1:
                dist_uv = inst.network._return_estimated_travel_time_drive(int(station1['osmid_drive']), int(station2['osmid_drive']))
                d[sv] = dist_uv
            else:
                d[sv] = 0
        travel_time_matrix.append(d)

    xt = pd.DataFrame(travel_time_matrix)
    pd_travel_time_matrix = pd_travel_time_matrix.append(xt, ignore_index=True)
    pd_travel_time_matrix.set_index(['station_index'], inplace=True)
    pd_travel_time_matrix.to_csv(path_tt_file)
    