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
    network_class_file = pickle_dir+'/'+place_name+'.network.class.pkl'

    network_directory = os.getcwd()+'/'+place_name

    if Path(network_class_file).is_file():
        inst = Instance(folder_to_network=place_name)

    save_dir_csv = os.path.join(inst.save_dir, 'csv')

    #zones = pd.read_csv(save_dir_csv+'/'+place_name+'.zones.csv')
    #stations = pd.read_csv(save_dir_csv+'/'+place_name+'.stations.csv')

    #zone1 70% requests
    pt = inst.network.polygon.centroid
    lon = pt.x
    lat = pt.y
    
    earth_radius = 6371009  # meters
    dist_lat = 2000
    dist_lon = 2000

    lat = lat
    lng = lon

    delta_lat = (dist_lat / earth_radius) * (180 / math.pi)
    delta_lng = (dist_lon / earth_radius) * (180 / math.pi) / math.cos(lat * math.pi / 180)
    
    north = lat + delta_lat
    south = lat - delta_lat
    east = lng + delta_lng
    west = lng - delta_lng
                
    #north, south, east, west = ox.utils_geo.bbox_from_point(zone_center_point, distance)
    polygon1 = Polygon([(west, south), (east, south), (east, north), (west, north)])

    #zone2 30% requests
    earth_radius = 6371009  # meters
    dist_lat = 2000
    dist_lon = 2000  

    lat = 41.767025
    lng = -87.689840
    

    delta_lat = (dist_lat / earth_radius) * (180 / math.pi)
    delta_lng = (dist_lon / earth_radius) * (180 / math.pi) / math.cos(lat * math.pi / 180)
    
    north = lat + delta_lat
    south = lat - delta_lat
    east = lng + delta_lng
    west = lng - delta_lng
                
    #north, south, east, west = ox.utils_geo.bbox_from_point(zone_center_point, distance)
    polygon2 = Polygon([(west, south), (east, south), (east, north), (west, north)])

    
    all_stations = []
    #for indexz, zone in zones.iterrows():
    stationslis = []
    for index, row in inst.network.bus_stations.iterrows():

        lon = inst.network.bus_stations.loc[index, 'lon']
        lat = inst.network.bus_stations.loc[index, 'lat']
        #polygon = shapely.wkt.loads(zone['polygon'])
        print(type(polygon1))
        pnt = Point(lon, lat)
        if polygon1.contains(pnt):
            stationslis.append(index)



    stationslis2 = []
    for index, row in inst.network.bus_stations.iterrows():

        lon = inst.network.bus_stations.loc[index, 'lon']
        lat = inst.network.bus_stations.loc[index, 'lat']
        #polygon = shapely.wkt.loads(zone['polygon'])
        print(type(polygon2))
        pnt = Point(lon, lat)
        if polygon2.contains(pnt):
            stationslis2.append(index)

    data = {
        'Name': ['zone70', 'zone30'],
        'stations': [stationslis, stationslis2]
    }

    df = pd.DataFrame(data)

    df.to_csv('zones_uneven_demand.csv', index=False)

    #all_stations.append(stationslis)

    #zones['stations'] = all_stations

    #zones = pd.DataFrame(zones)
    #zones.to_csv(save_dir_csv+'/'+place_name+'.zones.csv')
    

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
    '''