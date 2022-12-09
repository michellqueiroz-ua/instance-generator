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

if __name__ == '__main__':

    place_name = "Diest, Belgium"
    save_dir = os.getcwd()+'/'+place_name
    pickle_dir = os.path.join(save_dir, 'pickle')
    network_class_file = pickle_dir+'/'+place_name+'.network.class.pkl'

    network_directory = os.getcwd()+'/'+place_name

    if Path(network_class_file).is_file():
        inst = Instance(folder_to_network=place_name)

    lons = []
    lats = []

    lonlat = 0
    with open('locations_Diest.txt','r') as file:
        for line in file:
            for word in line.split():

                if lonlat == 0:
                    lats.append(word)
                    lonlat = 1
                else:
                    lons.append(word)
                    lonlat = 0
    adresses = []
    adresses = pd.DataFrame(adresses)

    for (u,v,k) in inst.network.G_drive.edges(data=True): 
        #print(u, v, k)
        '''
        print(k['highway'])
        try:
            print(k['maxspeed'])
        except KeyError:
            pass
        '''

        speed = 0
        type1speed = 55
        type2speed = 40
        type3speed = 20
        type4speed = 15
        type5speed = 10
        type6speed = 30
        if type(k['highway']) is not list:

            if (k['highway'] == 'motorway'):
                speed = type1speed

            elif (k['highway'] == 'motorway_link'):
                speed = type1speed

            elif (k['highway'] == 'trunk'):
                speed = type1speed

            elif (k['highway'] == 'trunk_link'):
                speed = type1speed

            elif (k['highway'] == 'primary'):
                speed = type2speed

            elif (k['highway'] == 'primary_link'):
                speed = type2speed

            elif (k['highway'] == 'secondary'):
                speed = type3speed

            elif (k['highway'] == 'secondary_link'):
                speed = type3speed

            elif (k['highway'] == 'tertiary'):
                speed = type4speed

            elif (k['highway'] == 'tertiary_link'):
                speed = type4speed

            elif (k['highway'] == 'residential'):
                speed = type5speed

            elif (k['highway'] == 'living_street'):
                speed = type5speed

            elif (k['highway'] == 'unclassified'):
                speed = type6speed

            elif (k['highway'] == 'no'):
                speed = type6speed

            elif (k['highway'] == 'emergency_bay'):
                speed = type6speed

            else:
                speed = type6speed
                #print(k['highway'])


            speed = speed/3.6
            inst.network.G_drive[u][v][0]['travel_time'] = int(math.ceil(inst.network.G_drive[u][v][0]['length']/(speed)))
        else:

            if ((k['highway'][0] == 'motorway') or (k['highway'][1] == 'motorway')):
                speed = type1speed

            elif ((k['highway'][0] == 'motorway_link') or (k['highway'][1] == 'motorway_link')) :
                speed = type1speed

            elif ((k['highway'][0] == 'trunk') or (k['highway'][1] == 'trunk')) :
                speed = type1speed

            elif ((k['highway'][0] == 'trunk_link') or (k['highway'][1] == 'trunk_link')):
                speed = type1speed

            elif ((k['highway'][0] == 'primary') or (k['highway'][1] == 'primary')):
                speed = type2speed

            elif ((k['highway'][0] == 'primary_link') or (k['highway'][1] == 'primary_link')):
                speed = type2speed

            elif ((k['highway'][0] == 'secondary') or (k['highway'][1] == 'secondary')):
                speed = type3speed

            elif ((k['highway'][0] == 'secondary_link') or (k['highway'][1] == 'secondary_link')):
                speed = type3speed

            elif ((k['highway'][0] == 'tertiary') or (k['highway'][1] == 'tertiary')):
                speed = type4speed

            elif ((k['highway'][0] == 'tertiary_link') or (k['highway'][1] == 'tertiary_link')):
                speed = type4speed

            elif ((k['highway'][0] == 'residential') or (k['highway'][1] == 'residential')):
                speed = type5speed

            elif ((k['highway'][0] == 'living_street') or (k['highway'][1] == 'living_street')):
                speed = type5speed

            elif ((k['highway'][0] == 'unclassified') or (k['highway'][1] == 'unclassified')):
                speed = type6speed

            elif ((k['highway'][0] == 'no') or (k['highway'][1] == 'no')):
                speed = type6speed

            elif ((k['highway'][0] == 'emergency_bay') or (k['highway'][1] == 'emergency_bay')):
                speed = type6speed

            else:
                speed = type6speed
                #print(k['highway'])


            speed = speed/3.6
            inst.network.G_drive[u][v][0]['travel_time'] = int(math.ceil(inst.network.G_drive[u][v][0]['length']/(speed)))

        
    node_list = []
    for idx in range(0, len(lats)):
        print(lats[idx])
        print(lons[idx])
        
        x = float(lons[idx])
        y = float(lats[idx])

        center_point = (y, x)
        north, south, east, west = ox.utils_geo.bbox_from_point(center_point, 1000)

        '''
        G = ox.graph_from_bbox(
            north,
            south,
            east,
            west,
            network_type='drive',
            retain_all=True
        )
        '''
        #print('here')
        polygon = ox.utils_geo.bbox_to_poly(north, south, east, west)

        G_drive = ox.graph_from_polygon(
            polygon,
            network_type='drive',
            retain_all=True
        )

        #u, v, key = ox.nearest_edges(inst.network.G_walk, x, y)
        #stop_node_walk = min((u, v), key=lambda n: ox.distance.great_circle_vec(y, x, inst.network.G_walk.nodes[n]['y'], inst.network.G_walk.nodes[n]['x']))
    
        u, v, key = ox.nearest_edges(inst.network.G_drive, x, y)
        stop_node_drive = min((u, v), key=lambda n: ox.distance.great_circle_vec(y, x, inst.network.G_drive.nodes[n]['y'], inst.network.G_drive.nodes[n]['x']))

        node_list.append(stop_node_drive)

        

        print('h')
        d = {
            'osmid_drive':stop_node_drive,
            'lat':y,
            'lon':x
        }
        
        adresses = adresses.append(d, ignore_index=True)
    
    for node in node_list:
        print(node)

    print('ttm')
    travel_time = inst.network._get_travel_time_matrix("list", node_list=node_list)
    ttmpd = pd.DataFrame(travel_time)
    output_file_ttm_csv = os.path.join(inst.save_dir_ttm, 'travel_time_matrix_min' + '.csv')
    ttmpd.to_csv(output_file_ttm_csv)
    
    print('leave ttm')
