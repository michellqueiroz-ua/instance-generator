import matplotlib.pyplot as plt
import os
import osmnx as ox
import pandas as pd
from shapely.geometry import Polygon


def retrieve_schools(G_walk, G_drive, place_name, save_dir, output_folder_base):

    '''
    retrieve information of educational establishment tagged as school on OpenStreetMaps
    '''
    schools = []
    
    save_dir_csv = os.path.join(save_dir, 'csv')
    if not os.path.isdir(save_dir_csv):
        os.mkdir(save_dir_csv)

    save_dir_images = os.path.join(save_dir, 'images')
    schools_folder = os.path.join(save_dir_images, 'schools')
    

    if not os.path.isdir(schools_folder):
        os.mkdir(schools_folder)

    path_schools_csv_file = os.path.join(save_dir_csv, output_folder_base+'.schools.csv')

    if os.path.isfile(path_schools_csv_file):
        
        print('is file schools')
        schools = pd.read_csv(path_schools_csv_file)

    else:

        print('creating file schools')

        tags = {
            'amenity':'school',
        }
        
        poi_schools = ox.geometries_from_place(place_name, tags=tags)
        print('poi schools len', len(poi_schools))
        
        if len(poi_schools) > 0:

            for index, poi in poi_schools.iterrows():

                #print(poi)

                school_point = (poi.geometry.centroid.y, poi.geometry.centroid.x)

                u, v, key = ox.get_nearest_edge(G_walk, school_point)
                school_node_walk = min((u, v), key=lambda n: ox.distance.great_circle_vec(poi.geometry.centroid.y, poi.geometry.centroid.x, G_walk.nodes[n]['y'], G_walk.nodes[n]['x']))
            
                u, v, key = ox.get_nearest_edge(G_drive, school_point)
                school_node_drive = min((u, v), key=lambda n: ox.distance.great_circle_vec(poi.geometry.centroid.y, poi.geometry.centroid.x, G_drive.nodes[n]['y'], G_drive.nodes[n]['x']))
            
                d = {
                    'school_id':index,
                    'school_name':poi['name'],
                    'osmid_walk':school_node_walk,
                    'osmid_drive':school_node_drive,
                    'lat':poi.geometry.centroid.y,
                    'lon':poi.geometry.centroid.x,
                }

                schools.append(d)
                    
            schools = pd.DataFrame(schools)
            schools.to_csv(path_schools_csv_file)
    
    if len(schools) > 0:
        schools.set_index(['school_id'], inplace=True)

    return schools





