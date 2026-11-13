import matplotlib.pyplot as plt
import os
import osmnx as ox
import pandas as pd
from shapely.geometry import Polygon
from osmnx.distance import great_circle
from .nearest import centroid_coordinates, nearest_edge_endpoints
from .osm_features import features_in_area


def retrieve_schools(G_walk, G_drive, place_name, save_dir, output_folder_base, polygon=None):

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
        
        poi_schools = features_in_area(place_name, tags, polygon)
        print('poi schools len', len(poi_schools))
        
        if len(poi_schools) > 0:

            lats, lons = centroid_coordinates(poi_schools)
            nodes_walk = nearest_edge_endpoints(G_walk, lats, lons)
            nodes_drive = nearest_edge_endpoints(G_drive, lats, lons)

            for (index, poi), walk, drive, lat, lon in zip(poi_schools.iterrows(), nodes_walk, nodes_drive, lats, lons):

                d = {
                    #'school_id':index,
                    'school_name':poi['name'],
                    'osmid_walk':walk,
                    'osmid_drive':drive,
                    'lat':lat,
                    'lon':lon,
                }

                schools.append(d)
                    
            schools = pd.DataFrame(schools)
            schools.to_csv(path_schools_csv_file)
    '''
    if len(schools) > 0:
        schools.set_index(['school_id'], inplace=True)
    '''
    
    return schools





