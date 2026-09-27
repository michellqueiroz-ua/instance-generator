import matplotlib.pyplot as plt
import os
import osmnx as ox
import pandas as pd
from shapely.geometry import Polygon

from . import snap


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
        
        poi_schools = ox.features_from_place(place_name, tags=tags)
        print('poi schools len', len(poi_schools))
        
        if len(poi_schools) > 0:

            # Snap every school in one pass. Per-point nearest_edges rebuilt the
            # walk network's edge index once per school; see REQreate/snap.py.
            rows = list(poi_schools.iterrows())
            centroids = [poi.geometry.centroid for _, poi in rows]
            lons = [c.x for c in centroids]
            lats = [c.y for c in centroids]
            nodes_walk = snap.nearest_edge_endpoints(G_walk, lons, lats, 'schools.walk')
            nodes_drive = snap.nearest_edge_endpoints(G_drive, lons, lats, 'schools.drive')

            for (index, poi), centroid, school_node_walk, school_node_drive in zip(
                    rows, centroids, nodes_walk, nodes_drive):

                d = {
                    #'school_id':index,
                    'school_name':poi['name'],
                    'osmid_walk':school_node_walk,
                    'osmid_drive':school_node_drive,
                    'lat':centroid.y,
                    'lon':centroid.x,
                }

                schools.append(d)
                    
            schools = pd.DataFrame(schools)
            schools.to_csv(path_schools_csv_file)
    '''
    if len(schools) > 0:
        schools.set_index(['school_id'], inplace=True)
    '''
    
    return schools





