import matplotlib.pyplot as plt
import os
import osmnx as ox
import pandas as pd
from shapely.geometry import Polygon

from . import snap


def retrieve_hospitals(G_walk, G_drive, place_name, save_dir, output_folder_base):
    '''
    Retrieve information of hospitals from OpenStreetMap.
    Includes hospitals, clinics, and other healthcare facilities.
    '''
    hospitals = []
    
    save_dir_csv = os.path.join(save_dir, 'csv')
    if not os.path.isdir(save_dir_csv):
        os.mkdir(save_dir_csv)

    save_dir_images = os.path.join(save_dir, 'images')
    hospitals_folder = os.path.join(save_dir_images, 'hospitals')
    
    if not os.path.isdir(hospitals_folder):
        os.mkdir(hospitals_folder)

    path_hospitals_csv_file = os.path.join(save_dir_csv, output_folder_base+'.hospitals.csv')

    if os.path.isfile(path_hospitals_csv_file):
        print('Hospital data already exists, loading from file')
        hospitals = pd.read_csv(path_hospitals_csv_file)
    else:
        print('Retrieving hospital data from OpenStreetMap')

        # Tags for healthcare facilities
        tags = {
            'amenity': ['hospital', 'clinic', 'doctors'],
        }
        
        try:
            poi_hospitals = ox.features_from_place(place_name, tags=tags)
            print(f'Found {len(poi_hospitals)} healthcare facilities')
            
            if len(poi_hospitals) > 0:
                # Snap every facility in one pass. Per-point nearest_edges rebuilt
                # the network's edge index once per facility; see REQreate/snap.py.
                rows = list(poi_hospitals.iterrows())
                centroids = [poi.geometry.centroid for _, poi in rows]
                lons = [c.x for c in centroids]
                lats = [c.y for c in centroids]
                nodes_walk = snap.nearest_edge_endpoints(G_walk, lons, lats, 'hospitals.walk')
                nodes_drive = snap.nearest_edge_endpoints(G_drive, lons, lats, 'hospitals.drive')

                for (index, poi), centroid, hospital_node_walk, hospital_node_drive in zip(
                        rows, centroids, nodes_walk, nodes_drive):

                    # Get name, use amenity type as fallback
                    hospital_name = poi.get('name', f"{poi.get('amenity', 'hospital')}_{index}")
                    
                    d = {
                        'hospital_name': hospital_name,
                        'amenity_type': poi.get('amenity', 'hospital'),
                        'osmid_walk': hospital_node_walk,
                        'osmid_drive': hospital_node_drive,
                        'lat': centroid.y,
                        'lon': centroid.x,
                    }

                    hospitals.append(d)
                        
                hospitals = pd.DataFrame(hospitals)
                hospitals.to_csv(path_hospitals_csv_file, index=False)
                print(f'Saved {len(hospitals)} hospitals to {path_hospitals_csv_file}')
        
        except Exception as e:
            print(f'Error retrieving hospitals: {e}')
            hospitals = pd.DataFrame(columns=['hospital_name', 'amenity_type', 'osmid_walk', 'osmid_drive', 'lat', 'lon'])
    
    return hospitals
