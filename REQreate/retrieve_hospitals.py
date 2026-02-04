import matplotlib.pyplot as plt
import os
import osmnx as ox
import pandas as pd
from shapely.geometry import Polygon
from osmnx.distance import great_circle


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
                for index, poi in poi_hospitals.iterrows():
                    hospital_point = (poi.geometry.centroid.y, poi.geometry.centroid.x)

                    # Find nearest nodes in walk and drive networks
                    u, v, key = ox.nearest_edges(G_walk, hospital_point[1], hospital_point[0])
                    hospital_node_walk = min((u, v), key=lambda n: great_circle(
                        poi.geometry.centroid.y, 
                        poi.geometry.centroid.x, 
                        G_walk.nodes[n]['y'], 
                        G_walk.nodes[n]['x']
                    ))
                
                    u, v, key = ox.nearest_edges(G_drive, hospital_point[1], hospital_point[0])
                    hospital_node_drive = min((u, v), key=lambda n: great_circle(
                        poi.geometry.centroid.y, 
                        poi.geometry.centroid.x, 
                        G_drive.nodes[n]['y'], 
                        G_drive.nodes[n]['x']
                    ))
                
                    # Get name, use amenity type as fallback
                    hospital_name = poi.get('name', f"{poi.get('amenity', 'hospital')}_{index}")
                    
                    d = {
                        'hospital_name': hospital_name,
                        'amenity_type': poi.get('amenity', 'hospital'),
                        'osmid_walk': hospital_node_walk,
                        'osmid_drive': hospital_node_drive,
                        'lat': poi.geometry.centroid.y,
                        'lon': poi.geometry.centroid.x,
                    }

                    hospitals.append(d)
                        
                hospitals = pd.DataFrame(hospitals)
                hospitals.to_csv(path_hospitals_csv_file, index=False)
                print(f'Saved {len(hospitals)} hospitals to {path_hospitals_csv_file}')
        
        except Exception as e:
            print(f'Error retrieving hospitals: {e}')
            hospitals = pd.DataFrame(columns=['hospital_name', 'amenity_type', 'osmid_walk', 'osmid_drive', 'lat', 'lon'])
    
    return hospitals
