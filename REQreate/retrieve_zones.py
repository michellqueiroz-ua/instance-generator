import math
import matplotlib.pyplot as plt
import os
import osmnx as ox
import pandas as pd
from shapely.geometry import Polygon


def retrieve_zones(G_walk, G_drive, place_name, save_dir, output_folder_base, BBx, BBy):

    zones = []
    zone_id = 0

    save_dir_csv = os.path.join(save_dir, 'csv')
    if not os.path.isdir(save_dir_csv):
        os.mkdir(save_dir_csv)

    save_dir_images = os.path.join(save_dir, 'images')
    zones_folder = os.path.join(save_dir_images, 'zones')
    
    if not os.path.isdir(zones_folder):
        os.mkdir(zones_folder)

    path_zones_csv_file = os.path.join(save_dir_csv, output_folder_base+'.zones.csv')

    if os.path.isfile(path_zones_csv_file):
        
        print('is file zones')
        zones = pd.read_csv(path_zones_csv_file)

        #updates the polygons
        for index, zone in zones.iterrows():

            earth_radius = 6371009  # meters
            dist_lat = zone['dist_lat']
            dist_lon = zone['dist_lon']  

            lat = zone['center_point_y']
            lng = zone['center_point_x']

            delta_lat = (dist_lat / earth_radius) * (180 / math.pi)
            delta_lng = (dist_lon / earth_radius) * (180 / math.pi) / math.cos(lat * math.pi / 180)
            
            north = lat + delta_lat
            south = lat - delta_lat
            east = lng + delta_lng
            west = lng - delta_lng
                        
            #north, south, east, west = ox.utils_geo.bbox_from_point(zone_center_point, distance)
            polygon = Polygon([(west, south), (east, south), (east, north), (west, north)])
            
            zones.loc[index, 'polygon'] = polygon
        
    else:

        print('creating file zones')

        tags = {
            'place':'borough',
            'place':'suburb',
            'place':'quarter',
            'place':'neighbourhood',
        }
        
        poi_zones = ox.features_from_place(place_name, tags=tags)
        print('poi zones len', len(poi_zones))

        if len(poi_zones) > 0:

            for index, poi in poi_zones.iterrows():
                
                if str(poi['name']) != 'nan':
                    zone_name = str(poi['name'])
                    
                    if not any((z.get('name', None) == zone_name) for z in zones):
                       
                        #future: see what to do with geometries that are not points
                        if poi['geometry'].geom_type == 'Point':
 
                            earth_radius = 6371009  # meters
                            dist_lat = BBx
                            dist_lon = BBy  

                            lat = poi.geometry.centroid.y
                            lng = poi.geometry.centroid.x

                            delta_lat = (dist_lat / earth_radius) * (180 / math.pi)
                            delta_lng = (dist_lon / earth_radius) * (180 / math.pi) / math.cos(lat * math.pi / 180)
                            
                            north = lat + delta_lat
                            south = lat - delta_lat
                            east = lng + delta_lng
                            west = lng - delta_lng

                            polygon = Polygon([(west, south), (east, south), (east, north), (west, north)])

                            zone_center_point = (poi.geometry.centroid.y, poi.geometry.centroid.x)
                            
                            #osmid nearest node walk
                            osmid_walk = ox.nearest_nodes(G_walk, zone_center_point[1], zone_center_point[0]) 

                            #osmid nearest node drive
                            osmid_drive = ox.nearest_nodes(G_drive, zone_center_point[1], zone_center_point[0])

                            #plot here the center point zone in the walk network
                            nc = ['r' if (node == osmid_walk) else '#336699' for node in G_walk.nodes()]
                            ns = [16 if (node == osmid_walk) else 1 for node in G_walk.nodes()]
                            zone_filename = str(zone_id)+'_'+zone_name+'_walk.png'
                            fig, ax = ox.plot_graph(G_walk, node_size=ns, show=False, node_color=nc, node_zorder=2, save=True, filepath=zones_folder+'/'+zone_filename)
                            plt.close(fig)

                            #plot here the center point zone in the drive network
                            nc = ['r' if (node == osmid_drive) else '#336699' for node in G_drive.nodes()]
                            ns = [16 if (node == osmid_drive) else 1 for node in G_drive.nodes()]
                            zone_filename = str(zone_id)+'_'+zone_name+'_drive.png'
                            fig, ax = ox.plot_graph(G_drive, node_size=ns, show=False, node_color=nc, node_zorder=2, save=True, filepath=zones_folder+'/'+zone_filename)
                            plt.close(fig)

                            n = {
                                'index': index,
                                'id': zone_id,
                                'name': zone_name,
                                'polygon': polygon,
                                'center_point_y': poi.geometry.centroid.y,
                                'center_point_x': poi.geometry.centroid.x,
                                'osmid_walk': osmid_walk,
                                'osmid_drive': osmid_drive,
                                'dist_lat': dist_lat,
                                'dist_lon': dist_lon,
                            }

                            zone_id += 1

                            zones.append(n)
                
            zones = pd.DataFrame(zones)
            zones.to_csv(path_zones_csv_file)
    
    if len(zones) > 0:
        zones.set_index(['id'], inplace=True)

    return zones





