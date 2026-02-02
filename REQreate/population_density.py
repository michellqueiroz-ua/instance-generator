import math
import numpy as np
import os
import pandas as pd

import urllib.request
import zipfile
import random
import itertools
#import shapefile
from shapely.geometry import Polygon
#from descartes.patch import PolygonPatch
from shapely.geometry import Point
import shapely.wkt
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.style.use('ggplot')

#import sqlalchemy as sqla
#from sqlalchemy_utils import database_exists
from datetime import datetime
from operator import mul
from scipy.stats import zscore
import networkx as nx
import osmnx as ox

#import seaborn as sns
#from fitter import Fitter, get_common_distributions, get_distributions
#import powerlaw
from pathlib import Path
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    class DummyRay:
        @staticmethod
        def remote(func):
            return func
        @staticmethod
        def shutdown():
            pass
        @staticmethod
        def init(**kwargs):
            pass
    ray = DummyRay()
from instance_class import Instance
from multiprocessing import cpu_count
import pickle

def attribute_population_density_zones(inst, blocksdf, popdf):

    inst.network.zones['number_population'] = 0
    
    for idx, row in blocksdf.iterrows():
        #pnt = (poi['lat'], poi['lon'])
        pnt = Point(row['lon'], row['lat'])
        for idx2, zone in inst.network.zones.iterrows():

            zone_polygon = zone['polygon']
        
            if zone_polygon.contains(pnt): 
                #print(popdf.loc[popdf['CENSUS BLOCK FULL'] == row['GEOID10']]['TOTAL POPULATION'])
                for idx3, row3 in popdf.iterrows():
                    if row['GEOID10'] == row3['CENSUS BLOCK FULL']:
                        inst.network.zones.loc[idx2, 'number_population'] += row3['TOTAL POPULATION']
                        print(inst.network.zones.loc[idx2, 'number_population'])
                break

    total_sum = inst.network.zones['number_population'].sum()
    print('total_sum population: ', total_sum)

    inst.network.zones['density_pop'] = (inst.network.zones['number_population']/total_sum)*100
    print(inst.network.zones['density_pop'].head())
    print(inst.network.zones['density_pois'].sum())

@ray.remote
def get_osmid_node(G, idx, point):

    osmid_node = ox.nearest_nodes(G, point[1], point[0])

    return (idx, osmid_node)

def add_osmid_nodes(inst, place_name, df):

   
    ray.shutdown()
    ray.init(num_cpus=cpu_count())
    G_drive_id = ray.put(inst.network.G_drive)
    osmid_nodes = ray.get([get_osmid_node.remote(G_drive_id, id1, (row1['lat'], row1['lon'])) for id1, row1 in df.iterrows()]) 

    for pairid in osmid_nodes:

        #origin_point = (row1['Pickup_Centroid_Latitude'], row1['Pickup_Centroid_Longitude'])
        #df.loc[id1, 'osmid_origin'] = ox.get_nearest_node(inst.network.G_drive, origin_point)
        idx = pairid[0]
        osmid_node = pairid[1]
        print(osmid_node)
        df.loc[idx, 'osmid'] = osmid_node
        
    ray.shutdown()

    return df


def read_data_population_density():

    #function that reads the pop density data

    blocksdf = pd.read_csv("CensusBlockTIGER2010.csv")

    popdf = pd.read_csv("Population_by_2010_Census_Block.csv")

    print(blocksdf)
    for index, line in blocksdf.iterrows():
        #print(line)
        #print("hier")
        polygon = shapely.wkt.loads(line['the_geom'])
        #polygon = Polygon(line['the_geom'])
        #print(polygon)
        centroid_polygon = polygon.centroid
        #print(centroid_polygon.x, centroid_polygon.y)
        blocksdf.loc[index, 'lon'] = centroid_polygon.x
        blocksdf.loc[index, 'lat'] = centroid_polygon.y



    place_name = "Chicago, Illinois"
    #print('add osmid nodes')
    save_dir = os.getcwd()+'/'+place_name
    pickle_dir = os.path.join(save_dir, 'pickle')
    network_class_file = pickle_dir+'/'+place_name+'.network.class.pkl'

    network_directory = os.getcwd()+'/'+place_name

    if Path(network_class_file).is_file():
        inst = Instance(folder_to_network=place_name)

    blocksdf = add_osmid_nodes(inst, "Chicago, Illinois", blocksdf)

    print("hier")
    print(blocksdf)

    #ATTRIBUTE TO EACH ZONE ITS POPULATION DENSITY
    attribute_population_density_zones(inst, blocksdf, popdf)

    print('out function attribute')
    print(inst.network.zones['density_pop'].head())

    pickle_dir = os.path.join(save_dir, 'pickle')
    output_folder_base = place_name
    network_class_file = pickle_dir+'/'+output_folder_base+'.network3.class.pkl'
    
    output_network_class = open(network_class_file, 'wb')
    pickle.dump(inst.network, output_network_class, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':


    read_data_population_density()