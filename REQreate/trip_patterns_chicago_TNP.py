import math
import numpy as np
import os
import pandas as pd

import urllib.request
import zipfile
import random
import itertools
import shapefile
from shapely.geometry import Polygon
from descartes.patch import PolygonPatch
from shapely.geometry import Point
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.style.use('ggplot')

import sqlalchemy as sqla
from sqlalchemy_utils import database_exists
from datetime import datetime
from operator import mul
from scipy.stats import zscore
import networkx as nx
import osmnx as ox

import seaborn as sns
from fitter import Fitter, get_common_distributions, get_distributions
import powerlaw
from pathlib import Path
from instance_class import Instance

from retrieve_POIs import get_POIs_matrix_csv
from retrieve_POIs import attribute_density_zones
from retrieve_POIs import calc_rank_between_zones
from retrieve_POIs import calc_probability_travel_between_zones
from retrieve_POIs import rank_of_displacements
from multiprocessing import cpu_count
from scipy.stats.kde import gaussian_kde

import ray
import gc

from matplotlib.colors import LogNorm
import pickle


def rank_model(place_name, df):

    output_folder_base = place_name
    save_dir = os.getcwd()+'/'+place_name
    pickle_dir = os.path.join(save_dir, 'pickle')
    network_class_file = pickle_dir+'/'+place_name+'.network.class.pkl'

    network_directory = os.getcwd()+'/'+place_name

    if Path(network_class_file).is_file():
        inst = Instance(folder_to_network=place_name)

    for node in inst.network.G_drive.nodes():
        #print(inst.network.G_drive.nodes[node])
        try:
            
            print(inst.network.G_drive.nodes[node]['population'])
            #print('found')
        except:
            #print('not found')
            pass


    rows = 25
    columns = 25
    print('smaller zones')
    inst.network.zones = inst.network.divide_network_grid(rows, columns, save_dir, output_folder_base)

    del inst.network.G_walk
    del inst.network.shortest_path_walk
    gc.collect()
    pois = get_POIs_matrix_csv(inst.network.G_drive, place_name, save_dir, output_folder_base)
    attribute_density_zones(inst, pois)
    #del attribute_density_zones

    zone_ranks = calc_rank_between_zones(inst)
    del pois
    #del calc_rank_between_zones
    gc.collect()

    alpha = 0.84
    print('zone_probabilities')
    zone_probabilities = calc_probability_travel_between_zones(inst, zone_ranks, alpha)
    #del calc_probability_travel_between_zones
    gc.collect()
    #df = pd.read_sql_query('SELECT Pickup_Centroid_Latitude, Pickup_Centroid_Longitude, Dropoff_Centroid_Latitude, Dropoff_Centroid_Longitude FROM table_record', database)
    
    '''
    print('rank displacements')           
    df = rank_of_displacements(inst, zone_ranks, df)

    print(df['rank_trip'].describe())
    print(df['rank_trip'].mean())
    print(df['rank_trip'].std())

    plt.close()
    ax = df['rank_trip'].hist(bins=30, figsize=(15,5))
    ax.set_yscale('log')
    ax.set_xlabel("trip rank")
    ax.set_ylabel("count")
    plt.savefig('rank_trips.png')
    plt.close()

    x = df['rank_trip'].values
    pdf = gaussian_kde(x)
    y = pdf(x)
    plt.scatter(x, y)
    plt.savefig('scatter_rank_trips.png')
    '''



    #dumping updated network file
    print(inst.network.zones['density_pois'].head())
    inst.network.zone_ranks = zone_ranks
    inst.network.zone_probabilities = zone_probabilities
    
    pickle_dir = os.path.join(save_dir, 'pickle')
    network_class_file = pickle_dir+'/'+output_folder_base+'.network3.class.pkl'
    
    output_network_class = open(network_class_file, 'wb')
    pickle.dump(inst.network, output_network_class, pickle.HIGHEST_PROTOCOL)
    
@ray.remote
def get_osmid_node(G, idx, point):

    osmid_node = ox.nearest_nodes(G, point[1], point[0])

    return (idx, osmid_node)

def add_osmid_nodes(place_name, df):

    #print('add osmid nodes')
    save_dir = os.getcwd()+'/'+place_name
    pickle_dir = os.path.join(save_dir, 'pickle')
    network_class_file = pickle_dir+'/'+place_name+'.network.class.pkl'

    network_directory = os.getcwd()+'/'+place_name

    if Path(network_class_file).is_file():
        inst = Instance(folder_to_network=place_name)

    ray.shutdown()
    ray.init(num_cpus=cpu_count())
    G_drive_id = ray.put(inst.network.G_drive)
    osmid_origins = ray.get([get_osmid_node.remote(G_drive_id, id1, (row1['Pickup_Centroid_Latitude'], row1['Pickup_Centroid_Longitude'])) for id1, row1 in df.iterrows()]) 

    ray.shutdown()
    ray.init(num_cpus=cpu_count())
    G_drive_id = ray.put(inst.network.G_drive)
    osmid_destinations = ray.get([get_osmid_node.remote(G_drive_id, id1, (row1['Dropoff_Centroid_Latitude'], row1['Dropoff_Centroid_Longitude'])) for id1, row1 in df.iterrows()]) 

    for pairid in osmid_origins:

        #origin_point = (row1['Pickup_Centroid_Latitude'], row1['Pickup_Centroid_Longitude'])
        #df.loc[id1, 'osmid_origin'] = ox.get_nearest_node(inst.network.G_drive, origin_point)
        idx = pairid[0]
        osmid_node = pairid[1]
        df.loc[idx, 'osmid_origin'] = osmid_node

    for pairid in osmid_destinations:

        #destination_point = (row1['Dropoff_Centroid_Latitude'], row1['Dropoff_Centroid_Longitude'])
        #df.loc[id1, 'osmid_destination'] = ox.get_nearest_node(inst.network.G_drive, destination_point)
        idx = pairid[0]
        osmid_node = pairid[1]
        df.loc[idx, 'osmid_destination'] = osmid_node
        
    ray.shutdown()

    return df

def new_heatmap(place_name, dfc):

    save_dir = os.getcwd()+'/'+place_name
    pickle_dir = os.path.join(save_dir, 'pickle')
    network_class_file = pickle_dir+'/'+place_name+'.network.class.pkl'

    network_directory = os.getcwd()+'/'+place_name

    if Path(network_class_file).is_file():
        inst = Instance(folder_to_network=place_name)
    
    
    #group by IDs and count
    #df_og = pd.read_sql_query('SELECT Pickup_Centroid_Latitude, Pickup_Centroid_Longitude FROM table_record', database)
    
    #df_de = pd.read_sql_query('SELECT Dropoff_Centroid_Latitude, Dropoff_Centroid_Longitude FROM table_record', database)
    df_og = dfc

    df_de = dfc

    pts_bhx = []
    pts_bhy = []   

    pts_og = []
    pts_ogx = []
    pts_ogy = []    
    for idx, row in df_og.iterrows():

        pt = (row['Pickup_Centroid_Longitude'], row['Pickup_Centroid_Latitude'])
        pts_og.append(pt)
        pts_ogx.append(row['Pickup_Centroid_Longitude'])
        pts_ogy.append(row['Pickup_Centroid_Latitude'])

        pts_bhx.append(row['Pickup_Centroid_Longitude'])
        pts_bhy.append(row['Pickup_Centroid_Latitude'])

    pts_de = []
    pts_dex = []   
    pts_dey = []      
    for idx, row in df_de.iterrows():

        pt = (row['Dropoff_Centroid_Longitude'], row['Dropoff_Centroid_Latitude'])
        pts_de.append(pt)
        pts_dex.append(row['Dropoff_Centroid_Longitude'])
        pts_dey.append(row['Dropoff_Centroid_Latitude'])

        pts_bhx.append(row['Dropoff_Centroid_Longitude'])
        pts_bhy.append(row['Dropoff_Centroid_Latitude'])

    minx, miny, maxx, maxy = inst.network.polygon.bounds
    #hm = Heatmap(libpath="cHeatmap.cpython-38-x86_64-linux-gnu.so")
    #img = hm.heatmap(pts_og, scheme='classic', dotsize=75, opacity=128, area=((minx, miny), (maxx, maxy)))
    #img.save("heatmap_og.png")

    #hm = Heatmap(libpath="cHeatmap.cpython-38-x86_64-linux-gnu.so")
    #img = hm.heatmap(pts_de, scheme='classic', dotsize=75, opacity=128, area=((minx, miny), (maxx, maxy)))
    #img.save("heatmap_de.png")

    print(' len points ', len(pts_ogx))
    print(' len points ', len(pts_bhx))
    #plt.hist2d(pts_ogx,pts_ogy, bins=[np.arange(minx,maxx,5),np.arange(miny,maxy,5)])
    '''
    h = plt.hist2d(pts_ogx,pts_ogy, bins=25, norm=LogNorm(), cmap='jet')
    plt.colorbar(h[3])
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.savefig('heatmap_origin.png')
    plt.close()

    #plt.hist2d(pts_dex,pts_dey, bins=[np.arange(minx,maxx,10),np.arange(miny,maxy,10)])
    h = plt.hist2d(pts_dex,pts_dey, bins=25, norm=LogNorm(), cmap='jet')
    plt.colorbar(h[3])
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.savefig('heatmap_destination.png')
    plt.close()
    '''

    h = plt.hist2d(pts_bhx,pts_bhy, bins=25, norm=LogNorm(), cmap='jet')
    plt.colorbar(h[3])
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.savefig('heatmap_both.png')
    plt.close()

    #curr_folder = os.getcwd()

    #fig, ax = ox.plot_graph(inst.network.G_drive,figsize=(8, 8), dpi=128, show=False, filepath='heatmap_origin_points.png', save=True)

    #fig, ax = ox.plot_graph(inst.network.G_drive,figsize=(8, 8), dpi=128, show=False, filepath='heatmap_destination_points.png', save=True)

def heatmap_osmnx(place_name, database):

    save_dir = os.getcwd()+'/'+place_name
    pickle_dir = os.path.join(save_dir, 'pickle')
    network_class_file = pickle_dir+'/'+place_name+'.network.class.pkl'

    network_directory = os.getcwd()+'/'+place_name

    if Path(network_class_file).is_file():
        inst = Instance(folder_to_network=place_name)
    
    #group by IDs and count
    df_og = pd.read_sql_query('SELECT osmid_origin AS osmid, count(*) AS OGcount \
                        FROM table_record \
                        GROUP BY osmid_origin', database)
    
    df_de = pd.read_sql_query('SELECT osmid_destination AS osmid, count(*) AS DEcount \
                        FROM table_record \
                        GROUP BY osmid_destination', database)
    
    #print('before')
    #print(df_og.head())

    df_og['osmid'] = df_og['osmid'].astype(int)
    df_de['osmid'] = df_de['osmid'].astype(int)
    df_og.set_index(['osmid'], inplace=True)
    df_de.set_index(['osmid'], inplace=True)

    #print('after')
    #print(df_og.head())

    for node in inst.network.G_drive.nodes():

        try:
            
            inst.network.G_drive.nodes[node]['OGcount'] = df_og.loc[node, 'OGcount']
        
        except KeyError:

            inst.network.G_drive.nodes[node]['OGcount'] = 0

        try:
            
            inst.network.G_drive.nodes[node]['DEcount'] = df_de.loc[node, 'DEcount']
        
        except KeyError:

            inst.network.G_drive.nodes[node]['DEcount'] = 0
    
     

    #do the heatmap (one for pickup one for dropoff)
    #Make geodataframes from graph data
    nodes, edges = ox.graph_to_gdfs(inst.network.G_drive, nodes=True, edges=True)

    print('OGCount')
    print(nodes['OGcount'])

    curr_folder = os.getcwd()

    #Then plot a graph where node size and node color are related to the number of visits
    nc = ox.plot.get_node_colors_by_attr(inst.network.G_drive,'OGcount',num_bins=40)
    fig, ax = ox.plot_graph(inst.network.G_drive,figsize=(8, 8),node_size=nodes['OGcount'], node_color=nc, show=False,filepath='heatmap_origin_points.png', save=True)

    #plt.savefig('heatmap_origin_points.png')
    #plt.close(fig)

    nc = ox.plot.get_node_colors_by_attr(inst.network.G_drive,'DEcount',num_bins=40)
    fig, ax = ox.plot_graph(inst.network.G_drive,figsize=(8, 8),node_size=nodes['DEcount'], node_color=nc, show=False, filepath='heatmap_destination_points.png', save=True)

    #plt.savefig('heatmap_destination_points.png')
    #plt.close(fig)

def remove_false_records(df):

    df = df.loc[(df['Trip_Miles'] > 0.3)]

    df['Fare'] = df['Fare'].astype(str)
    new = df["Fare"].str.split("$", n = 1, expand = True)

    df['Fare'] = new[0]
    df['Fare'] = df['Fare'].astype(float)
    df.dropna(subset=['Fare'], inplace=True)
    df['Fare'] = df['Fare'].astype(int)

    #print(df['Fare'].head())

    df = df.loc[(df['Fare'] > 0)]

    df = df.loc[(df['ih'] <= df['idh'])]

    df.dropna(subset=['Trip_Seconds'], inplace=True)
    df['Trip_Seconds'] = df['Trip_Seconds'].astype(int)
    #print(df['Trip_Seconds'].head())
    df = df.loc[(df['Trip_Seconds'] > 0)]

    df['Pickup_Centroid_Latitude'].replace('', np.nan, inplace=True)
    df['Pickup_Centroid_Longitude'].replace('', np.nan, inplace=True)
    df['Dropoff_Centroid_Latitude'].replace('', np.nan, inplace=True)
    df['Dropoff_Centroid_Longitude'].replace('', np.nan, inplace=True)
    df.dropna(subset=['Pickup_Centroid_Latitude'], inplace=True)
    df.dropna(subset=['Pickup_Centroid_Longitude'], inplace=True)
    df.dropna(subset=['Dropoff_Centroid_Latitude'], inplace=True)
    df.dropna(subset=['Dropoff_Centroid_Longitude'], inplace=True)

    return df

def powelaw_best_fitting_distribution(dists):
    
    results = powerlaw.Fit(dists)

    #print(results.power_law.alpha)
    #print(results.power_law.xmin)
    #print(results.truncated_power_law.parameter1_name)
    #print(results.truncated_power_law.parameter2_name)
    #print(results.truncated_power_law.parameter3_name)
    print(results.supported_distributions)
    R, p = results.distribution_compare('power_law', 'truncated_power_law')
    print(R, p)

def Fitter_best_fitting_distribution(dists):
    f = Fitter(dists, timeout=180, distributions= get_common_distributions())
    #f = Fitter(dists, timeout=180)

    bins = 100
    density = True 

    f.fit()
    #print(f.summary(plot=True))
    _summary, _data = f.summary(plot=True)

    print(f.get_best(method = 'sumsquare_error'))

    return _data

def ratio_eta_real_time(place_name, df_ratio):

    save_dir = os.getcwd()+'/'+place_name
    pickle_dir = os.path.join(save_dir, 'pickle')
    network_class_file = pickle_dir+'/'+place_name+'.network.class.pkl'

    network_directory = os.getcwd()+'/'+place_name

    if Path(network_class_file).is_file():
        inst = Instance(folder_to_network=place_name)

    ratios = []
    for id1, row1 in df_ratio.iterrows():

        #origin_point = (row1['Pickup_Centroid_Latitude'], row1['Pickup_Centroid_Longitude'])
        node_origin = row1['osmid_origin']

        #destination_point = (row1['Dropoff_Centroid_Latitude'], row1['Dropoff_Centroid_Longitude'])
        node_destination = row1['osmid_destination']

        speed = 6.94444 #Mps
        #eta = inst.network._return_estimated_travel_time_drive(int(node_origin), int(node_destination))
        eta = inst.network._return_estimated_distance_drive(int(node_origin), int(node_destination))
        eta = eta/speed

        real = row1['Trip_Seconds']

        if eta > 0:
            ratio = real/eta
            ratios.append(ratio)
            #print(real) 

    #print(ratios)
    mean = sum(ratios) / len(ratios)
    variance = sum([((x - mean) ** 2) for x in ratios]) / len(ratios)
    res = variance ** 0.5

    print('mean ratio', mean)
    print('std ratio', res)

@ray.remote
def GD_single(network, inst1, idx1, row1, earliest_departure, time_gap):

    ltro = []
    ltrd = []
    
    #print(count)
    for idx2, row2 in inst1.iterrows():

        if idx2 != idx1:

            latest_arrival1 = row1[earliest_departure] + row1['Trip_Seconds']
            latest_arrival2 = row2[earliest_departure] + row2['Trip_Seconds']
            #print(row2['earliest_departure'])
            if ((row2[earliest_departure] >= row1[earliest_departure] - time_gap) and (row2[earliest_departure] <= row1[earliest_departure] + time_gap)):
                #if (row2['originnode_drive'] != row1['originnode_drive']) and (row2['originnode_drive'] != row1['destinationnode_drive']):
                #ltro.append(row2['originnode_drive'])
                #origin_point = (row2['Pickup_Centroid_Latitude'], row2['Pickup_Centroid_Longitude'])
                node_origin = row2['osmid_origin']
    
                ltro.append(node_origin)

            if ((latest_arrival2 >= row1[earliest_departure] - time_gap) and (latest_arrival2 <= row1[earliest_departure] + time_gap)):
                #if (row2['destinationnode_drive'] != row1['originnode_drive']) and (row2['destinationnode_drive'] != row1['destinationnode_drive']):
                #ltro.append(row2['destinationnode_drive'])
                #destination_point = (row2['Dropoff_Centroid_Latitude'], row2['Dropoff_Centroid_Longitude'])
                node_destination = row2['osmid_destination']
                
                ltro.append(node_destination)

            if ((latest_arrival2 >= latest_arrival1 - time_gap) and (latest_arrival2 <= latest_arrival1 + time_gap)):
                #if (row2['destinationnode_drive'] != row1['originnode_drive']) and (row2['destinationnode_drive'] != row1['destinationnode_drive']):
                #ltrd.append(row2['destinationnode_drive'])
                #destination_point = (row2['Dropoff_Centroid_Latitude'], row2['Dropoff_Centroid_Longitude'])
                node_destination = row2['osmid_destination']

                ltro.append(node_destination)

            if ((row2[earliest_departure] >= latest_arrival1 - time_gap) and (row2[earliest_departure] <= latest_arrival1 + time_gap)):
                #if (row2['originnode_drive'] != row1['originnode_drive']) and (row2['originnode_drive'] != row1['destinationnode_drive']):
                #ltrd.append(row2['originnode_drive'])
                #origin_point = (row2['Pickup_Centroid_Latitude'], row2['Pickup_Centroid_Longitude'])
                node_origin = row2['osmid_origin']

                ltro.append(node_origin)

    #count += 1
    #ltro = list(dict.fromkeys(ltro))
    #ltrd = list(dict.fromkeys(ltrd))
    #print(ltro)
    #print(ltrd)

    ltrot = []
    ltrdt = []
    
    #org_row1 = int(row1['originnode_drive'])
    #origin_point = (row1['Pickup_Centroid_Latitude'], row1['Pickup_Centroid_Longitude'])
    #org_row1 = ox.get_nearest_node(inst.network.G_drive, origin_point)
    org_row1 = row1['osmid_origin']
    
    for x in ltro:

        #tuplx = (x, inst.network._return_estimated_travel_time_drive(int(org_row1), int(x)))
        tuplx = (x, network._return_estimated_distance_drive(int(org_row1), int(x)))
        ltrot.append(tuplx)

    #dest_row1 = int(row1['destinationnode_drive'])
    #destination_point = (row1['Dropoff_Centroid_Latitude'], row1['Dropoff_Centroid_Longitude'])
    #dest_row1 = ox.get_nearest_node(inst.network.G_drive, destination_point)
    dest_row1 = row1['osmid_destination']

    for y in ltrd:

        #tuply = (y, inst.network._return_estimated_travel_time_drive(int(dest_row1), int(y)))
        tuply = (y, network._return_estimated_distance_drive(int(dest_row1), int(y)))
        ltrdt.append(tuply)


    #ordenar as tuplas
    ltrot.sort(key = lambda x: x[1]) 
    ltrdt.sort(key = lambda x: x[1])
    
    #pegar a media das 5 primeiras
    n_neig = 5
    avgo = 0
    for i in range(min(n_neig, len(ltrot))):
        avgo += ltrot[i][1]
    
    if len(ltrot) > 0:
        avgo = avgo/min(n_neig, len(ltrot))

    avgd = 0
    for j in range(min(n_neig, len(ltrdt))):
        avgd += ltrdt[j][1]
    #adicionar numa variavel de soma
    if len(ltrdt) > 0:
        avgd = avgd/min(n_neig, len(ltrdt))
    
    #print(avgo, avgd)
    #print(avgd)
    #sumnn += avgo + avgd
    return avgo + avgd

def geographic_dispersion(place_name, inst1, day):

    save_dir = os.getcwd()+'/'+place_name
    pickle_dir = os.path.join(save_dir, 'pickle')
    network_class_file = pickle_dir+'/'+place_name+'.network.class.pkl'

    network_directory = os.getcwd()+'/'+place_name

    if Path(network_class_file).is_file():
        inst = Instance(folder_to_network=place_name)

    csv_directory = network_directory+'/csv_format'
    ttm_directory = network_directory+'/travel_time_matrix'
    directory = os.fsencode(csv_directory)


    #ttm_file_inst1 = 'travel_time_matrix_'+filename1
    #ttmfilename1 = os.fsdecode(ttm_file_inst1)
    #ttm1 = pd.read_csv(ttm_directory+'/'+ttmfilename1)
    #ttm1.set_index(['osmid_origin'], inplace=True)


    #mu
    #average travel time between origin and destinations
    dtt = []
    for idx, req in inst1.iterrows():
        dtt.append(req['Trip_Seconds'])


    mudarp = sum(dtt) / len(dtt)
    mu2 = inst1['Trip_Seconds'].mean()

    #average travel time between x nearest neighbors
    #nyc -> compute for the 5 nearest zones
    earliest_departure = 'pu_time_sec'
    #latest_arrival = 'do_time_sec'
    time_gap = 600
    #node_origin = 
    #node_destination = 
    
    sumnn = 0
    count = 0
    '''
    for idx1, row1 in inst1.iterrows():

        ltro = []
        ltrd = []
        print(count)
        for idx2, row2 in inst1.iterrows():

            if idx2 != idx1:

                latest_arrival1 = row1[earliest_departure] + row1['Trip_Seconds']
                latest_arrival2 = row2[earliest_departure] + row2['Trip_Seconds']
                #print(row2['earliest_departure'])
                if ((row2[earliest_departure] >= row1[earliest_departure] - time_gap) and (row2[earliest_departure] <= row1[earliest_departure] + time_gap)):
                    #if (row2['originnode_drive'] != row1['originnode_drive']) and (row2['originnode_drive'] != row1['destinationnode_drive']):
                    #ltro.append(row2['originnode_drive'])
                    #origin_point = (row2['Pickup_Centroid_Latitude'], row2['Pickup_Centroid_Longitude'])
                    node_origin = row2['osmid_origin']
        
                    ltro.append(node_origin)

                if ((latest_arrival2 >= row1[earliest_departure] - time_gap) and (latest_arrival2 <= row1[earliest_departure] + time_gap)):
                    #if (row2['destinationnode_drive'] != row1['originnode_drive']) and (row2['destinationnode_drive'] != row1['destinationnode_drive']):
                    #ltro.append(row2['destinationnode_drive'])
                    #destination_point = (row2['Dropoff_Centroid_Latitude'], row2['Dropoff_Centroid_Longitude'])
                    node_destination = row2['osmid_destination']
                    
                    ltro.append(node_destination)

                if ((latest_arrival2 >= latest_arrival1 - time_gap) and (latest_arrival2 <= latest_arrival1 + time_gap)):
                    #if (row2['destinationnode_drive'] != row1['originnode_drive']) and (row2['destinationnode_drive'] != row1['destinationnode_drive']):
                    #ltrd.append(row2['destinationnode_drive'])
                    #destination_point = (row2['Dropoff_Centroid_Latitude'], row2['Dropoff_Centroid_Longitude'])
                    node_destination = row2['osmid_destination']

                    ltro.append(node_destination)

                if ((row2[earliest_departure] >= latest_arrival1 - time_gap) and (row2[earliest_departure] <= latest_arrival1 + time_gap)):
                    #if (row2['originnode_drive'] != row1['originnode_drive']) and (row2['originnode_drive'] != row1['destinationnode_drive']):
                    #ltrd.append(row2['originnode_drive'])
                    #origin_point = (row2['Pickup_Centroid_Latitude'], row2['Pickup_Centroid_Longitude'])
                    node_origin = row2['osmid_origin']

                    ltro.append(node_origin)

        count += 1
        #ltro = list(dict.fromkeys(ltro))
        #ltrd = list(dict.fromkeys(ltrd))
        #print(ltro)
        #print(ltrd)

        ltrot = []
        ltrdt = []
        
        #org_row1 = int(row1['originnode_drive'])
        #origin_point = (row1['Pickup_Centroid_Latitude'], row1['Pickup_Centroid_Longitude'])
        #org_row1 = ox.get_nearest_node(inst.network.G_drive, origin_point)
        org_row1 = row1['osmid_origin']
        
        for x in ltro:

            #tuplx = (x, inst.network._return_estimated_travel_time_drive(int(org_row1), int(x)))
            tuplx = (x, inst.network._return_estimated_distance_drive(int(org_row1), int(x)))
            ltrot.append(tuplx)

        #dest_row1 = int(row1['destinationnode_drive'])
        #destination_point = (row1['Dropoff_Centroid_Latitude'], row1['Dropoff_Centroid_Longitude'])
        #dest_row1 = ox.get_nearest_node(inst.network.G_drive, destination_point)
        dest_row1 = row1['osmid_destination']

        for y in ltrd:

            #tuply = (y, inst.network._return_estimated_travel_time_drive(int(dest_row1), int(y)))
            tuply = (y, inst.network._return_estimated_distance_drive(int(dest_row1), int(y)))
            ltrdt.append(tuply)


        #ordenar as tuplas
        ltrot.sort(key = lambda x: x[1]) 
        ltrdt.sort(key = lambda x: x[1])
        
        #pegar a media das 5 primeiras
        n_neig = 5
        avgo = 0
        for i in range(min(n_neig, len(ltrot))):
            avgo += ltrot[i][1]
        
        if len(ltrot) > 0:
            avgo = avgo/min(n_neig, len(ltrot))

        avgd = 0
        for j in range(min(n_neig, len(ltrdt))):
            avgd += ltrdt[j][1]
        #adicionar numa variavel de soma
        if len(ltrdt) > 0:
            avgd = avgd/min(n_neig, len(ltrdt))
        
        #print(avgo, avgd)
        #print(avgd)
        sumnn += avgo + avgd
    '''

    ray.shutdown()
    ray.init(num_cpus=cpu_count())
    del inst.network.G_walk
    del inst.network.shortest_path_walk
    gc.collect()
    inst1_id = ray.put(inst1)
    network_id = ray.put(inst.network)
    
    values = ray.get([GD_single.remote(network_id, inst1_id, idx1, row1, earliest_departure, time_gap)  for idx1, row1 in inst1.iterrows()]) 

    #calc sum here
    for value in values:
        sumnn += value

    omegadarp = sumnn/(len(inst1)*2)
    #ttm1['mean'] = ttm1.mean(axis=1)
    #varchi = 0.7
    #omega = ttm1['mean'].mean()
    
    #print(mudarp)
    #print(omegadarp)
    gd = mudarp + omegadarp
    print(gd)
    return gd

@ray.remote
def single_edge(network, inst1, id1, o1, d1, req1, id2, req2, thtt, thts, the, earliest_departure, number_reqs):

    #o2 = req2['originnode_drive']
    #d2 = req2['destinationnode_drive']

    #origin_point = (req2['Pickup_Centroid_Latitude'], req2['Pickup_Centroid_Longitude'])
    #o2 = ox.get_nearest_node(inst.network.G_drive, origin_point)

    #destination_point = (req2['Dropoff_Centroid_Latitude'], req2['Dropoff_Centroid_Longitude'])
    #d2 = ox.get_nearest_node(inst.network.G_drive, destination_point)
    o2 = req2['osmid_origin'] 
    d2 = req2['osmid_destination']

    #oott = inst.network._return_estimated_travel_time_drive(int(o1), int(o2))  
    #ddtt = inst.network._return_estimated_travel_time_drive(int(d1), int(d2)) 

    #oott2 = inst.network._return_estimated_travel_time_drive(int(o2), int(o1))  
    #ddtt2 = inst.network._return_estimated_travel_time_drive(int(d2), int(d1))

    oott = network._return_estimated_distance_drive(int(o1), int(o2))  
    ddtt = network._return_estimated_distance_drive(int(d1), int(d2)) 

    oott2 = network._return_estimated_distance_drive(int(o2), int(o1))  
    ddtt2 = network._return_estimated_distance_drive(int(d2), int(d1))   

    phi = min(oott + ddtt, oott2 + ddtt2)

    n1 = int(id1)
    n2 = int(id2+number_reqs)
    #print(n1, n2)
    if phi < thtt:
        #print("here")
        #tau = abs(req1['time_stamp'] - req2['time_stamp'])

        eu1 = abs(req1[earliest_departure])
        eu2 = abs(req2[earliest_departure])
        vartheta = abs(eu1 - eu2)

        #print(tau, vartheta)

        if (vartheta < the):

            #G.add_edge(n1, n2, weight=100)
            return (n1, n2, 100)

        else:

            #if (tau < thts) or (vartheta < the):
                #print("here")
                #G.add_edge(n1, n2, weight=75)

            #else:
                #print("here")
            #G.add_edge(n1, n2, weight=50)
            return (n1, n2, 50)
    else:

        #G.add_edge(n1, n2, weight=0)
        return (n1, n2, 0)

def similarity(place_name, inst1, inst2):

    save_dir = os.getcwd()+'/'+place_name
    pickle_dir = os.path.join(save_dir, 'pickle')
    network_class_file = pickle_dir+'/'+place_name+'.network.class.pkl'

    network_directory = os.getcwd()+'/'+place_name

    if Path(network_class_file).is_file():
        inst = Instance(folder_to_network=place_name)

    thtt = 360
    thts = 60
    the = 60
    #columns for computation
    earliest_departure = 'pu_time_sec'
    
    number_reqs = len(inst1)

    G = nx.Graph()
    for i in range(number_reqs*2):
        G.add_node(int(i))

    count = 0
    for id1, req1 in inst1.iterrows():

        o1 = req1['osmid_origin']
        d1 = req1['osmid_destination']
        
        print(count)
        for id2, req2 in inst2.iterrows():

            #o2 = req2['originnode_drive']
            #d2 = req2['destinationnode_drive']

            #origin_point = (req2['Pickup_Centroid_Latitude'], req2['Pickup_Centroid_Longitude'])
            #o2 = ox.get_nearest_node(inst.network.G_drive, origin_point)
        
            #destination_point = (req2['Dropoff_Centroid_Latitude'], req2['Dropoff_Centroid_Longitude'])
            #d2 = ox.get_nearest_node(inst.network.G_drive, destination_point)
            o2 = req2['osmid_origin'] 
            d2 = req2['osmid_destination']

            #oott = inst.network._return_estimated_travel_time_drive(int(o1), int(o2))  
            #ddtt = inst.network._return_estimated_travel_time_drive(int(d1), int(d2)) 

            #oott2 = inst.network._return_estimated_travel_time_drive(int(o2), int(o1))  
            #ddtt2 = inst.network._return_estimated_travel_time_drive(int(d2), int(d1))

            oott = inst.network._return_estimated_distance_drive(int(o1), int(o2))  
            ddtt = inst.network._return_estimated_distance_drive(int(d1), int(d2)) 

            oott2 = inst.network._return_estimated_distance_drive(int(o2), int(o1))  
            ddtt2 = inst.network._return_estimated_distance_drive(int(d2), int(d1))   

            
            phi = min(oott + ddtt, oott2 + ddtt2)
           
            n1 = int(id1)
            n2 = int(id2+number_reqs)
            #print(n1, n2)
            if phi < thtt:
                #print("here")
                #tau = abs(req1['time_stamp'] - req2['time_stamp'])

                eu1 = abs(req1[earliest_departure])
                eu2 = abs(req2[earliest_departure])
                vartheta = abs(eu1 - eu2)

                #print(tau, vartheta)

                if (vartheta < the):

                    G.add_edge(n1, n2, weight=100)

                else:

                    #if (tau < thts) or (vartheta < the):
                        #print("here")
                        #G.add_edge(n1, n2, weight=75)

                    #else:
                        #print("here")
                    G.add_edge(n1, n2, weight=50)
            else:

                G.add_edge(n1, n2, weight=0)
        
        count += 1

    '''
    del inst.network.G_walk
    del inst.network.shortest_path_walk
    
    for id1, req1 in inst1.iterrows():

        print('id1', id1)
        o1 = req1['osmid_origin']
        d1 = req1['osmid_destination']

        ray.shutdown()
        ray.init(num_cpus=cpu_count())
        gc.collect()
        inst1_id = ray.put(inst1)
        network_id = ray.put(inst.network)
        
        edges = ray.get([single_edge.remote(network_id, inst1_id, id1, o1, d1, req1, id2, req2, thtt, thts, the, earliest_departure, number_reqs)  for id2, req2 in inst2.iterrows()]) 
        #print(edges)

        for edge in edges:
            G.add_edge(edge[0], edge[1], weight=edge[2])

        del inst1_id
        del network_id
        del edges
        gc.collect()
    '''

    print('left')
    M = nx.max_weight_matching(G, weight='weight', maxcardinality=True)
    #M = nx.bipartite.minimum_weight_full_matching(G, weight='weight')

    si1i2 = 0
    #print(len(M))
    #print(M)
    count = 0
    for e in M:

        peso = G.edges[int(e[0]), int(e[1])]['weight']
        #if peso > 1: 
        si1i2 += peso
        count += 1
        #print(si1i2)

    #print(count)
    
    si1i2 = si1i2/count
    print(si1i2)

    return si1i2

def dynamism(inst1, ed, ld):

    time_stamp = 'pu_time_sec'
    Te = abs(ld - ed)

    inst1 = inst1.sort_values(time_stamp)

    sorted_ts = inst1[time_stamp].tolist()
    #sorted_ts = [i for i in sorted_ts if i != 0]
    #exclude time stamp 0

    DELTA = []
    for ts in range(len(sorted_ts)-1):
        DELTA.append(abs(sorted_ts[ts+1] - sorted_ts[ts]))

    number_reqs = len(inst1)
    
    theta = Te/len(sorted_ts)

    SIGMA = []
    for k in range(len(DELTA)):

        if ((k == 0) and (DELTA[k] < theta)): 

            SIGMA.append(theta - DELTA[k])

        else:

            if ((k > 0) and (DELTA[k] < theta)): 

                 SIGMA.append(theta - DELTA[k] + SIGMA[k-1]*((theta - DELTA[k])/theta))

            else:

                 SIGMA.append(0)

    #print(SIGMA)
    #lambdax = 0
    #for sk in SIGMA:

    #    lambdax += sk

    NEGSIGMA = []
    for k in range(len(DELTA)):

        if ((k > 0) and (DELTA[k] < theta)): 

            NEGSIGMA.append(theta + SIGMA[k-1]*((theta - DELTA[k])/theta))

        else:

            NEGSIGMA.append(theta)

    #print(NEGSIGMA)
    #eta = 0
    #for nsk in NEGSIGMA:

    #    eta += nsk

    rho = 1 - (sum(SIGMA)/sum(NEGSIGMA)) 

    #print(DELTA)
    #print(SIGMA)
    #print(NEGSIGMA)
    #print(lambdax)
    #print(eta)
    print(rho)

def real_data_tests_chicago_database2(time_intervals):

    if database_exists('sqlite:///chicago_database_tnp.db'):
        
        chicago_database = sqla.create_engine('sqlite:///chicago_database_tnp.db')

    else:

        chicago_database = sqla.create_engine('sqlite:///chicago_database_tnp.db')
        j, chunksize = 1, 100000
        #for month in range(9,10):
        fp = "Taxi_Trips_-_2019.csv"
        for df in pd.read_csv(fp, chunksize=chunksize, iterator=True):
            df = df.rename(columns={c: c.replace(' ', '_') for c in df.columns})
            
            #print(df.columns)
            df['Trip_Start_Timestamp'] = df['Trip_Start_Timestamp'].astype(str)
            df['Trip_End_Timestamp'] = df['Trip_End_Timestamp'].astype(str)

            df['pickup_day'] = [x[0:10] for x in df['Trip_Start_Timestamp']]
            df['dropoff_day'] = [x[0:10] for x in df['Trip_End_Timestamp']]
            df['pickup_time'] = [x[11:19] for x in df['Trip_Start_Timestamp']]
            df['dropoff_time'] = [x[11:19] for x in df['Trip_End_Timestamp']]

            #pickup time min
            df['h'] = [x[11:13] for x in df['Trip_Start_Timestamp']]
            #df['h'] = [x[11:13] for x in df['Trip_Start_Timestamp']]

            df['min'] = [x[14:16] for x in df['Trip_Start_Timestamp']]
            #df['min'] = [x[14:16] for x in df['Trip_Start_Timestamp']]

            df['sec'] = [x[17:19] for x in df['Trip_Start_Timestamp']]
            #df['sec'] = [x[17:19] for x in df['Trip_Start_Timestamp']]

            df['spriod'] = [x[20:22] for x in df['Trip_Start_Timestamp']]

            df['h'].replace('', np.nan, inplace=True)
            df.dropna(subset=['h'], inplace=True)
            df['ih'] = df['h'].astype(int)
            for idx, row in df.iterrows():
                if row['spriod'] == 'PM':
                    if row['ih'] != 12:
                        df.loc[idx, 'ih'] = row['ih'] + 12
                elif row['spriod'] == 'AM':
                    if row['ih'] == 12:
                        df.loc[idx, 'ih'] = 0
            df['imin'] = df['min'].astype(int)
            df['isec'] = df['sec'].astype(int)

            #df['ih'] = df['ih'] * 3600
            #df['imin'] = df['imin'] * 60
            
            df['pu_time_sec'] = (df['ih'] * 3600) + (df['imin'] * 60) + df['isec']
            df['pu_time_sec'] = df['pu_time_sec'].astype(int)

            #dropoff time min
            df['dh'] = [x[11:13] for x in df['Trip_End_Timestamp']]
            #df['dh'] = [x[11:13] for x in df['Trip_End_Timestamp']]

            df['dmin'] = [x[14:16] for x in df['Trip_End_Timestamp']]
            #df['dmin'] = [x[14:16] for x in df['Trip_End_Timestamp']]

            df['dsec'] = [x[17:19] for x in df['Trip_End_Timestamp']]
            #df['dsec'] = [x[17:19] for x in df['Trip_End_Timestamp']]

            df['epriod'] = [x[20:22] for x in df['Trip_End_Timestamp']]

            df['dh'].replace('', np.nan, inplace=True)
            df.dropna(subset=['dh'], inplace=True)
            df['idh'] = df['dh'].astype(int)
            for idx, row in df.iterrows():
                if row['epriod'] == 'PM':
                    if row['idh'] != 12:
                        df.loc[idx, 'idh'] = row['idh'] + 12
                elif row['epriod'] == 'AM':
                    if row['idh'] == 12:
                        df.loc[idx, 'idh'] = 0
            df['idmin'] = df['dmin'].astype(int)
            df['idsec'] = df['dsec'].astype(int)

            #df['idh'] = df['idh'] 
            #df['idmin'] = df['idmin'] 
            
            df['do_time_sec'] = (df['idh'] * 3600) + (df['idmin'] * 60) + df['idsec']
            df['do_time_sec'] = df['do_time_sec'].astype(int)

            df['Trip_Miles'].astype(float)

            df = remove_false_records(df)

            df['Trip_Miles'] = 1.609*df['Trip_Miles']*1000 #milest to meters
            df['speed'] = df['Trip_Miles']/df['Trip_Seconds']
            df = add_osmid_nodes("Chicago, Illinois", df)
            df.index += j
            df.to_sql('table_record', chicago_database, if_exists='append')
            j = df.index[-1] + 1
        del df


    data_dist = []

    data_pu_lon = []
    data_pu_lat = []

    dfc = pd.read_sql_query('SELECT ih, pickup_day, pu_time_sec, do_time_sec, Trip_Seconds, Pickup_Centroid_Latitude, Pickup_Centroid_Longitude, Dropoff_Centroid_Latitude, Dropoff_Centroid_Longitude, osmid_origin, osmid_destination, Trip_Miles, speed FROM table_record', chicago_database)
    dfc['speed'] = dfc['speed']*3.6
    dfc['speed'].replace([np.inf, -np.inf], np.nan, inplace=True)
    dfc.dropna(subset=['speed'], inplace=True)

    cols = ['Trip_Miles', 'speed', 'Trip_Seconds']
    #dfc = dfc.loc[(dfc['Trip_Miles'] > 2000)]

    print('len before:', len(dfc))
    for col in cols:
        col_zscore = col + '_zscore'
        dfc[col_zscore] = (dfc[col] - dfc[col].mean())/dfc[col].std(ddof=0)

    zcols = ['Trip_Miles_zscore', 'speed_zscore', 'Trip_Seconds_zscore']

    for zcol in zcols:
        dfc = dfc.loc[(dfc[zcol] < 3)]

    print('len after 1:', len(dfc))

    for ti in time_intervals:

        #print(time_intervals)
        dayx = ti[0]
        ed = ti[1]
        ld = ti[2]
        print(dayx, ed, ld)

        #fitting pickups and dropoff times
        #pus = dfc["pu_time_sec"].values
        #Fitter_best_fitting_distribution(pus)
        #print('out fitter1')

        final_dfc = pd.DataFrame()
        for day in range(1,31):

            weekends_holidays = [1, 2, 7, 8, 14, 15, 21, 22, 28, 29]

            if dayx == 0:

                if day not in weekends_holidays:
                    d1 = '09/{0:0=2d}/2019'.format(day)

                    dfct = dfc.loc[(dfc['pickup_day'] == d1)]
                    final_dfc = final_dfc.append(dfct, ignore_index=True)
            else:

                if day in weekends_holidays:
                    d1 = '09/{0:0=2d}/2019'.format(day)

                    dfct = dfc.loc[(dfc['pickup_day'] == d1)]
                    final_dfc = final_dfc.append(dfct, ignore_index=True)


        #dfc = final_dfc
        print('len after 2:', len(final_dfc))

        #understand peak hours // off - peak 
        #Observar se os requests seguem normal distribution during peak hours and uniform during off peak. Pegar sample dos horários e plotar
        dfc2 = final_dfc
        dfc2 = dfc2.groupby('ih').count()
        ax = dfc2.plot(y='pickup_day', kind='line', style="-o", figsize=(15,5))
        plt.xlabel('hour')
        plt.ylabel('number of trips')
        plt.savefig('pickup_trips_time.png')
        plt.close()


        final_dfc = final_dfc.loc[(final_dfc['pu_time_sec'] >= ed) & (final_dfc['pu_time_sec'] <= ld)]
        print('len after 3:', len(final_dfc))

        
        #distance
        print(final_dfc['Trip_Miles'].describe())
        print(final_dfc['Trip_Miles'].mean())
        print(final_dfc['Trip_Miles'].std())

        ax = final_dfc['Trip_Miles'].hist(bins=30, figsize=(15,5))
        ax.set_yscale('log')
        ax.set_xlabel("trip distance (meters)")
        ax.set_ylabel("count")
        plt.savefig('Trip_Miles.png')
        plt.close()

        #speed
        #df_speed = pd.read_sql_query('SELECT speed FROM table_record', chicago_database)
        print(final_dfc['speed'].describe())
        print(final_dfc['speed'].mean())
        print(final_dfc['speed'].std())

        ax = final_dfc['speed'].hist(bins=30, figsize=(15,5))
        ax.set_yscale('log')
        ax.set_xlabel("trip distance (kmh)")
        ax.set_ylabel("count")
        plt.savefig('speed.png')
        plt.close()
        #plt.show()

        
        #similarity
        #ed = 420
        #ld = 540

        '''
        similarities = []
        for day in range(1,31):
            
            for day2 in range(1,31):
                
                if day2 > day:

                    d1 = '09/{0:0=2d}/2019'.format(day)
                    d2 = '09/{0:0=2d}/2019'.format(day2)

                    #df_1 = pd.read_sql_query('SELECT pickup_day, pu_time_sec, Pickup_Centroid_Latitude, Pickup_Centroid_Longitude, Dropoff_Centroid_Latitude, Dropoff_Centroid_Longitude, osmid_origin, osmid_destination  \
                            #FROM table_record', chicago_database)

                    #df_2 = pd.read_sql_query('SELECT pickup_day, pu_time_sec, Pickup_Centroid_Latitude, Pickup_Centroid_Longitude, Dropoff_Centroid_Latitude, Dropoff_Centroid_Longitude, osmid_origin, osmid_destination \
                            #FROM table_record', chicago_database)

                    df_1 = dfc.loc[(dfc['pickup_day'] == d1) & (dfc['pu_time_sec'] >= ed) & (dfc['pu_time_sec'] <= ld)]

                    df_2 = dfc.loc[(dfc['pickup_day'] == d2) & (dfc['pu_time_sec'] >= ed) & (dfc['pu_time_sec'] <= ld)]

                    #what if there are different numbers of requests?
                    #sample randomly
                    print(d1)
                    print(d2)
                    row_nr = min(len(df_1), len(df_2))
                    print(len(df_1), len(df_2))
                    if (len(df_2) < len(df_1)):
                        df_1 = df_1.sample(n = row_nr, replace = False)
                    else:
                        df_2 = df_2.sample(n = row_nr, replace = False)

                    print('equal lengths:', len(df_1), len(df_2))
                    simvalued1d2 = similarity("Chicago, Illinois", df_1, df_2) 
                    print('similarity: ', simvalued1d2)
                    similarities.append(simvalued1d2)
    
        print(similarities)
        meansim = sum(similarities) / len(similarities)
        print('mean similarities', meansim)
        '''
        
        #df_gd = pd.read_sql_query('SELECT pickup_day, pu_time_sec, do_time_sec, Trip_Seconds, Pickup_Centroid_Latitude, Pickup_Centroid_Longitude, Dropoff_Centroid_Latitude, Dropoff_Centroid_Longitude, osmid_origin, osmid_destination  \
        #                    FROM table_record', chicago_database)

        #print(df_loc.columns)
        #print(df_gd.columns)
        
        #geographic dispersion
        '''
        gds = []
        for day in range(1,31):
            
            sd = '09/{0:0=2d}/2019'.format(day)
            #ed = ed
            #ld = 540*60

            #print(sd)
            
            df_gd_d = dfc.loc[(dfc['pickup_day'] == sd)]
            #df_gd_d_loc = pd.merge(df_gd_d, df_loc)
            #print(df_gd_d_loc.head())
            
            print(sd)
            print('geographic dispersion: ', len(df_gd_d))
            gd = geographic_dispersion("Chicago, Illinois", df_gd_d, day)
            gds.append(gd)


        meangd = sum(gds) / len(gds)
        print('mean gds', meangd)
        variance = sum([((x - meangd) ** 2) for x in gds]) / len(gds)
        res = variance ** 0.5
        print('std gds', res)
        '''

        #df_dyn = pd.read_sql_query('SELECT pickup_day, pickup_time, pu_time_sec, do_time_sec, Trip_Seconds, Pickup_Centroid_Latitude, Pickup_Centroid_Longitude, Dropoff_Centroid_Latitude, Dropoff_Centroid_Longitude, osmid_origin, osmid_destination \
        #                    FROM table_record', chicago_database)
        '''
        # requests regarding the population
        population = 2710000

        #montly per pop
        mpp = len(dfc)/population
        print(mpp)

        #dynamism
        #average number of trips per day at a given time slot
        avg_trips = 0
        for day in range(1,31):

            sd = '09/{0:0=2d}/2019'.format(day)
            #ed = 420
            #ld = 540

            print(sd)
            
            df_dyn_d = dfc.loc[(dfc['pickup_day'] == sd) & (dfc['pu_time_sec'] >= ed) & (dfc['pu_time_sec'] <= ld)]

            if len(df_dyn_d) > 0:
                avg_trips += len(df_dyn_d)
            
            #daily per pop
            #dpp = len(df_dyn_d)/population
            #print(dpp)
            print('len ', len(df_dyn_d))
            print('dynamism')
            
            dynamism(df_dyn_d, ed, ld)
            #ratio between real vs estimated travel time
            #print('ratio eta vs real')
            #ratio_eta_real_time("Chicago, Illinois", df_dyn_d)

        #change the formulas (for measures of features), and put it in appendix
        print('average number of trips per day between ' + str(ed) + ' and ' + str(ld))
        print(avg_trips)
        ''' 
        #heatmap
        print('heatmap')
        #heatmap_osmnx("Chicago, Illinois", chicago_database)
        #new_heatmap("Chicago, Illinois", dfc)

        #fitting
        #df_fit = pd.read_sql_query('SELECT pu_time_sec, Trip_Miles FROM table_record', chicago_database)

        #z_scores = zscore(df_fit)
        #abs_z_scores = np.abs(z_scores)
        #filtered_entries = (abs_z_scores < 3).all(axis=1)
        #df_fit = df_fit[filtered_entries]

        dists = final_dfc["Trip_Miles"].values
        data_dist.append(Fitter_best_fitting_distribution(dists))

        #coords = dfc["Pickup_Centroid_Longitude"].values
        #data_pu_lon.append(Fitter_best_fitting_distribution(coords))

        #coords = dfc["Pickup_Centroid_Latitude"].values
        #data_pu_lat.append(Fitter_best_fitting_distribution(coords))

        #Dropoff_Centroid_Latitude, Dropoff_Centroid_Longitude


    #rank model
    print('rank model')
    rank_model("Chicago, Illinois", dfc)

    bins = 100
    density = True 

    plt.close()

    for d1 in data_dist:
        plt.hist(d1 , bins=bins, density=density, histtype="step")

    #plt.yscale('log')
    plt.xlabel("distance (meters)")
    plt.ylabel("probability")
    plt.tight_layout()
    plt.savefig('fitting_dist_curve.png')
    plt.close()
    print('out fitter2')
    #powelaw_best_fitting_distribution(dists)
    
    

if __name__ == '__main__':
    
    time_intervals = []
    


    #7 and 10 am weekday
    ed = 25200
    ld = 36000
    day = 0
    time_intervals.append((day,ed,ld))

    #16 and 20 am weekday
    ed = 57600
    ld = 72000
    day = 0
    time_intervals.append((day,ed,ld))

    

    #16 and 20 am weekend
    ed = 57600
    ld = 72000
    day = 1
    time_intervals.append((day,ed,ld))
    

    real_data_tests_chicago_database2(time_intervals)

    



