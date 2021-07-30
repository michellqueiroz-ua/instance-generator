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
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import sqlalchemy as sqla
from sqlalchemy_utils import database_exists
from datetime import datetime
from operator import mul
from scipy.stats import zscore
import networkx as nx

import seaborn as sns
from fitter import Fitter, get_common_distributions, get_distributions
import powerlaw
import re
from pathlib import Path
from instance_class import Instance
import osmnx as ox

def add_osmid_nodes(place_name, df, df_loc):

    save_dir = os.getcwd()+'/'+place_name
    pickle_dir = os.path.join(save_dir, 'pickle')
    network_class_file = pickle_dir+'/'+place_name+'.network.class.pkl'

    network_directory = os.getcwd()+'/'+place_name

    if Path(network_class_file).is_file():
        inst = Instance(folder_to_network=place_name)

    for id1, row1 in df.iterrows():

        longitude = df_loc.loc[df_loc['Kiosk ID'] == row1['Checkout_Kiosk_ID'], 'longitude']
        latitude = df_loc.loc[df_loc['Kiosk ID'] == row1['Checkout_Kiosk_ID'], 'latitude']
        origin_point = (latitude, longitude)
        df.loc[id1, 'osmid_origin'] = ox.get_nearest_node(inst.network.G_drive, origin_point)

        longitude = df_loc.loc[df_loc['Kiosk ID'] == row1['Return_Kiosk_ID'], 'longitude']
        latitude = df_loc.loc[df_loc['Kiosk ID'] == row1['Return_Kiosk_ID'], 'latitude']
        destination_point = (latitude, longitude)
        df.loc[id1, 'osmid_destination'] = ox.get_nearest_node(inst.network.G_drive, destination_point)

    return df

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

    for node in inst.network.G_drive.nodes():

        try:
            
            inst.network.G_drive.nodes[node]['OGcount'] = df_og.loc[node, 'OGcount']
        
        except KeyError:

            inst.network.G_drive.nodes[node]['OGcount'] = 0

        try:
            
            inst.network.G_drive.nodes[node]['DEcount'] = df_og.loc[node, 'DEcount']
        
        except KeyError:

            inst.network.G_drive.nodes[node]['DEcount'] = 0
    
     

    #do the heatmap (one for pickup one for dropoff)
    #Make geodataframes from graph data
    nodes, edges = ox.graph_to_gdfs(inst.network.G_drive, nodes=True, edges=True)

    #Then plot a graph where node size and node color are related to the number of visits
    nc = ox.plot.get_node_colors_by_attr(inst.network.G_drive,'OGcount',num_bins = 10)
    fig, ax = ox.plot_graph(inst.network.G_drive,fig_height=8,fig_width=8,node_size=nodes['OGcount'], node_color=nc)

    plt.savefig(os.getcwd()+'/heatmap_origin_points.png')
    plt.close(fig)

    nc = ox.plot.get_node_colors_by_attr(inst.network.G_drive,'DEcount',num_bins = 10)
    fig, ax = ox.plot_graph(inst.network.G_drive,fig_height=8,fig_width=8,node_size=nodes['DEcount'], node_color=nc)

    plt.savefig(os.getcwd()+'/heatmap_destination_points.png')
    plt.close(fig)

def remove_false_records(df):

    #df = df.loc[(df['trip'] > 0.3)]

    df = df.loc[(df['Checkout_Kiosk_ID'] != df['Return_Kiosk_ID'])]

    df = df.loc[(df['Trip_Duration_Seconds'] > 0)]

    return df

def distance(place_name, df_dist, df_loc):

    save_dir = os.getcwd()+'/'+place_name
    pickle_dir = os.path.join(save_dir, 'pickle')
    network_class_file = pickle_dir+'/'+place_name+'.network.class.pkl'

    network_directory = os.getcwd()+'/'+place_name

    if Path(network_class_file).is_file():
        inst = Instance(folder_to_network=place_name)

    df_dist['trip_distance'] = np.nan
    for idxs, row in df_dist.iterrows():

        r1 = df_loc.loc[df_loc['Kiosk ID'] == row['Checkout_Kiosk_ID']]
        #r2 = df_loc.loc[df_loc['Kiosk ID'] == row['Checkout_Kiosk_ID']]
        print(r1.index)
        latitude = df_loc.loc[r1.index, 'latitude']
        print(latitude)
        #print(longitude)

        origin_point = (latitude, r1['longitude'])
        print(origin_point)
        node_origin = ox.get_nearest_node(inst.network.G_drive, origin_point)
        
        r2 = df_loc.loc[df_loc['Kiosk ID'] == row['Return_Kiosk_ID']]
        #latitude = df_loc.loc[df_loc['Kiosk ID'] == row['Return_Kiosk_ID'], 'latitude']
        destination_point = (r2['latitude'], r2['longitude'])
        print(destination_point)
        node_destination = ox.get_nearest_node(inst.network.G_drive, destination_point)

        df_dist['trip_distance'] = inst.network._return_estimated_distance_drive(int(node_origin), int(node_destination))

    df_dist.dropna(subset=['trip_distance'], inplace=True)
    df_dist = df_dist.loc[(df['trip_distance'] > 500)]

    return df_dist

def ratio_eta_real_time(place_name, df_ratio, df_loc):

    save_dir = os.getcwd()+'/'+place_name
    pickle_dir = os.path.join(save_dir, 'pickle')
    network_class_file = pickle_dir+'/'+place_name+'.network.class.pkl'

    network_directory = os.getcwd()+'/'+place_name

    if Path(network_class_file).is_file():
        inst = Instance(folder_to_network=place_name)

    ratios = []
    for id1, row1 in df_ratio.iterrows():

        longitude = df_loc.loc[df_loc['Kiosk ID'] == row1['Checkout_Kiosk_ID'], 'longitude']
        latitude = df_loc.loc[df_loc['Kiosk ID'] == row1['Checkout_Kiosk_ID'], 'latitude']
        origin_point = (latitude, longitude)
        node_origin = ox.get_nearest_node(inst.network.G_drive, origin_point)
        
        longitude = df_loc.loc[df_loc['Kiosk ID'] == row1['Return_Kiosk_ID'], 'longitude']
        latitude = df_loc.loc[df_loc['Kiosk ID'] == row1['Return_Kiosk_ID'], 'latitude']
        destination_point = (latitude, longitude)
        node_destination = ox.get_nearest_node(inst.network.G_drive, destination_point)

        eta = inst.network._return_estimated_travel_time_drive(int(node_origin), int(node_destination))
        real = row1['Trip_Duration_Seconds']

        ratio = real/eta
        ratios.append(ratio)
        #print(real) 

    print(ratios)
    
def geographic_dispersion(place_name, inst1, df_loc):


    save_dir = os.getcwd()+'/'+place_name
    pickle_dir = os.path.join(save_dir, 'pickle')
    network_class_file = pickle_dir+'/'+place_name+'.network.class.pkl'

    network_directory = os.getcwd()+'/'+place_name

    if Path(network_class_file).is_file():
        inst = Instance(folder_to_network=place_name)

    csv_directory = network_directory+'/csv_format'
    ttm_directory = network_directory+'/travel_time_matrix'
    directory = os.fsencode(csv_directory)


    ttm_file_inst1 = 'travel_time_matrix_'+filename1
    ttmfilename1 = os.fsdecode(ttm_file_inst1)
    ttm1 = pd.read_csv(ttm_directory+'/'+ttmfilename1)
    ttm1.set_index(['osmid_origin'], inplace=True)


    #mu
    #average travel time between origin and destinations
    dtt = []
    for idx, req in inst1.iterrows():
        dtt.append(req['Trip_Duration_Seconds'])


    mudarp = sum(dtt) / len(dtt)
    mu2 = inst1['Trip_Duration_Seconds'].mean()

    #average travel time between x nearest neighbors
    #nyc -> compute for the 5 nearest zones
    earliest_departure = 'pu_time_sec'
    #latest_arrival = 'do_time_sec'
    time_gap = 10
    #node_origin = 
    #node_destination = 
    
    sumnn = 0
    for idx1, row1 in inst1.iterrows():

        ltro = []
        ltrd = []
        for idx2, row2 in inst1.iterrows():

            if idx2 != idx1:

                latest_arrival1 = row1[earliest_departure] + row1['Trip_Duration_Seconds']
                latest_arrival2 = row2[earliest_departure] + row2['Trip_Duration_Seconds']
                #print(row2['earliest_departure'])
                if (row2[earliest_departure] >= row1[earliest_departure] - time_gap) and (row2[earliest_departure] <= row1[earliest_departure] + time_gap):
                    #if (row2['originnode_drive'] != row1['originnode_drive']) and (row2['originnode_drive'] != row1['destinationnode_drive']):
                    #ltro.append(row2['originnode_drive'])
                    longitude = df_loc.loc[df_loc['Kiosk_ID'] == row2['Checkout_Kiosk_ID'], 'longitude']
                    latitude = df_loc.loc[df_loc['Kiosk_ID'] == row2['Checkout_Kiosk_ID'], 'latitude']
                    origin_point = (latitude, longitude)
                    node_origin = ox.get_nearest_node(inst.network.G_drive, origin_point)
                    ltro.append(node_origin)

                if (latest_arrival2 >= row1[earliest_departure] - time_gap) and (latest_arrival2 <= row1[earliest_departure] + time_gap):
                    #if (row2['destinationnode_drive'] != row1['originnode_drive']) and (row2['destinationnode_drive'] != row1['destinationnode_drive']):
                    #ltro.append(row2['destinationnode_drive'])
                    longitude = df_loc.loc[df_loc['Kiosk_ID'] == row2['Return_Kiosk_ID'], 'longitude']
                    latitude = df_loc.loc[df_loc['Kiosk_ID'] == row2['Return_Kiosk_ID'], 'latitude']
                    destination_point = (latitude, longitude)
                    node_destination = ox.get_nearest_node(inst.network.G_drive, destination_point)
                    ltro.append(node_destination)

                if (latest_arrival2 >= latest_arrival1 - time_gap) and (latest_arrival2 <= latest_arrival1 + time_gap):
                    #if (row2['destinationnode_drive'] != row1['originnode_drive']) and (row2['destinationnode_drive'] != row1['destinationnode_drive']):
                    #ltrd.append(row2['destinationnode_drive'])
                    longitude = df_loc.loc[df_loc['Kiosk_ID'] == row2['Return_Kiosk_ID'], 'longitude']
                    latitude = df_loc.loc[df_loc['Kiosk_ID'] == row2['Return_Kiosk_ID'], 'latitude']
                    destination_point = (latitude, longitude)
                    node_destination = ox.get_nearest_node(inst.network.G_drive, destination_point)
                    ltro.append(node_destination)

                if (row2[earliest_departure] >= latest_arrival1 - time_gap) and (row2[earliest_departure] <= latest_arrival1 + time_gap):
                    #if (row2['originnode_drive'] != row1['originnode_drive']) and (row2['originnode_drive'] != row1['destinationnode_drive']):
                    #ltrd.append(row2['originnode_drive'])
                    longitude = df_loc.loc[df_loc['Kiosk_ID'] == row2['Checkout_Kiosk_ID'], 'longitude']
                    latitude = df_loc.loc[df_loc['Kiosk_ID'] == row2['Checkout_Kiosk_ID'], 'latitude']
                    origin_point = (latitude, longitude)
                    node_origin = ox.get_nearest_node(inst.network.G_drive, origin_point)
                    ltro.append(node_origin)

        #ltro = list(dict.fromkeys(ltro))
        #ltrd = list(dict.fromkeys(ltrd))
        #print(ltro)
        #print(ltrd)

        ltrot = []
        ltrdt = []
        
        #org_row1 = int(row1['originnode_drive'])
        longitude = df_loc.loc[df_loc['Kiosk_ID'] == row1['Checkout_Kiosk_ID'], 'longitude']
        latitude = df_loc.loc[df_loc['Kiosk_ID'] == row1['Checkout_Kiosk_ID'], 'latitude']
        origin_point = (latitude, longitude)
        org_row1 = ox.get_nearest_node(inst.network.G_drive, origin_point)
        
        for x in ltro:

            tuplx = (x, inst.network._return_estimated_travel_time_drive(int(org_row1), int(x)))
            ltrot.append(tuplx)

        #dest_row1 = int(row1['destinationnode_drive'])
        longitude = df_loc.loc[df_loc['Kiosk_ID'] == row1['Return_Kiosk_ID'], 'longitude']
        latitude = df_loc.loc[df_loc['Kiosk_ID'] == row1['Return_Kiosk_ID'], 'latitude']
        destination_point = (latitude, longitude)
        dest_row1 = ox.get_nearest_node(inst.network.G_drive, destination_point)
        
        for y in ltrd:

            tuply = (y, inst.network._return_estimated_travel_time_drive(int(dest_row1), int(y)))
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

    omegadarp = sumnn/(len(inst1)*2)
    #ttm1['mean'] = ttm1.mean(axis=1)
    #varchi = 0.7
    #omega = ttm1['mean'].mean()
    

    print(mudarp)
    print(omegadarp)
    gd = mudarp + omegadarp
    print(gd)

def similarity(place_name, inst1, inst2, df_loc):

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

    #top_nodes = [i for i in range(number_reqs)]
    #bottom_nodes = [i+500 for i in range(number_reqs)]
    
    for id1, req1 in inst1.iterrows():

        #o1 = req1['originnode_drive']
        longitude = df_loc.loc[df_loc['Kiosk_ID'] == req1['Checkout_Kiosk_ID'], 'longitude']
        latitude = df_loc.loc[df_loc['Kiosk_ID'] == req1['Checkout_Kiosk_ID'], 'latitude']
        origin_point = (latitude, longitude)
        o1 = ox.get_nearest_node(inst.network.G_drive, origin_point)
        
        #d1 = req1['destinationnode_drive']
        longitude = df_loc.loc[df_loc['Kiosk_ID'] == req1['Return_Kiosk_ID'], 'longitude']
        latitude = df_loc.loc[df_loc['Kiosk_ID'] == req1['Return_Kiosk_ID'], 'latitude']
        destination_point = (latitude, longitude)
        d1 = ox.get_nearest_node(inst.network.G_drive, destination_point)

        for id2, req2 in inst2.iterrows():

            #o2 = req2['originnode_drive']
            #d2 = req2['destinationnode_drive']
            longitude = df_loc.loc[df_loc['Kiosk_ID'] == req2['Checkout_Kiosk_ID'], 'longitude']
            latitude = df_loc.loc[df_loc['Kiosk_ID'] == req2['Checkout_Kiosk_ID'], 'latitude']
            origin_point = (latitude, longitude)
            o2 = ox.get_nearest_node(inst.network.G_drive, origin_point)
            
            #d1 = req1['destinationnode_drive']
            longitude = df_loc.loc[df_loc['Kiosk_ID'] == req2['Return_Kiosk_ID'], 'longitude']
            latitude = df_loc.loc[df_loc['Kiosk_ID'] == req2['Return_Kiosk_ID'], 'latitude']
            destination_point = (latitude, longitude)
            d2 = ox.get_nearest_node(inst.network.G_drive, destination_point)

            oott = inst.network._return_estimated_travel_time_drive(int(o1), int(o2))  
            ddtt = inst.network._return_estimated_travel_time_drive(int(d1), int(d2)) 

            oott2 = inst.network._return_estimated_travel_time_drive(int(o2), int(o1))  
            ddtt2 = inst.network._return_estimated_travel_time_drive(int(d2), int(d1))  

            #odtt = inst.network._return_estimated_travel_time_drive(int(o1), int(d2))  
            #dott = inst.network._return_estimated_travel_time_drive(int(d1), int(o2))


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


    M = nx.max_weight_matching(G, weight='weight', maxcardinality=True)
    #M = nx.bipartite.minimum_weight_full_matching(G, weight='weight')

    si1i2 = 0
    print(len(M))
    #print(M)
    count = 0
    for e in M:
        #print(e)
        #print(e[0])
        #print(e[1])
        #print(e)
        #print(e)
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
    lambdax = 0
    for sk in SIGMA:

        lambdax += sk

    NEGSIGMA = []
    for k in range(len(DELTA)):

        if ((k > 0) and (DELTA[k] < theta)): 

            NEGSIGMA.append(theta + SIGMA[k-1]*((theta - DELTA[k])/theta))

        else:

            NEGSIGMA.append(theta)

    #print(NEGSIGMA)
    eta = 0
    for nsk in NEGSIGMA:

        eta += nsk

    rho = 1 - (sum(SIGMA)/sum(NEGSIGMA)) 

    #print(DELTA)
    #print(SIGMA)
    #print(NEGSIGMA)
    #print(lambdax)
    #print(eta)
    print(rho)
    
def real_data_tests_austin_database(ed, ld):

    locs = "Austin_MetroBike_Kiosk_Locations.csv"
    df_loc = pd.read_csv(locs)

    new = df_loc["Location"].str.split(",", n = 1, expand = True)
    df_loc['latitude'] = new[0]
    df_loc['longitude'] = new[1]
    
    df_loc['longitude'] = df_loc['longitude'].map(lambda x: x.lstrip('(°').rstrip('°)'))
    df_loc['latitude'] = df_loc['latitude'].map(lambda x: x.lstrip('(°').rstrip('°)'))
    print(df_loc.head())

    if database_exists('sqlite:///austin_database.db'):
        
        austin_database = sqla.create_engine('sqlite:///austin_database.db')

    else:

        austin_database = sqla.create_engine('sqlite:///austin_database.db')
        j, chunksize = 1, 100000
        #for month in range(9,10):
        fp = "Austin_MetroBike_Trips.csv"
        for df in pd.read_csv(fp, chunksize=chunksize, iterator=True):
            df = df.rename(columns={c: c.replace(' ', '_') for c in df.columns})
            
            #print(df.columns)
            #df['Trip_Start_Timestamp'] = df['Trip_Start_Timestamp'].astype(str)
            #df['Trip_End_Timestamp'] = df['Trip_End_Timestamp'].astype(str)

            #df['Checkout_Date'] = [x[0:10] for x in df['Checkout_Time']]
            #df['dropoff_day'] = [x[0:10] for x in df['Trip_End_Timestamp']]
            #df['pickup_time'] = [x[11:19] for x in df['Checkout_Time']]
            #df['dropoff_time'] = [x[11:19] for x in df['Trip_End_Timestamp']]

            #pickup time min
            new = df["Checkout_Time"].str.split(":", n = 2, expand = True)

            df['h'] = new[0]
            #df['h'] = [x[0:2] for x in df['Checkout_Time']]

            df['min'] = new[1]
            #df['min'] = [x[3:5] for x in df['Checkout_Time']]

            df['sec'] = new[2]
            #df['sec'] = [x[6:8] for x in df['Checkout_Time']]

            df['ih'] = df['h'].astype(int)
            df['imin'] = df['min'].astype(int)
            df['isec'] = df['sec'].astype(int)

            df['ih'] = df['ih'] * 3600
            df['imin'] = df['imin'] * 60
            
            df['pu_time_sec'] = df['ih'] + df['imin'] + df['isec']
            df['pu_time_sec'] = df['pu_time_sec'].astype(int)

            df['Trip_Duration_Seconds'] = df['Trip_Duration_Minutes']*60
            df = distance("Austin, Texas", df, df_loc)

            df = remove_false_records(df)

            df['speed'] = df['trip_distance']/df['Trip_Duration_Seconds']
            df = add_osmid_nodes("Austin, Texas", df, df_loc)
            df.index += j
            df.to_sql('table_record', austin_database, if_exists='append')
            j = df.index[-1] + 1
        del df

    #understand peak hours // off - peak 
    #Observar se os requests seguem normal distribution during peak hours and uniform during off peak. Pegar sample dos horários e plotar
    df_pu = pd.read_sql_query('SELECT pu_time_sec AS time, count(*) AS PUcount \
                        FROM table_record \
                        GROUP BY pu_time_sec', austin_database)
    print(df_pu.head())
    print(len(df_pu))


    ax = df_pu.plot(x='time', y='PUcount', kind='line', style="-o", figsize=(15,5))
    plt.savefig('number_trips_time.png')
    plt.close()
    #plt.show()

    #distance
    df_dist = pd.read_sql_query('SELECT trip_distance  \
                        FROM table_record', austin_database)
    z_scores = zscore(df_dist)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    df_dist = df_dist[filtered_entries]
    print(df_dist['trip_distance'].describe())
    print(df_dist['trip_distance'].mean())
    print(df_dist['trip_distance'].std())

    ax = df_dist['trip_distance'].hist(bins=30, figsize=(15,5))
    ax.set_yscale('log')
    ax.set_xlabel("trip distance (meters)")
    ax.set_ylabel("count")
    plt.savefig('trip_distance.png')
    plt.close()

    #speed
    df_speed = pd.read_sql_query('SELECT speed FROM table_record', austin_database)
    df['speed'] = df['speed']*3.6
    z_scores = zscore(df_speed)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    df_speed = df_speed[filtered_entries]
    print(df_speed['speed'].describe())
    print(df_speed['speed'].mean())
    print(df_speed['speed'].std())

    ax = df_dist['speed'].hist(bins=30, figsize=(15,5))
    ax.set_yscale('log')
    ax.set_xlabel("trip distance (kmh)")
    ax.set_ylabel("count")
    plt.savefig('speed.png')
    plt.close()
    #plt.show()

    #similarity
    similarities = []
    for day in range(1,2):
        for day2 in range(1,2):
            if day2 > day:

                d1 = '09/{0:0=2d}/2019'.format(day)
                d2 = '09/{0:0=2d}/2019'.format(day2)

                df_1 = pd.read_sql_query('SELECT Checkout_Date, pu_time_sec, Checkout_Kiosk_ID, Return_Kiosk_ID, Trip_Duration_Seconds  \
                        FROM table_record', austin_database)

                df_2 = pd.read_sql_query('SELECT Checkout_Date, pu_time_sec, Checkout_Kiosk_ID, Return_Kiosk_ID, Trip_Duration_Seconds \
                        FROM table_record', austin_database)

                df_1 = df_1.loc[(df_1['Checkout_Date'] == d1) & (df_1['pu_time_sec'] >= ed) & (df_1['pu_time_sec'] <= ld)]

                df_2 = df_2.loc[(df_2['Checkout_Date'] == d2) & (df_2['pu_time_sec'] >= ed) & (df_2['pu_time_sec'] <= ld)]

                #what if there are different numbers of requests?
                #sample randomly
                row_nr = min(len(df_1), len(df_2))
                print(len(df_1), len(df_2))
                if (len(df_2) < len(df_1)):
                    df_1 = df_1.sample(n = row_nr, replace = False)
                else:
                    df_2 = df_2.sample(n = row_nr, replace = False)

                print(len(df_1), len(df_2))
                #similarities.append(similarity("Austin, Texas", df_1, df_2, df_loc))

    #geographic dispersion
    df_gd = pd.read_sql_query('SELECT Checkout_Date, pu_time_sec, Checkout_Kiosk_ID, Return_Kiosk_ID, Trip_Duration_Seconds  \
                        FROM table_record', austin_database)

    print(df_loc.columns)
    print(df_gd.columns)
    for day in range(1,2):

        sd = '09/{0:0=2d}/2019'.format(day)
        #ed = ed
        #ld = 540*60

        print(sd)
        
        df_gd_d = df_gd.loc[(df_gd['Checkout_Date'] == sd) & (df_gd['pu_time_sec'] >= ed) & (df_gd['pu_time_sec'] <= ld)]
        #df_gd_d_loc = pd.merge(df_gd_d, df_loc)
        #print(df_gd_d_loc.head())
        
        print('geographic dispersion')
        #geographic_dispersion("Austin, Texas", df_gd_d, df_loc)

    df_dyn = pd.read_sql_query('SELECT Checkout_Date, pu_time_sec, do_time_sec, Checkout_Kiosk_ID, Return_Kiosk_ID, Trip_Duration_Seconds \
                        FROM table_record', austin_database)

    # requests regarding the population
    population = 2710000

    #montly per pop
    mpp = len(df_dyn)/population
    print(mpp)

    #dynamism
    #average number of trips per day at a given time slot
    avg_trips = 0
    for day in range(1,3):

        sd = '09/{0:0=2d}/2019'.format(day)
        #ed = 420
        #ld = 540

        print(sd)
        
        df_dyn_d = df_dyn.loc[(df_dyn['Checkout_Date'] == sd) & (df_dyn['pu_time_sec'] >= ed) & (df_dyn['pu_time_sec'] <= ld)]

        if len(df_dyn_d) > 0:
            avg_trips += len(df_dyn_d)
        
        #daily per pop
        dpp = len(df_dyn_d)/population
        print(dpp)
        print('dynamism')
        dynamism(df_dyn_d, ed, ld)
        #ratio between real vs estimated travel time
        #ratio_eta_real_time("Austin, Texas", df_dyn_d)

    #change the formulas (for measures of features), and put it in appendix

    print('average number of trips per day between ' + str(ed) + ' and ' + str(ld))
    print(avg_trips)

if __name__ == '__main__':
    
    ed = 420*60
    ld = 540*60
    real_data_tests_austin_database(ed, ld)

    



