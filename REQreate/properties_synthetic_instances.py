import networkx as nx
import os
import osmnx as ox
import pandas as pd
import pickle
import numpy as np
import random
import statistics

from pathlib import Path
from instance_class import Instance

from output_files import JsonConverter

from shapely.geometry import Polygon
from descartes.patch import PolygonPatch
from shapely.geometry import Point
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.style.use('ggplot')

from matplotlib.colors import LogNorm
from fitter import Fitter, get_common_distributions, get_distributions
from multiprocessing import cpu_count

import ray
import gc

@ray.remote
def compute_distances(network, idx, origin, destination):

    dist = network._return_estimated_distance_drive(origin, destination)
    print(idx)
    tuple_re = (idx, dist)
    return tuple_re

def geographic_dispersion(inst, inst1, problem, filename1):

    if problem == 'DARP':
        #mu
        #average travel time between origin and destinations
        dtt = []
        for idx, req in inst1.iterrows():
            dtt.append(req['direct_travel_time'])

        mudarp = sum(dtt) / len(dtt)
        mu2 = inst1['direct_travel_time'].mean()

        #average travel time between x nearest neighbors
        #nyc -> compute for the 5 nearest zones
        earliest_departure = 'earliest_departure'
        #latest_arrival = 'do_time_sec'
        time_gap = 600
        #node_origin = 
        #node_destination = 
        osmid_origin = 'originnode_drive'
        osmid_destination = 'destinationnode_drive'
        speed = 7.22 #26kmh
        
        sumnn = 0
        for idx1, row1 in inst1.iterrows():

            ltro = []
            ltrd = []
            for idx2, row2 in inst1.iterrows():

                if idx2 != idx1:

                    latest_arrival1 = row1[earliest_departure] + row1['direct_travel_time']
                    latest_arrival2 = row2[earliest_departure] + row2['direct_travel_time']
                    #print(row2['earliest_departure'])
                    if (row2[earliest_departure] >= row1[earliest_departure] - time_gap) and (row2[earliest_departure] <= row1[earliest_departure] + time_gap):
                        #if (row2['originnode_drive'] != row1['originnode_drive']) and (row2['originnode_drive'] != row1['destinationnode_drive']):
                        #ltro.append(row2['originnode_drive'])
                        #origin_point = (row2['Pickup_Centroid_Latitude'], row2['Pickup_Centroid_Longitude'])
                        node_origin = row2[osmid_origin]
            
                        ltro.append(node_origin)

                    if (latest_arrival2 >= row1[earliest_departure] - time_gap) and (latest_arrival2 <= row1[earliest_departure] + time_gap):
                        #if (row2['destinationnode_drive'] != row1['originnode_drive']) and (row2['destinationnode_drive'] != row1['destinationnode_drive']):
                        #ltro.append(row2['destinationnode_drive'])
                        #destination_point = (row2['Dropoff_Centroid_Latitude'], row2['Dropoff_Centroid_Longitude'])
                        node_destination = row2[osmid_destination]
                        
                        ltro.append(node_destination)

                    if (latest_arrival2 >= latest_arrival1 - time_gap) and (latest_arrival2 <= latest_arrival1 + time_gap):
                        #if (row2['destinationnode_drive'] != row1['originnode_drive']) and (row2['destinationnode_drive'] != row1['destinationnode_drive']):
                        #ltrd.append(row2['destinationnode_drive'])
                        #destination_point = (row2['Dropoff_Centroid_Latitude'], row2['Dropoff_Centroid_Longitude'])
                        node_destination = row2[osmid_destination]

                        ltro.append(node_destination)

                    if (row2[earliest_departure] >= latest_arrival1 - time_gap) and (row2[earliest_departure] <= latest_arrival1 + time_gap):
                        #if (row2['originnode_drive'] != row1['originnode_drive']) and (row2['originnode_drive'] != row1['destinationnode_drive']):
                        #ltrd.append(row2['originnode_drive'])
                        #origin_point = (row2['Pickup_Centroid_Latitude'], row2['Pickup_Centroid_Longitude'])
                        node_origin = row2[osmid_origin]

                        ltro.append(node_origin)

            #ltro = list(dict.fromkeys(ltro))
            #ltrd = list(dict.fromkeys(ltrd))
            #print(ltro)
            #print(ltrd)

            ltrot = []
            ltrdt = []
            
            #org_row1 = int(row1['originnode_drive'])
            #origin_point = (row1['Pickup_Centroid_Latitude'], row1['Pickup_Centroid_Longitude'])
            #org_row1 = ox.get_nearest_node(inst.network.G_drive, origin_point)
            org_row1 = row1[osmid_origin]
            
            for x in ltro:

                #tuplx = (x, inst.network._return_estimated_travel_time_drive(int(org_row1), int(x)))
                dist = inst.network._return_estimated_distance_drive(int(org_row1), int(x))
                tt = dist/speed
                tuplx = (x, tt)
                ltrot.append(tuplx)

            #dest_row1 = int(row1['destinationnode_drive'])
            #destination_point = (row1['Dropoff_Centroid_Latitude'], row1['Dropoff_Centroid_Longitude'])
            #dest_row1 = ox.get_nearest_node(inst.network.G_drive, destination_point)
            dest_row1 = row1[osmid_destination]

            for y in ltrd:

                #tuply = (y, inst.network._return_estimated_travel_time_drive(int(dest_row1), int(y)))
                dist = inst.network._return_estimated_distance_drive(int(dest_row1), int(y))
                tt = dist/speed
                tuply = (y, tt)
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

    else:

        ttm_file_inst1 = 'travel_time_matrix_'+filename1
        ttmfilename1 = os.fsdecode(ttm_file_inst1)
        ttm1 = pd.read_csv(ttm_directory+'/'+ttmfilename1)
        ttm1.set_index(['osmid_origin'], inplace=True)

        earliest_departure = 'earliest_departure'
        latest_arrival = 'latest_arrival'
        time_gap = 600
        #node_origin = 
        #node_destination = 
        osmid_origin = 'originnode_drive'
        osmid_destination = 'destinationnode_drive'
        speed = 7.22 #26kmh
        stops_orgn = 'stops_orgn'
        stops_dest = 'stops_dest'

        stations = []
        ovrsumm = 0
        for idx, row in inst1.iterrows():
            
            orgn = row[stops_orgn].strip('][').split(', ')
            
            dest = row[stops_dest].strip('][').split(', ')

            
            orgn = [int(i) for i in orgn]
            dest = [int(i) for i in dest]

            summ = 0
            count = 0
            for so in orgn:
                for sd in dest:

                    if so != sd:
                        summ += ttm1.loc[so, str(sd)]
                        count += 1

            ovrsumm += summ/count

        
        muodbrp = ovrsumm/(len(inst1))

        sumnn = 0
        for idx1, row1 in inst1.iterrows():
            ltro = []
            ltrd = []
            for idx2, row2 in inst1.iterrows():

                if idx2 != idx1:

                    
                    if (row2[earliest_departure] >= row1[earliest_departure] - time_gap) and (row2[earliest_departure] <= row1[earliest_departure] + time_gap):
                        
                        stps = row2[stops_orgn].strip('][').split(', ')
                        ltro.extend(stps)

                    if (row2[latest_arrival] >= row1[earliest_departure] - time_gap) and (row2[latest_arrival] <= row1[earliest_departure] + time_gap):
                        stps = row2[stops_dest].strip('][').split(', ')
                        ltro.extend(stps)

                    if (row2[latest_arrival] >= row1[latest_arrival] - time_gap) and (row2[latest_arrival] <= row1[latest_arrival] + time_gap):
                        stps = row2[stops_dest].strip('][').split(', ')
                        ltrd.extend(stps)

                    if (row2[earliest_departure] >= row1[latest_arrival] - time_gap) and (row2[earliest_departure] <= row1[latest_arrival] + time_gap):
                        stps = row2[stops_orgn].strip('][').split(', ')
                        ltrd.extend(stps)

            ltro = list(dict.fromkeys(ltro))
            ltrd = list(dict.fromkeys(ltrd))

            ltrot = []
            ltrdt = []

            org_stps = row1[stops_orgn].strip('][').split(', ')
            org_stps = [int(i) for i in org_stps]
            ltro = [int(i) for i in ltro if int(i) not in org_stps]
            for s in org_stps:
                for x in ltro:

                    tuplx = (x, ttm1.loc[int(s), str(x)])
                    ltrot.append(tuplx)

            
            dest_stps = row1[stops_dest].strip('][').split(', ')
            dest_stps = [int(i) for i in dest_stps]
            ltrd = [int(i) for i in ltrd if int(i) not in dest_stps]
            for s in dest_stps:
                for y in ltrd:

                    tuply = (y, ttm1.loc[int(s), str(y)])
                    ltrdt.append(tuply)


            #sort tuples
            ltrot.sort(key = lambda x: x[1]) 
            ltrdt.sort(key = lambda x: x[1])
            
            #avg 5 first
            n_neig = 5
            avgo = 0
            for i in range(min(n_neig, len(ltrot))):
                avgo += ltrot[i][1]
            
            if len(ltrot) > 0:
                avgo = avgo/min(n_neig, len(ltrot))

            avgd = 0
            for j in range(min(n_neig, len(ltrdt))):
                avgd += ltrdt[j][1]
            
            if len(ltrdt) > 0:
                avgd = avgd/min(n_neig, len(ltrdt))
            
            sumnn += avgo + avgd

        omegaodbrp = sumnn/(len(inst1)*2)
        
        print(muodbrp)
        print(omegaodbrp)
        gd = muodbrp + omegaodbrp
        print(gd)

def similarity(inst, inst1, inst2):

    thtt = 360
    thts = 60
    the = 60
    speed = 7.22 #26kmh
    #columns for computation
    earliest_departure = 'earliest_departure'
    osmid_origin = 'originnode_drive'
    osmid_destination = 'destinationnode_drive'
    
    number_reqs = len(inst1)

    G = nx.Graph()
    for i in range(number_reqs*2):
        G.add_node(int(i))

    #top_nodes = [i for i in range(number_reqs)]
    #bottom_nodes = [i+500 for i in range(number_reqs)]
    
    for id1, req1 in inst1.iterrows():

        #o1 = req1['originnode_drive']
        #d1 = req1['destinationnode_drive']

        #origin_point = (req1['Pickup_Centroid_Latitude'], req1['Pickup_Centroid_Longitude'])
        #o1 = ox.get_nearest_node(inst.network.G_drive, origin_point)
        
        #destination_point = (req1['Dropoff_Centroid_Latitude'], req1['Dropoff_Centroid_Longitude'])
        #d1 = ox.get_nearest_node(inst.network.G_drive, destination_point)
        o1 = req1[osmid_origin]
        d1 = req1[osmid_destination]
        

        for id2, req2 in inst2.iterrows():

            #o2 = req2['originnode_drive']
            #d2 = req2['destinationnode_drive']

            #origin_point = (req2['Pickup_Centroid_Latitude'], req2['Pickup_Centroid_Longitude'])
            #o2 = ox.get_nearest_node(inst.network.G_drive, origin_point)
        
            #destination_point = (req2['Dropoff_Centroid_Latitude'], req2['Dropoff_Centroid_Longitude'])
            #d2 = ox.get_nearest_node(inst.network.G_drive, destination_point)
            o2 = req2[osmid_origin] 
            d2 = req2[osmid_destination]

            #oott = inst.network._return_estimated_travel_time_drive(int(o1), int(o2))  
            #ddtt = inst.network._return_estimated_travel_time_drive(int(d1), int(d2)) 

            #oott2 = inst.network._return_estimated_travel_time_drive(int(o2), int(o1))  
            #ddtt2 = inst.network._return_estimated_travel_time_drive(int(d2), int(d1))

            oott = inst.network._return_estimated_distance_drive(int(o1), int(o2))  
            ddtt = inst.network._return_estimated_distance_drive(int(d1), int(d2)) 

            oott2 = inst.network._return_estimated_distance_drive(int(o2), int(o1))  
            ddtt2 = inst.network._return_estimated_distance_drive(int(d2), int(d1))   

            oott = oott/speed
            ddtt = ddtt/speed

            oott2 = oott2/speed
            ddtt2 = ddtt2/speed

            phi = min(oott + ddtt, oott2 + ddtt2)
           
            n1 = int(id1)
            n2 = int(id2+number_reqs)
            #print(n1, n2)
            if phi < thtt:
                #print("here")
                tau = abs(req1['time_stamp'] - req2['time_stamp'])

                eu1 = abs(req1[earliest_departure])
                eu2 = abs(req2[earliest_departure])
                vartheta = abs(eu1 - eu2)

                #print(tau, vartheta)

                if (vartheta < the):

                    G.add_edge(n1, n2, weight=100)

                else:

                    if (tau < thts) or (vartheta < the):
                        #print("here")
                        G.add_edge(n1, n2, weight=75)

                    else:
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

    time_stamp = 'time_stamp'
    Te = abs(ld - ed)

    inst1 = inst1.sort_values(time_stamp)

    sorted_ts = inst1[time_stamp].tolist()
    #sorted_ts = [i for i in sorted_ts if i != 0]
    #exclude time stamp 0

    DELTA = []
    for ts in range(len(sorted_ts)-1):
        DELTA.append(float(abs(sorted_ts[ts+1] - sorted_ts[ts])))

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

    print(rho)

def new_heatmap(inst, dfc):

    df_og = dfc

    df_de = dfc

    pts_bhx = []
    pts_bhy = []

    pts_og = []
    pts_ogx = []
    pts_ogy = []    
    for idx, row in df_og.iterrows():

        pt = (row['originx'], row['originy'])
        pts_og.append(pt)
        pts_ogx.append(row['originx'])
        pts_ogy.append(row['originy'])

        pts_bhx.append(row['originx'])
        pts_bhy.append(row['originy'])
    

    pts_de = []
    pts_dex = []   
    pts_dey = []      
    for idx, row in df_de.iterrows():

        pt = (row['destinationx'], row['destinationy'])
        pts_de.append(pt)
        pts_dex.append(row['destinationx'])
        pts_dey.append(row['destinationy'])

        pts_bhx.append(row['destinationx'])
        pts_bhy.append(row['destinationy'])

    minx, miny, maxx, maxy = inst.network.polygon.bounds
    #hm = Heatmap(libpath="cHeatmap.cpython-38-x86_64-linux-gnu.so")
    #img = hm.heatmap(pts_og, scheme='classic', dotsize=75, opacity=128, area=((minx, miny), (maxx, maxy)))
    #img.save("heatmap_og.png")

    #hm = Heatmap(libpath="cHeatmap.cpython-38-x86_64-linux-gnu.so")
    #img = hm.heatmap(pts_de, scheme='classic', dotsize=75, opacity=128, area=((minx, miny), (maxx, maxy)))
    #img.save("heatmap_de.png")

    #print(' len points ', len(pts_ogx))
    #plt.hist2d(pts_ogx,pts_ogy, bins=[np.arange(minx,maxx,5),np.arange(miny,maxy,5)])
    #print(len(pts_ogx))
    h = plt.hist2d(pts_ogx,pts_ogy, bins=25, norm=LogNorm(), cmap='jet')
    plt.colorbar(h[3])
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.savefig('heatmap_origin_syn.png')
    plt.close()

    #plt.hist2d(pts_dex,pts_dey, bins=[np.arange(minx,maxx,10),np.arange(miny,maxy,10)])
    h = plt.hist2d(pts_dex,pts_dey, bins=25, norm=LogNorm(), cmap='jet')
    plt.colorbar(h[3])
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.savefig('heatmap_destination_syn.png')
    plt.close()

    h = plt.hist2d(pts_bhx,pts_bhy, bins=25, norm=LogNorm(), cmap='jet')
    plt.colorbar(h[3])
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.savefig('heatmap_both_syn.png')
    plt.close()

    #curr_folder = os.getcwd()

    #fig, ax = ox.plot_graph(inst.network.G_drive,figsize=(8, 8), dpi=128, show=False, filepath='heatmap_origin_points.png', save=True)

    #fig, ax = ox.plot_graph(inst.network.G_drive,figsize=(8, 8), dpi=128, show=False, filepath='heatmap_destination_points.png', save=True)

def new_heatmap_pois(inst, place_name):

    output_folder_base = place_name
    save_dir_csv = os.path.join(save_dir, 'csv')

    path_pois = os.path.join(save_dir_csv, output_folder_base+'.pois.csv')

    if os.path.isfile(path_pois):
        print('is file POIs')
        pois = pd.read_csv(path_pois)

    pts_bhx = []
    pts_bhy = []

    for idx, row in pois.iterrows():

        pt = (row['lon'], row['lat'])

        pts_bhx.append(row['lon'])
        pts_bhy.append(row['lat'])
    
    minx, miny, maxx, maxy = inst.network.polygon.bounds

    h = plt.hist2d(pts_bhx,pts_bhy, bins=25, norm=LogNorm(), cmap='jet')
    plt.colorbar(h[3])
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.savefig('heatmap_both_syn_pois.png')
    plt.close()

    #curr_folder = os.getcwd()

    #fig, ax = ox.plot_graph(inst.network.G_drive,figsize=(8, 8), dpi=128, show=False, filepath='heatmap_origin_points.png', save=True)

    #fig, ax = ox.plot_graph(inst.network.G_drive,figsize=(8, 8), dpi=128, show=False, filepath='heatmap_destination_points.png', save=True)

def Fitter_best_fitting_distribution(dists):
    f = Fitter(dists, timeout=180, distributions= get_common_distributions())

    f.fit()
    _summary, _data = f.summary(plot=True)
    print(f.get_best(method = 'sumsquare_error'))

    return _data

def urgency(inst1):

    chi = []
    for idx, req in inst1.iterrows():

        #if req['time_stamp'] > 0:
        #    er = abs(req['latest_departure'] - req['time_stamp'])
        #    chi.append(er)
        er = abs(req['reaction_time'])
        chi.append(er)
    
    mean = sum(chi) / len(chi)
    variance = sum([((x - mean) ** 2) for x in chi]) / len(chi)
    stdv = variance ** 0.5
    #stdv2 = statistics.pstdev(chi)

    '''
    inst1['reaction_time'] = abs(inst1['latest_departure'] - inst1['time_stamp'])
    chi2 = inst1['reaction_time'].tolist()
    mean2 = inst1['reaction_time'].mean()
    variance2 = sum([((x - mean2) ** 2) for x in chi2]) / len(chi2)
    stdv3 = variance ** 0.5
    stdv4 = statistics.pstdev(chi2)
    '''
    
    print('mean:', mean)
    print('std:', stdv)
    return mean
    #print(stdv)

    #print(mean2)
    #print(stdv4)

if __name__ == '__main__':

    place_name = "Chicago, Illinois"
    save_dir = os.getcwd()+'/'+place_name
    pickle_dir = os.path.join(save_dir, 'pickle')
    network_class_file = pickle_dir+'/'+place_name+'.network.class.pkl'

    #standard_filename = 'Chicago,Illinois_DARP_500'
    #param_class_file = pickle_dir+'/'+standard_filename+'.param.class.pkl'

    network_directory = os.getcwd()+'/'+place_name
    all_trips = pd.DataFrame()

    if Path(network_class_file).is_file():
        inst = Instance(folder_to_network=place_name)

    save_dir_csv = os.path.join(inst.save_dir, 'csv_format')
    if not os.path.isdir(save_dir_csv):
        os.mkdir(save_dir_csv)
    
    inst.sorted_attributes = ['destination', 'origin', 'destinationx', 'originx', 'destinationy', 'originy', 'destinationnode_drive', 'originnode_drive', 'reaction_time', 'direct_distance', 'direct_travel_time', 'walk_speed', 'max_walking', 'time_walking', 'stops_orgn', 'stops_dest', 'lead_time', 'time_window_length', 'earliest_departure', 'time_stamp', 'latest_arrival', 'earliest_arrival', 'latest_departure']
    for instance in os.listdir(os.path.join(inst.save_dir, 'json_format')):
        
        if instance != ".DS_Store":
            input_name = os.path.join(inst.save_dir, 'json_format', instance)
            
            output_name_csv = instance.split('.')[0] + '.csv'
            output_name_csv = output_name_csv.replace(" ", "")
            
            converter = JsonConverter(file_name=input_name)
            converter.convert_normal(inst=inst, problem_type="DARP", path_instance_csv_file=os.path.join(save_dir_csv, output_name_csv))
            #converter.convert_localsolver(output_file_name=os.path.join(save_dir_localsolver, output_name_ls))

    csv_directory = network_directory+'/csv_format'
    directory = os.fsencode(csv_directory)

    #del inst.network.shortest_path_walk
    gc.collect()

    num = 0
    data_dist = []
    for file_inst1 in os.listdir(directory):

        filename1 = os.fsdecode(file_inst1)

        if (filename1.endswith(".csv")):
        
            inst1 = pd.read_csv(csv_directory+'/'+filename1)

            #all_trips = all_trips.append(inst1, ignore_index=True)


            speed = 7.22 #mps = 26 kmh
            #speed = 5.55 #mps = 20 kmh
            
            dists = inst1["direct_distance"].values
            data_dist.append(Fitter_best_fitting_distribution(dists))
            print('out fitter2')
            '''
            new_heatmap(inst, inst1)
            '''

            #print('urgency')
            #urgency(inst1)

            #print('dynamism')
            #dynamism(inst1, 25200, 25560)

            problem = "DARP"
            #geographic_dispersion(inst, inst1, problem, filename1)
    plt.close()

    #print(all_trips.columns)
    for d1 in data_dist:
        plt.hist(d1 , bins=100, density=True, histtype="step")
    
    #plt.yscale('log')
    plt.xlabel("distance (meters)")
    plt.ylabel("probability")
    plt.tight_layout()
    plt.savefig('fitting_distsyn_curve.png')
    plt.close()
    #ax = inst1['direct_distance'].hist(bins=100, density=True, histype="step")
    
    '''
    print(inst1['direct_distance'].describe())
    print(inst1['direct_distance'].mean())
    print(inst1['direct_distance'].std()
    '''