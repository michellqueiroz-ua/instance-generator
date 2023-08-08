import networkx as nx
import os
#import osmnx as ox
import pandas as pd
import pickle
import numpy as np
import random

from pathlib import Path

from shapely.geometry import Polygon
from descartes.patch import PolygonPatch
from shapely.geometry import Point
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.style.use('ggplot')

from matplotlib.colors import LogNorm
import geopy.distance
import math

def geographic_dispersion(inst1):

    #mu
    #average travel time between origin and destinations
    dtt = []
    for idx, req in inst1.iterrows():
        dtt.append(req['Trip_Seconds'])

    mudarp = sum(dtt) / len(dtt)
    mu2 = inst1['Trip_Seconds'].mean()


    #average travel time between x nearest neighbors
    #nyc -> compute for the 5 nearest zones
    earliest_departure = 'earliest_departure'
    latest_arrival = 'latest_arrival'
    time_gap = 15
    #node_origin = 
    #node_destination = 
    #osmid_origin = 'originnode_drive'
    #osmid_destination = 'destinationnode_drive'
    speed = 7.22 #26kmh
    
    sumnn = 0
    for idx1, row1 in inst1.iterrows():

        ltro = []
        ltrd = []
        for idx2, row2 in inst1.iterrows():

            if idx2 != idx1:

                #latest_arrival1 = row1[earliest_departure] + row1['Trip_Seconds']
                #latest_arrival2 = row2[earliest_departure] + row2['Trip_Seconds']
                #print(row2['earliest_departure'])
                if (row2[earliest_departure] >= row1[earliest_departure] - time_gap) and (row2[earliest_departure] <= row1[earliest_departure] + time_gap):
                    #if (row2['originnode_drive'] != row1['originnode_drive']) and (row2['originnode_drive'] != row1['destinationnode_drive']):
                    #ltro.append(row2['originnode_drive'])
                    #origin_point = (row2['Pickup_Centroid_Latitude'], row2['Pickup_Centroid_Longitude'])
                    #node_origin = row2[osmid_origin]
        
                    ltro.append(row2['origin'])

                if (row2[latest_arrival] >= row1[earliest_departure] - time_gap) and (row2[latest_arrival] <= row1[earliest_departure] + time_gap):
                    #if (row2['destinationnode_drive'] != row1['originnode_drive']) and (row2['destinationnode_drive'] != row1['destinationnode_drive']):
                    #ltro.append(row2['destinationnode_drive'])
                    #destination_point = (row2['Dropoff_Centroid_Latitude'], row2['Dropoff_Centroid_Longitude'])
                    #node_destination = row2[osmid_destination]
                    
                    ltro.append(row2['destination'])

                if (row2[latest_arrival] >= row1[latest_arrival] - time_gap) and (row2[latest_arrival] <= row1[latest_arrival] + time_gap):
                    #if (row2['destinationnode_drive'] != row1['originnode_drive']) and (row2['destinationnode_drive'] != row1['destinationnode_drive']):
                    #ltrd.append(row2['destinationnode_drive'])
                    #destination_point = (row2['Dropoff_Centroid_Latitude'], row2['Dropoff_Centroid_Longitude'])
                    #node_destination = row2[osmid_destination]

                    ltro.append(row2['destination'])

                if (row2[earliest_departure] >= row1[latest_arrival] - time_gap) and (row2[earliest_departure] <= row1[latest_arrival] + time_gap):
                    #if (row2['originnode_drive'] != row1['originnode_drive']) and (row2['originnode_drive'] != row1['destinationnode_drive']):
                    #ltrd.append(row2['originnode_drive'])
                    #origin_point = (row2['Pickup_Centroid_Latitude'], row2['Pickup_Centroid_Longitude'])
                    #node_origin = row2[osmid_origin]

                    ltro.append(row2['origin'])

        #ltro = list(dict.fromkeys(ltro))
        #ltrd = list(dict.fromkeys(ltrd))
        #print(ltro)
        #print(ltrd)

        ltrot = []
        ltrdt = []
        
        #org_row1 = int(row1['originnode_drive'])
        #origin_point = (row1['Pickup_Centroid_Latitude'], row1['Pickup_Centroid_Longitude'])
        #org_row1 = ox.get_nearest_node(inst.network.G_drive, origin_point)
        org_row1 = row1['origin']
        
        for x in ltro:

            #tuplx = (x, inst.network._return_estimated_travel_time_drive(int(org_row1), int(x)))
            distance = math.sqrt( ((org_row1[0]-x[0])**2)+((org_row1[1]-x[1])**2) )
            #print(org_row1[0], org_row1[1])
            #print(distance)
            #dist = inst.network._return_estimated_distance_drive(int(org_row1), int(x))
            #tt = (distance*1000)/speed
            tt = distance*60
            tuplx = (x, tt)
            ltrot.append(tuplx)

        #dest_row1 = int(row1['destinationnode_drive'])
        #destination_point = (row1['Dropoff_Centroid_Latitude'], row1['Dropoff_Centroid_Longitude'])
        #dest_row1 = ox.get_nearest_node(inst.network.G_drive, destination_point)
        dest_row1 = row1['destination']

        for y in ltrd:

            #tuply = (y, inst.network._return_estimated_travel_time_drive(int(dest_row1), int(y)))
            #dist = inst.network._return_estimated_distance_drive(int(dest_row1), int(y))
            distance = math.sqrt( ((dest_row1[0]-y[0])**2)+((dest_row1[1]-y[1])**2) )
            #tt = (distance*1000)/speed
            tt = distance*60
            tuply = (y, tt)
            ltrdt.append(tuply)


        #ordenar as tuplas
        ltrot.sort(key = lambda x: x[1]) 
        ltrdt.sort(key = lambda x: x[1])

        #print(ltrot)
        #print(ltrdt)
        
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
    
    #print(mudarp)
    #print(omegadarp)
    gd = mudarp + omegadarp
    #print(gd)
    
    #gd = mudarp
    return gd

if __name__ == '__main__':


    directory = 'instances_dcdarp'

    num_req = 0
    num_veh = 0
    gds = []
    for filename in os.listdir(directory):  
        if (filename != '.DS_Store'):
            filenameff = os.path.join(directory, filename)
            print(filename)
            with open(filenameff) as fileobj:
                count=0
                df = pd.DataFrame(columns = ['Trip_Seconds', 'earliest_departure', 'latest_arrival', 'origin', 'destination'])
                for line in fileobj:
                    count+=1

                    if count==2:
                        x = line.split()
                        num_req = x[0]
                        print(num_req)

                    if count==4:
                        x = line.split()
                        num_veh = int(x[0])
                        print(num_veh)

                    if count>4+num_veh:
                        #print(line)
                        x = line.split()

                        o_x = x[2]
                        o_y = x[3]

                        d_x = x[7]
                        d_y = x[8]

                        earliest_departure = x[5]
                        latest_arrival = x[11]

                        p1 = [float(o_x), float(o_y)]
                        p2 = [float(d_x), float(d_y)]

                        #p1 = [4, 0]
                        #p2 = [6, 6]
                        distance = math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2) )
                        #print(distance)

                        speed = 7.22 #26kmh
                        #trip_seconds = (distance*1000)/speed
                        trip_seconds = distance*60
                        
                        new_row = {'Trip_Seconds':trip_seconds, 'earliest_departure':int(earliest_departure), 'latest_arrival':int(latest_arrival), 'origin': p1, 'destination': p2}
                        df = df.append(new_row, ignore_index=True)
                        #print(trip_seconds)
                        #print (geopy.distance.geodesic(coords_1, coords_2))

                        #print(earliest_departure, latest_arrival)

                #print(df)
                
                gds.append(geographic_dispersion(df))
                #print(gds)

    print(gds)
    print(min(gds)/60)
    print(max(gds)/60)




