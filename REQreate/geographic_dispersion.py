import networkx as nx
import os
import osmnx as ox
import pandas as pd

from pathlib import Path
from instance_class import Instance


def geographic_dispersion(inst, problem, filename1):

    network_directory = os.getcwd()+'/'+inst.network.place_name
    csv_directory = network_directory+'/csv_format'
    ttm_directory = network_directory+'/travel_time_matrix'
    directory = os.fsencode(csv_directory)

    #filename1 = os.fsdecode(file_inst1)

    #if (filename1.endswith(".csv")):
    
    inst1 = pd.read_csv(csv_directory+'/'+filename1)
    ttm_file_inst1 = 'travel_time_matrix_'+filename1
    ttmfilename1 = os.fsdecode(ttm_file_inst1)
    #ttm1 = pd.read_csv(ttm_directory+'/'+ttmfilename1)
    #ttm1.set_index(['osmid_origin'], inplace=True)

    if problem == 'ODBRP':

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
                        #summ += ttm1.loc[so, str(sd)]
                        stopo = int(inst.network.bus_stations.loc[so, 'osmid_drive'])
                        stopd = int(inst.network.bus_stations.loc[sd, 'osmid_drive'])
                        summ += round(inst.network._return_estimated_travel_time_drive(int(stopo), int(stopd)),1) 
                        count += 1

            ovrsumm += round(summ/count,1)

        
        muodbrp = round(ovrsumm/(len(inst1)),1)

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

                    #tuplx = (x, ttm1.loc[int(s), str(x)])
                    stopo = int(inst.network.bus_stations.loc[s, 'osmid_drive'])
                    stopd = int(inst.network.bus_stations.loc[x, 'osmid_drive'])
                    tuplx = (x, round(inst.network._return_estimated_travel_time_drive(int(stopo), int(stopd)),1))
                    ltrot.append(tuplx)

            
            dest_stps = row1[stops_dest].strip('][').split(', ')
            dest_stps = [int(i) for i in dest_stps]
            ltrd = [int(i) for i in ltrd if int(i) not in dest_stps]
            for s in dest_stps:
                for y in ltrd:

                    #tuply = (y, ttm1.loc[int(s), str(y)])
                    stopo = int(inst.network.bus_stations.loc[s, 'osmid_drive'])
                    stopd = int(inst.network.bus_stations.loc[y, 'osmid_drive'])
                    tuply = (y, round(inst.network._return_estimated_travel_time_drive(int(stopo), int(stopd)), 1))
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
        gd = muodbrp + omegaodbrp

    else:
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
        
        #print(mudarp)
        #print(omegadarp)
        gd = mudarp + omegadarp
        #print(gd)

    return gd
'''
if __name__ == '__main__':

    place_name = "Rennes, France"
    problem = 'ODBRP'
    save_dir = os.getcwd()+'/'+place_name
    pickle_dir = os.path.join(save_dir, 'pickle')
    network_class_file = pickle_dir+'/'+place_name+'.network.class.pkl'

    
    if Path(network_class_file).is_file():
        inst = Instance(folder_to_network=place_name)

    network_directory = os.getcwd()+'/'+place_name
    csv_directory = network_directory+'/csv_format'
    ttm_directory = network_directory+'/travel_time_matrix'
    directory = os.fsencode(csv_directory)
    for file_inst1 in os.listdir(directory):

        filename1 = os.fsdecode(file_inst1)

        if (filename1.endswith(".csv")):
        
            inst1 = pd.read_csv(csv_directory+'/'+filename1)

            ttm_file_inst1 = 'travel_time_matrix_'+filename1
            ttmfilename1 = os.fsdecode(ttm_file_inst1)
            ttm1 = pd.read_csv(ttm_directory+'/'+ttmfilename1)
            ttm1.set_index(['osmid_origin'], inplace=True)

            if problem == 'ODBRP':

                stations = []
                ovrsumm = 0
                for idx, row in inst1.iterrows():
                    
                    orgn = row['stops_orgn'].strip('][').split(', ')
                    
                    dest = row['stops_dest'].strip('][').split(', ')

                    
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

                            
                            if (row2['earliest_departure'] >= row1['earliest_departure'] - 600) and (row2['earliest_departure'] <= row1['earliest_departure'] + 600):
                                
                                stps = row2['stops_orgn'].strip('][').split(', ')
                                ltro.extend(stps)

                            if (row2['latest_arrival'] >= row1['earliest_departure'] - 600) and (row2['latest_arrival'] <= row1['earliest_departure'] + 600):
                                stps = row2['stops_dest'].strip('][').split(', ')
                                ltro.extend(stps)

                            if (row2['latest_arrival'] >= row1['latest_arrival'] - 600) and (row2['latest_arrival'] <= row1['latest_arrival'] + 600):
                                stps = row2['stops_dest'].strip('][').split(', ')
                                ltrd.extend(stps)

                            if (row2['earliest_departure'] >= row1['latest_arrival'] - 600) and (row2['earliest_departure'] <= row1['latest_arrival'] + 600):
                                stps = row2['stops_orgn'].strip('][').split(', ')
                                ltrd.extend(stps)

                    ltro = list(dict.fromkeys(ltro))
                    ltrd = list(dict.fromkeys(ltrd))

                    ltrot = []
                    ltrdt = []

                    org_stps = row1['stops_orgn'].strip('][').split(', ')
                    org_stps = [int(i) for i in org_stps]
                    ltro = [int(i) for i in ltro if int(i) not in org_stps]
                    for s in org_stps:
                        for x in ltro:

                            tuplx = (x, ttm1.loc[int(s), str(x)])
                            ltrot.append(tuplx)

                    
                    dest_stps = row1['stops_dest'].strip('][').split(', ')
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
                gd = muodbrp + omegaodbrp
            
            else:

                #mu
                #average travel time between origin and destinations
                dtt = []
                for idx, req in inst1.iterrows():
                    dtt.append(req['direct_travel_time'])


                mudarp = sum(dtt) / len(dtt)
                mu2 = inst1['direct_travel_time'].mean()

                #average travel time between x nearest neighbors
                sumnn = 0
                for idx1, row1 in inst1.iterrows():

                    ltro = []
                    ltrd = []
                    for idx2, row2 in inst1.iterrows():

                        if idx2 != idx1:

                            if (row2['earliest_departure'] >= row1['earliest_departure'] - 600) and (row2['earliest_departure'] <= row1['earliest_departure'] + 600):
                                if (row2['originnode_drive'] != row1['originnode_drive']) and (row2['originnode_drive'] != row1['destinationnode_drive']):
                                    ltro.append(row2['originnode_drive'])

                            if (row2['latest_arrival'] >= row1['earliest_departure'] - 600) and (row2['latest_arrival'] <= row1['earliest_departure'] + 600):
                                if (row2['destinationnode_drive'] != row1['originnode_drive']) and (row2['destinationnode_drive'] != row1['destinationnode_drive']):
                                    ltro.append(row2['destinationnode_drive'])

                            if (row2['latest_arrival'] >= row1['latest_arrival'] - 600) and (row2['latest_arrival'] <= row1['latest_arrival'] + 600):
                                if (row2['destinationnode_drive'] != row1['originnode_drive']) and (row2['destinationnode_drive'] != row1['destinationnode_drive']):
                                    ltrd.append(row2['destinationnode_drive'])

                            if (row2['earliest_departure'] >= row1['latest_arrival'] - 600) and (row2['earliest_departure'] <= row1['latest_arrival'] + 600):
                                if (row2['originnode_drive'] != row1['originnode_drive']) and (row2['originnode_drive'] != row1['destinationnode_drive']):
                                    ltrd.append(row2['originnode_drive'])

                    ltrot = []
                    ltrdt = []
                    
                    org_row1 = int(row1['originnode_drive'])
                    for x in ltro:

                        tuplx = (x, inst.network._return_estimated_travel_time_drive(int(org_row1), int(x)))
                        ltrot.append(tuplx)

                    dest_row1 = int(row1['destinationnode_drive'])
                    for y in ltrd:

                        tuply = (y, inst.network._return_estimated_travel_time_drive(int(dest_row1), int(y)))
                        ltrdt.append(tuply)


                    #sort tuples
                    ltrot.sort(key = lambda x: x[1]) 
                    ltrdt.sort(key = lambda x: x[1])
                    
                    #average of the 5 first
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

                omegadarp = sumnn/(len(inst1)*2)
                
                gd = mudarp + omegadarp
'''
