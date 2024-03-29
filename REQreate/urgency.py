import networkx as nx
import os
import osmnx as ox
import pandas as pd
import statistics

from pathlib import Path
from instance_class import Instance

def urgency(inst1):

    chi = []
    for idx, req in inst1.iterrows():

        if req['time_stamp'] > 0:
            er = abs(req['latest_departure'] - req['time_stamp'])
            chi.append(er)

    
    mean = sum(chi) / len(chi)
    variance = sum([((x - mean) ** 2) for x in chi]) / len(chi)
    stdv = variance ** 0.5
    stdv2 = statistics.pstdev(chi)

    return mean, stdv

'''
if __name__ == '__main__':

    place_name = "Rennes, France"
    save_dir = os.getcwd()+'/'+place_name
    pickle_dir = os.path.join(save_dir, 'pickle')
    network_class_file = pickle_dir+'/'+place_name+'.network.class.pkl'

    network_directory = os.getcwd()+'/'+place_name

    if Path(network_class_file).is_file():
        inst = Instance(folder_to_network=place_name)

    csv_directory = network_directory+'/csv_format'
    directory = os.fsencode(csv_directory)
    for file_inst1 in os.listdir(directory):

        filename1 = os.fsdecode(file_inst1)

        if (filename1.endswith(".csv")):
        
            inst1 = pd.read_csv(csv_directory+'/'+filename1)

            chi = []
            for idx, req in inst1.iterrows():

                if req['time_stamp'] > 0:
                    er = abs(req['latest_departure'] - req['time_stamp'])
                    chi.append(er)

            
            mean = sum(chi) / len(chi)
            variance = sum([((x - mean) ** 2) for x in chi]) / len(chi)
            stdv = variance ** 0.5
            stdv2 = statistics.pstdev(chi)

            
            print(mean)
            #print(stdv)

            #print(mean2)
            #print(stdv4)
'''