import networkx as nx
import os
import osmnx as ox
import pandas as pd
import pickle

from pathlib import Path

def dynamism(inst1, ed, ld):

    #time_stamp = 'time_stamp'
    Te = abs(ld - ed)

    inst1.sort()

    sorted_ts = inst1

    #sorted_ts = inst1[time_stamp].tolist()
    #sorted_ts = [i for i in sorted_ts if i != 0]
    #exclude time stamp 0

    DELTA = []
    for ts in range(len(sorted_ts)-1):
        DELTA.append(float(abs(sorted_ts[ts+1] - sorted_ts[ts])))

    number_reqs = len(inst1)

    theta = Te/len(sorted_ts)

    #print(theta)
    #print(DELTA)

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

    #print(rho)
    return rho

'''
if __name__ == '__main__':

    place_name = "Rennes, France"
    save_dir = os.getcwd()+'/'+place_name
    pickle_dir = os.path.join(save_dir, 'pickle')
    network_class_file = pickle_dir+'/'+place_name+'.network.class.pkl'

    standard_filename = 'Rennes,France_DARP_500'
    param_class_file = pickle_dir+'/'+standard_filename+'.param.class.pkl'

    network_directory = os.getcwd()+'/'+place_name

    if Path(param_class_file).is_file():
        with open(param_class_file, 'rb') as param_class_file:
            param = pickle.load(param_class_file)

    csv_directory = network_directory+'/csv_format'
    directory = os.fsencode(csv_directory)
    for file_inst1 in os.listdir(directory):

        filename1 = os.fsdecode(file_inst1)

        if (filename1.endswith(".csv")):
        
            inst1 = pd.read_csv(csv_directory+'/'+filename1)

            inst1 = inst1.sort_values("time_stamp")

            sorted_ts = inst1['time_stamp'].tolist()
            sorted_ts = [i for i in sorted_ts if i != 0]
            #exclude time stamp 0

            DELTA = []
            for ts in range(len(sorted_ts)-1):
                DELTA.append(abs(sorted_ts[ts+1] - sorted_ts[ts]))

            number_reqs = len(inst1)
            Te = param.parameters['max_early_departure']['value'] - param.parameters['min_early_departure']['value']

            theta = int(Te/len(sorted_ts))

            SIGMA = []
            for k in range(len(DELTA)):

                if ((k == 0) and (DELTA[k] < theta)): 

                    SIGMA.append(theta - DELTA[k])

                else:

                    if ((k > 0) and (DELTA[k] < theta)): 

                         SIGMA.append(theta - DELTA[k] + SIGMA[k-1]*((theta - DELTA[k])/theta))

                    else:

                         SIGMA.append(0)

            lambdax = 0
            for sk in SIGMA:

                lambdax += sk

            NEGSIGMA = []
            for k in range(len(DELTA)):

                if ((k > 0) and (DELTA[k] < theta)): 

                    NEGSIGMA.append(theta + SIGMA[k-1]*((theta - DELTA[k])/theta))

                else:

                    NEGSIGMA.append(theta)

            eta = 0
            for nsk in NEGSIGMA:

                eta += nsk

            rho = 1 - (sum(SIGMA)/sum(NEGSIGMA)) 

            print(rho)
'''


            

                

                


