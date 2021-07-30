import networkx as nx
import os
import osmnx as ox
import pandas as pd

from pathlib import Path
from instance_class import Instance


if __name__ == '__main__':

    place_name = "Rennes, France"
    save_dir = os.getcwd()+'/'+place_name
    pickle_dir = os.path.join(save_dir, 'pickle')
    network_class_file = pickle_dir+'/'+place_name+'.network.class.pkl'

    network_directory = os.getcwd()+'/'+place_name

    if Path(network_class_file).is_file():
        inst = Instance(folder_to_network=place_name)

    thtt = 360
    thts = 60
    the = 60
    csv_directory = network_directory+'/csv_format'
    directory = os.fsencode(csv_directory)
    for file_inst1 in os.listdir(directory):
        
        filename1 = os.fsdecode(file_inst1)
        
        
        print(filename1)
        if (filename1.endswith(".csv")):

            inst1 = pd.read_csv(csv_directory+'/'+filename1)

            number_reqs = len(inst1)


            for file_inst2 in os.listdir(directory):

                filename2 = os.fsdecode(file_inst2)

                if (filename1 != filename2):

                    G = nx.Graph()
                    for i in range(number_reqs*2):
                        G.add_node(int(i))

                    top_nodes = [i for i in range(number_reqs)]
                    bottom_nodes = [i+500 for i in range(number_reqs)]
                    #print(top_nodes)
                    #print(bottom_nodes)
                    #G.add_nodes_from(top_nodes, bipartite=0)
                    #G.add_nodes_from(bottom_nodes, bipartite=1)

                    if (filename2.endswith(".csv")):

                        inst2 = pd.read_csv(csv_directory+'/'+filename2)

                        for id1, req1 in inst1.iterrows():

                            o1 = req1['originnode_drive']
                            d1 = req1['destinationnode_drive']

                            for id2, req2 in inst2.iterrows():

                                #if id2 >= id1:
                                    #print(id1, id2)
                                o2 = req2['originnode_drive']
                                d2 = req2['destinationnode_drive']

                                oott = inst.network._return_estimated_travel_time_drive(int(o1), int(o2))  
                                ddtt = inst.network._return_estimated_travel_time_drive(int(d1), int(d2)) 

                                oott2 = inst.network._return_estimated_travel_time_drive(int(o2), int(o1))  
                                ddtt2 = inst.network._return_estimated_travel_time_drive(int(d2), int(d1))  

                                #odtt = inst.network._return_estimated_travel_time_drive(int(o1), int(d2))  
                                #dott = inst.network._return_estimated_travel_time_drive(int(d1), int(o2))


                                phi = min(oott + ddtt, oott2 + ddtt2)
                                #phi = oott + ddtt

                                n1 = int(id1)
                                n2 = int(id2+number_reqs)
                                #print(n1, n2)
                                if phi < thtt:
                                    #print("here")
                                    tau = abs(req1['time_stamp'] - req2['time_stamp'])

                                    eu1 = abs(req1['earliest_departure'])
                                    eu2 = abs(req2['earliest_departure'])
                                    vartheta = abs(eu1 - eu2)

                                    #print(tau, vartheta)

                                    if (tau < thts) and (vartheta < the):

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
                        print(si1i2)
                        si1i2 = si1i2/count

                        print(si1i2)
