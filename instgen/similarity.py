import networkx as nx
import os
import osmnx as ox
import pandas as pd

from pathlib import Path
from instance_class import Instance


if __name__ == '__main__':

    G2 = nx.Graph()
    G2.add_node("A")
    G2.add_node("B")
    G2.add_node("C")
    G2.add_node("D")
    G2.add_node("E")
    G2.add_node("F")
    G2.add_node("G")
    G2.add_node("H")
    G2.add_node("I")
    G2.add_node("J")
    G2.add_node("K")
    G2.add_node("L")
    G2.add_edge("A", "B")
    G2.add_edge("B", "D")
    G2.add_edge("B", "C")
    G2.add_edge("D", "F")
    G2.add_edge("C", "E")
    #G2.add_edge("F", "E")
    G2.add_edge("E", "G")
    G2.add_edge("F", "G")
    G2.add_edge("F", "J")
    G2.add_edge("J", "L")
    G2.add_edge("G", "I")
    G2.add_edge("G", "H")
    G2.add_edge("H", "K")
    G2.add_edge("J", "I")

    print(nx.center(G2))


    place_name = "Rennes, France"
    save_dir = os.getcwd()+'/'+place_name
    pickle_dir = os.path.join(save_dir, 'pickle')
    network_class_file = pickle_dir+'/'+place_name+'.network.class.pkl'

    network_directory = os.getcwd()+'/'+place_name

    if Path(network_class_file).is_file():
        inst = Instance(folder_to_network=place_name)

    thtt = 180
    thts = 60
    the = 30
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

                    if (filename2.endswith(".csv")):

                        inst2 = pd.read_csv(csv_directory+'/'+filename2)

                        for id1, req1 in inst1.iterrows():

                            o1 = req1['originnode_drive']
                            d1 = req1['destinationnode_drive']

                            for id2, req2 in inst2.iterrows():

                                if id2 >= id1:
                                    #print(id1, id2)
                                    o2 = req2['originnode_drive']
                                    d2 = req2['destinationnode_drive']

                                    oott = inst.network._return_estimated_travel_time_drive(int(o1), int(o2))  
                                    ddtt = inst.network._return_estimated_travel_time_drive(int(d1), int(d2))  

                                    odtt = inst.network._return_estimated_travel_time_drive(int(o1), int(d2))  
                                    dott = inst.network._return_estimated_travel_time_drive(int(d1), int(o2))


                                    phi = min(oott + ddtt, odtt + dott)

                                    n1 = int(id1)
                                    n2 = int(id2+number_reqs)
                                    if phi < thtt:

                                        tau = abs(req1['time_stamp'] - req2['time_stamp'])

                                        eu1 = abs(req1['latest_departure'] - req1['time_stamp'])
                                        eu2 = abs(req2['latest_departure'] - req2['time_stamp'])
                                        vartheta = abs(eu1 - eu2)

                                        if (tau < thts) and (vartheta < the):

                                            G.add_edge(n1, n2, similarity=100)

                                        else:

                                            if (tau < thts) or (vartheta < the):

                                                G.add_edge(n1, n2, similarity=75)

                                            else:

                                                G.add_edge(n1, n2, similarity=50)
                                    else:

                                        G.add_edge(n1, n2, similarity=0)


                        M = nx.max_weight_matching(G, maxcardinality=True, weight='similarity')

                        si1i2 = 0
                        #print(len(M))
                        for e in M:
                            #print(e[0])
                            #print(e[1])
                            #print(e)
                            si1i2 += G.edges[int(e[0]), int(e[1])]['similarity']

                        si1i2 = si1i2/len(M)

                        print(si1i2)
