import networkx as nx
import os
import osmnx as ox
import pandas as pd


if __name__ == '__main__':

	if Path(network_class_file).is_file():
        inst = Instance(folder_to_network=place_name)

    thtt = 120
    thts = 60
    the = 30
	directory = "Rennes, France/csv_instance"
	for file_inst1 in os.listdir(directory):
		G = nx.DiGraph()

		number_reqs = len(inst1)

		for i in range(number_reqs*2):
			G.add_node(i)

		for file_inst2 in os.listdir(directory):

			if (file_inst1 != file_inst2):
				if (file_inst1.endswith(".csv") and file_inst2.endswith(".csv")):

					inst1 = pd.read_csv(file_inst1)

					inst2 = pd.read_csv(file_inst2)

					for id1, req1 in inst1.iterrows():

						o1 = inst1.loc[int(id1), 'originnode_drive']
						d1 = inst1.loc[int(id1), 'destinationnode_drive']

						for id2, req2 in inst2.iterrows():

							o2 = inst1.loc[int(id1), 'originnode_drive']
							d2 = inst1.loc[int(id1), 'destinationnode_drive']

							oott = inst.network._return_estimated_travel_time_drive(int(o1), int(o2))  
							ddtt = inst.network._return_estimated_travel_time_drive(int(d1), int(d2))  

							odtt = inst.network._return_estimated_travel_time_drive(int(o1), int(d2))  
							dott = inst.network._return_estimated_travel_time_drive(int(d1), int(o2))


							phi = min(oott + ddtt, odtt + dott)

							if phi < thtt:


								tau = abs(inst1.loc[int(id1), 'time_stamp'] - inst2.loc[int(id2), 'time_stamp'])

								eu1 = abs(inst1.loc[int(id1), 'time_stamp'] - inst1.loc[int(id1), 'latest_departure'])
								eu2 = abs(inst2.loc[int(id2), 'time_stamp'] - inst2.loc[int(id2), 'latest_departure'])
								vartheta = abs(eu1 - eu2)

								if (tau < thts) and (vartheta < the):

									G.add_edge(id1, id2+number_reqs, similarity=1)

								else:

									if (tau < thts) or (vartheta < the):

										G.add_edge(id1, id2+number_reqs, similarity=0.75)

									else:

										G.add_edge(id1, id2+number_reqs, similarity=0.5)
							else:

								G.add_edge(id1, id2+number_reqs, similarity=0)


		M = nx.max_weight_matching(G, weight='similarity')


		si1i2 = 0

		for e in M:

			si1i2 += G.edges[e]['similarity']

		si1i2 = si1i2/len(M)

		print(si1i2)