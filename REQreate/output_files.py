import json
import math
import networkx as nx
import pandas as pd

def output_fixed_route_network(output_file_name, network):

    with open(output_file_name, 'w') as file:

        fixed_lines_stations = []
        file.write(str(len(network.linepieces)))
        file.write('\n')

        i = 0
        for lp in network.linepieces:

            file.write(str(len(lp)))
            file.write(' ')

            for station in lp:
                if station not in fixed_lines_stations:
                    fixed_lines_stations.append(station)
                file.write(str(station))
                file.write(' ')

            file.write('\n')

            for distuv in network.linepieces_dist[i]:
                file.write(str(int(distuv)))
                file.write(' ')

            file.write('\n')

            i += 1

        file.write(str(len(network.connecting_nodes)))
        file.write('\n')

        for station in network.connecting_nodes:
            file.write(str(station))
            file.write(' ')
        file.write('\n')

        file.write(str(len(network.direct_lines)))
        file.write('\n')
        for dl in network.direct_lines:

            file.write(str(len(dl)))
            file.write(' ')

            for station in dl:
                file.write(str(station))
                file.write(' ')

            file.write('\n')

        file.write(str(len(network.transfer_nodes)))
        file.write('\n')

        for station in network.transfer_nodes:
            file.write(str(station))
            file.write(' ')
        file.write('\n')

        '''
        for ids in network.subway_lines:

            file.write(str(ids))
            file.write(' ')
            
            nodes_path = nx.dijkstra_path(network.subway_lines[ids]['route_graph'], network.subway_lines[ids]['begin_route'], network.subway_lines[ids]['end_route'], weight='duration_avg')

            file.write(str(len(nodes_path)))
            file.write(' ')

            for u in range(len(nodes_path)):
                file.write(str(int(network.deconet_network_nodes.loc[int(nodes_path[u]), 'bindex'])))
                file.write(' ')
            file.write('\n')
        '''

        for station in fixed_lines_stations:

            file.write(str(station))
            file.write(' ')

            file.write(str(network.bus_stations.loc[int(station), 'lat']))
            file.write(' ')

            file.write(str(network.bus_stations.loc[int(station), 'lon']))
            file.write('\n')


class JsonConverter(object):

    def __init__(self, file_name):

        with open(file_name, 'rb') as file:
            self.json_data = json.load(file)
            file.close()

    def convert_normal(self, inst, problem_type, path_instance_csv_file):

        instance = []
        requests = self.json_data.get('requests')
        for request in requests.values():

            d = {}
            for att in inst.sorted_attributes:

                if request.get(att) is not None:
                    
                    #if inst.GA.nodes[att]['output_csv'] is True:
                        d[att] = request.get(att)
                                   
                        if ((att == 'origin') or (att == 'destination')):

                            d[att+'node_drive'] = request.get(att+'node_drive')

                        if ((att == 'stops_orgn') or (att == 'stops_dest')):

                            d[att+'_walking_distance'] = request.get(att+'_walking_distance')

            d['seed_location'] = request.get('seed_location')
            d['reqid'] = request.get('reqid')
            instance.append(d)

        instance = pd.DataFrame(instance)
        instance = instance.sort_values("reqid")
        instance.to_csv(path_instance_csv_file)
        