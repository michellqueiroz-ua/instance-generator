import os
import numpy as np
import random

from retrieve_network import download_network_information
from output_files import output_fixed_route_network



if __name__ == '__main__':

    #retrieve the instance's network
    
    random.seed(100)
    np.random.seed(100)    
    #place_name='Lisbon, Portugal'
    get_fixed_lines = 'deconet'

    place_name='Lisbon, Portugal'
    #get_fixed_lines=None
    network = download_network_information(place_name=place_name, max_speed_factor=0.5, get_fixed_lines=get_fixed_lines)

    save_dir = os.getcwd()+'/'+place_name
    save_dir_fr = os.path.join(save_dir, 'fr_network')
    if not os.path.isdir(save_dir_fr):
        os.mkdir(save_dir_fr)

    output_name_fr = place_name+'.frn'

    if get_fixed_lines is not None:
        output_fixed_route_network(output_file_name=os.path.join(save_dir_fr, output_name_fr), network=network)