from retrieve_network import download_network_information
       

if __name__ == '__main__':

    #retrieve the instance's network
    network = download_network_information(place_name='Rennes, France', get_fixed_lines="deconet")
