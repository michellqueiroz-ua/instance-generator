import math
import matplotlib.pyplot as plt
import os
import osmnx as ox
import pandas as pd
import shapely
from shapely.geometry import Polygon
from shapely.geometry import Point
from pathlib import Path
import sys
sys.path.append('REQreate')
from instance_class import Instance


if __name__ == '__main__':

    place_name = "Maastricht, Netherlands"
    save_dir = os.getcwd()+'/'+place_name
    pickle_dir = os.path.join(save_dir, 'pickle')
    network_class_file = pickle_dir+'/'+place_name+'.network.class.pkl'

    print(f"Looking for network class file: {network_class_file}")
    
    if Path(network_class_file).is_file():
        print("Network class file found, loading...")
        inst = Instance(folder_to_network=place_name)
    else:
        print("ERROR: Network class file not found!")
        exit(1)

    save_dir_csv = os.path.join(inst.save_dir, 'csv')

    zones_file = save_dir_csv+'/'+place_name+'.zones.csv'
    stations_file = save_dir_csv+'/'+place_name+'.stations.csv'
    
    print(f"\nLoading zones from: {zones_file}")
    print(f"Loading stations from: {stations_file}")
    
    zones = pd.read_csv(zones_file)
    stations = pd.read_csv(stations_file)
    
    print(f"\nTotal zones: {len(zones)}")
    print(f"Total stations: {len(stations)}")
    print(f"\nAssigning bus stations to zones...")

    all_stations = []
    for indexz, zone in zones.iterrows():
        if indexz % 50 == 0:
            print(f"Processing zone {indexz}/{len(zones)}")
        
        stationslis = []
        for indexs, station in stations.iterrows():

            polygon = shapely.wkt.loads(zone['polygon'])
            pnt = Point(station['lon'], station['lat'])
            if polygon.contains(pnt):
                stationslis.append(indexs)

        all_stations.append(stationslis)
        
    print(f"\nAssignment complete!")
    
    # Calculate statistics
    zones_with_stations = sum(1 for s in all_stations if len(s) > 0)
    total_assignments = sum(len(s) for s in all_stations)
    
    print(f"Statistics:")
    print(f"  - Zones with at least one station: {zones_with_stations}/{len(zones)}")
    print(f"  - Total station assignments: {total_assignments}")

    zones['stations'] = all_stations

    output_file = save_dir_csv+'/'+place_name+'.zones.csv'
    print(f"\nSaving updated zones file to: {output_file}")
    zones = pd.DataFrame(zones)
    zones.to_csv(output_file)
    
    print("\nDONE! The zones CSV now has a 'stations' column.")
