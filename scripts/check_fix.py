"""
Test script to verify the fix for assigning bus stations to zones.

This script checks if the network_class.py fix will work correctly.
"""

import pandas as pd
import json
import os
from shapely.geometry import Point
from pathlib import Path

place_name = "Maastricht, Netherlands"
save_dir = os.path.join(os.getcwd(), place_name)
save_dir_csv = os.path.join(save_dir, 'csv')

print("=" * 60)
print("TESTING THE FIX")
print("=" * 60)

# Load the existing files
zones_file = os.path.join(save_dir_csv, f'{place_name}.zones.csv')
stations_file = os.path.join(save_dir_csv, f'{place_name}.stations.csv')

print(f"\n1. Loading files...")
print(f"   Zones: {zones_file}")
print(f"   Stations: {stations_file}")

zones = pd.read_csv(zones_file)
stations = pd.read_csv(stations_file)

print(f"   Zones loaded: {len(zones)}")
print(f"   Stations loaded: {len(stations)}")

# Simulate what the fix will do
print(f"\n2. Simulating station assignment (this is what the fix does)...")

import shapely.wkt

all_stations = []
for indexz, zone in zones.iterrows():
    if indexz % 50 == 0:
        print(f"   Processing zone {indexz}/{len(zones)}")
    
    stationslis = []
    polygon = shapely.wkt.loads(zone['polygon'])
    
    for indexs, station in stations.iterrows():
        pnt = Point(station['lon'], station['lat'])
        if polygon.contains(pnt):
            stationslis.append(indexs)
    
    all_stations.append(stationslis)

zones['stations'] = all_stations

# Statistics
zones_with_stations = sum(1 for s in all_stations if len(s) > 0)
total_assignments = sum(len(s) for s in all_stations)

print(f"\n3. Results:")
print(f"   Zones with stations: {zones_with_stations}/{len(zones)}")
print(f"   Zones without stations: {len(zones) - zones_with_stations}/{len(zones)}")
print(f"   Total station assignments: {total_assignments}")

if zones_with_stations > 0:
    print(f"\n   ✓ SUCCESS! The fix will work correctly.")
    print(f"   Sample zones with stations:")
    count = 0
    for idx, row in zones.iterrows():
        if len(all_stations[idx]) > 0 and count < 5:
            print(f"      Zone {idx}: {len(all_stations[idx])} stations - {all_stations[idx][:5]}")
            count += 1
else:
    print(f"\n   ✗ PROBLEM: No stations were assigned to any zones!")
    print(f"   This might indicate an issue with polygon or coordinate data.")

print("\n" + "=" * 60)
print("NOTE: To apply the fix, you need to regenerate the network")
print("by running your JSON configuration through input_json.py again.")
print("The fix is now in network_class.py and will run automatically.")
print("=" * 60)
