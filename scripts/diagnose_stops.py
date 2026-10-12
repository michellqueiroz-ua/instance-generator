import pandas as pd
import json
import os
import pickle
from pathlib import Path

# Configuration
place_name = "Maastricht, Netherlands"
save_dir = os.path.join(os.getcwd(), place_name)
save_dir_csv = os.path.join(save_dir, 'csv')
pickle_dir = os.path.join(save_dir, 'pickle')

print("=" * 60)
print("DIAGNOSTIC SCRIPT FOR EMPTY STOPS ISSUE")
print("=" * 60)

# Check if directories exist
print(f"\n1. Checking directories...")
print(f"   Save dir exists: {os.path.exists(save_dir)}")
print(f"   CSV dir exists: {os.path.exists(save_dir_csv)}")

# Find zones CSV file
print(f"\n2. Looking for zones CSV file...")
zones_files = [f for f in os.listdir(save_dir_csv) if '.zones.csv' in f]
print(f"   Found zones files: {zones_files}")

if not zones_files:
    print("   ERROR: No zones CSV file found!")
    exit(1)

zones_file = os.path.join(save_dir_csv, zones_files[0])
print(f"   Using: {zones_file}")

# Load zones CSV
print(f"\n3. Loading zones CSV...")
zonescsv = pd.read_csv(zones_file)
print(f"   Total zones: {len(zonescsv)}")
print(f"   Columns: {list(zonescsv.columns)}")

# Check if 'stations' column exists
if 'stations' not in zonescsv.columns:
    print("   ERROR: 'stations' column not found in zones CSV!")
    print("   You may need to run 'assing_bus_stops_zones.py' first")
    exit(1)

print(f"   'stations' column exists: YES")

# Check stations data in zones
print(f"\n4. Checking stations data in zones...")
zones_with_stations = 0
zones_without_stations = 0
total_stations_count = 0

for idx, row in zonescsv.iterrows():
    try:
        stations_data = row['stations']
        if pd.isna(stations_data) or stations_data == '' or stations_data == '[]':
            zones_without_stations += 1
        else:
            stations_list = json.loads(stations_data) if isinstance(stations_data, str) else stations_data
            if len(stations_list) > 0:
                zones_with_stations += 1
                total_stations_count += len(stations_list)
                if idx < 5:  # Show first 5 zones
                    print(f"   Zone {idx}: {len(stations_list)} stations - {stations_list}")
            else:
                zones_without_stations += 1
    except Exception as e:
        print(f"   Zone {idx}: ERROR - {e}")
        zones_without_stations += 1

print(f"\n   Summary:")
print(f"   - Zones WITH stations: {zones_with_stations}")
print(f"   - Zones WITHOUT stations: {zones_without_stations}")
print(f"   - Total stations across all zones: {total_stations_count}")

if zones_with_stations == 0:
    print("\n   ERROR: NO ZONES HAVE STATIONS ASSIGNED!")
    print("   This is why len(stops) is always 0.")
    print("   Solution: Run 'assing_bus_stops_zones.py' to assign stations to zones")
    exit(1)

# Find and load bus stations CSV
print(f"\n5. Looking for bus stations CSV file...")
stations_files = [f for f in os.listdir(save_dir_csv) if '.stations.csv' in f]
print(f"   Found stations files: {stations_files}")

if stations_files:
    stations_file = os.path.join(save_dir_csv, stations_files[0])
    print(f"   Using: {stations_file}")
    stations = pd.read_csv(stations_file)
    print(f"   Total bus stations: {len(stations)}")
    print(f"   Columns: {list(stations.columns)}")
else:
    print("   WARNING: No stations CSV file found!")
    stations = None

# Load network class if available
print(f"\n6. Checking network class...")
network_class_file = os.path.join(pickle_dir, place_name + '.network.class.pkl')
if Path(network_class_file).is_file():
    print(f"   Network class file exists: YES")
    try:
        with open(network_class_file, 'rb') as f:
            network = pickle.load(f)
        print(f"   Network loaded successfully")
        print(f"   Bus stations in network: {len(network.bus_stations)}")
    except Exception as e:
        print(f"   Error loading network: {e}")
        network = None
else:
    print(f"   Network class file exists: NO")
    network = None

# Simulate the stops finding logic
print(f"\n7. Simulating stops search for a sample request...")
if zones_with_stations > 0:
    # Find a zone with stations
    sample_zone_idx = None
    for idx, row in zonescsv.iterrows():
        try:
            stations_data = row['stations']
            if not pd.isna(stations_data) and stations_data != '' and stations_data != '[]':
                stations_list = json.loads(stations_data) if isinstance(stations_data, str) else stations_data
                if len(stations_list) > 0:
                    sample_zone_idx = idx
                    break
        except:
            pass
    
    if sample_zone_idx is not None:
        print(f"   Using zone {sample_zone_idx} as sample")
        stations_data = zonescsv.loc[sample_zone_idx]['stations']
        stationsz = json.loads(stations_data)
        print(f"   Stations in this zone: {stationsz}")
        print(f"   Number of stations: {len(stationsz)}")
        
        # Check if these station IDs exist in the stations dataframe
        if stations is not None:
            print(f"\n   Checking if station IDs are valid...")
            for station_id in stationsz[:5]:  # Check first 5
                if station_id in stations.index:
                    print(f"   Station {station_id}: EXISTS in stations CSV")
                else:
                    print(f"   Station {station_id}: NOT FOUND in stations CSV - THIS IS A PROBLEM!")

print("\n" + "=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)

if zones_without_stations > zones_with_stations:
    print("\nPOTENTIAL ISSUE: Most zones don't have stations assigned!")
    print("This could cause len(stops) = 0 if requests fall in zones without stations.")
