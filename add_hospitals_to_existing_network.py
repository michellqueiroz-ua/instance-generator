"""
Quick script to add hospital data to an existing network without regenerating everything.
Usage: python add_hospitals_to_existing_network.py "Location Name"
Example: python add_hospitals_to_existing_network.py "Maastricht, Netherlands"
"""

import sys
import os
import pickle

# Add REQreate to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'REQreate'))
from retrieve_hospitals import retrieve_hospitals

def add_hospitals_to_network(location_name):
    """Add hospitals to an existing network"""
    
    # Try original name first, then cleaned version
    network_dir = os.path.join(os.getcwd(), location_name)
    
    if not os.path.exists(network_dir):
        # Try cleaned version
        folder_name = location_name.replace(", ", "_").replace(" ", "_")
        network_dir = os.path.join(os.getcwd(), folder_name)
        
        if not os.path.exists(network_dir):
            print(f"❌ Network folder not found for: {location_name}")
            print(f"Tried:")
            print(f"  - {location_name}")
            print(f"  - {folder_name}")
            print(f"\nAvailable folders:")
            for item in os.listdir(os.getcwd()):
                if os.path.isdir(item) and not item.startswith('.') and not item.startswith('__'):
                    print(f"  - {item}")
            return False
    
    # Use the actual folder name for file operations
    actual_folder_name = os.path.basename(network_dir)
    
    # Load the network class
    pickle_dir = os.path.join(network_dir, 'pickle')
    
    # Try to find the network pickle file
    network_file = None
    if os.path.exists(pickle_dir):
        for f in os.listdir(pickle_dir):
            if f.endswith('.network.class.pkl'):
                network_file = os.path.join(pickle_dir, f)
                break
    
    if not network_file or not os.path.exists(network_file):
        print(f"❌ Network pickle file not found in: {pickle_dir}")
        return False
    
    print(f"📂 Loading network: {location_name}")
    with open(network_file, 'rb') as f:
        network = pickle.load(f)
    
    print(f"🏥 Retrieving hospitals from OpenStreetMap...")
    hospitals = retrieve_hospitals(network.G_walk, network.G_drive, location_name, network_dir, actual_folder_name)
    
    if len(hospitals) > 0:
        print(f"✅ Successfully retrieved {len(hospitals)} hospitals!")
        
        # Add to network object
        network.hospitals = hospitals
        
        # Save updated network
        print(f"💾 Saving updated network...")
        with open(network_file, 'wb') as f:
            pickle.dump(network, f, pickle.HIGHEST_PROTOCOL)
        
        print(f"✅ Done! Hospitals saved to:")
        print(f"   CSV: {network_dir}/csv/{actual_folder_name}.hospitals.csv")
        print(f"   Network class updated: {network_file}")
        
        # Show sample
        print(f"\n📋 Sample hospitals:")
        print(hospitals.head(5).to_string())
        
        return True
    else:
        print(f"⚠️ No hospitals found in {location_name}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python add_hospitals_to_existing_network.py \"Location Name\"")
        print("Example: python add_hospitals_to_existing_network.py \"Maastricht, Netherlands\"")
        sys.exit(1)
    
    location = sys.argv[1]
    success = add_hospitals_to_network(location)
    sys.exit(0 if success else 1)
