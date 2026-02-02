import streamlit as st
import json
import os
import subprocess
import pandas as pd
from pathlib import Path
import time

st.set_page_config(
    page_title="REQreate - Instance Generator",
    page_icon="🚌",
    layout="wide"
)

# Title and description
st.title("🚌 REQreate Instance Generator")
st.markdown("""
Generate realistic on-demand transportation instances based on real-world network data.
This tool downloads OpenStreetMap data and creates synthetic passenger requests for your specified location.
""")

# Sidebar for navigation
page = st.sidebar.radio("Navigation", ["Create New Instance", "View Existing Instances", "Documentation"])

if page == "Create New Instance":
    st.header("Create New Instance")
    
    with st.form("instance_form"):
        st.subheader("📍 Location Settings")
        col1, col2 = st.columns(2)
        
        with col1:
            place_name = st.text_input(
                "Location (City, Country)",
                value="Maastricht, Netherlands",
                help="Enter the location in the format: City, Country"
            )
            
        with col2:
            output_name = st.text_input(
                "Output Folder Name",
                value=place_name if place_name else "instance",
                help="Name for the output folder (defaults to location name)"
            )
        
        st.subheader("🚍 Network Settings")
        col1, col2 = st.columns(2)
        
        with col1:
            vehicle_speed = st.number_input("Vehicle Speed (km/h)", min_value=5.0, value=20.0, step=5.0, help="Average vehicle speed for travel time calculations")
            max_walking_distance = st.number_input("Max Walking Distance (m)", min_value=100, value=1000, step=100, help="Maximum distance passengers can walk to/from bus stops")
            
        with col2:
            walk_speed = st.number_input("Walking Speed (km/h)", min_value=1.0, value=5.0, step=0.5, help="Average walking speed for passengers")
        
        st.caption("ℹ️ Note: Number of bus stations is automatically retrieved from OpenStreetMap")
        
        st.subheader("👥 Request Settings")
        col1, col2 = st.columns(2)
        
        with col1:
            num_requests = st.number_input("Number of Requests", min_value=1, value=10, help="Total passenger requests to generate")
            replicate_num = st.number_input("Replicate Number", min_value=0, value=1, help="Generate multiple instances with same configuration (used as random seed)")
            problem_type = st.selectbox(
                "Problem Type / Request Format",
                ["DARP", "ODBRP"],
                help="DARP: Origin/Destination coordinates (general ride-sharing). ODBRP: Bus stops origin/destination (bus-based systems)"
            )
            
        with col2:
            st.markdown("**Request Arrival Time Horizon**")
            start_time = st.time_input("Start Time", value=pd.to_datetime("07:00").time(), help="Earliest request arrival time")
            end_time = st.time_input("End Time", value=pd.to_datetime("08:00").time(), help="Latest request arrival time")
        
        st.subheader("⚙️ Advanced Settings")
        col1, col2 = st.columns(2)
        
        with col1:
            dynamism = st.slider("Dynamism", min_value=0.0, max_value=1.0, value=0.0, step=0.1, 
                               help="Fraction of requests that arrive dynamically during operation. 0 = all requests known in advance (static), 1 = all requests arrive in real-time (dynamic)")
            max_delay = st.number_input("Max Delay (seconds)", min_value=0, value=600,
                                       help="Maximum ride time increase allowed (compared to direct trip). Represents passenger tolerance for detours")
            
        with col2:
            reaction_time = st.number_input("Reaction Time / Urgency (seconds)", min_value=0, value=120,
                                           help="Time between request arrival and desired pickup. Lower values = more urgent requests")
            request_distribution = st.selectbox(
                "Request Distribution",
                ["uniform", "rank_model"],
                help="How to distribute requests spatially: uniform = random across network, rank_model = based on POI density"
            )
        
        use_travel_time = st.checkbox("Use Travel Time Matrix", value=True, 
                                    help="Use precomputed travel times (recommended for accuracy)")
        
        # Submit button
        submitted = st.form_submit_button("🚀 Generate Instance", use_container_width=True)
        
        if submitted:
            # Convert times to seconds and hours
            start_seconds = start_time.hour * 3600 + start_time.minute * 60
            end_seconds = end_time.hour * 3600 + end_time.minute * 60
            start_hours = start_time.hour + start_time.minute / 60.0
            end_hours = end_time.hour + end_time.minute / 60.0
            
            # Create output folder name (clean version for filenames)
            folder_name = output_name if output_name else place_name.replace(", ", "_").replace(" ", "_")
            folder_name_clean = folder_name.replace(",", "").replace(" ", "_")
            
            # Build configuration JSON based on problem type
            if problem_type == "ODBRP":
                # ODBRP configuration with bus stops
                config = {
                    "seed": replicate_num,
                    "network": place_name,
                    "problem": "ODBRP",
                    "set_fixed_speed": {
                        "vehicle_speed_data": vehicle_speed,
                        "vehicle_speed_data_unit": "kmh"
                    },
                    "replicas": 1,
                    "requests": num_requests,
                    "instance_filename": ["network", "problem", "requests", "min_early_departure", "max_early_departure"],
                    "parameters": [
                        {
                            "name": "min_early_departure",
                            "type": "float",
                            "value": start_hours,
                            "time_unit": "h"
                        },
                        {
                            "name": "max_early_departure",
                            "type": "float",
                            "value": end_hours,
                            "time_unit": "h"
                        },
                        {
                            "name": "graphml",
                            "type": "graphml",
                            "value": True
                        }
                    ],
                    "attributes": [
                        {
                            "name": "time_stamp",
                            "type": "integer",
                            "time_unit": "s",
                            "pdf": [{
                                "type": "uniform",
                                "loc": start_seconds,
                                "scale": end_seconds - start_seconds
                            }],
                            "constraints": [
                                "time_stamp >= 0",
                                "time_stamp >= min_early_departure",
                                "time_stamp <= max_early_departure"
                            ],
                            "dynamism": int(dynamism * 100)
                        },
                        {
                            "name": "reaction_time",
                            "type": "integer",
                            "time_unit": "s",
                            "pdf": [{
                                "type": "uniform",
                                "loc": reaction_time,
                                "scale": 0
                            }]
                        },
                        {
                            "name": "earliest_departure",
                            "type": "integer",
                            "time_unit": "s",
                            "expression": "time_stamp",
                            "constraints": [
                                "earliest_departure >= 0",
                                "earliest_departure >= min_early_departure"
                            ]
                        },
                        {
                            "name": "latest_departure",
                            "type": "integer",
                            "time_unit": "s",
                            "expression": "time_stamp + reaction_time"
                        },
                        {
                            "name": "latest_arrival",
                            "type": "integer",
                            "time_unit": "s",
                            "expression": f"earliest_departure + direct_travel_time + {max_delay}"
                        },
                        {
                            "name": "origin",
                            "type": "location"
                        },
                        {
                            "name": "destination",
                            "type": "location"
                        },
                        {
                            "name": "stops_orgn",
                            "type": "array_primitives",
                            "expression": "stops(origin)",
                            "constraints": ["len(stops_orgn) > 0"]
                        },
                        {
                            "name": "stops_dest",
                            "type": "array_primitives",
                            "expression": "stops(destination)",
                            "constraints": ["len(stops_dest) > 0", "not (set(stops_orgn) & set(stops_dest))"]
                        },
                        {
                            "name": "max_walking",
                            "type": "integer",
                            "time_unit": "s",
                            "pdf": [{
                                "type": "uniform",
                                "loc": max_walking_distance / walk_speed,
                                "scale": 0
                            }],
                            "output_csv": False
                        },
                        {
                            "name": "walk_speed",
                            "type": "real",
                            "speed_unit": "mps",
                            "pdf": [{
                                "type": "uniform",
                                "loc": walk_speed / 3.6,
                                "scale": 0
                            }],
                            "output_csv": False
                        },
                        {
                            "name": "direct_travel_time",
                            "type": "integer",
                            "time_unit": "s",
                            "expression": "dtt(origin,destination)"
                        },
                        {
                            "name": "direct_distance",
                            "type": "integer",
                            "length_unit": "m",
                            "expression": "dist_drive(origin,destination)"
                        }
                    ],
                    "travel_time_matrix": ["bus_stations"] if use_travel_time else []
                }
            else:  # DARP
                # DARP configuration with coordinates
                config = {
                    "seed": replicate_num,
                    "network": place_name,
                    "problem": "DARP",
                    "set_fixed_speed": {
                        "vehicle_speed_data": vehicle_speed,
                        "vehicle_speed_data_unit": "kmh"
                    },
                    "replicas": 1,
                    "requests": num_requests,
                    "instance_filename": ["network", "problem", "requests", "min_early_departure", "max_early_departure"],
                    "parameters": [
                        {
                            "name": "min_early_departure",
                            "type": "float",
                            "value": start_hours,
                            "time_unit": "h"
                        },
                        {
                            "name": "max_early_departure",
                            "type": "float",
                            "value": end_hours,
                            "time_unit": "h"
                        },
                        {
                            "name": "graphml",
                            "type": "graphml",
                            "value": True
                        }
                    ],
                    "attributes": [
                        {
                            "name": "time_stamp",
                            "type": "integer",
                            "time_unit": "s",
                            "pdf": [{
                                "type": "uniform",
                                "loc": start_seconds,
                                "scale": end_seconds - start_seconds
                            }],
                            "constraints": [
                                "time_stamp >= 0",
                                "time_stamp >= min_early_departure",
                                "time_stamp <= max_early_departure"
                            ],
                            "dynamism": int(dynamism * 100)
                        },
                        {
                            "name": "pickup_from",
                            "type": "integer",
                            "time_unit": "s",
                            "expression": "time_stamp"
                        },
                        {
                            "name": "pickup_to",
                            "type": "integer",
                            "time_unit": "s",
                            "expression": f"pickup_from + {reaction_time}"
                        },
                        {
                            "name": "dropoff_from",
                            "type": "integer",
                            "time_unit": "s",
                            "expression": "pickup_to + drivingDuration"
                        },
                        {
                            "name": "dropoff_to",
                            "type": "integer",
                            "time_unit": "s",
                            "expression": f"dropoff_from + {max_delay}"
                        },
                        {
                            "name": "origin",
                            "type": "location"
                        },
                        {
                            "name": "destination",
                            "type": "location"
                        },
                        {
                            "name": "drivingDuration",
                            "type": "integer",
                            "time_unit": "s",
                            "expression": "dtt(origin,destination)"
                        },
                        {
                            "name": "drivingDistance",
                            "type": "integer",
                            "length_unit": "m",
                            "expression": "dist_drive(origin,destination)"
                        }
                    ]
                }
            
            # Create configuration directory if it doesn't exist
            config_dir = "examples/webapp_instances/"
            os.makedirs(config_dir, exist_ok=True)
            
            # Save configuration file with descriptive name (no spaces or commas)
            config_filename = f"{folder_name_clean}_{problem_type}_{num_requests}req.json"
            config_path = os.path.join(config_dir, config_filename)
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            st.success(f"✅ Configuration created: {config_filename}")
            
            # Show configuration
            with st.expander("View Configuration JSON"):
                st.json(config)
            
            # Run generation
            st.info("🔄 Starting instance generation... This may take several minutes.")
            
            progress_placeholder = st.empty()
            log_placeholder = st.empty()
            
            # Create a progress bar
            progress_bar = st.progress(0)
            
            try:
                # Import and run input_json directly with our configuration
                import sys
                sys.path.insert(0, 'REQreate')
                from input_json import input_json
                
                # Run the generation (input_json expects directory with trailing slash)
                base_save_folder = f"instances_{folder_name_clean}"
                input_json("examples/webapp_instances/", config_filename, base_save_folder)
                
                progress_bar.progress(1.0)
                st.success(f"✅ Instance generated successfully!")
                st.info(f"📁 Instance saved in: {place_name}/")
                
            except Exception as e:
                st.error(f"❌ Generation failed: {str(e)}")
                st.exception(e)

elif page == "View Existing Instances":
    st.header("📂 Existing Instances")
    
    # Find all instance folders
    instance_folders = []
    for item in os.listdir("."):
        if os.path.isdir(item) and item not in ["REQreate", "cache", "environment", "examples", ".git", "__pycache__"]:
            csv_folder = os.path.join(item, "csv")
            if os.path.exists(csv_folder):
                instance_folders.append(item)
    
    if not instance_folders:
        st.info("No instances found. Create your first instance using the 'Create New Instance' page!")
    else:
        selected_instance = st.selectbox("Select Instance", instance_folders)
        
        if selected_instance:
            st.subheader(f"Instance: {selected_instance}")
            
            # Show folder structure
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 CSV Files")
                csv_folder = os.path.join(selected_instance, "csv")
                if os.path.exists(csv_folder):
                    csv_files = sorted([f for f in os.listdir(csv_folder) if f.endswith('.csv')])
                    for file in csv_files:
                        file_path = os.path.join(csv_folder, file)
                        file_size = os.path.getsize(file_path) / 1024  # KB
                        st.text(f"📄 {file} ({file_size:.1f} KB)")
            
            with col2:
                st.markdown("### 🗂️ Other Files")
                for subfolder in ["pickle", "json_format", "graphml_format"]:
                    folder_path = os.path.join(selected_instance, subfolder)
                    if os.path.exists(folder_path):
                        file_count = len([f for f in os.listdir(folder_path)])
                        st.text(f"📁 {subfolder}/ ({file_count} files)")
            
            # Preview CSV files
            st.markdown("### 👀 Preview Data")
            csv_files_to_preview = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]
            
            if csv_files_to_preview:
                selected_csv = st.selectbox("Select file to preview", csv_files_to_preview)
                
                if selected_csv:
                    try:
                        df = pd.read_csv(os.path.join(csv_folder, selected_csv))
                        st.dataframe(df.head(20), use_container_width=True)
                        
                        # Show basic stats
                        st.markdown("#### Statistics")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Rows", len(df))
                        with col2:
                            st.metric("Columns", len(df.columns))
                        with col3:
                            st.metric("Size", f"{os.path.getsize(os.path.join(csv_folder, selected_csv)) / 1024:.1f} KB")
                        
                    except Exception as e:
                        st.error(f"Error loading file: {str(e)}")

elif page == "Documentation":
    st.header("📚 Documentation")
    
    st.markdown("""
    ## REQreate Instance Generator
    
    This tool generates realistic on-demand transportation instances based on real-world network data from OpenStreetMap.
    
    ### Quick Start
    
    1. **Go to "Create New Instance"**
    2. **Enter your location** (e.g., "Maastricht, Netherlands")
    3. **Configure parameters** (or use defaults)
    4. **Click "Generate Instance"**
    5. **Wait for completion** (typically 5-15 minutes)
    
    ### Key Parameters
    
    - **Location**: City and country name (as it appears in OpenStreetMap)
    - **Number of Requests**: How many passenger requests to generate
    - **Time Window**: When requests should occur
    - **Dynamism**: 0 = all requests known in advance, 1 = all arrive during operation
    - **Request Distribution**: 
        - `uniform`: Random distribution across network
        - `rank_model`: Based on POI density and trip patterns
    
    ### Generated Files
    
    The tool creates several output files:
    
    - **CSV files**: Bus stations, zones, requests, travel time matrix
    - **Network files**: GraphML format for network visualization
    - **Pickle files**: Python objects for further processing
    - **Images**: Network visualizations
    
    ### What Gets Downloaded?
    
    - Road network (walk + drive)
    - Bus stations (transit stops)
    - Schools
    - Points of Interest (POIs): shops, restaurants, offices, etc.
    
    ### Troubleshooting
    
    - **Generation takes too long**: Reduce number of requests or bus stations
    - **Out of memory**: Use smaller area or fewer zones
    - **Network not found**: Check location name spelling (must match OpenStreetMap)
    
    ### Requirements
    
    Install required packages:
    ```bash
    pip install streamlit osmnx networkx pandas numpy shapely
    ```
    
    ### Running the App
    
    ```bash
    streamlit run app.py
    ```
    
    ### Citation
    
    If you use this tool in research, please cite the REQreate framework.
    """)
    
    st.markdown("---")
    st.markdown("Made with ❤️ using [Streamlit](https://streamlit.io)")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**REQreate Instance Generator**

Generate realistic on-demand transportation instances from real-world data.

Version 1.0
""")
