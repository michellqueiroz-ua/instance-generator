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
        
        st.subheader("🚍 Vehicle & Network Settings")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            vehicle_capacity = st.number_input("Vehicle Capacity", min_value=1, value=6, help="Maximum passengers per vehicle")
            num_vehicles = st.number_input("Number of Vehicles", min_value=1, value=10)
            
        with col2:
            num_bus_stations = st.number_input("Number of Bus Stations", min_value=10, value=237)
            max_walking_distance = st.number_input("Max Walking Distance (m)", min_value=100, value=1000, step=100)
            
        with col3:
            vehicle_speed = st.number_input("Vehicle Speed (km/h)", min_value=5.0, value=20.0, step=5.0)
            walk_speed = st.number_input("Walking Speed (km/h)", min_value=1.0, value=5.0, step=0.5)
        
        st.subheader("👥 Request Settings")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            num_requests = st.number_input("Number of Requests", min_value=1, value=10, help="Total passenger requests to generate")
            replicate_num = st.number_input("Replicate Number", min_value=0, value=1, help="Seed for reproducibility")
            
        with col2:
            start_time = st.time_input("Start Time", value=pd.to_datetime("07:00").time())
            end_time = st.time_input("End Time", value=pd.to_datetime("08:00").time())
            
        with col3:
            time_horizon = st.number_input("Time Horizon (seconds)", min_value=0, value=14400, help="Total simulation time")
            dynamism = st.slider("Dynamism", min_value=0.0, max_value=1.0, value=0.0, step=0.1, 
                               help="0 = all requests known in advance, 1 = all requests arrive dynamically")
        
        st.subheader("⚙️ Advanced Settings")
        col1, col2 = st.columns(2)
        
        with col1:
            max_delay = st.number_input("Max Delay (seconds)", min_value=0, value=600)
            reaction_time = st.number_input("Reaction Time (seconds)", min_value=0, value=120)
            
        with col2:
            request_distribution = st.selectbox(
                "Request Distribution",
                ["uniform", "rank_model"],
                help="How to distribute requests across the network"
            )
            use_travel_time = st.checkbox("Use Travel Time Matrix", value=True, 
                                        help="Use precomputed travel times (recommended)")
        
        # Submit button
        submitted = st.form_submit_button("🚀 Generate Instance", use_container_width=True)
        
        if submitted:
            # Convert times to seconds
            start_seconds = start_time.hour * 3600 + start_time.minute * 60
            end_seconds = end_time.hour * 3600 + end_time.minute * 60
            
            # Create configuration JSON
            config = {
                "location_name": place_name,
                "output_folder": output_name if output_name else place_name,
                "vehicle_capacity": vehicle_capacity,
                "number_vehicles": num_vehicles,
                "num_bus_stations": num_bus_stations,
                "max_walking_distance": max_walking_distance,
                "vehicle_speed": vehicle_speed,
                "walk_speed": walk_speed,
                "num_requests": num_requests,
                "replicate_num": replicate_num,
                "start_time": start_seconds,
                "end_time": end_seconds,
                "time_horizon": time_horizon,
                "dynamism": dynamism,
                "max_delay": max_delay,
                "reaction_time": reaction_time,
                "request_distribution": request_distribution,
                "use_travel_time": use_travel_time
            }
            
            # Save temporary config file
            config_path = "temp_config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            st.success(f"✅ Configuration created for {place_name}")
            
            # Show configuration
            with st.expander("View Configuration"):
                st.json(config)
            
            # Run generation
            st.info("🔄 Starting instance generation... This may take several minutes.")
            
            progress_placeholder = st.empty()
            log_placeholder = st.empty()
            
            # Create a progress bar
            progress_bar = st.progress(0)
            
            try:
                # Run the generation script
                process = subprocess.Popen(
                    ["python", "REQreate/REQreate.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1
                )
                
                log_lines = []
                progress_value = 0
                
                # Read output in real-time
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        log_lines.append(output.strip())
                        log_placeholder.text_area("Generation Log", "\n".join(log_lines[-20:]), height=300)
                        
                        # Update progress based on keywords
                        if "Network Graphs" in output:
                            progress_value = 0.1
                        elif "Bus Stations" in output:
                            progress_value = 0.2
                        elif "Zones" in output:
                            progress_value = 0.3
                        elif "POIs" in output:
                            progress_value = 0.5
                        elif "Generating request" in output:
                            progress_value = 0.8
                        elif "leave ttm" in output:
                            progress_value = 1.0
                        
                        progress_bar.progress(progress_value)
                
                # Check exit code
                if process.returncode == 0:
                    st.success("✅ Instance generated successfully!")
                    st.balloons()
                    
                    # Show output location
                    output_folder = output_name if output_name else place_name
                    st.info(f"📁 Output saved to: `{output_folder}/`")
                    
                    # Show generated files
                    csv_folder = os.path.join(output_folder, "csv")
                    if os.path.exists(csv_folder):
                        csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]
                        if csv_files:
                            st.subheader("Generated Files")
                            for file in csv_files[:5]:  # Show first 5 files
                                st.text(f"✓ {file}")
                else:
                    st.error("❌ Generation failed. Check the log above for errors.")
                    stderr = process.stderr.read()
                    if stderr:
                        st.code(stderr, language="text")
                        
            except Exception as e:
                st.error(f"❌ Error running generation: {str(e)}")
            
            finally:
                # Clean up temp config
                if os.path.exists(config_path):
                    os.remove(config_path)

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
