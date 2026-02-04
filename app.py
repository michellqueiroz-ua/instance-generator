import streamlit as st
import json
import os
import subprocess
import pandas as pd
from pathlib import Path
import time
from streamlit_folium import st_folium
from map_utils import create_request_map, create_heatmap
import ast
import re
from attribute_library import get_attributes_for_problem, get_required_attributes, build_attributes_list

def prepare_requests_for_map(df):
    """
    Prepare requests dataframe for mapping by extracting/parsing coordinates.
    Handles different coordinate formats.
    """
    df = df.copy()
    
    # Check if coordinates are already in separate columns
    if all(col in df.columns for col in ['origin_lat', 'origin_lon', 'dest_lat', 'dest_lon']):
        return df
    
    # Parse origin coordinates
    if 'origin' in df.columns:
        def parse_coord(coord_str):
            if isinstance(coord_str, str):
                # Remove brackets and quotes, split by comma
                coord_str = coord_str.strip('"[]')
                parts = [float(x.strip()) for x in coord_str.split(',')]
                return parts[0], parts[1]  # lat, lon
            return None, None
        
        df[['origin_lat', 'origin_lon']] = df['origin'].apply(lambda x: pd.Series(parse_coord(x)))
    
    # Parse destination coordinates
    if 'destination' in df.columns:
        df[['dest_lat', 'dest_lon']] = df['destination'].apply(lambda x: pd.Series(parse_coord(x)))
    
    return df

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
    
    # Problem Type Selection - OUTSIDE FORM for immediate updates
    st.subheader("🎯 Problem Type")
    problem_type = st.selectbox(
        "Select Problem Type",
        ["DARP", "ODBRP", "Patient Transport"],
        help="DARP: Origin/Destination coordinates (general ride-sharing). ODBRP: Bus stops origin/destination (bus-based systems). Patient Transport: Non-urgent medical transportation with hospital destinations"
    )
    
    # Show info for Patient Transport
    if problem_type == "Patient Transport":
        st.info("""
        🏥 **Patient Transport Mode**
        
        This mode generates instances for non-urgent medical transportation:
        - **Origins**: Random locations (patient homes)
        - **Destinations**: Hospital/clinic locations from OpenStreetMap
        - **Time Windows**: Flexible pickup windows with appointment deadlines
        - **Ride Constraints**: Max ride time (150% of direct time), max daily ride time
        - **Service Time**: Longer service duration for patient assistance (3-5 minutes)
        
        ⚠️ Requires hospital data in network. Run: `python add_hospitals_to_existing_network.py "Location"`
        """)
    
    # Attribute Selection - OUTSIDE FORM for immediate updates
    st.subheader("📋 Request Attributes")
    st.markdown(f"**Configure attributes for {problem_type} problem**")
    
    # Get available attributes for selected problem type
    available_attrs = get_attributes_for_problem(problem_type)
    required_attrs = get_required_attributes(problem_type)
    
    # Display attributes in a more readable format
    st.markdown("**Standard Attributes:**")
    
    # Create columns for better layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Create multiselect with shorter labels
        attr_options = list(available_attrs.keys())
        
        # Pre-select required attributes
        default_selection = required_attrs.copy()
        
        selected_attrs = st.multiselect(
            "Select attributes to include:",
            options=attr_options,
            default=default_selection,
            help="Select which attributes to include in your instance. Required attributes are pre-selected.",
            key="selected_attrs"
        )
    
    with col2:
        st.markdown("**Legend:**")
        for attr_name in attr_options[:5]:  # Show first 5 as examples
            is_required = "⭐" if attr_name in required_attrs else "○"
            st.caption(f"{is_required} {attr_name}")
        if len(attr_options) > 5:
            st.caption(f"...and {len(attr_options)-5} more")
    
    # Show descriptions in expandable section
    with st.expander("ℹ️ View Attribute Descriptions", expanded=False):
        for attr_name in attr_options:
            attr_info = available_attrs[attr_name]
            required_badge = "⭐ **Required**" if attr_name in required_attrs else "○ Optional"
            st.markdown(f"**{attr_name}** - {required_badge}")
            st.caption(attr_info['description'])
            st.divider()
    
    # Validate that all required attributes are selected
    missing_required = [attr for attr in required_attrs if attr not in selected_attrs]
    if missing_required:
        st.error(f"❌ Missing required attributes: {', '.join(missing_required)}")
    
    # Custom Attributes Section
    st.subheader("➕ Custom Attributes (Optional)")
    
    # Initialize session state for custom attributes
    if 'custom_attrs' not in st.session_state:
        st.session_state.custom_attrs = []
    
    # Show existing custom attributes
    if st.session_state.custom_attrs:
        st.markdown("**Current Custom Attributes:**")
        for idx, attr in enumerate(st.session_state.custom_attrs):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.code(f"{attr['name']} ({attr['type']})", language="text")
            with col2:
                if st.button("❌", key=f"remove_{idx}", help="Remove this attribute"):
                    st.session_state.custom_attrs.pop(idx)
                    st.rerun()
    
    # Custom attribute builder
    with st.expander("➕ Add New Custom Attribute"):
        custom_name = st.text_input("Attribute Name", placeholder="e.g., patient_priority", key="custom_name")
        custom_type = st.selectbox("Type", ["integer", "real", "location", "array_primitives", "array_locations"], key="custom_type")
        custom_description = st.text_area("Description", placeholder="What does this attribute represent?", key="custom_desc")
        
        custom_template = {"name": custom_name, "type": custom_type}
        
        if custom_type in ["integer", "real"]:
            col1, col2 = st.columns(2)
            with col1:
                use_expression = st.checkbox("Use expression instead of PDF", help="Define value as formula", key="use_expr")
            with col2:
                if custom_type == "integer":
                    custom_unit = st.selectbox("Unit", ["", "time_unit", "length_unit"], key="custom_unit")
                    if custom_unit:
                        unit_value = st.selectbox("Value", ["s", "m", "km"], key="unit_val")
                        custom_template[custom_unit] = unit_value
                else:
                    custom_unit = st.selectbox("Unit", ["", "speed_unit"], key="custom_unit")
                    if custom_unit:
                        unit_value = st.selectbox("Value", ["mps", "kmh"], key="unit_val")
                        custom_template[custom_unit] = unit_value
            
            if use_expression:
                custom_expr = st.text_input("Expression", placeholder="e.g., drivingDuration * 2", key="custom_expr")
                if custom_expr:
                    custom_template["expression"] = custom_expr
            else:
                dist_type = st.selectbox("Distribution", ["uniform", "normal"], key="dist_type")
                col1, col2 = st.columns(2)
                with col1:
                    loc_val = st.number_input("Location (loc)", value=0.0, key="loc_val")
                with col2:
                    scale_val = st.number_input("Scale", value=1.0, key="scale_val")
                custom_template["pdf"] = [{"type": dist_type, "loc": loc_val, "scale": scale_val}]
        
        elif custom_type == "location":
            st.info("Location attributes are automatically generated from the network")
        
        elif custom_type == "array_primitives":
            custom_expr = st.text_input("Expression", placeholder="e.g., stops(origin)", key="custom_expr_arr")
            if custom_expr:
                custom_template["expression"] = custom_expr
        
        elif custom_type == "array_locations":
            n_locs = st.number_input("Number of locations (N)", min_value=1, value=2, key="n_locs")
            method = st.selectbox("Generation method", ["random", "clustered"], key="gen_method")
            custom_template["N"] = n_locs
            custom_template["method"] = method
        
        if st.button("➕ Add Attribute", type="primary", key="add_custom_btn"):
            if custom_name:
                st.session_state.custom_attrs.append({
                    "name": custom_name,
                    "type": custom_type,
                    "description": custom_description,
                    "template": custom_template
                })
                st.success(f"✅ Added: {custom_name}")
                st.rerun()
            else:
                st.error("Please enter an attribute name")
    
    st.markdown("---")
    
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
        
        st.caption("ℹ️ Note: Bus stations, schools, and hospitals are automatically retrieved from OpenStreetMap")
        
        st.subheader("👥 Request Settings")
        col1, col2 = st.columns(2)
        
        with col1:
            num_requests = st.number_input("Number of Requests", min_value=1, value=10, help="Total passenger requests to generate")
            replicate_num = st.number_input("Replicate Number", min_value=0, value=1, help="Generate multiple instances with same configuration (used as random seed)")
            
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
    
    # Process form submission
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
                # Build attributes list from selections
                attributes_list = build_attributes_list(problem_type, selected_attrs)
                
                # Add custom attributes from session state
                if 'custom_attrs' in st.session_state:
                    for custom_attr in st.session_state.custom_attrs:
                        attributes_list.append(custom_attr["template"])
                
                # Override time-related attributes with form values
                for attr in attributes_list:
                    if attr["name"] == "time_stamp":
                        attr["pdf"] = [{"type": "uniform", "loc": start_seconds, "scale": end_seconds - start_seconds}]
                        attr["dynamism"] = int(dynamism * 100)
                    elif attr["name"] == "reaction_time":
                        attr["pdf"] = [{"type": "uniform", "loc": reaction_time, "scale": 0}]
                    elif attr["name"] == "latest_arrival":
                        attr["expression"] = f"earliest_departure + direct_travel_time + {max_delay}"
                    elif attr["name"] == "max_walking":
                        attr["pdf"] = [{"type": "uniform", "loc": max_walking_distance / walk_speed, "scale": 0}]
                    elif attr["name"] == "walk_speed":
                        attr["pdf"] = [{"type": "uniform", "loc": walk_speed / 3.6, "scale": 0}]
                
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
                    "attributes": attributes_list,
                    "travel_time_matrix": ["bus_stations"] if use_travel_time else []
                }
            elif problem_type == "Patient Transport":
                # Build attributes list from selections
                attributes_list = build_attributes_list(problem_type, selected_attrs)
                
                # Add custom attributes from session state
                if 'custom_attrs' in st.session_state:
                    for custom_attr in st.session_state.custom_attrs:
                        attributes_list.append(custom_attr["template"])
                
                # Override time-related attributes with form values
                for attr in attributes_list:
                    if attr["name"] == "time_stamp":
                        attr["pdf"] = [{"type": "uniform", "loc": start_seconds, "scale": end_seconds - start_seconds}]
                        attr["dynamism"] = int(dynamism * 100)
                    elif attr["name"] == "pickup_to":
                        attr["expression"] = f"pickup_from + {reaction_time}"
                    elif attr["name"] == "dropoff_to":
                        attr["expression"] = f"dropoff_from + {max_delay}"
                    elif attr["name"] == "max_ride_time":
                        # Use 150% of direct time as specified
                        attr["expression"] = "drivingDuration * 1.5"
                
                # Patient Transport configuration (DARP variant with hospital destinations)
                config = {
                    "seed": replicate_num,
                    "network": place_name,
                    "problem": "DARP",  # Uses DARP solver but with hospital constraints
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
                        },
                        {
                            "name": "hospitals",
                            "type": "array_locations",
                            "locs": "hospitals"
                        }
                    ],
                    "attributes": attributes_list
                }
            else:  # DARP
                # Build attributes list from selections
                attributes_list = build_attributes_list(problem_type, selected_attrs)
                
                # Add custom attributes from session state
                if 'custom_attrs' in st.session_state:
                    for custom_attr in st.session_state.custom_attrs:
                        attributes_list.append(custom_attr["template"])
                
                # Override time-related attributes with form values
                for attr in attributes_list:
                    if attr["name"] == "time_stamp":
                        attr["pdf"] = [{"type": "uniform", "loc": start_seconds, "scale": end_seconds - start_seconds}]
                        attr["dynamism"] = int(dynamism * 100)
                    elif attr["name"] == "pickup_to":
                        attr["expression"] = f"pickup_from + {reaction_time}"
                    elif attr["name"] == "dropoff_to":
                        attr["expression"] = f"dropoff_from + {max_delay}"
                
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
                    "attributes": attributes_list
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
                
                # Run the generation - it saves to place_name folder automatically
                input_json("examples/webapp_instances/", config_filename, "")
                
                progress_bar.progress(1.0)
                st.success(f"✅ Instance generated successfully!")
                st.info(f"📁 Instance saved in: {place_name}/csv_format/")
                st.info(f"📄 File: {config_filename.replace('.json', '.csv')}")
                
            except Exception as e:
                st.error(f"❌ Generation failed: {str(e)}")
                st.exception(e)

elif page == "View Existing Instances":
    st.header("📂 Existing Instances")
    
    # Find all instance folders (location folders with csv_format)
    instance_folders = []
    
    # Check all directories in root
    for item in os.listdir("."):
        if os.path.isdir(item) and item not in ["REQreate", "cache", "environment", "examples", ".git", "__pycache__", "output"]:
            csv_format_folder = os.path.join(item, "csv_format")
            if os.path.exists(csv_format_folder):
                # Count CSV files that are instances (not subdirectories)
                csv_files = [f for f in os.listdir(csv_format_folder) if f.endswith('.csv') and os.path.isfile(os.path.join(csv_format_folder, f))]
                has_requests = False
                if csv_files:
                    # Check if any file has request data
                    try:
                        for f in csv_files:
                            sample = pd.read_csv(os.path.join(csv_format_folder, f), nrows=1)
                            if 'origin' in sample.columns or 'origin_lat' in sample.columns:
                                has_requests = True
                                break
                    except:
                        pass
                
                instance_folders.append({
                    'name': item,
                    'path': item,
                    'csv_count': len(csv_files),
                    'has_requests': has_requests
                })
    
    if not instance_folders:
        st.info("No instances found. Create your first instance using the 'Create New Instance' page!")
    else:
        # Show instance info
        st.markdown(f"Found **{len(instance_folders)}** instances")
        
        # Format instance display with additional info
        def format_instance(inst):
            status = "✅ Has requests" if inst['has_requests'] else "⚠️ Network only"
            return f"{inst['name']} - {inst['csv_count']} files ({status})"
        
        selected_idx = st.selectbox(
            "Select Instance",
            options=range(len(instance_folders)),
            format_func=lambda i: format_instance(instance_folders[i])
        )
        
        selected_instance_info = instance_folders[selected_idx]
        selected_instance = selected_instance_info['name']
        selected_instance_path = selected_instance_info['path']
        
        if selected_instance:
            st.subheader(f"Instance: {selected_instance}")
            
            # Create tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Preview", "🗺️ Map View", "🔥 Heatmap", "📁 Files"])
            
            csv_format_folder = os.path.join(selected_instance_path, "csv_format")
            
            with tab1:
                # Preview CSV files
                st.markdown("### 👀 Preview Data")
                
                # Check for network data files in csv folder
                csv_folder = os.path.join(selected_instance_path, "csv")
                network_files = {}
                if os.path.exists(csv_folder):
                    for filename in os.listdir(csv_folder):
                        if filename.endswith('.csv'):
                            if 'hospital' in filename.lower():
                                network_files['🏥 Hospitals'] = os.path.join(csv_folder, filename)
                            elif 'school' in filename.lower():
                                network_files['🏫 Schools'] = os.path.join(csv_folder, filename)
                            elif 'station' in filename.lower() or 'stops' in filename.lower():
                                network_files['🚏 Bus Stations'] = os.path.join(csv_folder, filename)
                            elif 'zone' in filename.lower():
                                network_files['📍 Zones'] = os.path.join(csv_folder, filename)
                            elif 'poi' in filename.lower():
                                network_files['📌 POIs'] = os.path.join(csv_folder, filename)
                
                # Show network data overview if available
                if network_files:
                    st.markdown("#### 🌐 Network Data")
                    cols = st.columns(len(network_files))
                    for idx, (name, filepath) in enumerate(network_files.items()):
                        with cols[idx]:
                            try:
                                df_net = pd.read_csv(filepath)
                                st.metric(name, len(df_net))
                                with st.expander(f"View {name}"):
                                    st.dataframe(df_net.head(10), use_container_width=True)
                            except:
                                st.metric(name, "Error")
                    st.markdown("---")
                
                # Only show CSV files, not subdirectories
                all_items = os.listdir(csv_format_folder)
                csv_files_to_preview = [f for f in all_items if f.endswith('.csv') and os.path.isfile(os.path.join(csv_format_folder, f))]
                
                if csv_files_to_preview:
                    selected_csv = st.selectbox("Select file to preview", csv_files_to_preview)
                    
                    if selected_csv:
                        try:
                            df = pd.read_csv(os.path.join(csv_format_folder, selected_csv))
                            st.dataframe(df.head(20), use_container_width=True)
                            
                            # Show basic stats
                            st.markdown("#### Statistics")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Rows", len(df))
                            with col2:
                                st.metric("Columns", len(df.columns))
                            with col3:
                                st.metric("Size", f"{os.path.getsize(os.path.join(csv_format_folder, selected_csv)) / 1024:.1f} KB")
                            
                            # Download button
                            csv_data = df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download CSV",
                                data=csv_data,
                                file_name=selected_csv,
                                mime="text/csv"
                            )
                            
                        except Exception as e:
                            st.error(f"Error loading file: {str(e)}")
            
            with tab2:
                st.markdown("### 🗺️ Request Distribution Map")
                
                # Look for requests file in csv_format folder only
                requests_files = []
                
                if os.path.exists(csv_format_folder):
                    all_files = [f for f in os.listdir(csv_format_folder) if f.endswith('.csv') and os.path.isfile(os.path.join(csv_format_folder, f))]
                    # Request files typically have many columns
                    for f in all_files:
                        try:
                            sample = pd.read_csv(os.path.join(csv_format_folder, f), nrows=1)
                            if 'origin' in sample.columns or 'origin_lat' in sample.columns:
                                requests_files.append(f)
                        except:
                            pass
                
                if requests_files:
                    # If multiple request files, let user choose
                    if len(requests_files) > 1:
                        selected_request_file = st.selectbox("Select request file:", requests_files)
                    else:
                        selected_request_file = requests_files[0]
                    
                    requests_file = os.path.join(csv_format_folder, selected_request_file)
                    
                    try:
                        requests_df = pd.read_csv(requests_file)
                        
                        # Parse coordinates if they're in string format
                        requests_df = prepare_requests_for_map(requests_df)
                        
                        # Check for required columns
                        required_cols = ['origin_lat', 'origin_lon', 'dest_lat', 'dest_lon']
                        if all(col in requests_df.columns for col in required_cols):
                            
                            # Map options
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                show_lines = st.checkbox("Show connection lines", value=True)
                            with col2:
                                show_stations = st.checkbox("Show bus stations", value=False)
                            with col3:
                                show_hospitals = st.checkbox("Show hospitals", value=False)
                            
                            # Load bus stations if available (from network data in csv folder)
                            bus_stations_df = None
                            hospitals_df = None
                            
                            if show_stations:
                                network_csv_folder = os.path.join(selected_instance_path, "csv")
                                if os.path.exists(network_csv_folder):
                                    all_csv_files = os.listdir(network_csv_folder)
                                    bus_files = [f for f in all_csv_files if 'station' in f.lower() and f.endswith('.csv')]
                                    if bus_files:
                                        try:
                                            bus_stations_df = pd.read_csv(os.path.join(network_csv_folder, bus_files[0]))
                                        except Exception as e:
                                            st.warning(f"Could not load bus stations: {e}")
                            
                            # Load hospitals if available
                            if show_hospitals:
                                network_csv_folder = os.path.join(selected_instance_path, "csv")
                                if os.path.exists(network_csv_folder):
                                    all_csv_files = os.listdir(network_csv_folder)
                                    hospital_files = [f for f in all_csv_files if 'hospital' in f.lower() and f.endswith('.csv')]
                                    if hospital_files:
                                        try:
                                            hospitals_df = pd.read_csv(os.path.join(network_csv_folder, hospital_files[0]))
                                        except Exception as e:
                                            st.warning(f"Could not load hospitals: {e}")
                            
                            # Create and display map
                            with st.spinner("Loading map..."):
                                map_obj = create_request_map(
                                    requests_df, 
                                    bus_stations_df=bus_stations_df,
                                    hospitals_df=hospitals_df,
                                    show_lines=show_lines
                                )
                                st_folium(map_obj, width=800, height=600)
                            
                            # Show statistics
                            st.markdown("### 📊 Request Statistics")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Requests", len(requests_df))
                            with col2:
                                if 'earliest_start' in requests_df.columns:
                                    st.metric("Earliest Request", f"{requests_df['earliest_start'].min()}s")
                                elif 'pickup_from' in requests_df.columns:
                                    st.metric("Earliest Pickup", f"{requests_df['pickup_from'].min()}s")
                            with col3:
                                if 'latest_end' in requests_df.columns:
                                    st.metric("Latest Request", f"{requests_df['latest_end'].max()}s")
                                elif 'dropoff_to' in requests_df.columns:
                                    st.metric("Latest Dropoff", f"{requests_df['dropoff_to'].max()}s")
                        else:
                            st.error(f"Requests file missing required columns: {required_cols}")
                            st.info(f"Available columns: {list(requests_df.columns)}")
                            
                    except Exception as e:
                        st.error(f"Error loading map: {str(e)}")
                        with st.expander("Show error details"):
                            st.exception(e)
                else:
                    st.warning("⚠️ No requests file found in this instance.")
                    st.info("""
                    **Why is this happening?**
                    
                    This instance appears to contain only network data (roads, bus stations, POIs) but no passenger requests yet.
                    
                    **To generate requests:**
                    1. Go to "Create New Instance" page
                    2. Select this location: `{}`
                    3. Configure the number of requests and other parameters
                    4. Generate a new instance with requests
                    
                    **Available files in csv_format folder:**
                    """.format(selected_instance))
                    if os.path.exists(csv_format_folder):
                        for f in sorted(os.listdir(csv_format_folder)):
                            if os.path.isfile(os.path.join(csv_format_folder, f)):
                                st.text(f"  • {f}")
            
            with tab3:
                st.markdown("### 🔥 Request Density Heatmap")
                
                # Look for requests file in csv_format folder only
                requests_files = []
                
                if os.path.exists(csv_format_folder):
                    all_files = [f for f in os.listdir(csv_format_folder) if f.endswith('.csv') and os.path.isfile(os.path.join(csv_format_folder, f))]
                    # Request files typically have many columns
                    for f in all_files:
                        try:
                            sample = pd.read_csv(os.path.join(csv_format_folder, f), nrows=1)
                            if 'origin' in sample.columns or 'origin_lat' in sample.columns:
                                requests_files.append(f)
                        except:
                            pass
                
                if requests_files:
                    # If multiple request files, let user choose
                    if len(requests_files) > 1:
                        selected_request_file = st.selectbox("Select request file for heatmap:", requests_files, key="heatmap_file")
                    else:
                        selected_request_file = requests_files[0]
                    
                    try:
                        requests_df = pd.read_csv(os.path.join(csv_format_folder, selected_request_file))
                        
                        # Parse coordinates if they're in string format
                        requests_df = prepare_requests_for_map(requests_df)
                        
                        required_cols = ['origin_lat', 'origin_lon', 'dest_lat', 'dest_lon']
                        if all(col in requests_df.columns for col in required_cols):
                            
                            heatmap_type = st.radio(
                                "Show heatmap for:",
                                options=['origins', 'destinations', 'both'],
                                horizontal=True
                            )
                            
                            with st.spinner("Generating heatmap..."):
                                heatmap_obj = create_heatmap(requests_df, layer_type=heatmap_type)
                                st_folium(heatmap_obj, width=800, height=600)
                        else:
                            st.error(f"Requests file missing required columns: {required_cols}")
                            st.info(f"Available columns: {list(requests_df.columns)}")
                                
                    except Exception as e:
                        st.error(f"Error creating heatmap: {str(e)}")
                        with st.expander("Show error details"):
                            st.exception(e)
                else:
                    st.warning("⚠️ No requests file found in this instance.")
                    st.info("""
                    Heatmaps require passenger request data. This instance only contains network data.
                    
                    Generate a new instance with requests using the "Create New Instance" page.
                    """)
            
            with tab4:
                # Show folder structure
                st.markdown("### 📁 Available Files")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📊 Instance Files (csv_format)")
                    if os.path.exists(csv_format_folder):
                        csv_files = sorted([f for f in os.listdir(csv_format_folder) if f.endswith('.csv') and os.path.isfile(os.path.join(csv_format_folder, f))])
                        for file in csv_files:
                            file_path = os.path.join(csv_format_folder, file)
                            file_size = os.path.getsize(file_path) / 1024  # KB
                            st.text(f"📄 {file} ({file_size:.1f} KB)")
                
                with col2:
                    st.markdown("#### 🗂️ Other Files")
                    for subfolder in ["pickle", "json_format", "graphml_format", "images", "csv"]:
                        folder_path = os.path.join(selected_instance_path, subfolder)
                        if os.path.exists(folder_path):
                            file_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
                            if file_count > 0:
                                st.text(f"📁 {subfolder}/ ({file_count} files)")

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
