"""
Map visualization utilities for REQreate
"""
import folium
from folium import plugins
import pandas as pd
import streamlit as st


def create_request_map(requests_df, network_graph=None, bus_stations_df=None, hospitals_df=None, show_lines=True):
    """
    Create an interactive map showing request distribution
    
    Args:
        requests_df: DataFrame with columns ['origin_lat', 'origin_lon', 'dest_lat', 'dest_lon']
        network_graph: Optional OSMnx graph to show network
        bus_stations_df: Optional DataFrame with bus station locations
        hospitals_df: Optional DataFrame with hospital locations
        show_lines: Whether to draw lines between origins and destinations
    
    Returns:
        folium.Map object
    """
    
    # Calculate map center
    all_lats = list(requests_df['origin_lat']) + list(requests_df['dest_lat'])
    all_lons = list(requests_df['origin_lon']) + list(requests_df['dest_lon'])
    center_lat = sum(all_lats) / len(all_lats)
    center_lon = sum(all_lons) / len(all_lons)
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles='OpenStreetMap'
    )
    
    # Add layer control
    origin_layer = folium.FeatureGroup(name='Origins (Green)')
    dest_layer = folium.FeatureGroup(name='Destinations (Red)')
    line_layer = folium.FeatureGroup(name='Request Paths')
    
    # Add bus stations if provided (using simple CircleMarkers for faster loading)
    if bus_stations_df is not None and len(bus_stations_df) > 0:
        bus_layer = folium.FeatureGroup(name='Bus Stations', show=True)
        for idx, station in bus_stations_df.iterrows():
            folium.Marker(
                location=[station['lat'], station['lon']],
                icon=folium.DivIcon(html=f'''
                    <div style="font-size: 16px; color: blue; text-shadow: 1px 1px 2px white;">
                        <b>◼</b>
                    </div>
                '''),
                popup=f"Station {station.get('station_id', idx)}",
                tooltip="🚌 Bus Station"
            ).add_to(bus_layer)
        bus_layer.add_to(m)
    
    # Add hospitals if provided
    if hospitals_df is not None and len(hospitals_df) > 0:
        hospital_layer = folium.FeatureGroup(name='Hospitals', show=True)
        for idx, hospital in hospitals_df.iterrows():
            folium.Marker(
                location=[hospital['lat'], hospital['lon']],
                icon=folium.DivIcon(html=f'''
                    <div style="font-size: 20px; color: red; text-shadow: 1px 1px 2px white;">
                        <b>✚</b>
                    </div>
                '''),
                popup=folium.Popup(
                    f"<b>{hospital.get('hospital_name', 'Hospital')}</b><br>"
                    f"Type: {hospital.get('amenity_type', 'N/A')}",
                    max_width=200
                ),
                tooltip=f"🏥 {hospital.get('hospital_name', 'Hospital')}"
            ).add_to(hospital_layer)
        hospital_layer.add_to(m)
    
    # Add request origins and destinations
    for idx, row in requests_df.iterrows():
        # Origin marker (green)
        folium.CircleMarker(
            location=[row['origin_lat'], row['origin_lon']],
            radius=6,
            color='darkgreen',
            fill=True,
            fillColor='green',
            fillOpacity=0.7,
            popup=folium.Popup(
                f"<b>Request {idx}</b><br>"
                f"Origin<br>"
                f"Time: {row.get('earliest_start', row.get('pickup_from', 'N/A'))}",
                max_width=200
            ),
            tooltip=f"Origin - Request {idx}"
        ).add_to(origin_layer)
        
        # Destination marker (red)
        folium.CircleMarker(
            location=[row['dest_lat'], row['dest_lon']],
            radius=6,
            color='darkred',
            fill=True,
            fillColor='red',
            fillOpacity=0.7,
            popup=folium.Popup(
                f"<b>Request {idx}</b><br>"
                f"Destination<br>"
                f"Latest arrival: {row.get('latest_end', row.get('dropoff_to', 'N/A'))}",
                max_width=200
            ),
            tooltip=f"Destination - Request {idx}"
        ).add_to(dest_layer)
        
        # Draw line between origin and destination
        if show_lines:
            folium.PolyLine(
                locations=[
                    [row['origin_lat'], row['origin_lon']],
                    [row['dest_lat'], row['dest_lon']]
                ],
                color='gray',
                weight=2,
                opacity=0.4,
                popup=f"Request {idx}"
            ).add_to(line_layer)
    
    # Add layers to map
    origin_layer.add_to(m)
    dest_layer.add_to(m)
    if show_lines:
        line_layer.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add fullscreen button
    plugins.Fullscreen().add_to(m)
    
    # Add measure control
    plugins.MeasureControl(position='topleft').add_to(m)
    
    return m


def create_heatmap(requests_df, layer_type='origins'):
    """
    Create a heatmap of request density
    
    Args:
        requests_df: DataFrame with request locations
        layer_type: 'origins', 'destinations', or 'both'
    """
    
    center_lat = requests_df['origin_lat'].mean()
    center_lon = requests_df['origin_lon'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles='OpenStreetMap'
    )
    
    heat_data = []
    
    if layer_type in ['origins', 'both']:
        heat_data.extend(
            [[row['origin_lat'], row['origin_lon']] 
             for _, row in requests_df.iterrows()]
        )
    
    if layer_type in ['destinations', 'both']:
        heat_data.extend(
            [[row['dest_lat'], row['dest_lon']] 
             for _, row in requests_df.iterrows()]
        )
    
    plugins.HeatMap(heat_data, radius=15).add_to(m)
    
    return m
