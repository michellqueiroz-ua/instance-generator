---
title: REQreate Instance Generator
emoji: 🚌
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# REQreate Instance Generator

Generate realistic on-demand transportation problem instances based on real-world network data from OpenStreetMap.

## Features

- 🗺️ **Real Network Data**: Downloads street networks, POIs, and transit stops from OpenStreetMap
- 🎯 **Multiple Problem Types**: DARP, ODBRP, and Patient Transport scenarios
- 📊 **Customizable Attributes**: Configure request attributes, time windows, and constraints
- 🏥 **Hospital Integration**: Patient Transport mode with hospital/clinic destinations
- 🗺️ **Interactive Maps**: Visualize generated instances with Folium maps
- 📦 **Multiple Formats**: Outputs CSV, JSON, GraphML, and pickle formats
- 💾 **Easy Download**: Get all generated files in a single ZIP

## Usage

1. **Select Problem Type**: Choose between DARP (general ride-sharing), ODBRP (bus-based), or Patient Transport
2. **Configure Attributes**: Select standard attributes or add custom ones
3. **Set Location**: Enter a city name (e.g., "Maastricht, Netherlands")
4. **Generate**: Click to start instance generation (takes 10-15 minutes)
5. **Download**: Get all files in ZIP format

## Problem Types

### DARP (Dial-a-Ride Problem)
General ride-sharing with origin/destination coordinates, suitable for flexible routing problems.

### ODBRP (On-Demand Bus Routing Problem)
Bus-based systems where pickup/drop-off occurs at designated bus stops.

### Patient Transport
Non-urgent medical transportation with:
- Random origin locations (patient homes)
- Hospital/clinic destinations from OpenStreetMap
- Flexible time windows with appointment constraints
- Extended service times for patient assistance

## Instance Components

Generated instances include:
- **Network data**: Street graph, nodes, edges
- **POI data**: Hospitals, schools, bus stations, zones
- **Request data**: Passenger requests with attributes
- **Travel time matrix**: Pre-computed distances
- **Visualizations**: Maps and heatmaps

## Citation

If you use this tool for research, please cite the REQreate framework:
[Citation information to be added]

## License

MIT License - see LICENSE file for details
