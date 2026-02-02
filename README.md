# REQreate 🚌

REQreate is a tool to generate instances for on-demand transportation problems. Such problems consist of optimizing the routes of vehicles according to passengers' demand for transportation under space and time restrictions (requests). REQreate is flexible and can be configured to generate instances for a large number of problems in this problem class. For example, the Dial-a-Ride Problem (DARP) and On-demand Bus Routing Problem (ODBRP). The tool makes use of real-life networks from OpenStreetMaps to generate instances for an extensive catalogue of existing and upcoming on-demand transportation problems.

## ✨ New: Web Interface

REQreate now includes a user-friendly web interface built with Streamlit! No need to edit JSON files manually.

**Quick Start:**
```bash
pip install streamlit
python -m streamlit run app.py
```

Or simply double-click `run_webapp.bat` (Windows)

The web interface provides:
- 🎨 Visual forms for all parameters
- 📊 Browse and preview existing instances
- 🔄 Real-time generation progress
- 📚 Built-in documentation
- 💾 Easy data exploration

See [WEBAPP.md](WEBAPP.md) for detailed instructions.

## How to use REQreate?

### Option 1: Web Interface (Recommended for new users)
Use the Streamlit web app for a guided, visual experience:
```bash
python -m streamlit run app.py
```

### Option 2: Command Line (Advanced users)
Attributes and parameters that define an instance are described in a configuration file given as input to REQreate. The used syntax is JSON.
Each configuration file can generate one or more instances and it should be given as parameter in the REQreate.py file.
Edit "REQreate.py" indicating the desired configuration file and a name for the output folder and simply run "python REQreate.py" on your terminal.
Check out [this folder](https://github.com/michellqueiroz-ua/instance-generator/tree/master/examples/basic_examples) for examples of configuration files.

# Questions? Do not hesitate to send an e-mail to:

michell.queiroz@uantwerpen.be

## Installation requirements

### Quick Install (Recommended)

1. **Install Python 3.8+** (if not already installed)

2. **Install required packages:**
```bash
pip install osmnx networkx pandas numpy shapely matplotlib streamlit
```

3. **Run the tool:**
   - Web Interface: `python -m streamlit run app.py`
   - Command Line: `python REQreate/REQreate.py`

### Detailed Installation

1. Have python 3.8 or a newer version installed

2. Install anaconda (optional but recommended)
	- [Tutorial 1 anaconda](https://problemsolvingwithpython.com/01-Orientation/01.00-Welcome/)
	- [Tutorial 2 anaconda](https://docs.anaconda.com/anaconda/install/)

3. Install OSMnx (necessary to retrieve the networks)
	- [Tutorial OSMnx](https://osmnx.readthedocs.io/en/stable/installation.html)

4. Install Streamlit (for web interface)
```bash
pip install streamlit
```

5. Activate your environment (if using conda):
```bash
conda activate ox
```

## Features

- 🌍 **Real-world networks** from OpenStreetMap
- 🚍 **Flexible configuration** for various on-demand transport problems
- 📊 **Multiple output formats** (CSV, JSON, GraphML, Pickle)
- 🎯 **Realistic request patterns** based on POI density
- ⏱️ **Accurate travel times** using actual road speeds
- 🖥️ **User-friendly web interface** for easy instance generation

## Output Files

REQreate generates comprehensive datasets including:
- Bus station locations
- Network topology (walk + drive)
- Passenger requests with time windows
- Travel time matrices
- Zone definitions
- Points of Interest (POIs)
- Network visualizations


