# REQreate 🚌

REQreate is a tool to generate instances for on-demand transportation problems. Such problems consist of optimizing the routes of vehicles according to passengers' demand for transportation under space and time restrictions (requests). REQreate is flexible and can be configured to generate instances for a large number of problems in this problem class. For example, the Dial-a-Ride Problem (DARP) and On-demand Bus Routing Problem (ODBRP). The tool makes use of real-life networks from OpenStreetMaps to generate instances for an extensive catalogue of existing and upcoming on-demand transportation problems.

## ✨ New: Web Interface

REQreate now includes a user-friendly web interface built with Streamlit! No need to edit JSON files manually.

**Quick Start:**
```bash
pip install .[app]
reqreate app
```

Or simply double-click `run_webapp.bat` (Windows)

The web interface provides:
- 🎨 Visual forms for all parameters
- 📊 Browse and preview existing instances
- �️ Interactive maps with request distribution visualization
- 🔥 Heatmaps for demand density analysis
- �🔄 Real-time generation progress
- 📚 Built-in documentation
- 💾 Easy data exploration

See [WEBAPP.md](WEBAPP.md) for detailed instructions.

## 🌐 Deploy Online

Want to share REQreate with others? Deploy it to Hugging Face Spaces for free!

**Hugging Face Spaces** (Recommended):
- ✅ Free tier with 16GB RAM
- ✅ No timeout issues
- ✅ 50GB persistent storage
- ✅ Automatic HTTPS

**Quick Deploy:**
1. Create account at [huggingface.co](https://huggingface.co/join)
2. Create new Space with Streamlit SDK
3. Push your code to the Space repository
4. App automatically builds and deploys

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed deployment instructions.

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

1. **Install Python 3.9+** (if not already installed)

2. **Install REQreate** from a checkout of this repository:
```bash
pip install .[app]
```

3. **Run the tool:**
   - Web Interface: `reqreate app` — opens the interface in your browser, running entirely on your machine
   - Command Line: `reqreate generate my_config.json`
   - `reqreate --help` lists every option

Running the interface locally is the recommended way to use REQreate. Besides
not depending on a hosted deployment, OpenStreetMap downloads then leave from
your own IP address rather than one shared with every other user of a hosting
platform, which is what gets requests refused by the public Overpass API.

**Optional extras:**

| Extra | Installs | Needed for |
|---|---|---|
| `parallel` | ray | parallel processing (the code falls back to sequential without it) |
| `analysis` | scikit-learn, sqlalchemy | the taxi-dataset trip-pattern and `uber_movement` modules |

```bash
pip install .[app,parallel]
```

### Troubleshooting: Overpass API errors

If generation fails with a connection error mentioning `overpass-api.de`, the
public OpenStreetMap query servers are refusing or rate-limiting the request.
This is unrelated to the location you asked for. REQreate retries and fails
over across several mirrors automatically; if none of them work, set
`REQREATE_OVERPASS_URL` to an Overpass instance you can reach:

```bash
export REQREATE_OVERPASS_URL=https://overpass.kumi.systems/api/interpreter
```

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


