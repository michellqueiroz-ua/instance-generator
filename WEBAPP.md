# REQreate Web Interface

## Quick Start

1. **Install REQreate** with the web interface (requires Python 3.11+):
```bash
pip install "reqreate[app]"
```

2. **Run the web app** from the folder where you want the instances saved:
```bash
reqreate app
```

3. **Open your browser** - it should automatically open at `http://localhost:8501`

## Features

- 🎨 **Visual Interface**: User-friendly forms instead of editing JSON files
- 📍 **Easy Configuration**: All parameters in one place with helpful descriptions
- 🔄 **Real-time Progress**: See generation progress as it happens
- 📊 **Instance Browser**: View and explore existing instances
- 📚 **Built-in Documentation**: Help and guides right in the app

## Usage

### Create New Instance
1. Enter your location (e.g., "Maastricht, Netherlands")
2. Adjust parameters or use defaults
3. Click "Generate Instance"
4. Monitor progress in real-time
5. Files are saved automatically

### View Existing Instances
- Browse all generated instances
- Preview CSV files
- See file statistics
- Download data

## Tips

- Start with default parameters for your first test
- Reduce number of requests (10-50) for faster generation
- Use "rank_model" distribution for more realistic patterns
- Check logs if generation fails

## Requirements

See `pyproject.toml` for the full list. Main dependencies:
- streamlit
- osmnx
- pandas
- networkx

Enjoy! 🚌
