# REQreate Web Interface

For installation, startup instructions, output locations, and expected generation
times, see the [web-interface quick start](index.md#quick-start).

## Interface features

- 🎨 **Visual Interface**: User-friendly forms instead of editing JSON files
- 📍 **Easy Configuration**: All parameters in one place with helpful descriptions
- 🔄 **Real-time Progress**: See generation progress as it happens
- 📊 **Instance Browser**: View and explore existing instances
- 📚 **Built-in Documentation**: Help and guides right in the app

## Usage

### Create New Instance

1. Open **Networks** and download a network by place name or by providing a point and radius. Optionally include public transport fixed lines.
2. Open **Create New Instance** and select a network that has already been built.
3. Set vehicle speed and other instance parameters, then click **Generate Instance**.
4. Monitor progress; generated files are saved automatically.

Network downloads and instance generation are separate operations. The instance form only lists networks already present in the working directory.
Networks can also be built from a terminal with `reqreate network "Maastricht, Netherlands"`.

### Networks

- Browse built networks with their last-modified date and folder size.
- Download a network by place name or around a longitude/latitude point and radius.
- Bus stop locations are included with the built network.

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
