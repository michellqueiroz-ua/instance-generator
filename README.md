# REQreate

REQreate is a tool to generate instances for on-demand transportation problems. Such problems consist of optimizing the routes of vehicles according to passengers' demand for transportation under space and time restrictions (requests). REQreate is flexible and can be configured to generate instances for a large number of problems in this problem class. For example, the Dial-a-Ride Problem (DARP) and On-demand Bus Routing Problem (ODBRP). The tool makes use of real-life networks from OpenStreetMaps to generate instances for an extensive catalogue of existing and upcoming on-demand transportation problems.

# How to use REQreate?

Attributes and parameters that define an instance are described in a configuration file given as input to REQreate. The used syntax is JSON.
Each configuration file can generate one or more instances and it should be given as parameter in the REQreate.py file.
Edit "REQreate.py" indicating the desired configuration file and a name for the output folder and simply run "python REQreate.py" on your terminal.
Check out [this folder](https://github.com/michellqueiroz-ua/instance-generator/tree/master/examples) for examples of configuration files.

# Installation requirements

1. Have python 3.8 or a newer version installed

2. Install anaconda
	[Tutorial 1](https://problemsolvingwithpython.com/01-Orientation/01.00-Welcome/)
	[Tutorial 2](https://docs.anaconda.com/anaconda/install/)

2. Download the conda environment file [here]().

3. Create the environment from the environment.yml file:

	conda env create -f REQreate_environment.yml

4. Everytime you use the tool, activate the environment

	conda activate reqreate


