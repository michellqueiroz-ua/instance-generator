#examples to run the code

#case using "--get_fixed_lines 'deconet' place a folder named "deconet" inside the folder with the base file name of the city"

#before running the bash file, activate the osmnx envrinoment by running the command: "conda activate ox"

#python3 generator_lines_mq_with_osmnx.py --is_network_generation --base_file_name helsinki --place_name 'Helsinki, Finland'
#python3 generator_lines_mq_with_osmnx.py --is_request_generation --base_file_name helsinki --request_demand "normal" 8 h 30 min 1 --origin "random" -1 --destination "random" -1 "EDT" --request_demand "uniform" 8 h 10 h 350 --origin "random" -1 --destination "random" -1 "EDT" --time_window 7 10 h
#python3 generator_lines_mq_with_osmnx.py --is_network_generation --base_file_name rennes --place_name 'Rennes, France' --vehicle_speed_data "set" 30 kmh
#python3 generator_lines_mq_with_osmnx.py --is_network_generation --base_file_name rennes --place_name 'Rennes, France' --get_fixed_lines 'deconet'
python3 generator_lines_mq_with_osmnx.py --is_network_generation --place_name 'Rennes, France'
#python3 generator_lines_mq_with_osmnx.py  --is_request_generation --base_file_name rennes  --request_demand "normal" 8 h 30 min 100 --origin "random" -1 --destination "random" -1 "EDT" --time_window 7 11 h
