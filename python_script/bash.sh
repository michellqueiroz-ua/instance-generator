#python3 generator_lines_mq_with_osmnx.py --is_network_generation --base_file_name antwerp_inst --place_name 'Antwerp, Belgium' --get_fixed_lines 'osm'
#python3 generator_lines_mq_with_osmnx.py --is_network_generation --base_file_name berlin --place_name 'Berlin, Germany' --get_fixed_lines 'deconet'

#python3 generator_lines_mq_with_osmnx.py --is_network_generation --base_file_name rennes --place_name 'Rennes, France'
#python3 generator_lines_mq_with_osmnx.py --is_request_generation ---base_file_name rennes --request_demand "normal" 8 h 30 min 100 --origin "set" 4 0 1 2 3 --destination "set" 2 4 5 "EDT" --request_demand "uniform" 8 h 9 h 100 --origin "random" -1 --destination "random" -1 "EDT" --time_window 7 10 h
#python3 generator_lines_mq_with_osmnx.py --is_request_generation ---base_file_name rennes  --request_demand "normal" 8 h 30 min 100 --origin "set" 4 6 11 17 3 --destination "set" 2 6 7 "EDT" --request_demand "uniform" 8 h 9 h 100 --origin "random" 3 --destination "random" 1 "EDT" --time_window 7 10 h
#'Diadema, São Paulo
#Central Business District, Cincinnati, Ohio
#'Diadema, São Paulo'
#python3 generator_lines_mq_with_osmnx.py --is_network_generation --base_file_name toulouse --place_name 'Toulouse, France'
#python3 generator_lines_mq_with_osmnx.py --is_request_generation --base_file_name toulouse --request_demand "normal" 8 h 30 min 1 --origin "random" -1 --destination "random" -1 "EDT" --request_demand "uniform" 8 h 10 h 350 --origin "random" -1 --destination "random" -1 "EDT" --time_window 7 10 h
#ython3 generator_lines_mq_with_osmnx.py --is_network_generation --base_file_name helsinki --place_name 'Helsinki, Finland'
#python3 generator_lines_mq_with_osmnx.py --is_network_generation --base_file_name paris --place_name 'Paris, France' --get_fixed_lines 'deconet'
python3 generator_lines_mq_with_osmnx.py --is_network_generation --base_file_name paris --place_name 'Paris, France'
#python3 generator_lines_mq_with_osmnx.py --is_network_generation --base_file_name rennes --place_name 'Rennes, France' --vehicle_speed_data "set" 30 kmh
#python3 generator_lines_mq_with_osmnx.py --is_network_generation --base_file_name rennes --place_name 'Rennes, France' --get_fixed_lines 'deconet'
#python3 generator_lines_mq_with_osmnx.py  --is_request_generation --base_file_name rennes  --request_demand "normal" 8 h 30 min 100 --origin "random" -1 --destination "random" -1 "EDT" --request_demand "uniform" 8 h 9 h 300 --origin "random" -1 --destination "random" -1 "EDT" --time_window 7 10 h