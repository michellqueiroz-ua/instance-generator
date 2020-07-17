#python3 generator_lines_mq_with_osmnx.py --is_network_generation --base_file_name antwerp_inst --place_name 'Antwerp, Belgium' --get_fixed_lines 'osm'
#python3 generator_lines_mq_with_osmnx.py --is_network_generation --base_file_name berlin --place_name 'Berlin, Germany' --get_fixed_lines 'deconet'
python3 generator_lines_mq_with_osmnx.py --is_network_generation --base_file_name rennes --place_name 'Rennes, France' --get_fixed_lines 'deconet'
#python3 generator_lines_mq_with_osmnx.py --is_request_generation --network_class_file 'cohio_inst/pickle/cohio_inst.network.class.pkl' --request_demand "normal" 8 h 30 min 100 --origin "set" 4 0 1 2 3 --destination "set" 2 4 5 "EDT" --request_demand "uniform" 8 h 9 h 100 --origin "random" -1 --destination "random" -1 "EDT" --time_window 7 10 h
#python3 generator_lines_mq_with_osmnx.py --is_request_generation --network_class_file 'antwerp_inst.network.class.pkl' --request_demand "normal" 8 h 30 min 100 --origin "set" 4 6 11 17 3 --destination "set" 2 6 7 "EDT" --request_demand "uniform" 8 h 9 h 100 --origin "random" 3 --destination "random" 1 "EDT" --time_window 7 10 h
#'Diadema, São Paulo
#Central Business District, Cincinnati, Ohio
#'Diadema, São Paulo'