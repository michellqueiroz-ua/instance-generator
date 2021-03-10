import os
from streamlit import caching
from instance_class import Instance
from output_files import JsonConverter

if __name__ == '__main__':

    caching.clear_cache()
    
    inst1 = Instance(folder_to_network='Rennes, France')
    inst1.set_problem_type(problem_type="ODBRP")
    inst1.add_request_demand_uniform(min_time=8, max_time=10, number_of_requests=100, time_unit="h")
    #inst1.add_request_demand_normal(mean=8, std=0.5, number_of_requests=100, time_unit="h")
    inst1.add_spatial_distribution(num_origins=1, num_destinations=1, prob=90, is_random_origin_zones=True, is_random_destination_zones=True)
    inst1.add_spatial_distribution(num_origins=-1, num_destinations=-1, prob=5)
    inst1.add_spatial_distribution(num_origins=-1, num_destinations=-1, prob=5)
    inst1.add_spatial_distribution(num_origins=-1, num_destinations=-1, prob=5)
    inst1.set_time_window(min_early_departure=5, max_early_departure=11, time_unit="h") #planning horizon
    inst1.set_interval_lead_time(min_lead_time=0, max_lead_time=5, time_unit="min")
    inst1.set_interval_walk_speed(min_walk_speed=5, max_walk_speed=5, speed_unit="kmh")
    inst1.set_interval_max_walking(lb_max_walking=300, ub_max_walking=600, time_unit="s")
    inst1.set_return_factor(return_factor=0.0)
    inst1.set_number_replicas(number_replicas=1)
    inst1.set_delay_vehicle_factor(delay_vehicle_factor=0.5)
    inst1.set_delay_walk_factor(delay_walk_factor=0.5)
    #inst1.add_vehicle_requirements("wheelchair")
    #inst1.add_vehicle_requirements("ambulatory")
    inst1.generate_requests()

    caching.clear_cache()
        
    # convert instances from json to cpp and localsolver formats
    save_dir_cpp = os.path.join(inst1.save_dir, 'cpp_format')
    if not os.path.isdir(save_dir_cpp):
        os.mkdir(save_dir_cpp)

    save_dir_localsolver = os.path.join(inst1.save_dir, 'localsolver_format')
    if not os.path.isdir(save_dir_localsolver):
        os.mkdir(save_dir_localsolver)

    for instance in os.listdir(os.path.join(inst1.save_dir, 'json_format')):
        
        if instance != ".DS_Store":
            input_name = os.path.join(inst1.save_dir, 'json_format', instance)
            
            output_name_cpp = instance.split('.')[0] + '_cpp.pass'
            output_name_cpp = output_name_cpp.replace(" ", "")
            
            output_name_ls = instance.split('.')[0] + '_ls.pass'

            converter = JsonConverter(file_name=input_name)
            converter.convert_normal(output_file_name=os.path.join(save_dir_cpp, output_name_cpp), network=inst1.network, problem_type=inst1.problem_type)
            #converter.convert_localsolver(output_file_name=os.path.join(save_dir_localsolver, output_name_ls))
             
    #print('placement of stops - testing')
    #cluster_travel_demand(param, network)

