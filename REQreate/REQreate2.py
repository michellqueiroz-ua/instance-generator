from input_json import input_json
#from streamlit import caching
import os

if __name__ == '__main__':

    #caching.clear_cache()

    #EDIT HERE
    place_name = "Chicago, Illinois"
    inst_directory = '../examples/high_dimension_instances/nightlife2/'
    #inst_directory = '../examples/testes/'
    base_save_folder_name = 'nightlife2' #give a unique folder name to save the instances
    #EDIT HERE

    directory = os.fsencode(inst_directory)

    #input_json(inst_directory+'/'+'Chicago,Illinois_ODBRP_7.0_8.0_300_0.0_300.0_0.0_180_1000.json')
    
    for instance in os.listdir(directory):

        instance_filename = os.fsdecode(instance)
        instance_filename = instance_filename.replace("._", "")

        #print(os.listdir(os.path.join(place_name, 'json_format', base_save_folder_name)))

        final_list = []

        if os.path.isdir(place_name+'/json_format/'+base_save_folder_name):
            
            file_list = os.listdir(os.path.join(place_name, 'json_format', base_save_folder_name))

            
            for filex in file_list:
                filex = filex.replace('_1.json', "")
                filex = filex.replace('_2.json', "")
                filex = filex.replace('_3.json', "")
                filex = filex.replace('_4.json', "")
                filex = filex.replace('_5.json', "")
                filex = filex+'.json'

                final_list.append(filex)

        if instance_filename not in final_list:

            if (instance_filename.endswith(".json")):
                print(instance_filename)
                #print("here")
                try:
                    input_json(inst_directory, instance_filename, base_save_folder_name)  
                except FileNotFoundError:
                    
                    pass
        else:
            print('already FOUND')
    
