from input_json import input_json
#from streamlit import caching
import os

if __name__ == '__main__':

    #caching.clear_cache()

    #EDIT HERE
    place_name = "Chicago, Illinois"
    inst_directory = '../examples/basic_examples/example_4' #directory to which the configuration files are saved
    base_save_folder_name = 'instancesEXAMPLES' #give a unique folder name to save the instances
    #EDIT HERE

    directory = os.fsencode(inst_directory)

    for instance in os.listdir(directory):

        instance_filename = os.fsdecode(instance)
        instance_filename = instance_filename.replace("._", "")

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
    
