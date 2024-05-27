import os
from instance_class import Instance

if __name__ == '__main__':

    place_name = "Chicago, Illinois"
    base_save_folder_name = 'uneven_demand' #give a unique folder name to save the instances
    inst_directory = '../examples/uneven_demand/'
    directory = os.fsencode(inst_directory)
    inst = Instance(folder_to_network=place_name)

    save_dir_csv = os.path.join(inst.save_dir, 'csv_format')
    if not os.path.isdir(save_dir_csv):
        os.mkdir(save_dir_csv)

    '''
    for instance in os.listdir(directory):

        instance_filename = os.fsdecode(instance)
        instance_filename = instance_filename.replace("._", "")
    '''

    replicate_num = 1
    for instance in os.listdir(os.path.join(inst.save_dir, 'json_format', base_save_folder_name)):

        '''
        filename_json = inst_directory+instance_filename
        base_filename = filename_json.replace(inst_directory, "")
        inst_base = instance.replace(inst.save_dir+'/json_format/'+base_save_folder_name+'/', "")
        inst_base = inst_base.replace('_1.json', "")
        inst_base = inst_base.replace('_2.json', "")
        inst_base = inst_base.replace('_3.json', "")
        inst_base = inst_base.replace('_4.json', "")
        inst_base = inst_base.replace('_5.json', "")
        inst_base = inst_base.replace('_6.json', "")
        inst_base = inst_base.replace('_7.json', "")
        inst_base = inst_base.replace('_8.json', "")
        inst_base = inst_base.replace('_9.json', "")
        inst_base = inst_base.replace('_10.json', "")
        inst_base = inst_base+'.json'
        '''

        if (instance != ".DS_Store"):

            input_name = os.path.join(inst.save_dir, 'json_format', base_save_folder_name, instance)
            
            output_name_csv = instance.split('.json')[0] + '.csv'
            output_name_csv = output_name_csv.replace(" ", "")
            
            print(instance)
            print(output_name_csv)

