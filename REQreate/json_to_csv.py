import os
from instance_class import Instance
from output_files import JsonConverter

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

    '''
    final_filename = ''
    for p in inst.instance_filename:

        if p in inst.parameters:
            if 'value' in inst.parameters[p]:
                strv = str(inst.parameters[p]['value'])
                strv = strv.replace(" ", "")

                if len(final_filename) > 0:
                    if p == 'min_early_departure':
                        strv = inst.parameters[p]['value']/3600
                        strv = str(strv)

                    if p == 'max_early_departure':
                        strv = inst.parameters[p]['value']/3600
                        strv = str(strv)

                    final_filename = final_filename + '_' + strv
                else: final_filename = strv
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

            converter = JsonConverter(file_name=input_name)
            converter.convert_normal(inst=inst, problem_type=inst.parameters['problem']['value'], path_instance_csv_file=os.path.join(save_dir_csv, base_save_folder_name, output_name_csv))
            
            inst1 = pd.read_csv(os.path.join(save_dir_csv, output_name_csv))


            full_final_filename = inst.filename_json.replace(inst_directory, "")
            full_final_filename = full_final_filename.replace(".json", "")
            full_final_filename = full_final_filename + '_' + str(replicate_num) + '.csv'


            print(instance)
            print(output_name_csv)
            print(full_final_filename)

            os.rename(os.path.join(save_dir_csv, output_name_csv), os.path.join(save_dir_csv, full_final_filename))
            replicate_num = replicate_num + 1
