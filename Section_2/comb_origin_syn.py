
import numpy as np
import pandas as pd
import os



# read files ./Pure_Syn_dataset/bayesian_network_num_rows_50000-500000/
def read_files(path):
    files = os.listdir(path)
    files = [file for file in files if file.endswith('.csv')]
    return files


# read data
def read_data(file):
    data = pd.read_csv(file)
    return data

if __name__ == '__main__':

    current_dir = os.getcwd()

    exp="Section_2/Syn_Data_for_ML_task/synthetic_data_using_Bayesian_Network"

    # path for newz_train.csv
    original_data_path = '../FORJACC/DATAPOOL_z_tr/newz_train.csv'

    original_data = pd.read_csv(original_data_path)
    main_path =f'{current_dir}/{exp}'


    files=read_files(main_path)
    print(files)

    if 'Combined_synthetic_data_using_Bayesian_Network' not in os.listdir(f'{current_dir}/Section_2/Syn_Data_for_ML_task'):
        os.mkdir(f'{current_dir}/Section_2/Syn_Data_for_ML_task/Combined_synthetic_data_using_Bayesian_Network')

    for file in files:
        print(f'Processing file: {file}')
        data = read_data(f'{main_path}/{file}')

        # combine the original data with the data
        combined_data = pd.concat([original_data, data]).sample(frac=1, random_state=42).reset_index(drop=True)

        # length of the combined data
        print(f'Length of the combined data: {len(combined_data)}')

        combined_data.to_csv(f'{current_dir}/Section_2/Syn_Data_for_ML_task/Combined_synthetic_data_using_Bayesian_Network/combined_{file}', index=False)

    


