# load file in DATAPOOL_z_tr

import numpy as np
import pandas as pd
import os
import pickle
import math
import csv


from sdv.evaluation.single_table import run_diagnostic, evaluate_quality
from sdv.evaluation.single_table import get_column_plot
from sdv.metadata import SingleTableMetadata

from sdmetrics.single_column import RangeCoverage
from sdmetrics.single_column import StatisticSimilarity
from sdmetrics.single_table import NewRowSynthesis


# import all functions from utils.performance_metrics
from utils.performance_metrics_edited import *

def calculate_average_coverage(metric_coverage):
    # Extract the first dictionary from the input
    coverage_values = metric_coverage
    
    # Calculate the sum of all coverage values
    total_sum = sum(coverage_values.values())
    
    # Calculate the number of features
    number_of_features = len(coverage_values)
    
    # Calculate the average coverage
    average_coverage = total_sum / number_of_features
    
    return average_coverage


def sdv_range_coverage_continous_features(data_a, data_b,metadata):
    
    list_of_range_coverage_continous_features = []
    # in metadata.columns for the key 'sdtype' is 'numerical'
    for column in metadata.columns:
        if metadata.columns[column]['sdtype'] == 'numerical':
            # print(column)
            coverage=RangeCoverage.compute(
                real_data=data_a[column],
                synthetic_data=data_b[column]
            )
            list_of_range_coverage_continous_features.append(coverage)
    
    # calculate the average of the list_of_range_coverage_continous_features
    average_coverage=sum(list_of_range_coverage_continous_features)/len(list_of_range_coverage_continous_features)
    
    return average_coverage


def sdv_stat_sim_continous_features(data_a, data_b,metadata):
    
    list_of_stat_sim_continous_features = []
    # in metadata.columns for the key 'sdtype' is 'numerical'
    for column in metadata.columns:
        if metadata.columns[column]['sdtype'] == 'numerical':
            # print(column)
            coverage=StatisticSimilarity.compute(
                real_data=data_a[column],
                synthetic_data=data_b[column],
                statistic='mean'
            )
            list_of_stat_sim_continous_features.append(coverage)
    
    # calculate the average of the list_of_stat_sim_continous_features
    average_stat_sim=sum(list_of_stat_sim_continous_features)/len(list_of_stat_sim_continous_features)
    
    return average_stat_sim


def sdv_newrow(data_a, data_b,metadata):
    
    coverage=NewRowSynthesis.compute(
            real_data=data_a,
            synthetic_data=data_b,
            metadata=metadata,
            numerical_match_tolerance=0.01,
            synthetic_sample_size=50
        )
    
    return coverage

if __name__ == '__main__':
    # current directory
    exp_dir = os.getcwd()

    exp_syn_dataset_path = exp_dir + '/Section_1/Syn_Data_by_methods'

    each_num_rows = 70

    num_of_samples = 100

    confidence_level = 1.96


    method_exp_list =[f"Gaussian_Copula_num_rows_{each_num_rows}",f"CTGAN_num_rows_{each_num_rows}",f"Bayesian_Network_num_rows_{each_num_rows}", f"TVAE_num_rows_{each_num_rows}", f"rtvae_num_rows_{each_num_rows}", f"ddpm_num_rows_{each_num_rows}"]

    len_method_exp_list = len(method_exp_list)

    # for loop with method_exp_list
    for method_exp in method_exp_list:
        print(method_exp)

        syn_data_dir = exp_syn_dataset_path + '/' + method_exp

        # read all csv files in the directory syn_data_dir
        whole_files = [f for f in os.listdir(syn_data_dir ) if f.endswith('.csv')]

        # remove the file whose name starts with 'newz_sampled_data_num_rows'
        syn_data_files = [f for f in whole_files if not f.startswith('newz_sampled_data_num_rows')]

        # get file whose name starts with 'newz_sampled_data_num_rows'
        sampled_data_files = [f for f in whole_files if f.startswith('newz_sampled_data_num_rows')]

        syn_data_files=sorted(syn_data_files)

        # Synthetic data quality metrics from research Generation and evaluation of synthetic patient data Andre Goncalves1*, Priyadip Ray1, Braden Soper1, Jennifer Stevens2,LindaCoyle2 and Ana Paula Sales1
        list_of_kl_divergence = []
        list_of_pairwise_correlation_difference = []
        list_of_cluster_measure = []
        list_of_coverage = []
        list_of_cross_classification = []

        # added by me
        list_of_cross_regression = []

        # Synthetic data quality metrics from research SDV
        list_of_sdv_continous_features_coverage = []
        list_of_sdv_stat_sim_continous_features = []
        list_of_sdv_newrow = []
        list_of_sdv_diagnostic_score = []
        list_of_sdv_quality_report_score = []

        list_index=[]

        metadata = ""

        categorical_columns = ['ecg_manual_qtcb_binary', 'sex', 'tele_manual_qtcfb']
        # nemerical columns are the columns that are not in the categorical_columns
        numerical_columns = ""

        for syn_data_file in syn_data_files:
        
            syn_data_index = int(syn_data_file.split('_')[-1].split('.')[0])
            list_index.append(syn_data_index)
            

            # load syn_data_file 
            syn_data = pd.read_csv(syn_data_dir + '/' + syn_data_file)

            # find the corresponding sampled_data_file
            sampled_data_file = f'newz_sampled_data_num_rows_{each_num_rows}_index_' + str(syn_data_index) + '.csv'

            # load sampled_data_file
            sampled_data = pd.read_csv(syn_data_dir + '/' + sampled_data_file)

            # 

            # Define metadata
            if metadata == "":
                # # nemerical columns are the columns that are not in the categorical_columns
                numerical_columns = [column for column in sampled_data.columns if column not in categorical_columns]
                # Create metadata object
                metadata = SingleTableMetadata()

                # Extract metadata from the dataframe
                metadata.detect_from_dataframe(data=sampled_data)


                # Manually update each column's metadata
                metadata.update_column('ecg_manual_qtcb_binary', sdtype='categorical')
                metadata.update_column('age', sdtype='numerical', computer_representation='Float')
                metadata.update_column('sex', sdtype='categorical')
                metadata.update_column('number_qtc_meds', sdtype='numerical', computer_representation='Float')
                metadata.update_column('ecg_auto_hr', sdtype='numerical', computer_representation='Float')
                metadata.update_column('ecg_auto_pr', sdtype='numerical', computer_representation='Float')
                metadata.update_column('ecg_auto_qrs', sdtype='numerical', computer_representation='Float')
                metadata.update_column('ecg_auto_qt', sdtype='numerical', computer_representation='Float')
                metadata.update_column('ecg_auto_p_axis', sdtype='numerical', computer_representation='Float')
                metadata.update_column('ecg_auto_r_axis', sdtype='numerical', computer_representation='Float')
                metadata.update_column('ecg_auto_t_axis', sdtype='numerical', computer_representation='Float')
                metadata.update_column('ecg_auto_qtc', sdtype='numerical', computer_representation='Float')
                metadata.update_column('tele_auto_hr', sdtype='numerical', computer_representation='Float')
                metadata.update_column('tele_auto_qt', sdtype='numerical', computer_representation='Float')
                metadata.update_column('tele_auto_qtc', sdtype='numerical', computer_representation='Float')
                metadata.update_column('tele_auto_map', sdtype='numerical', computer_representation='Float')
                metadata.update_column('tele_manual_qrs', sdtype='numerical', computer_representation='Float')
                metadata.update_column('tele_manual_qt', sdtype='numerical', computer_representation='Float')
                metadata.update_column('tele_manual_rr', sdtype='numerical', computer_representation='Float')
                metadata.update_column('tele_manual_qtcb', sdtype='numerical', computer_representation='Float')
                metadata.update_column('tele_manual_qtcf', sdtype='numerical', computer_representation='Float')
                metadata.update_column('tele_manual_qtcfb', sdtype='categorical')
                metadata.update_column('auto_teleqt_minus_auto_ecgqt', sdtype='numerical', computer_representation='Float')
                metadata.update_column('auto_telehr_minus_auto_ecghr', sdtype='numerical', computer_representation='Float')

            # Load the data in sampled_data_file
            data_a = sampled_data
            data_b = syn_data

            # KL Divergence origin
            metric_kl=kl_divergence(data_a=data_a, data_b=data_b)
            # get values from the dictionary metric_kl only for the key ecg_manual_qtcb_binary, sex, tele_manual_qtcfb
            metric_kl = {key: value for key, value in metric_kl.items() if key in ['ecg_manual_qtcb_binary','sex','tele_manual_qtcfb']}

            # KL divergence edited
            # metric_kl=kl_divergence_edited(data_a=data_a, data_b=data_b, bandwidth=0.1)
            metric_kl=kl_divergence_edited_v2(data_a=data_a, data_b=data_b)

            # check if any value in the dictionary metric_kl is nan
            # if nan, print below
            if any(np.isnan(value) for key, value in metric_kl.items()):
                print("KL Divergence is nan")
                # continue

            # sum of all the kl_divergence values in the dictionary metric_kl
            metric_kl_sum = sum([value for key, value in metric_kl.items()])


            # Pairwise Correlation Difference
            metric_pcd=pairwise_correlation_difference(data_a=data_a, data_b=data_b) 
            # nan
            if np.isnan(metric_pcd['-']):
                print("Pairwise Correlation Difference is None")
                continue

            # Log Cluster? or Cluster Measure?
            metric_cm=cluster_measure(data_a=data_a, data_b=data_b)
            # calculate log metric_cm
            metric_cm['-']=math.log(metric_cm['-'])


            if np.isnan(metric_cm['-']):
                print("Cluster Measure is None")
                continue



            # Support Coverage
            # metric_coverage=coverage(data_a=data_a, data_b=data_b)
            # Support Coverage defined by me.
            metric_coverage=coverage_categorical_continous(data_a=data_a, data_b=data_b)
            average_coverage=calculate_average_coverage(metric_coverage)

            # Cross classification
            # need to check how the output is generated
            # metric_crossclass=cross_classification(data_a=data_a, data_b=data_b)
            # metric_crossclass=cca_accuracy(data_a=data_a, data_b=data_b)
            metric_crossclass=cca_accuracy_edited(data_a=data_a, data_b=data_b, base_metric='balanced_accuracy')
            metric_crossclass_average = sum([value for key, value in metric_crossclass.items() if key in categorical_columns]) / len(categorical_columns)


            # Cross regression
            metric_crossreg=ccr_diff_edited(data_a=data_a, data_b=data_b, base_metric='r2_score')
            metric_crossreg_average= sum([value for key, value in metric_crossreg.items() if key in numerical_columns]) / len(numerical_columns)
            

            #perform basic validity checks
            diagnostic = run_diagnostic(sampled_data, syn_data, metadata)
            print("diagnostic._overall_score",diagnostic._overall_score)
            # list_of_sdv_diagnostic_score.append(diagnostic._overall_score)


            #measure the statistical similarity
            quality_report = evaluate_quality(sampled_data, syn_data, metadata)
            print("quality_report._overall_score",quality_report._overall_score)



            
            quality_report.get_details(property_name='Column Shapes')



            quality_report.get_details(property_name='Column Pair Trends')

            

            sdv_avg_con_coverage=sdv_range_coverage_continous_features(sampled_data, syn_data, metadata)
            


            sdv_avg_stat_sim=sdv_stat_sim_continous_features(sampled_data, syn_data, metadata)
    

            sdv_avg_newrow=sdv_newrow(sampled_data, syn_data, metadata)


            

            list_of_kl_divergence.append(metric_kl_sum)
            list_of_pairwise_correlation_difference.append(metric_pcd)
            list_of_cluster_measure.append(metric_cm)
            list_of_coverage.append(average_coverage)
            list_of_cross_classification.append(metric_crossclass_average)
            list_of_cross_regression.append(metric_crossreg_average)
            list_of_sdv_diagnostic_score.append(diagnostic._overall_score)
            list_of_sdv_quality_report_score.append(quality_report._overall_score)
            list_of_sdv_continous_features_coverage.append(sdv_avg_con_coverage)
            list_of_sdv_stat_sim_continous_features.append(sdv_avg_stat_sim)
            list_of_sdv_newrow.append(sdv_avg_newrow)
        


        # store the results in a dictionary
        lists_each_method_exp = {
            "method_exp": method_exp,
            'list_of_kl_divergence': list_of_kl_divergence,
            'list_of_pairwise_correlation_difference': list_of_pairwise_correlation_difference,
            'list_of_cluster_measure': list_of_cluster_measure,
            'list_of_coverage': list_of_coverage,
            'list_of_cross_classification': list_of_cross_classification,
            'list_of_cross_regression': list_of_cross_regression,
            'list_of_sdv_diagnostic_score': list_of_sdv_diagnostic_score,
            'list_of_sdv_quality_report_score': list_of_sdv_quality_report_score,
            'list_of_sdv_continous_features_coverage': list_of_sdv_continous_features_coverage,
            'list_of_sdv_stat_sim_continous_features': list_of_sdv_stat_sim_continous_features,
            'list_of_sdv_newrow': list_of_sdv_newrow

        }

        with open(f'lists_each_method_exp_{method_exp}.pkl', 'wb') as f:
            pickle.dump(lists_each_method_exp, f)
            

        print(f'lists_each_method_exp_{method_exp}.pkl is saved')
        print(list_index)




