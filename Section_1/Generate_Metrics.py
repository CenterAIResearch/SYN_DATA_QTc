# load file in DATAPOOL_z_tr

import numpy as np
import pandas as pd
import os

from sdv.evaluation.single_table import run_diagnostic, evaluate_quality
from sdv.evaluation.single_table import get_column_plot
from sdv.metadata import SingleTableMetadata

import math

# import all functions from utils.performance_metrics
from utils.performance_metrics_edited import *
import csv

from sdmetrics.single_column import RangeCoverage
from sdmetrics.single_column import StatisticSimilarity
from sdmetrics.single_table import NewRowSynthesis

import pickle

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


# origin_data_path = '/Users/choih2/Documents/GitHub/CSICU_Data_Science/PSB2025/DATAPOOL_z_tr/newz_train.csv'

exp_dir ='/Users/choih2/Documents/GitHub/CSICU_Data_Science/PSB2025/Experiments/Exp_3_New'

exp_syn_dataset_path = exp_dir + '/Syn_Datasets'

each_num_rows = 70

num_of_samples = 100

confidence_level = 1.96





# method_exp_list =["Gaussian_Copula_num_rows_50","CTGAN_num_rows_50"]
method_exp_list =[f"Gaussian_Copula_num_rows_{each_num_rows}",f"CTGAN_num_rows_{each_num_rows}",f"Bayesian_Network_num_rows_{each_num_rows}", f"TVAE_num_rows_{each_num_rows}", f"rtvae_num_rows_{each_num_rows}", f"ddpm_num_rows_{each_num_rows}"]

len_method_exp_list = len(method_exp_list)

# method_exp_list =[f"CTGAN_num_rows_{each_num_rows}",f"Gaussian_Copula_num_rows_{each_num_rows}"]
# method_exp_list =["CTGAN_num_rows_88","Gaussian_Copula_num_rows_88"]


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
    # use index to get the syn_data_file and sampled_data_file
    # for index in range(num_of_samples):
        # Load the data in syn_data_file
        # print("syn_data_file[index]",syn_data_files[index])
        # syn_data = pd.read_csv(syn_data_dir + '/' + syn_data_files[index])
        # split the syn_data_file to get the index
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


        # if metric_kl is nan print below
        # if np.isnan(metric_kl):
        #     print("KL Divergence is nan")
        #     continue


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

        # save syn_data_file name and sampled_data_file in the 


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
        

        


        # From the SDV,     

        #perform basic validity checks
        diagnostic = run_diagnostic(sampled_data, syn_data, metadata)
        print("diagnostic._overall_score",diagnostic._overall_score)
        # list_of_sdv_diagnostic_score.append(diagnostic._overall_score)


        #measure the statistical similarity
        quality_report = evaluate_quality(sampled_data, syn_data, metadata)
        print("quality_report._overall_score",quality_report._overall_score)



        
        quality_report.get_details(property_name='Column Shapes')
        # fig=quality_report.get_visualization(property_name='Column Shapes')
        # fig.show()


        quality_report.get_details(property_name='Column Pair Trends')
        # save quality_report.get_details(property_name='Column Pair Trends') as a csv file 
        # file_path ="/Users/choih2/Documents/GitHub/CSICU_Data_Science/PSB2025/quality_report_Column_Pair_Trends.csv"
        # quality_report.get_details(property_name='Column Pair Trends').to_csv(file_path, index=False)



        # quality_report.get_details(property_name='Column Pair Trends').iloc[0]
        # fig=quality_report.get_visualization(property_name='Column Pair Trends')
        # fig.show()

        sdv_avg_con_coverage=sdv_range_coverage_continous_features(sampled_data, syn_data, metadata)
        


        sdv_avg_stat_sim=sdv_stat_sim_continous_features(sampled_data, syn_data, metadata)
 

        sdv_avg_newrow=sdv_newrow(sampled_data, syn_data, metadata)


        print("hi")

        # 3. plot the data
        # fig = get_column_plot(
        #     real_data=sampled_data,
        #     synthetic_data=syn_data,
        #     metadata=metadata,
        #     column_name='ecg_manual_qtcb_binary'
        # )
        # fig.show()

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
        

    # calcuate CI 
    # CI for kl_divergence
    kl_divergence_mean = np.mean(list_of_kl_divergence)
    kl_divergence_std = np.std(list_of_kl_divergence)
    kl_divergence_ci = confidence_level * kl_divergence_std / np.sqrt(num_of_samples)
    kl_divergence_interval = [kl_divergence_mean - kl_divergence_ci, kl_divergence_mean + kl_divergence_ci]

    # CI for pairwise_correlation_difference
    list_of_pairwise_correlation_difference = list_of_pairwise_correlation_difference
    pairwise_correlation_difference_values_list = [value['-'] for value in list_of_pairwise_correlation_difference]
    pairwise_correlation_difference_mean = np.mean(pairwise_correlation_difference_values_list)
    pairwise_correlation_difference_std = np.std(pairwise_correlation_difference_values_list)
    pairwise_correlation_difference_ci = confidence_level * pairwise_correlation_difference_std / np.sqrt(num_of_samples)
    pairwise_correlation_difference_interval = [pairwise_correlation_difference_mean - pairwise_correlation_difference_ci, pairwise_correlation_difference_mean + pairwise_correlation_difference_ci]

    # CI for cluster_measure
    list_of_cluster_measure = list_of_cluster_measure
    cluster_measure_values_list = [value['-'] for value in list_of_cluster_measure]
    cluster_measure_mean = np.mean(cluster_measure_values_list)
    cluster_measure_std = np.std(cluster_measure_values_list)
    cluster_measure_ci = confidence_level * cluster_measure_std / np.sqrt(num_of_samples)
    cluster_measure_interval = [cluster_measure_mean - cluster_measure_ci, cluster_measure_mean + cluster_measure_ci]

    # CI for coverage (exclude)
    coverage_mean = np.mean(list_of_coverage)
    coverage_std = np.std(list_of_coverage)
    coverage_ci = confidence_level * coverage_std / np.sqrt(num_of_samples)
    coverage_interval = [coverage_mean - coverage_ci, coverage_mean + coverage_ci]

    


    # CI for cross_classification (for three categorical columns)
    cross_classification_mean = np.mean(list_of_cross_classification)
    cross_classification_std = np.std(list_of_cross_classification)
    cross_classification_ci = confidence_level * cross_classification_std / np.sqrt(num_of_samples)
    cross_classification_interval = [cross_classification_mean - cross_classification_ci, cross_classification_mean + cross_classification_ci]

    # CI for corss_regression (for remaining numerical columns)
    cross_regression_mean = np.mean(list_of_cross_regression)
    cross_regression_std = np.std(list_of_cross_regression)
    cross_regression_ci = confidence_level * cross_regression_std / np.sqrt(num_of_samples)
    cross_regression_interval = [cross_regression_mean - cross_regression_ci, cross_regression_mean + cross_regression_ci]

    # CI for sdv_diagnostic_score (exclude)
    sdv_diagnostic_score_mean = np.mean(list_of_sdv_diagnostic_score)
    sdv_diagnostic_score_std = np.std(list_of_sdv_diagnostic_score)
    sdv_diagnostic_score_ci = confidence_level * sdv_diagnostic_score_std / np.sqrt(num_of_samples)
    sdv_diagnostic_score_interval = [sdv_diagnostic_score_mean - sdv_diagnostic_score_ci, sdv_diagnostic_score_mean + sdv_diagnostic_score_ci]

    # CI for sdv_quality_report_score
    sdv_quality_report_score_mean = np.mean(list_of_sdv_quality_report_score)
    sdv_quality_report_score_std = np.std(list_of_sdv_quality_report_score)
    sdv_quality_report_score_ci = confidence_level * sdv_quality_report_score_std / np.sqrt(num_of_samples)
    sdv_quality_report_score_interval = [sdv_quality_report_score_mean - sdv_quality_report_score_ci, sdv_quality_report_score_mean + sdv_quality_report_score_ci]

    # CI for sdv_continous_features_coverage
    sdv_continous_features_coverage_mean = np.mean(list_of_sdv_continous_features_coverage)
    sdv_continous_features_coverage_std = np.std(list_of_sdv_continous_features_coverage)
    sdv_continous_features_coverage_ci = confidence_level * sdv_continous_features_coverage_std / np.sqrt(num_of_samples)
    sdv_continous_features_coverage_interval = [sdv_continous_features_coverage_mean - sdv_continous_features_coverage_ci, sdv_continous_features_coverage_mean + sdv_continous_features_coverage_ci]

    # CI for sdv_stat_sim_continous_features
    sdv_stat_sim_continous_features_mean = np.mean(list_of_sdv_stat_sim_continous_features)
    sdv_stat_sim_continous_features_std = np.std(list_of_sdv_stat_sim_continous_features)
    sdv_stat_sim_continous_features_ci = confidence_level * sdv_stat_sim_continous_features_std / np.sqrt(num_of_samples)
    sdv_stat_sim_continous_features_interval = [sdv_stat_sim_continous_features_mean - sdv_stat_sim_continous_features_ci, sdv_stat_sim_continous_features_mean + sdv_stat_sim_continous_features_ci]

    # CI for sdv_newrow
    sdv_newrow_mean = np.mean(list_of_sdv_newrow)
    sdv_newrow_std = np.std(list_of_sdv_newrow)
    sdv_newrow_ci = confidence_level * sdv_newrow_std / np.sqrt(num_of_samples)
    sdv_newrow_interval = [sdv_newrow_mean - sdv_newrow_ci, sdv_newrow_mean + sdv_newrow_ci]

    # add explanation for each metric in the ppt.



    # # PCD
    # print("pairwise_correlation_difference_interval",pairwise_correlation_difference_interval)
    # # mean
    # print("pairwise_correlation_difference_mean",pairwise_correlation_difference_mean)
    # # std
    # print("pairwise_correlation_difference_std",pairwise_correlation_difference_std)


    # # log CM
    # print("cluster_measure_interval",cluster_measure_interval)
    # # mean
    # print("cluster_measure_mean",cluster_measure_mean)
    # # std
    # print("cluster_measure_std",cluster_measure_std)


    # # support coverage
    # print("coverage_interval",coverage_interval)
    # # mean
    # print("coverage_mean",coverage_mean)
    # # std
    # print("coverage_std",coverage_std)

    # # kl divergence
    # print("kl_divergence_interval",kl_divergence_interval)
    # # mean
    # print("kl_divergence_mean",kl_divergence_mean)
    # # std
    # print("kl_divergence_std",kl_divergence_std)

    
    # # cross classification
    # # confidence interval
    # print("cross_classification_interval",cross_classification_interval)

    # # mean
    # print("cross_classification_mean",cross_classification_mean)

    # # std
    # print("cross_classification_std",cross_classification_std)


    # # confidence interval
    # print("sdv_diagnostic_score_interval",sdv_diagnostic_score_interval)
    # # mean
    # print("sdv_diagnostic_score_mean",sdv_diagnostic_score_mean)
    # # std
    # print("sdv_diagnostic_score_std",sdv_diagnostic_score_std)

    # # confidence interval
    # print("sdv_quality_report_score_interval",sdv_quality_report_score_interval)
    # # mean
    # print("sdv_quality_report_score_mean",sdv_quality_report_score_mean)
    # # std
    # print("sdv_quality_report_score_std",sdv_quality_report_score_std)

    # Create a dictionary with the data
    data = {
        "method_exp": method_exp,
        "pairwise_correlation_difference_interval": pairwise_correlation_difference_interval,
        "pairwise_correlation_difference_mean": pairwise_correlation_difference_mean,
        "pairwise_correlation_difference_std": pairwise_correlation_difference_std,

        "cluster_measure_interval": cluster_measure_interval,
        "cluster_measure_mean": cluster_measure_mean,
        "cluster_measure_std": cluster_measure_std,

        "coverage_interval": coverage_interval,
        "coverage_mean": coverage_mean,
        "coverage_std": coverage_std,
        
        "sdv_continous_features_coverage_interval": sdv_continous_features_coverage_interval,
        "sdv_continous_features_coverage_mean": sdv_continous_features_coverage_mean,
        "sdv_continous_features_coverage_std": sdv_continous_features_coverage_std,

        "sdv_stat_sim_continous_features_interval": sdv_stat_sim_continous_features_interval,
        "sdv_stat_sim_continous_features_mean": sdv_stat_sim_continous_features_mean,
        "sdv_stat_sim_continous_features_std": sdv_stat_sim_continous_features_std,

        "sdv_newrow_interval": sdv_newrow_interval,
        "sdv_newrow_mean": sdv_newrow_mean,
        "sdv_newrow_std": sdv_newrow_std,

        "kl_divergence_interval": kl_divergence_interval,
        "kl_divergence_mean": kl_divergence_mean,
        "kl_divergence_std": kl_divergence_std,

        "cross_classification_interval": cross_classification_interval,
        "cross_classification_mean": cross_classification_mean,
        "cross_classification_std": cross_classification_std,

        "cross_regression_interval": cross_regression_interval,
        "cross_regression_mean": cross_regression_mean,
        "cross_regression_std": cross_regression_std,

        "sdv_diagnostic_score_interval": sdv_diagnostic_score_interval,
        "sdv_diagnostic_score_mean": sdv_diagnostic_score_mean,
        "sdv_diagnostic_score_std": sdv_diagnostic_score_std,
        
        "sdv_quality_report_score_interval": sdv_quality_report_score_interval,
        "sdv_quality_report_score_mean": sdv_quality_report_score_mean,
        "sdv_quality_report_score_std": sdv_quality_report_score_std


    }

    # Write data to CSV file 
    with open(f'output_each_num_rows_{each_num_rows}_num_methods_{len_method_exp_list}.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data.keys())
        writer.writerow([data[key] if isinstance(data[key], (int, float)) else str(data[key]) for key in data.keys()])

    print(f'Data has been written to output_each_num_rows_{each_num_rows}_num_methods_{len_method_exp_list}.csv')


    print("Completed!")
    print(list_index)




