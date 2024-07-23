import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import ast
from matplotlib.patches import Patch
import os
import pickle
import logging


def convert_string_to_list(value):
    try:
        # Replace 'nan' with 'None' and convert the string to a list
        cleaned_value = value.replace('nan', 'None')
        list_value = ast.literal_eval(cleaned_value)
        
        return [np.nan if x is None else x for x in list_value]
    except (ValueError, SyntaxError):
        return value


def format_number(x):
    if abs(x) >= 100000:
        return f'{x:.1e}'
    else:
        return f'{x:.3f}'

# Function to format interval as string
def format_interval(interval):
    return f'({format_number(interval[0])}, {format_number(interval[1])})'

def format_mean_std(mean, std):
    return f'{format_number(mean)} ({format_number(std)})'

if __name__ == '__main__':
    # get all files in the current directory
    files = os.listdir()


    # filter only pickle files
    pickle_files = [file for file in files if file.endswith('.pkl')]
    print(pickle_files)

    # load all pickle files 
    for file in pickle_files:
        with open(file, 'rb') as f:
            data = pickle.load(f)
            print("here")

            print(data['method_exp'])

            # if data['method_exp'] includes 'Gaussian_Copula' then assign to gc_data_combined
            # if lower() is used, it will be case insensitive
            if 'Gaussian_Copula'.lower() in data['method_exp'].lower():

                # metrics = ["pairwise_correlation_difference", "cluster_measure", "kl_divergence", "cross_classification","cross_regression", "sdv_quality_report_score", "sdv_continous_features_coverage", "sdv_stat_sim_continous_features"]
                gc_data_combined = [
                    [item['-'] for item in data["list_of_pairwise_correlation_difference"]],
                    [item['-'] for item in data["list_of_cluster_measure"]],
                    data["list_of_kl_divergence"],
                    # data["list_of_coverage"],
                    data["list_of_cross_classification"],
                    data["list_of_cross_regression"],
                    # data["list_of_sdv_diagnostic_score"],
                    data["list_of_sdv_quality_report_score"],
                    data["list_of_sdv_continous_features_coverage"],
                    data["list_of_sdv_stat_sim_continous_features"],
                    data["list_of_sdv_newrow"]
                ]
            # if data['method_exp'] includes 'CTGAN' then assign to ctgan_data_combined
            elif 'CTGAN'.lower() in data['method_exp'].lower():
                ctgan_data_combined = [
                    [item['-'] for item in data["list_of_pairwise_correlation_difference"]],
                    [item['-'] for item in data["list_of_cluster_measure"]],
                    data["list_of_kl_divergence"],
                    # data["list_of_coverage"],
                    data["list_of_cross_classification"],
                    data["list_of_cross_regression"],
                    # data["list_of_sdv_diagnostic_score"],
                    data["list_of_sdv_quality_report_score"],
                    data["list_of_sdv_continous_features_coverage"],
                    data["list_of_sdv_stat_sim_continous_features"],
                    data["list_of_sdv_newrow"]
                ]

            # if data['method_exp'] includes 'Bayesian_Network' then assign to bayesian_network_data_combined
            elif 'Bayesian_Network'.lower() in data['method_exp'].lower():
                bayesian_network_data_combined = [
                    [item['-'] for item in data["list_of_pairwise_correlation_difference"]],
                    [item['-'] for item in data["list_of_cluster_measure"]],
                    data["list_of_kl_divergence"],
                    # data["list_of_coverage"],
                    data["list_of_cross_classification"],
                    data["list_of_cross_regression"],
                    # data["list_of_sdv_diagnostic_score"],
                    data["list_of_sdv_quality_report_score"],
                    data["list_of_sdv_continous_features_coverage"],
                    data["list_of_sdv_stat_sim_continous_features"],
                    data["list_of_sdv_newrow"]
                ]
            # if data['method_exp'] includes 'TVAE' then assign to tvae_data_combined
            elif 'TVAE'.lower() in data['method_exp'].lower() and 'RTVAE'.lower() not in data['method_exp'].lower():
                tvae_data_combined = [
                    [item['-'] for item in data["list_of_pairwise_correlation_difference"]],
                    [item['-'] for item in data["list_of_cluster_measure"]],
                    data["list_of_kl_divergence"],
                    # data["list_of_coverage"],
                    data["list_of_cross_classification"],
                    data["list_of_cross_regression"],
                    # data["list_of_sdv_diagnostic_score"],
                    data["list_of_sdv_quality_report_score"],
                    data["list_of_sdv_continous_features_coverage"],
                    data["list_of_sdv_stat_sim_continous_features"],
                    data["list_of_sdv_newrow"]
                ]
            # if data['method_exp'] includes 'RTVAE' then assign to rtvae_data_combined
            elif 'RTVAE'.lower() in data['method_exp'].lower() :
                rtvae_data_combined = [
                                [item['-'] for item in data["list_of_pairwise_correlation_difference"]],
                    [item['-'] for item in data["list_of_cluster_measure"]],
                    data["list_of_kl_divergence"],
                    # data["list_of_coverage"],
                    data["list_of_cross_classification"],
                    data["list_of_cross_regression"],
                    # data["list_of_sdv_diagnostic_score"],
                    data["list_of_sdv_quality_report_score"],
                    data["list_of_sdv_continous_features_coverage"],
                    data["list_of_sdv_stat_sim_continous_features"],
                    data["list_of_sdv_newrow"]
                ]

            # if data['method_exp'] includes 'DDPM' then assign to ddpm_data_combined
            elif 'DDPM'.lower() in data['method_exp'].lower():
                ddpm_data_combined = [
                                [item['-'] for item in data["list_of_pairwise_correlation_difference"]],
                    [item['-'] for item in data["list_of_cluster_measure"]],
                    data["list_of_kl_divergence"],
                    # data["list_of_coverage"],
                    data["list_of_cross_classification"],
                    data["list_of_cross_regression"],
                    # data["list_of_sdv_diagnostic_score"],
                    data["list_of_sdv_quality_report_score"],
                    data["list_of_sdv_continous_features_coverage"],
                    data["list_of_sdv_stat_sim_continous_features"],
                    data["list_of_sdv_newrow"]
                ]



    # Prepare data for combined box plot
    # metrics = ["pairwise_correlation_difference", "cluster_measure", "coverage", "kl_divergence", "cross_classification"]
    metrics = ["pairwise_correlation_difference", "cluster_measure", "kl_divergence", "cross_classification","cross_regression", "sdv_quality_report_score", "sdv_continous_features_coverage", "sdv_stat_sim_continous_features"]

    # Y-axis limits for each metric
    y_limits = {
        "pairwise_correlation_difference": (0, 10),
        "cluster_measure": (-5, 1),
        "kl_divergence": (0, 500),
        "cross_classification": (0.2, 1.5),
        # "cross_regression": (-10, 20),
        "cross_regression": (-3, 5),
        "sdv_quality_report_score": (0.5, 1),
        "sdv_continous_features_coverage": (0.5, 1.1),
        "sdv_stat_sim_continous_features": (0.5, 1.1)

    }

    # Define custom colors
    custom_colors = {
        "GC": "#1f77b4",  # Blue
        "CTGAN": "#ff7f0e",  # Orange
        "BN": "#2ca02c",  # Green
        "TVAE": "#d62728",  # Red
        "RTVAE": "#9467bd",  # Purple
        "DDPM": "#8c564b"  # Brown
    }


    methods = ["GC", "CTGAN", "BN", "TVAE", "RTVAE", "DDPM"]
    colors = [custom_colors[method] for method in methods]




    logging.basicConfig(filename='log_file.txt', level=logging.INFO, format='%(message)s')

    for i, metric in enumerate(metrics):
        

        # for overleaf box plot
        sorted_gc_data_combined_i = sorted(gc_data_combined[i])
        sorted_ctgan_data_combined_i = sorted(ctgan_data_combined[i])
        sorted_bayesian_network_data_combined_i = sorted(bayesian_network_data_combined[i])
        sorted_tvae_data_combined_i = sorted(tvae_data_combined[i])
        sorted_rtvae_data_combined_i = sorted(rtvae_data_combined[i])
        sorted_ddpm_data_combined_i = sorted(ddpm_data_combined[i])

        Q1_gc_data_combined_i = np.percentile(sorted_gc_data_combined_i, 25)
        median_gc_data_combined_i = np.percentile(sorted_gc_data_combined_i, 50)
        Q3_gc_data_combined_i = np.percentile(sorted_gc_data_combined_i, 75)

        IQR_gc_data_combined_i = Q3_gc_data_combined_i - Q1_gc_data_combined_i

        lower_whisker_gc_data_combined_i = max(min(sorted_gc_data_combined_i), Q1_gc_data_combined_i - 1.5 * IQR_gc_data_combined_i)
        upper_whisker_gc_data_combined_i = min(max(sorted_gc_data_combined_i), Q3_gc_data_combined_i + 1.5 * IQR_gc_data_combined_i)

        # please make the above prints saved in a log file
        # create a log file and save the above prints in the log file


        logging.info(f"metric: {metric}")
        logging.info(f"Lower whisker_gc: {lower_whisker_gc_data_combined_i}")
        logging.info(f"Lower quartile (Q1)_gc: {Q1_gc_data_combined_i}")
        logging.info(f"Median_gc: {median_gc_data_combined_i}")
        logging.info(f"Upper quartile (Q3)_gc: {Q3_gc_data_combined_i}")
        logging.info(f"Upper whisker_gc: {upper_whisker_gc_data_combined_i}")
        logging.info("\n")

        Q1_ctgan_data_combined_i = np.percentile(sorted_ctgan_data_combined_i, 25)
        median_ctgan_data_combined_i = np.percentile(sorted_ctgan_data_combined_i, 50)
        Q3_ctgan_data_combined_i = np.percentile(sorted_ctgan_data_combined_i, 75)

        IQR_ctgan_data_combined_i = Q3_ctgan_data_combined_i - Q1_ctgan_data_combined_i

        lower_whisker_ctgan_data_combined_i = max(min(sorted_ctgan_data_combined_i), Q1_ctgan_data_combined_i - 1.5 * IQR_ctgan_data_combined_i)
        upper_whisker_ctgan_data_combined_i = min(max(sorted_ctgan_data_combined_i), Q3_ctgan_data_combined_i + 1.5 * IQR_ctgan_data_combined_i)

        logging.info(f"metric: {metric}")
        logging.info(f"Lower whisker_ctgan: {lower_whisker_ctgan_data_combined_i}")
        logging.info(f"Lower quartile (Q1)_ctgan: {Q1_ctgan_data_combined_i}")
        logging.info(f"Median_ctgan: {median_ctgan_data_combined_i}")
        logging.info(f"Upper quartile (Q3)_ctgan: {Q3_ctgan_data_combined_i}")
        logging.info(f"Upper whisker_ctgan: {upper_whisker_ctgan_data_combined_i}")
        logging.info("\n")

        Q1_bayesian_network_data_combined_i = np.percentile(sorted_bayesian_network_data_combined_i, 25)
        median_bayesian_network_data_combined_i = np.percentile(sorted_bayesian_network_data_combined_i, 50)
        Q3_bayesian_network_data_combined_i = np.percentile(sorted_bayesian_network_data_combined_i, 75)

        IQR_bayesian_network_data_combined_i = Q3_bayesian_network_data_combined_i - Q1_bayesian_network_data_combined_i

        lower_whisker_bayesian_network_data_combined_i = max(min(sorted_bayesian_network_data_combined_i), Q1_bayesian_network_data_combined_i - 1.5 * IQR_bayesian_network_data_combined_i)
        upper_whisker_bayesian_network_data_combined_i = min(max(sorted_bayesian_network_data_combined_i), Q3_bayesian_network_data_combined_i + 1.5 * IQR_bayesian_network_data_combined_i)

        logging.info(f"metric: {metric}")
        logging.info(f"Lower whisker_bayesian_network: {lower_whisker_bayesian_network_data_combined_i}")
        logging.info(f"Lower quartile (Q1)_bayesian_network: {Q1_bayesian_network_data_combined_i}")
        logging.info(f"Median_bayesian_network: {median_bayesian_network_data_combined_i}")
        logging.info(f"Upper quartile (Q3)_bayesian_network: {Q3_bayesian_network_data_combined_i}")
        logging.info(f"Upper whisker_bayesian_network: {upper_whisker_bayesian_network_data_combined_i}")
        logging.info("\n")


        Q1_tvae_data_combined_i = np.percentile(sorted_tvae_data_combined_i, 25)
        median_tvae_data_combined_i = np.percentile(sorted_tvae_data_combined_i, 50)
        Q3_tvae_data_combined_i = np.percentile(sorted_tvae_data_combined_i, 75)

        IQR_tvae_data_combined_i = Q3_tvae_data_combined_i - Q1_tvae_data_combined_i

        lower_whisker_tvae_data_combined_i = max(min(sorted_tvae_data_combined_i), Q1_tvae_data_combined_i - 1.5 * IQR_tvae_data_combined_i)
        upper_whisker_tvae_data_combined_i = min(max(sorted_tvae_data_combined_i), Q3_tvae_data_combined_i + 1.5 * IQR_tvae_data_combined_i)

        logging.info(f"metric: {metric}")
        logging.info(f"Lower whisker_tvae: {lower_whisker_tvae_data_combined_i}")
        logging.info(f"Lower quartile (Q1)_tvae: {Q1_tvae_data_combined_i}")
        logging.info(f"Median_tvae: {median_tvae_data_combined_i}")
        logging.info(f"Upper quartile (Q3)_tvae: {Q3_tvae_data_combined_i}")
        logging.info(f"Upper whisker_tvae: {upper_whisker_tvae_data_combined_i}")
        logging.info("\n")

        Q1_rtvae_data_combined_i = np.percentile(sorted_rtvae_data_combined_i, 25)
        median_rtvae_data_combined_i = np.percentile(sorted_rtvae_data_combined_i, 50)
        Q3_rtvae_data_combined_i = np.percentile(sorted_rtvae_data_combined_i, 75)

        IQR_rtvae_data_combined_i = Q3_rtvae_data_combined_i - Q1_rtvae_data_combined_i

        lower_whisker_rtvae_data_combined_i = max(min(sorted_rtvae_data_combined_i), Q1_rtvae_data_combined_i - 1.5 * IQR_rtvae_data_combined_i)
        upper_whisker_rtvae_data_combined_i = min(max(sorted_rtvae_data_combined_i), Q3_rtvae_data_combined_i + 1.5 * IQR_rtvae_data_combined_i)

        logging.info(f"metric: {metric}")
        logging.info(f"Lower whisker_rtvae: {lower_whisker_rtvae_data_combined_i}")
        logging.info(f"Lower quartile (Q1)_rtvae: {Q1_rtvae_data_combined_i}")
        logging.info(f"Median_rtvae: {median_rtvae_data_combined_i}")
        logging.info(f"Upper quartile (Q3)_rtvae: {Q3_rtvae_data_combined_i}")
        logging.info(f"Upper whisker_rtvae: {upper_whisker_rtvae_data_combined_i}")
        logging.info("\n")

        Q1_ddpm_data_combined_i = np.percentile(sorted_ddpm_data_combined_i, 25)
        median_ddpm_data_combined_i = np.percentile(sorted_ddpm_data_combined_i, 50)
        Q3_ddpm_data_combined_i = np.percentile(sorted_ddpm_data_combined_i, 75)

        IQR_ddpm_data_combined_i = Q3_ddpm_data_combined_i - Q1_ddpm_data_combined_i

        lower_whisker_ddpm_data_combined_i = max(min(sorted_ddpm_data_combined_i), Q1_ddpm_data_combined_i - 1.5 * IQR_ddpm_data_combined_i)
        upper_whisker_ddpm_data_combined_i = min(max(sorted_ddpm_data_combined_i), Q3_ddpm_data_combined_i + 1.5 * IQR_ddpm_data_combined_i)

        logging.info(f"metric: {metric}")
        logging.info(f"Lower whisker_ddpm: {lower_whisker_ddpm_data_combined_i}")
        logging.info(f"Lower quartile (Q1)_ddpm: {Q1_ddpm_data_combined_i}")
        logging.info(f"Median_ddpm: {median_ddpm_data_combined_i}")
        logging.info(f"Upper quartile (Q3)_ddpm: {Q3_ddpm_data_combined_i}")
        logging.info(f"Upper whisker_ddpm: {upper_whisker_ddpm_data_combined_i}")
        logging.info("\n")
