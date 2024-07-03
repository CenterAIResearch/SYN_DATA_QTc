import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import ast
from matplotlib.patches import Patch
import os
import pickle


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



# read the output_each_num_rows_70_num_methods_6.csv file
total_metrics_output = pd.read_csv("output_each_num_rows_70_num_methods_6.csv")


# # get 0, 2, 4, 6, 8, 10 rows
# gc_metrics = total_metrics_output.iloc[0]
# ctgan_metrics = total_metrics_output.iloc[2]
# bayesian_network_metrics = total_metrics_output.iloc[4]
# tvae_metrics = total_metrics_output.iloc[6]
# rtvae_metrics = total_metrics_output.iloc[8]
# ddpm_metrics = total_metrics_output.iloc[10]

# # make each metrics as a dictionary for each method
# gc_metrics = gc_metrics.to_dict()
# ctgan_metrics = ctgan_metrics.to_dict()
# bayesian_network_metrics = bayesian_network_metrics.to_dict()
# tvae_metrics = tvae_metrics.to_dict()
# rtvae_metrics = rtvae_metrics.to_dict()
# ddpm_metrics = ddpm_metrics.to_dict()


# gc_metrics = {
#     "pairwise_correlation_difference": {
#         "mean": float(gc_metrics["pairwise_correlation_difference_mean"]),
#         "std": float(gc_metrics["pairwise_correlation_difference_std"]),
#         "interval": convert_string_to_list(gc_metrics["pairwise_correlation_difference_interval"])
#     },
#     "cluster_measure": {
#         "mean": float(gc_metrics["cluster_measure_mean"]),
#         "std": float(gc_metrics["cluster_measure_std"]),
#         "interval": convert_string_to_list(gc_metrics["cluster_measure_interval"])
#     },
#     # "coverage": {
#     #     "mean": float(gc_metrics["coverage_mean"]),
#     #     "std": float(gc_metrics["coverage_std"]),
#     #     "interval": convert_string_to_list(gc_metrics["coverage_interval"])
#     # },
#     "sdv_continous_features_coverage": {
#         "mean": float(gc_metrics["sdv_continous_features_coverage_mean"]),
#         "std": float(gc_metrics["sdv_continous_features_coverage_std"]),
#         "interval": convert_string_to_list(gc_metrics["sdv_continous_features_coverage_interval"])
#     },
#     "sdv_stat_sim_continous_features": {
#         "mean": float(gc_metrics["sdv_stat_sim_continous_features_mean"]),
#         "std": float(gc_metrics["sdv_stat_sim_continous_features_std"]),
#         "interval": convert_string_to_list(gc_metrics["sdv_stat_sim_continous_features_interval"])
#     },
#     "sdv_newrow": {
#         "mean": float(gc_metrics["sdv_newrow_mean"]),
#         "std": float(gc_metrics["sdv_newrow_std"]),
#         "interval": convert_string_to_list(gc_metrics["sdv_newrow_interval"])
#     },
#     "kl_divergence": {
#         "mean": float(gc_metrics["kl_divergence_mean"]),
#         "std": float(gc_metrics["kl_divergence_std"]),
#         "interval": convert_string_to_list(gc_metrics["kl_divergence_interval"])
#     },
#     "cross_classification": {
#         "mean": float(gc_metrics["cross_classification_mean"]),
#         "std": float(gc_metrics["cross_classification_std"]),
#         "interval": convert_string_to_list(gc_metrics["cross_classification_interval"])
#     },
#     "cross_regression": {
#         "mean": float(gc_metrics["cross_regression_mean"]),
#         "std": float(gc_metrics["cross_regression_std"]),
#         "interval": convert_string_to_list(gc_metrics["cross_regression_interval"])
#     },
#     # "sdv_diagnostic_score": {
#     #     "mean": float(gc_metrics["sdv_diagnostic_score_mean"]),
#     #     "std":  float(gc_metrics["sdv_diagnostic_score_std"]),
#     #     "interval": convert_string_to_list(gc_metrics["sdv_diagnostic_score_interval"])
#     # },
#     "sdv_quality_report_score": {
#         "mean": float(gc_metrics["sdv_quality_report_score_mean"]),
#         "std": float(gc_metrics["sdv_quality_report_score_std"]),
#         "interval": convert_string_to_list(gc_metrics["sdv_quality_report_score_interval"])
#     }
# }


# ctgan_metrics = {
#     "pairwise_correlation_difference": {
#         "mean": float(ctgan_metrics["pairwise_correlation_difference_mean"]),
#         "std": float(ctgan_metrics["pairwise_correlation_difference_std"]),
#         "interval": convert_string_to_list(ctgan_metrics["pairwise_correlation_difference_interval"])
#     },
#     "cluster_measure": {
#         "mean": float(ctgan_metrics["cluster_measure_mean"]),
#         "std": float(ctgan_metrics["cluster_measure_std"]),
#         "interval": convert_string_to_list(ctgan_metrics["cluster_measure_interval"])
#     },
#     # "coverage": {
#     #     "mean": float(ctgan_metrics["coverage_mean"]),
#     #     "std": float(ctgan_metrics["coverage_std"]),
#     #     "interval": convert_string_to_list(ctgan_metrics["coverage_interval"])
#     # },
#     "sdv_continous_features_coverage": {
#         "mean": float(ctgan_metrics["sdv_continous_features_coverage_mean"]),
#         "std": float(ctgan_metrics["sdv_continous_features_coverage_std"]),
#         "interval": convert_string_to_list(ctgan_metrics["sdv_continous_features_coverage_interval"])
#     },
#     "sdv_stat_sim_continous_features": {
#         "mean": float(ctgan_metrics["sdv_stat_sim_continous_features_mean"]),
#         "std": float(ctgan_metrics["sdv_stat_sim_continous_features_std"]),
#         "interval": convert_string_to_list(ctgan_metrics["sdv_stat_sim_continous_features_interval"])
#     },
#     "sdv_newrow": {
#         "mean": float(ctgan_metrics["sdv_newrow_mean"]),
#         "std": float(ctgan_metrics["sdv_newrow_std"]),
#         "interval": convert_string_to_list(ctgan_metrics["sdv_newrow_interval"])
#     },
#     "kl_divergence": {
#         "mean": float(ctgan_metrics["kl_divergence_mean"]),
#         "std": float(ctgan_metrics["kl_divergence_std"]),
#         "interval": convert_string_to_list(ctgan_metrics["kl_divergence_interval"])
#     },
#     "cross_classification": {
#         "mean": float(ctgan_metrics["cross_classification_mean"]),
#         "std": float(ctgan_metrics["cross_classification_std"]),
#         "interval": convert_string_to_list(ctgan_metrics["cross_classification_interval"])
#     },
#     "cross_regression": {
#         "mean": float(ctgan_metrics["cross_regression_mean"]),
#         "std": float(ctgan_metrics["cross_regression_std"]),
#         "interval": convert_string_to_list(ctgan_metrics["cross_regression_interval"])
#     },
#     # "sdv_diagnostic_score": {
#     #     "mean": float(ctgan_metrics["sdv_diagnostic_score_mean"]),
#     #     "std":  float(ctgan_metrics["sdv_diagnostic_score_std"]),
#     #     "interval": convert_string_to_list(ctgan_metrics["sdv_diagnostic_score_interval"])
#     # },
#     "sdv_quality_report_score": {
#         "mean": float(ctgan_metrics["sdv_quality_report_score_mean"]),
#         "std": float(ctgan_metrics["sdv_quality_report_score_std"]),
#         "interval": convert_string_to_list(ctgan_metrics["sdv_quality_report_score_interval"])
#     }

# }

# bayesian_network_metrics = {
#     "pairwise_correlation_difference": {
#         "mean": float(bayesian_network_metrics["pairwise_correlation_difference_mean"]),
#         "std": float(bayesian_network_metrics["pairwise_correlation_difference_std"]),
#         "interval": convert_string_to_list(bayesian_network_metrics["pairwise_correlation_difference_interval"])
#     },
#     "cluster_measure": {
#         "mean": float(bayesian_network_metrics["cluster_measure_mean"]),
#         "std": float(bayesian_network_metrics["cluster_measure_std"]),
#         "interval": convert_string_to_list(bayesian_network_metrics["cluster_measure_interval"])
#     },
#     # "coverage": {
#     #     "mean": float(bayesian_network_metrics["coverage_mean"]),
#     #     "std": float(bayesian_network_metrics["coverage_std"]),
#     #     "interval": convert_string_to_list(bayesian_network_metrics["coverage_interval"])
#     # },
#     "sdv_continous_features_coverage": {
#         "mean": float(bayesian_network_metrics["sdv_continous_features_coverage_mean"]),
#         "std": float(bayesian_network_metrics["sdv_continous_features_coverage_std"]),
#         "interval": convert_string_to_list(bayesian_network_metrics["sdv_continous_features_coverage_interval"])
#     },
#     "sdv_stat_sim_continous_features": {
#         "mean": float(bayesian_network_metrics["sdv_stat_sim_continous_features_mean"]),
#         "std": float(bayesian_network_metrics["sdv_stat_sim_continous_features_std"]),
#         "interval": convert_string_to_list(bayesian_network_metrics["sdv_stat_sim_continous_features_interval"])
#     },
#     "sdv_newrow": {
#         "mean": float(bayesian_network_metrics["sdv_newrow_mean"]),
#         "std": float(bayesian_network_metrics["sdv_newrow_std"]),
#         "interval": convert_string_to_list(bayesian_network_metrics["sdv_newrow_interval"])
#     },
#     "kl_divergence": {
#         "mean": float(bayesian_network_metrics["kl_divergence_mean"]),
#         "std": float(bayesian_network_metrics["kl_divergence_std"]),
#         "interval": convert_string_to_list(bayesian_network_metrics["kl_divergence_interval"])
#     },
#     "cross_classification": {
#         "mean": float(bayesian_network_metrics["cross_classification_mean"]),
#         "std": float(bayesian_network_metrics["cross_classification_std"]),
#         "interval": convert_string_to_list(bayesian_network_metrics["cross_classification_interval"])
#     },
#     "cross_regression": {
#         "mean": float(bayesian_network_metrics["cross_regression_mean"]),
#         "std": float(bayesian_network_metrics["cross_regression_std"]),
#         "interval": convert_string_to_list(bayesian_network_metrics["cross_regression_interval"])
#     },
#     # "sdv_diagnostic_score": {
#     #     "mean": float(bayesian_network_metrics["sdv_diagnostic_score_mean"]),
#     #     "std":  float(bayesian_network_metrics["sdv_diagnostic_score_std"]),
#     #     "interval": convert_string_to_list(bayesian_network_metrics["sdv_diagnostic_score_interval"])
#     # },
#     "sdv_quality_report_score": {
#         "mean": float(bayesian_network_metrics["sdv_quality_report_score_mean"]),
#         "std": float(bayesian_network_metrics["sdv_quality_report_score_std"]),
#         "interval": convert_string_to_list(bayesian_network_metrics["sdv_quality_report_score_interval"])
#     }
# }

# tvae_metrics = {
#     "pairwise_correlation_difference": {
#         "mean": float(tvae_metrics["pairwise_correlation_difference_mean"]),
#         "std": float(tvae_metrics["pairwise_correlation_difference_std"]),
#         "interval": convert_string_to_list(tvae_metrics["pairwise_correlation_difference_interval"])
#     },
#     "cluster_measure": {
#         "mean": float(tvae_metrics["cluster_measure_mean"]),
#         "std": float(tvae_metrics["cluster_measure_std"]),
#         "interval": convert_string_to_list(tvae_metrics["cluster_measure_interval"])
#     },
#     # "coverage": {
#     #     "mean": float(tvae_metrics["coverage_mean"]),
#     #     "std": float(tvae_metrics["coverage_std"]),
#     #     "interval": convert_string_to_list(tvae_metrics["coverage_interval"])
#     # },
#     "sdv_continous_features_coverage": {
#         "mean": float(tvae_metrics["sdv_continous_features_coverage_mean"]),
#         "std": float(tvae_metrics["sdv_continous_features_coverage_std"]),
#         "interval": convert_string_to_list(tvae_metrics["sdv_continous_features_coverage_interval"])
#     },
#     "sdv_stat_sim_continous_features": {
#         "mean": float(tvae_metrics["sdv_stat_sim_continous_features_mean"]),
#         "std": float(tvae_metrics["sdv_stat_sim_continous_features_std"]),
#         "interval": convert_string_to_list(tvae_metrics["sdv_stat_sim_continous_features_interval"])
#     },
#     "sdv_newrow": {
#         "mean": float(tvae_metrics["sdv_newrow_mean"]),
#         "std": float(tvae_metrics["sdv_newrow_std"]),
#         "interval": convert_string_to_list(tvae_metrics["sdv_newrow_interval"])
#     },
#     "kl_divergence": {
#         "mean": float(tvae_metrics["kl_divergence_mean"]),
#         "std": float(tvae_metrics["kl_divergence_std"]),
#         "interval": convert_string_to_list(tvae_metrics["kl_divergence_interval"])
#     },
#     "cross_classification": {
#         "mean": float(tvae_metrics["cross_classification_mean"]),
#         "std": float(tvae_metrics["cross_classification_std"]),
#         "interval": convert_string_to_list(tvae_metrics["cross_classification_interval"])
#     },
#     "cross_regression": {
#         "mean": float(tvae_metrics["cross_regression_mean"]),
#         "std": float(tvae_metrics["cross_regression_std"]),
#         "interval": convert_string_to_list(tvae_metrics["cross_regression_interval"])
#     },
#     # "sdv_diagnostic_score": {
#     #     "mean": float(tvae_metrics["sdv_diagnostic_score_mean"]),
#     #     "std":  float(tvae_metrics["sdv_diagnostic_score_std"]),
#     #     "interval": convert_string_to_list(tvae_metrics["sdv_diagnostic_score_interval"])
#     # },
#     "sdv_quality_report_score": {
#         "mean": float(tvae_metrics["sdv_quality_report_score_mean"]),
#         "std": float(tvae_metrics["sdv_quality_report_score_std"]),
#         "interval": convert_string_to_list(tvae_metrics["sdv_quality_report_score_interval"])
#     }
# }

# rtvae_metrics = {
#     "pairwise_correlation_difference": {
#         "mean": float(rtvae_metrics["pairwise_correlation_difference_mean"]),
#         "std": float(rtvae_metrics["pairwise_correlation_difference_std"]),
#         "interval": convert_string_to_list(rtvae_metrics["pairwise_correlation_difference_interval"])
#     },
#     "cluster_measure": {
#         "mean": float(rtvae_metrics["cluster_measure_mean"]),
#         "std": float(rtvae_metrics["cluster_measure_std"]),
#         "interval": convert_string_to_list(rtvae_metrics["cluster_measure_interval"])
#     },
#     "coverage": {
#         "mean": float(rtvae_metrics["coverage_mean"]),
#         "std": float(rtvae_metrics["coverage_std"]),
#         "interval": convert_string_to_list(rtvae_metrics["coverage_interval"])
#     },
#     "sdv_continous_features_coverage": {
#         "mean": float(rtvae_metrics["sdv_continous_features_coverage_mean"]),
#         "std": float(rtvae_metrics["sdv_continous_features_coverage_std"]),
#         "interval": convert_string_to_list(rtvae_metrics["sdv_continous_features_coverage_interval"])
#     },
#     "sdv_stat_sim_continous_features": {
#         "mean": float(rtvae_metrics["sdv_stat_sim_continous_features_mean"]),
#         "std": float(rtvae_metrics["sdv_stat_sim_continous_features_std"]),
#         "interval": convert_string_to_list(rtvae_metrics["sdv_stat_sim_continous_features_interval"])
#     },
#     "sdv_newrow": {
#         "mean": float(rtvae_metrics["sdv_newrow_mean"]),
#         "std": float(rtvae_metrics["sdv_newrow_std"]),
#         "interval": convert_string_to_list(rtvae_metrics["sdv_newrow_interval"])
#     },
#     "kl_divergence": {
#         "mean": float(rtvae_metrics["kl_divergence_mean"]),
#         "std": float(rtvae_metrics["kl_divergence_std"]),
#         "interval": convert_string_to_list(rtvae_metrics["kl_divergence_interval"])
#     },
#     "cross_classification": {
#         "mean": float(rtvae_metrics["cross_classification_mean"]),
#         "std": float(rtvae_metrics["cross_classification_std"]),
#         "interval": convert_string_to_list(rtvae_metrics["cross_classification_interval"])
#     },
#     "cross_regression": {
#         "mean": float(rtvae_metrics["cross_regression_mean"]),
#         "std": float(rtvae_metrics["cross_regression_std"]),
#         "interval": convert_string_to_list(rtvae_metrics["cross_regression_interval"])
#     },
#     # "sdv_diagnostic_score": {
#     #     "mean": float(rtvae_metrics["sdv_diagnostic_score_mean"]),
#     #     "std":  float(rtvae_metrics["sdv_diagnostic_score_std"]),
#     #     "interval": convert_string_to_list(rtvae_metrics["sdv_diagnostic_score_interval"])
#     # },
#     "sdv_quality_report_score": {
#         "mean": float(rtvae_metrics["sdv_quality_report_score_mean"]),
#         "std": float(rtvae_metrics["sdv_quality_report_score_std"]),
#         "interval": convert_string_to_list(rtvae_metrics["sdv_quality_report_score_interval"])
#     }
# }

# ddpm_metrics = {
#     "pairwise_correlation_difference": {
#         "mean": float(ddpm_metrics["pairwise_correlation_difference_mean"]),
#         "std": float(ddpm_metrics["pairwise_correlation_difference_std"]),
#         "interval": convert_string_to_list(ddpm_metrics["pairwise_correlation_difference_interval"])
#     },
#     "cluster_measure": {
#         "mean": float(ddpm_metrics["cluster_measure_mean"]),
#         "std": float(ddpm_metrics["cluster_measure_std"]),
#         "interval": convert_string_to_list(ddpm_metrics["cluster_measure_interval"])
#     },
#     # "coverage": {
#     #     "mean": float(ddpm_metrics["coverage_mean"]),
#     #     "std": float(ddpm_metrics["coverage_std"]),
#     #     "interval": convert_string_to_list(ddpm_metrics["coverage_interval"])
#     # },
#     "sdv_continous_features_coverage": {
#         "mean": float(ddpm_metrics["sdv_continous_features_coverage_mean"]),
#         "std": float(ddpm_metrics["sdv_continous_features_coverage_std"]),
#         "interval": convert_string_to_list(ddpm_metrics["sdv_continous_features_coverage_interval"])
#     },
#     "sdv_stat_sim_continous_features": {
#         "mean": float(ddpm_metrics["sdv_stat_sim_continous_features_mean"]),
#         "std": float(ddpm_metrics["sdv_stat_sim_continous_features_std"]),
#         "interval": convert_string_to_list(ddpm_metrics["sdv_stat_sim_continous_features_interval"])
#     },
#     "sdv_newrow": {
#         "mean": float(ddpm_metrics["sdv_newrow_mean"]),
#         "std": float(ddpm_metrics["sdv_newrow_std"]),
#         "interval": convert_string_to_list(ddpm_metrics["sdv_newrow_interval"])
#     },
#     "kl_divergence": {
#         "mean": float(ddpm_metrics["kl_divergence_mean"]),
#         "std": float(ddpm_metrics["kl_divergence_std"]),
#         "interval": convert_string_to_list(ddpm_metrics["kl_divergence_interval"])
#     },
#     "cross_classification": {
#         "mean": float(ddpm_metrics["cross_classification_mean"]),
#         "std": float(ddpm_metrics["cross_classification_std"]),
#         "interval": convert_string_to_list(ddpm_metrics["cross_classification_interval"])
#     },
#     "cross_regression": {
#         "mean": float(ddpm_metrics["cross_regression_mean"]),
#         "std": float(ddpm_metrics["cross_regression_std"]),
#         "interval": convert_string_to_list(ddpm_metrics["cross_regression_interval"])
#     },
#     # "sdv_diagnostic_score": {
#     #     "mean": float(ddpm_metrics["sdv_diagnostic_score_mean"]),
#     #     "std":  float(ddpm_metrics["sdv_diagnostic_score_std"]),
#     #     "interval": convert_string_to_list(ddpm_metrics["sdv_diagnostic_score_interval"])
#     # },
#     "sdv_quality_report_score": {
#         "mean": float(ddpm_metrics["sdv_quality_report_score_mean"]),
#         "std": float(ddpm_metrics["sdv_quality_report_score_std"]),
#         "interval": convert_string_to_list(ddpm_metrics["sdv_quality_report_score_interval"])
#     }
# }



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




# gc_data_combined = [
#     np.random.normal(gc_metrics[metric]["mean"], gc_metrics[metric]["std"], 100) 
#     for metric in metrics
# ]
# ctgan_data_combined = [
#     np.random.normal(ctgan_metrics[metric]["mean"], ctgan_metrics[metric]["std"], 100) 
#     for metric in metrics
# ]

# bayesian_network_data_combined = [
#     np.random.normal(bayesian_network_metrics[metric]["mean"], bayesian_network_metrics[metric]["std"], 100)
#     for metric in metrics
# ]

# tvae_data_combined = [
#     np.random.normal(tvae_metrics[metric]["mean"], tvae_metrics[metric]["std"], 100)
#     for metric in metrics
# ]

# rtvae_data_combined = [
#     np.random.normal(rtvae_metrics[metric]["mean"], rtvae_metrics[metric]["std"], 100)
#     for metric in metrics
# ]

# ddpm_data_combined = [
#     np.random.normal(ddpm_metrics[metric]["mean"], ddpm_metrics[metric]["std"], 100)
#     for metric in metrics
# ]





# Set Seaborn style
sns.set(style="whitegrid")

# Create combined box plots with adjusted y-axis limits
# fig, axes = plt.subplots(1, 5, figsize=(18, 7), sharey=False)
fig, axes = plt.subplots(1, 8, figsize=(20, 10), sharey=False)





for i, metric in enumerate(metrics):
    # check non of gc_data_combined[i], ctgan_data_combined[i], bayesian_network_data_combined[i], tvae_data_combined[i], rtvae_data_combined[i], ddpm_data_combined[i] have nan 
    # check whether gc_data_combined[i] has nan
    data = [gc_data_combined[i], ctgan_data_combined[i], bayesian_network_data_combined[i], tvae_data_combined[i], rtvae_data_combined[i], ddpm_data_combined[i]]
    # check whether data has nan and return the index 
    nan_index = [index for index, item in enumerate(data) if np.isnan(item).any()]
    # remove the nan from the data
    filtered_data = [item for index, item in enumerate(data) if index not in nan_index]
    # remove the method from the methods
    filtered_methods = [method for index, method in enumerate(methods) if index not in nan_index]
    # print("nan_index",nan_index)
    # remove the color from the colors
    filtered_colors = [color for index, color in enumerate(colors) if index not in nan_index]
    # data = [gc_data_combined[i], ctgan_data_combined[i], tvae_data_combined[i]]
    sns.boxplot(data=filtered_data, ax=axes[i], palette=filtered_colors)
    # axes[i].set_xticklabels(["Gaussian Copula", "CTGAN", "Bayesian Network", "TVAE", "RTVAE", "DDPM"], fontsize=5)
    axes[i].set_xticklabels(filtered_methods, fontsize=5, fontweight='bold')
    # axes[i].set_xticklabels(["Gaussian Copula", "CTGAN","TVAE"])
    axes[i].set_title(f"{metric.replace('_', ' ').title()}", fontsize=6, fontweight='bold')
    axes[i].set_ylim(y_limits[metric])





# plt.figure(figsize=(20, 10))
# legend_patches = [Patch(color=colors[j], label=methods[j]) for j in range(len(methods))]
# plt.legend(handles=legend_patches, ncol=6, loc='upper center', bbox_to_anchor=(0.2, 1), fontsize=8)


plt.tight_layout()
plt.savefig("box_plot_v5.png", dpi=300)
# plt.show()


# clear the plot
# plt.clf()