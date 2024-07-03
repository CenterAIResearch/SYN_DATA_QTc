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


    # get the current directory
    current_dir = os.getcwd()



    # Main dir to save png files
    main_dir = os.path.join(current_dir+"/Section_1", 'png_files')

    # make the directory if it does not exist
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)


    plt.tight_layout()
    # plt.savefig("box_plot_v5.png", dpi=300)
    plt.savefig(f"{main_dir}/box_plots.png", dpi=300)
    # plt.show()


    # clear the plot
    # plt.clf()