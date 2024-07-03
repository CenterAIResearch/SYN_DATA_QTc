import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast

import os
import pickle


import numpy as np
from scipy import stats




# CIs for single type metrics
def calculate_confidence_interval(data):
    
    data = np.array(data)
    m = np.mean(data)
    
    # standard error of the mean
    sem = stats.sem(data)

    # n = len(data)
    # s_calculated = s * np.sqrt(n)

    # standard deviation
    s = np.std(data)
    # s = np.std(data,  ddof=1)


    # ci95_hi = []
    # ci95_lo = []

    # for i in stats.index:
    #     m, c, s = stats.loc[i]
    #     ci95_hi.append(m + 1.96*s) #https://moderndive.com/8-confidence-intervals.html#se-method
    #     ci95_lo.append(m - 1.96*s)
    
    ci95_hi = m + 1.96*s #https://moderndive.com/8-confidence-intervals.html#se-method
    ci95_lo = m - 1.96*s

    # stats['ci95_hi'] = ci95_hi
    # stats['ci95_lo'] = ci95_lo
    
    return (ci95_lo, ci95_hi)

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


def format_mean_and_interval(metric):
    mean = format_number(metric["mean"])
    interval = metric["interval"]
    lower_bound = format_number(interval[0])
    upper_bound = format_number(interval[1])
    return f"Mean: {mean}, CI: {lower_bound} - {upper_bound}"

# Adjust the rendering function to prevent column name overflow
def render_mpl_table(data, col_width=5.0, row_height=0.625, font_size=10,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0:
            cell.set_text_props(weight='bold', color='w', fontsize=font_size-2)
            cell.set_facecolor(header_color)
            cell.set_height(0.075)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])
            cell.set_height(0.065)
    fig.tight_layout()
    return ax

def plot_confidence_intervals(df, metric, filename):
    methods = df['Method CI']
    values = df[metric].apply(lambda x: [float(x.split(": ")[1].split(",")[0]), 
                                         float(x.split(": ")[2].split(" - ")[0]), 
                                         float(x.split(": ")[2].split(" - ")[1])]).tolist()
    
    means = [v[0] for v in values]
    errors = [[v[0] - v[1], v[2] - v[0]] for v in values]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(len(methods))
    ax.errorbar(x_pos, means, yerr=np.array(errors).T, fmt='o', capsize=5, capthick=2, ecolor='blue')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    
    ax.set_ylabel(metric)
    ax.set_title(f'Confidence Interval for {metric}')
    
    plt.tight_layout()
    plt.savefig(filename)

# get the current directory
current_dir = os.getcwd()

# Main dir to save png files
main_dir = os.path.join(current_dir+"/Section_1", 'png_files')

# make the directory if it does not exist
if not os.path.exists(main_dir):
    os.makedirs(main_dir)

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
# total_metrics_output = pd.read_csv("output_each_num_rows_70_num_methods_6.csv")


# print(gc_data_combined)


# make the dictionary for each method as a list like the just above
gc_metrics = {
    "pairwise_correlation_difference": {
        # calcuare list of values for each metric

        "mean": np.mean(gc_data_combined[0]),
        "std": np.std(gc_data_combined[0]),
        "interval": calculate_confidence_interval(gc_data_combined[0])
    },
    "cluster_measure": {
        "mean": np.mean(gc_data_combined[1]),
        "std": np.std(gc_data_combined[1]),
        "interval": calculate_confidence_interval(gc_data_combined[1])
    },
    # "coverage": {
    #     "mean": np.mean(gc_metrics["coverage_mean"]),
    #     "std": np.std(gc_metrics["coverage_std"]),
    #     "interval": convert_string_to_list(gc_metrics["coverage_interval"])
    # },
    "sdv_continous_features_coverage": {
        "mean": np.mean(gc_data_combined[6]),
        "std": np.std(gc_data_combined[6]),
        "interval": calculate_confidence_interval(gc_data_combined[6])
    },
    "sdv_stat_sim_continous_features": {
        "mean": np.mean(gc_data_combined[7]),
        "std": np.std(gc_data_combined[7]),
        "interval": calculate_confidence_interval(gc_data_combined[7])
    },
    "sdv_newrow": {
        "mean": np.mean(gc_data_combined[8]),
        "std": np.std(gc_data_combined[8]),
        "interval": calculate_confidence_interval(gc_data_combined[8])
    },
    "kl_divergence": {
        "mean": np.mean(gc_data_combined[2]),
        "std": np.std(gc_data_combined[2]),
        "interval": calculate_confidence_interval(gc_data_combined[2])
    },
    "cross_classification": {
        "mean": np.mean(gc_data_combined[3]),
        "std": np.std(gc_data_combined[3]),
        "interval": calculate_confidence_interval(gc_data_combined[3])
    },
    "cross_regression": {
        "mean": np.mean(gc_data_combined[4]),
        "std": np.std(gc_data_combined[4]),
        "interval": calculate_confidence_interval(gc_data_combined[4])
    },
    # "sdv_diagnostic_score": {
    #     "mean": np.mean(gc_metrics["sdv_diagnostic_score_mean"]),
    #     "std":  float(gc_metrics["sdv_diagnostic_score_std"]),
    #     "interval": convert_string_to_list(gc_metrics["sdv_diagnostic_score_interval"])
    # },
    "sdv_quality_report_score": {
        "mean": np.mean(gc_data_combined[5]),
        "std": np.std(gc_data_combined[5]),
        "interval": calculate_confidence_interval(gc_data_combined[5])
    }
}


ctgan_metrics = {
    "pairwise_correlation_difference": {
        # calcuare list of values for each metric

        "mean": np.mean(ctgan_data_combined[0]),
        "std": np.std(ctgan_data_combined[0]),
        "interval": calculate_confidence_interval(ctgan_data_combined[0])
    },
    "cluster_measure": {
        "mean": np.mean(ctgan_data_combined[1]),
        "std": np.std(ctgan_data_combined[1]),
        "interval": calculate_confidence_interval(ctgan_data_combined[1])
    },
    # "coverage": {
    #     "mean": np.mean(gc_metrics["coverage_mean"]),
    #     "std": np.std(gc_metrics["coverage_std"]),
    #     "interval": convert_string_to_list(gc_metrics["coverage_interval"])
    # },
    "sdv_continous_features_coverage": {
        "mean": np.mean(ctgan_data_combined[6]),
        "std": np.std(ctgan_data_combined[6]),
        "interval": calculate_confidence_interval(ctgan_data_combined[6])
    },
    "sdv_stat_sim_continous_features": {
        "mean": np.mean(ctgan_data_combined[7]),
        "std": np.std(ctgan_data_combined[7]),
        "interval": calculate_confidence_interval(ctgan_data_combined[7])
    },
    "sdv_newrow": {
        "mean": np.mean(ctgan_data_combined[8]),
        "std": np.std(ctgan_data_combined[8]),
        "interval": calculate_confidence_interval(ctgan_data_combined[8])
    },
    "kl_divergence": {
        "mean": np.mean(ctgan_data_combined[2]),
        "std": np.std(ctgan_data_combined[2]),
        "interval": calculate_confidence_interval(ctgan_data_combined[2])
    },
    "cross_classification": {
        "mean": np.mean(ctgan_data_combined[3]),
        "std": np.std(ctgan_data_combined[3]),
        "interval": calculate_confidence_interval(ctgan_data_combined[3])
    },
    "cross_regression": {
        "mean": np.mean(ctgan_data_combined[4]),
        "std": np.std(ctgan_data_combined[4]),
        "interval": calculate_confidence_interval(ctgan_data_combined[4])
    },
    # "sdv_diagnostic_score": {
    #     "mean": np.mean(gc_metrics["sdv_diagnostic_score_mean"]),
    #     "std":  float(gc_metrics["sdv_diagnostic_score_std"]),
    #     "interval": convert_string_to_list(gc_metrics["sdv_diagnostic_score_interval"])
    # },
    "sdv_quality_report_score": {
        "mean": np.mean(ctgan_data_combined[5]),
        "std": np.std(ctgan_data_combined[5]),
        "interval": calculate_confidence_interval(ctgan_data_combined[5])
    }
}

bayesian_network_metrics = {
    "pairwise_correlation_difference": {
        # calcuare list of values for each metric

        "mean": np.mean(bayesian_network_data_combined[0]),
        "std": np.std(bayesian_network_data_combined[0]),
        "interval": calculate_confidence_interval(bayesian_network_data_combined[0])
    },
    "cluster_measure": {
        "mean": np.mean(bayesian_network_data_combined[1]),
        "std": np.std(bayesian_network_data_combined[1]),
        "interval": calculate_confidence_interval(bayesian_network_data_combined[1])
    },
    # "coverage": {
    #     "mean": np.mean(gc_metrics["coverage_mean"]),
    #     "std": np.std(gc_metrics["coverage_std"]),
    #     "interval": convert_string_to_list(gc_metrics["coverage_interval"])
    # },
    "sdv_continous_features_coverage": {
        "mean": np.mean(bayesian_network_data_combined[6]),
        "std": np.std(bayesian_network_data_combined[6]),
        "interval": calculate_confidence_interval(bayesian_network_data_combined[6])
    },
    "sdv_stat_sim_continous_features": {
        "mean": np.mean(bayesian_network_data_combined[7]),
        "std": np.std(bayesian_network_data_combined[7]),
        "interval": calculate_confidence_interval(bayesian_network_data_combined[7])
    },
    "sdv_newrow": {
        "mean": np.mean(bayesian_network_data_combined[8]),
        "std": np.std(bayesian_network_data_combined[8]),
        "interval": calculate_confidence_interval(bayesian_network_data_combined[8])
    },
    "kl_divergence": {
        "mean": np.mean(bayesian_network_data_combined[2]),
        "std": np.std(bayesian_network_data_combined[2]),
        "interval": calculate_confidence_interval(bayesian_network_data_combined[2])
    },
    "cross_classification": {
        "mean": np.mean(bayesian_network_data_combined[3]),
        "std": np.std(bayesian_network_data_combined[3]),
        "interval": calculate_confidence_interval(bayesian_network_data_combined[3])
    },
    "cross_regression": {
        "mean": np.mean(bayesian_network_data_combined[4]),
        "std": np.std(bayesian_network_data_combined[4]),
        "interval": calculate_confidence_interval(bayesian_network_data_combined[4])
    },
    # "sdv_diagnostic_score": {
    #     "mean": np.mean(gc_metrics["sdv_diagnostic_score_mean"]),
    #     "std":  float(gc_metrics["sdv_diagnostic_score_std"]),
    #     "interval": convert_string_to_list(gc_metrics["sdv_diagnostic_score_interval"])
    # },
    "sdv_quality_report_score": {
        "mean": np.mean(bayesian_network_data_combined[5]),
        "std": np.std(bayesian_network_data_combined[5]),
        "interval": calculate_confidence_interval(bayesian_network_data_combined[5])
    }
}

tvae_metrics = {
    "pairwise_correlation_difference": {
        # calcuare list of values for each metric

        "mean": np.mean(tvae_data_combined[0]),
        "std": np.std(tvae_data_combined[0]),
        "interval": calculate_confidence_interval(tvae_data_combined[0])
    },
    "cluster_measure": {
        "mean": np.mean(tvae_data_combined[1]),
        "std": np.std(tvae_data_combined[1]),
        "interval": calculate_confidence_interval(tvae_data_combined[1])
    },
    # "coverage": {
    #     "mean": np.mean(gc_metrics["coverage_mean"]),
    #     "std": np.std(gc_metrics["coverage_std"]),
    #     "interval": convert_string_to_list(gc_metrics["coverage_interval"])
    # },
    "sdv_continous_features_coverage": {
        "mean": np.mean(tvae_data_combined[6]),
        "std": np.std(tvae_data_combined[6]),
        "interval": calculate_confidence_interval(tvae_data_combined[6])
    },
    "sdv_stat_sim_continous_features": {
        "mean": np.mean(tvae_data_combined[7]),
        "std": np.std(tvae_data_combined[7]),
        "interval": calculate_confidence_interval(tvae_data_combined[7])
    },
    "sdv_newrow": {
        "mean": np.mean(tvae_data_combined[8]),
        "std": np.std(tvae_data_combined[8]),
        "interval": calculate_confidence_interval(tvae_data_combined[8])
    },
    "kl_divergence": {
        "mean": np.mean(tvae_data_combined[2]),
        "std": np.std(tvae_data_combined[2]),
        "interval": calculate_confidence_interval(tvae_data_combined[2])
    },
    "cross_classification": {
        "mean": np.mean(tvae_data_combined[3]),
        "std": np.std(tvae_data_combined[3]),
        "interval": calculate_confidence_interval(tvae_data_combined[3])
    },
    "cross_regression": {
        "mean": np.mean(tvae_data_combined[4]),
        "std": np.std(tvae_data_combined[4]),
        "interval": calculate_confidence_interval(tvae_data_combined[4])
    },
    # "sdv_diagnostic_score": {
    #     "mean": np.mean(gc_metrics["sdv_diagnostic_score_mean"]),
    #     "std":  float(gc_metrics["sdv_diagnostic_score_std"]),
    #     "interval": convert_string_to_list(gc_metrics["sdv_diagnostic_score_interval"])
    # },
    "sdv_quality_report_score": {
        "mean": np.mean(tvae_data_combined[5]),
        "std": np.std(tvae_data_combined[5]),
        "interval": calculate_confidence_interval(tvae_data_combined[5])
    }
}

rtvae_metrics = {
    "pairwise_correlation_difference": {
        # calcuare list of values for each metric

        "mean": np.mean(rtvae_data_combined[0]),
        "std": np.std(rtvae_data_combined[0]),
        "interval": calculate_confidence_interval(rtvae_data_combined[0])
    },
    "cluster_measure": {
        "mean": np.mean(rtvae_data_combined[1]),
        "std": np.std(rtvae_data_combined[1]),
        "interval": calculate_confidence_interval(rtvae_data_combined[1])
    },
    # "coverage": {
    #     "mean": np.mean(gc_metrics["coverage_mean"]),
    #     "std": np.std(gc_metrics["coverage_std"]),
    #     "interval": convert_string_to_list(gc_metrics["coverage_interval"])
    # },
    "sdv_continous_features_coverage": {
        "mean": np.mean(rtvae_data_combined[6]),
        "std": np.std(rtvae_data_combined[6]),
        "interval": calculate_confidence_interval(rtvae_data_combined[6])
    },
    "sdv_stat_sim_continous_features": {
        "mean": np.mean(rtvae_data_combined[7]),
        "std": np.std(rtvae_data_combined[7]),
        "interval": calculate_confidence_interval(rtvae_data_combined[7])
    },
    "sdv_newrow": {
        "mean": np.mean(rtvae_data_combined[8]),
        "std": np.std(rtvae_data_combined[8]),
        "interval": calculate_confidence_interval(rtvae_data_combined[8])
    },
    "kl_divergence": {
        "mean": np.mean(rtvae_data_combined[2]),
        "std": np.std(rtvae_data_combined[2]),
        "interval": calculate_confidence_interval(rtvae_data_combined[2])
    },
    "cross_classification": {
        "mean": np.mean(rtvae_data_combined[3]),
        "std": np.std(rtvae_data_combined[3]),
        "interval": calculate_confidence_interval(rtvae_data_combined[3])
    },
    "cross_regression": {
        "mean": np.mean(rtvae_data_combined[4]),
        "std": np.std(rtvae_data_combined[4]),
        "interval": calculate_confidence_interval(rtvae_data_combined[4])
    },
    # "sdv_diagnostic_score": {
    #     "mean": np.mean(gc_metrics["sdv_diagnostic_score_mean"]),
    #     "std":  float(gc_metrics["sdv_diagnostic_score_std"]),
    #     "interval": convert_string_to_list(gc_metrics["sdv_diagnostic_score_interval"])
    # },
    "sdv_quality_report_score": {
        "mean": np.mean(rtvae_data_combined[5]),
        "std": np.std(rtvae_data_combined[5]),
        "interval": calculate_confidence_interval(rtvae_data_combined[5])
    }
}

ddpm_metrics = {
    "pairwise_correlation_difference": {
        # calcuare list of values for each metric

        "mean": np.mean(ddpm_data_combined[0]),
        "std": np.std(ddpm_data_combined[0]),
        "interval": calculate_confidence_interval(ddpm_data_combined[0])
    },
    "cluster_measure": {
        "mean": np.mean(ddpm_data_combined[1]),
        "std": np.std(ddpm_data_combined[1]),
        "interval": calculate_confidence_interval(ddpm_data_combined[1])
    },
    # "coverage": {
    #     "mean": np.mean(gc_metrics["coverage_mean"]),
    #     "std": np.std(gc_metrics["coverage_std"]),
    #     "interval": convert_string_to_list(gc_metrics["coverage_interval"])
    # },
    "sdv_continous_features_coverage": {
        "mean": np.mean(ddpm_data_combined[6]),
        "std": np.std(ddpm_data_combined[6]),
        "interval": calculate_confidence_interval(ddpm_data_combined[6])
    },
    "sdv_stat_sim_continous_features": {
        "mean": np.mean(ddpm_data_combined[7]),
        "std": np.std(ddpm_data_combined[7]),
        "interval": calculate_confidence_interval(ddpm_data_combined[7])
    },
    "sdv_newrow": {
        "mean": np.mean(ddpm_data_combined[8]),
        "std": np.std(ddpm_data_combined[8]),
        "interval": calculate_confidence_interval(ddpm_data_combined[8])
    },
    "kl_divergence": {
        "mean": np.mean(ddpm_data_combined[2]),
        "std": np.std(ddpm_data_combined[2]),
        "interval": calculate_confidence_interval(ddpm_data_combined[2])
    },
    "cross_classification": {
        "mean": np.mean(ddpm_data_combined[3]),
        "std": np.std(ddpm_data_combined[3]),
        "interval": calculate_confidence_interval(ddpm_data_combined[3])
    },
    "cross_regression": {
        "mean": np.mean(ddpm_data_combined[4]),
        "std": np.std(ddpm_data_combined[4]),
        "interval": calculate_confidence_interval(ddpm_data_combined[4])
    },
    # "sdv_diagnostic_score": {
    #     "mean": np.mean(gc_metrics["sdv_diagnostic_score_mean"]),
    #     "std":  float(gc_metrics["sdv_diagnostic_score_std"]),
    #     "interval": convert_string_to_list(gc_metrics["sdv_diagnostic_score_interval"])
    # },
    "sdv_quality_report_score": {
        "mean": np.mean(ddpm_data_combined[5]),
        "std": np.std(ddpm_data_combined[5]),
        "interval": calculate_confidence_interval(ddpm_data_combined[5])
    }
}





# Create DataFrame for the confidence intervals (CI)
ci_data = {
    "Method CI": ["Gaussian Copula", "CTGAN", "BN", "TVAE", "RTVAE", "DDPM"],
    "PCD": [format_mean_and_interval(gc_metrics["pairwise_correlation_difference"]), 
            format_mean_and_interval(ctgan_metrics["pairwise_correlation_difference"]), 
            format_mean_and_interval(bayesian_network_metrics["pairwise_correlation_difference"]),
            format_mean_and_interval(tvae_metrics["pairwise_correlation_difference"]),
            format_mean_and_interval(rtvae_metrics["pairwise_correlation_difference"]),
            format_mean_and_interval(ddpm_metrics["pairwise_correlation_difference"])],
    "CM": [format_mean_and_interval(gc_metrics["cluster_measure"]), 
           format_mean_and_interval(ctgan_metrics["cluster_measure"]), 
           format_mean_and_interval(bayesian_network_metrics["cluster_measure"]),
           format_mean_and_interval(tvae_metrics["cluster_measure"]),
           format_mean_and_interval(rtvae_metrics["cluster_measure"]),
           format_mean_and_interval(ddpm_metrics["cluster_measure"])],
    "KL divergence": [format_mean_and_interval(gc_metrics["kl_divergence"]), 
                      format_mean_and_interval(ctgan_metrics["kl_divergence"]), 
                      format_mean_and_interval(bayesian_network_metrics["kl_divergence"]),
                      format_mean_and_interval(tvae_metrics["kl_divergence"]),
                      format_mean_and_interval(rtvae_metrics["kl_divergence"]),
                      format_mean_and_interval(ddpm_metrics["kl_divergence"])],
    "CrCl": [format_mean_and_interval(gc_metrics["cross_classification"]), 
             format_mean_and_interval(ctgan_metrics["cross_classification"]), 
             format_mean_and_interval(bayesian_network_metrics["cross_classification"]),
             format_mean_and_interval(tvae_metrics["cross_classification"]),
             format_mean_and_interval(rtvae_metrics["cross_classification"]),
             format_mean_and_interval(ddpm_metrics["cross_classification"])],
     "CrReg": [format_mean_and_interval(gc_metrics["cross_regression"]),
                         format_mean_and_interval(ctgan_metrics["cross_regression"]),
                         format_mean_and_interval(bayesian_network_metrics["cross_regression"]),
                         format_mean_and_interval(tvae_metrics["cross_regression"]),
                         format_mean_and_interval(rtvae_metrics["cross_regression"]),
                         format_mean_and_interval(ddpm_metrics["cross_regression"])],
    "Quality Score": [format_mean_and_interval(gc_metrics["sdv_quality_report_score"]),
                                 format_mean_and_interval(ctgan_metrics["sdv_quality_report_score"]),
                                 format_mean_and_interval(bayesian_network_metrics["sdv_quality_report_score"]),
                                 format_mean_and_interval(tvae_metrics["sdv_quality_report_score"]),
                                 format_mean_and_interval(rtvae_metrics["sdv_quality_report_score"]),
                                 format_mean_and_interval(ddpm_metrics["sdv_quality_report_score"])],
    "Continuous Features Coverage": [format_mean_and_interval(gc_metrics["sdv_continous_features_coverage"]),
                                         format_mean_and_interval(ctgan_metrics["sdv_continous_features_coverage"]),
                                         format_mean_and_interval(bayesian_network_metrics["sdv_continous_features_coverage"]),
                                         format_mean_and_interval(tvae_metrics["sdv_continous_features_coverage"]),
                                         format_mean_and_interval(rtvae_metrics["sdv_continous_features_coverage"]),
                                         format_mean_and_interval(ddpm_metrics["sdv_continous_features_coverage"])],
    "Stat Sim Continuous Features": [format_mean_and_interval(gc_metrics["sdv_stat_sim_continous_features"]),
                                         format_mean_and_interval(ctgan_metrics["sdv_stat_sim_continous_features"]),
                                         format_mean_and_interval(bayesian_network_metrics["sdv_stat_sim_continous_features"]),
                                         format_mean_and_interval(tvae_metrics["sdv_stat_sim_continous_features"]),
                                         format_mean_and_interval(rtvae_metrics["sdv_stat_sim_continous_features"]),
                                         format_mean_and_interval(ddpm_metrics["sdv_stat_sim_continous_features"])],
    "New Row": [format_mean_and_interval(gc_metrics["sdv_newrow"]),
                    format_mean_and_interval(ctgan_metrics["sdv_newrow"]),
                    format_mean_and_interval(bayesian_network_metrics["sdv_newrow"]),
                    format_mean_and_interval(tvae_metrics["sdv_newrow"]),
                    format_mean_and_interval(rtvae_metrics["sdv_newrow"]),
                    format_mean_and_interval(ddpm_metrics["sdv_newrow"])]
}

ci_df = pd.DataFrame(ci_data)

# Create DataFrame for the mean and standard deviation (Mean & STD)
# use 
mean_std_data = {
    "Method Mean & STD": ["GC-Mean & STD", "CTGAN-Mean & STD", "BN-Mean & STD", "TVAE-Mean & STD", "RTVAE-Mean & STD", "DDPM-Mean & STD"],
    "PCD": [format_mean_std(gc_metrics["pairwise_correlation_difference"]["mean"], gc_metrics["pairwise_correlation_difference"]["std"]),
            format_mean_std(ctgan_metrics["pairwise_correlation_difference"]["mean"], ctgan_metrics["pairwise_correlation_difference"]["std"]),
            format_mean_std(bayesian_network_metrics["pairwise_correlation_difference"]["mean"], bayesian_network_metrics["pairwise_correlation_difference"]["std"]),
            format_mean_std(tvae_metrics["pairwise_correlation_difference"]["mean"], tvae_metrics["pairwise_correlation_difference"]["std"]),
            format_mean_std(rtvae_metrics["pairwise_correlation_difference"]["mean"], rtvae_metrics["pairwise_correlation_difference"]["std"]),
            format_mean_std(ddpm_metrics["pairwise_correlation_difference"]["mean"], ddpm_metrics["pairwise_correlation_difference"]["std"])],
    "CM": [format_mean_std(gc_metrics["cluster_measure"]["mean"], gc_metrics["cluster_measure"]["std"]),
           format_mean_std(ctgan_metrics["cluster_measure"]["mean"], ctgan_metrics["cluster_measure"]["std"]),
           format_mean_std(bayesian_network_metrics["cluster_measure"]["mean"], bayesian_network_metrics["cluster_measure"]["std"]),
           format_mean_std(tvae_metrics["cluster_measure"]["mean"], tvae_metrics["cluster_measure"]["std"]),
           format_mean_std(rtvae_metrics["cluster_measure"]["mean"], rtvae_metrics["cluster_measure"]["std"]),
           format_mean_std(ddpm_metrics["cluster_measure"]["mean"], ddpm_metrics["cluster_measure"]["std"])],
    "KL divergence": [format_mean_std(gc_metrics["kl_divergence"]["mean"], gc_metrics["kl_divergence"]["std"]),
                      format_mean_std(ctgan_metrics["kl_divergence"]["mean"], ctgan_metrics["kl_divergence"]["std"]),
                      format_mean_std(bayesian_network_metrics["kl_divergence"]["mean"], bayesian_network_metrics["kl_divergence"]["std"]),
                      format_mean_std(tvae_metrics["kl_divergence"]["mean"], tvae_metrics["kl_divergence"]["std"]),
                      format_mean_std(rtvae_metrics["kl_divergence"]["mean"], rtvae_metrics["kl_divergence"]["std"]),
                      format_mean_std(ddpm_metrics["kl_divergence"]["mean"], ddpm_metrics["kl_divergence"]["std"])],
    "CrCl": [format_mean_std(gc_metrics["cross_classification"]["mean"], gc_metrics["cross_classification"]["std"]),
             format_mean_std(ctgan_metrics["cross_classification"]["mean"], ctgan_metrics["cross_classification"]["std"]),
             format_mean_std(bayesian_network_metrics["cross_classification"]["mean"], bayesian_network_metrics["cross_classification"]["std"]),
             format_mean_std(tvae_metrics["cross_classification"]["mean"], tvae_metrics["cross_classification"]["std"]),
             format_mean_std(rtvae_metrics["cross_classification"]["mean"], rtvae_metrics["cross_classification"]["std"]),
             format_mean_std(ddpm_metrics["cross_classification"]["mean"], ddpm_metrics["cross_classification"]["std"])],
    "CrReg": [format_mean_std(gc_metrics["cross_regression"]["mean"], gc_metrics["cross_regression"]["std"]),
                         format_mean_std(ctgan_metrics["cross_regression"]["mean"], ctgan_metrics["cross_regression"]["std"]),
                         format_mean_std(bayesian_network_metrics["cross_regression"]["mean"], bayesian_network_metrics["cross_regression"]["std"]),
                         format_mean_std(tvae_metrics["cross_regression"]["mean"], tvae_metrics["cross_regression"]["std"]),
                         format_mean_std(rtvae_metrics["cross_regression"]["mean"], rtvae_metrics["cross_regression"]["std"]),
                         format_mean_std(ddpm_metrics["cross_regression"]["mean"], ddpm_metrics["cross_regression"]["std"])],
    "Quality Score": [format_mean_std(gc_metrics["sdv_quality_report_score"]["mean"], gc_metrics["sdv_quality_report_score"]["std"]),
                                 format_mean_std(ctgan_metrics["sdv_quality_report_score"]["mean"], ctgan_metrics["sdv_quality_report_score"]["std"]),
                                 format_mean_std(bayesian_network_metrics["sdv_quality_report_score"]["mean"], bayesian_network_metrics["sdv_quality_report_score"]["std"]),
                                 format_mean_std(tvae_metrics["sdv_quality_report_score"]["mean"], tvae_metrics["sdv_quality_report_score"]["std"]),
                                 format_mean_std(rtvae_metrics["sdv_quality_report_score"]["mean"], rtvae_metrics["sdv_quality_report_score"]["std"]),
                                 format_mean_std(ddpm_metrics["sdv_quality_report_score"]["mean"], ddpm_metrics["sdv_quality_report_score"]["std"])],
    "Conti Col Coverage": [format_mean_std(gc_metrics["sdv_continous_features_coverage"]["mean"], gc_metrics["sdv_continous_features_coverage"]["std"]),
                                         format_mean_std(ctgan_metrics["sdv_continous_features_coverage"]["mean"], ctgan_metrics["sdv_continous_features_coverage"]["std"]),
                                         format_mean_std(bayesian_network_metrics["sdv_continous_features_coverage"]["mean"], bayesian_network_metrics["sdv_continous_features_coverage"]["std"]),
                                         format_mean_std(tvae_metrics["sdv_continous_features_coverage"]["mean"], tvae_metrics["sdv_continous_features_coverage"]["std"]),
                                         format_mean_std(rtvae_metrics["sdv_continous_features_coverage"]["mean"], rtvae_metrics["sdv_continous_features_coverage"]["std"]),
                                         format_mean_std(ddpm_metrics["sdv_continous_features_coverage"]["mean"], ddpm_metrics["sdv_continous_features_coverage"]["std"])],
    "Sim Conti Col": [format_mean_std(gc_metrics["sdv_stat_sim_continous_features"]["mean"], gc_metrics["sdv_stat_sim_continous_features"]["std"]),
                                         format_mean_std(ctgan_metrics["sdv_stat_sim_continous_features"]["mean"], ctgan_metrics["sdv_stat_sim_continous_features"]["std"]),
                                         format_mean_std(bayesian_network_metrics["sdv_stat_sim_continous_features"]["mean"], bayesian_network_metrics["sdv_stat_sim_continous_features"]["std"]),
                                         format_mean_std(tvae_metrics["sdv_stat_sim_continous_features"]["mean"], tvae_metrics["sdv_stat_sim_continous_features"]["std"]),
                                         format_mean_std(rtvae_metrics["sdv_stat_sim_continous_features"]["mean"], rtvae_metrics["sdv_stat_sim_continous_features"]["std"]),
                                         format_mean_std(ddpm_metrics["sdv_stat_sim_continous_features"]["mean"], ddpm_metrics["sdv_stat_sim_continous_features"]["std"])],
    "New Row": [format_mean_std(gc_metrics["sdv_newrow"]["mean"], gc_metrics["sdv_newrow"]["std"]),
                    format_mean_std(ctgan_metrics["sdv_newrow"]["mean"], ctgan_metrics["sdv_newrow"]["std"]),
                    format_mean_std(bayesian_network_metrics["sdv_newrow"]["mean"], bayesian_network_metrics["sdv_newrow"]["std"]),
                    format_mean_std(tvae_metrics["sdv_newrow"]["mean"], tvae_metrics["sdv_newrow"]["std"]),
                    format_mean_std(rtvae_metrics["sdv_newrow"]["mean"], rtvae_metrics["sdv_newrow"]["std"]),
                    format_mean_std(ddpm_metrics["sdv_newrow"]["mean"], ddpm_metrics["sdv_newrow"]["std"])]
}

mean_std_df = pd.DataFrame(mean_std_data)




# Render the confidence intervals table
render_mpl_table(ci_df, header_columns=0, col_width=3.0)
plt.savefig(f'{main_dir}/confidence_intervals_v7.png', bbox_inches='tight')
# Clear the plot
# plt.clf()

# Render the mean and standard deviation table
render_mpl_table(mean_std_df, header_columns=0, col_width=3.0)
plt.savefig(f'{main_dir}/mean_std_v7.png', bbox_inches='tight')


# confidence intervals visualization
metrics = ['PCD', 'CM', 'KL divergence', 'CrCl', 'CrReg', 'Quality Score', 
           'Continuous Features Coverage', 'Stat Sim Continuous Features', 'New Row']

for metric in metrics:
    plot_confidence_intervals(ci_df, metric, f'{main_dir}/{metric}_95PercentCI.png')

print("Done")


