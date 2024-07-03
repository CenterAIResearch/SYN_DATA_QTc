import os
import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.utils import check_X_y
from sklearn.metrics import make_scorer, confusion_matrix, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score
from sklearn.model_selection import cross_validate, StratifiedKFold
import sys


if __name__ == '__main__':
    print("Python version:",sys.version)

    print("sklearn version",sklearn.__version__)

    model_pickle_file = "model_668564998c68f70031896bd9_RFC_Origin.pkl"

    pickle_file = f'/appsrc/machine/test_trained_models_PSB25_Reproduction/Models_pkl/{model_pickle_file}'

    pickle_file_data_pickle_model = pickle_file.split('/')[-1].split('.')[0]
    print("pickle_file_data_pickle_model",pickle_file_data_pickle_model)




    dataset = '/appsrc/machine/test_trained_models_PSB25_Reproduction/DATAPOOL_z_test/newz_test.csv'  # 


    target_column = 'ecg_manual_qtcb_binary'
    seed = 42

    # load fitted model
    pickle_model = joblib.load(pickle_file)
    model = pickle_model['model']

    # if model is logistic regression, print parameters C, Dual, Penalty, AND Fit Intercept
    print("Model Parameters:", model.get_params())

    # print input features dimension
    if hasattr(model, 'coef_'):
        input_features_dim = model.coef_.shape[1]
    elif hasattr(model, 'feature_importances_'):
        input_features_dim = model.feature_importances_.shape[0]
    else:
        input_features_dim = "Unknown"

    print("Input feature dimension:", input_features_dim)

    # read input data
    input_data = pd.read_csv(dataset, sep=None, engine='python')

    print("input_data",input_data)

    input_data = pd.read_csv(dataset, sep=None, engine='python')
    input_data = input_data.dropna()


    # reproducing training score and testing score from Aliro
    features = input_data.drop(target_column, axis=1).values
    target = input_data[target_column].values
    # predict using features
    predicted = model.predict(features)

    # instead of predict, use predict_proba
    predicted_proba = model.predict_proba(features)


    print("predicted",predicted)
    print("target",target)
    print("predicted_proba",predicted_proba)
    # print("predicted_proba_threshold",predicted_proba_threshold)

    # Confusion Matrix
    # Compute and display the confusion matrix
    cm = confusion_matrix(target, predicted)
    # Display confusion matrix with labels for better understanding
    print("\nConfusion Matrix with Labels:")
    print(pd.DataFrame(cm, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive']))
    print("\n")

    # Precision, Recall, F1-Score
    precision = precision_score(target, predicted, average=None)
    recall = recall_score(target, predicted, average=None)
    f1 = f1_score(target, predicted, average=None)

    print("Class 0 - TP Rate (Recall):", recall[0])
    print("Class 0 - Precision:", precision[0])
    print("Class 0 - Recall:", recall[0])
    print("Class 0 - F-Measure:", f1[0])

    print("Class 1 - TP Rate (Recall):", recall[1])
    print("Class 1 - Precision:", precision[1])
    print("Class 1 - Recall:", recall[1])
    print("Class 1 - F-Measure:", f1[1])

    # Overall F-Measure
    overall_f_measure = f1_score(target, predicted, average='weighted')
    print("Overall F-Measure:", overall_f_measure)

    # Matthews Correlation Coefficient
    mcc = matthews_corrcoef(target, predicted)
    print("MCC:", mcc)

    # ROC Area, PRC Area
    roc_area = roc_auc_score(target, predicted)
    prc_area = average_precision_score(target, predicted)
    print("ROC Area:", roc_area)
    print("PRC Area:", prc_area)

    tn, fp, fn, tp = confusion_matrix(target, predicted).ravel()
    print("TN:", tn)
    print("FP:", fp)
    print("FN:", fn)
    print("TP:", tp)


    # balanced accuracy using tn, fp, fn, tp
    balanced_accuracy = (tp / (tp + fn) + tn / (tn + fp)) / 2

    sensitivity = tp / (tp + fn)
    ppv = tp / (tp + fp)
    fnr = fn / (fn + tp)
    forr = fn / (fn + tn)
    fdr = fp / (fp + tp)
    fpr = fp / (fp + tn)
    specificity = tn / (tn + fp)
    npv = tn / (tn + fn)

    f2_score = 5 * (precision * sensitivity) / (4 * precision + sensitivity)

    print("Balanced Accuracy:", balanced_accuracy)
    print("Sensitivity:", sensitivity)
    print("PPV:", ppv)
    print("FNR:", fnr)
    print("FOR:", forr)
    print("FDR:", fdr)
    print("FPR:", fpr)
    print("Specificity:", specificity)
    print("NPV:", npv)
    print("F2-Score:", f2_score)


    # Save below metrics to a file
    # ROC AUC, PRC AUC, Balanced Accuracy, F-Measure, Sensitivity, Specificity, PPV, NPV



    results = {
        "model_name": model_pickle_file,
        "ROC AUC": roc_area,
        "PRC AUC": prc_area,
        "Balanced Accuracy": balanced_accuracy,
        "F1-Measure": overall_f_measure,
        # "F2-Measure": f2_score,
        # calculate all metrics using tn, fp, fn, tp
        "Sensitivity": sensitivity,
        "PPV": ppv,

        "False Negative Rate": fnr,
        "False Omission Rate": forr,

        "False Discovery Rate": fdr,
        "False Positive Rate": fpr,

        "Specificity": specificity,
        "NPV": npv,

        "MCC": mcc,
    }   

    # Convert the results dictionary to a DataFrame. Using [results] to create a single row DataFrame.
    results_df = pd.DataFrame([results])

    # file_name = f'results_Syn_data_BN_Origin_data.csv'
    # file_name = f'results_Comb_data_BN_Origin_data.csv'
    file_name = f'results.csv'

    if os.path.exists(file_name):
        # File exists, append without writing the header
        results_df.to_csv(file_name, mode='a', index=False, header=False)
    else:
        # File doesn't exist, write with header
        results_df.to_csv(file_name, mode='w', index=False, header=True)