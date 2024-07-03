import os
import numpy as np
import pandas as pd
import joblib
from sklearn.utils import check_X_y
from sklearn.metrics import make_scorer, confusion_matrix, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score
from sklearn.model_selection import cross_validate, StratifiedKFold
import math


if __name__ == '__main__':
    # print python version
    import sys
    print("Python version:",sys.version)

    # print sklearn version
    import sklearn
    print("sklearn version",sklearn.__version__)



    # best models (Origin data)
    # best DTC
    pickle_file_data="origin_0625"
    pickle_file = f'/appsrc/machine/test_trained_models/picklefiles/{pickle_file_data}/model_66536fbd0d6aaf00316dceb4_DTC.pkl'




    pickle_file_data_pickle_model = pickle_file.split('/')[-1].split('.')[0]
    print("pickle_file_data_pickle_model",pickle_file_data_pickle_model)

    # if pickle_file_data == "origin"
    # if pickle_file_data includes "origin" or "origin_0625":
    if pickle_file_data == "origin" or pickle_file_data == "origin_0625" or pickle_file_data == "origin_0626":
        model_name = pickle_file.split('/')[-1].split('.')[0]
        model_name = model_name.split('_')[-1]
        print("Model Name:", model_name)

    elif pickle_file_data == "Syn" or pickle_file_data == "SMOTE" or pickle_file_data == "CTGAN" or pickle_file_data == "Syn_GauCopula" or pickle_file_data == "Syn_GauCopula_112+112" :
        model_name = pickle_file.split('/')[-1].split('.')[0]
        model_name = model_name.split('_')[-2]
        print("Model Name:", model_name)
    elif pickle_file_data == "EXP2-Syn_BN_SYN_500-1000" or pickle_file_data == "EXP3-Comb_Real_Syn_BN_SYN_500-1000":
        # for example, from pickle_file=='/appsrc/machine/test_trained_models/picklefiles/Syn_BN_SYN_500-1000/SYN_5000/model_667b36365ba83f0038e01e04_SVC_SYN_5000.pkl', please get 'SVC'
        # Get the file name without the directory path
        file_name = pickle_file.split('/')[-1]

        # Get the part of the file name without the extension
        model_name = file_name.split('.')[0]
        
        # Get the model type by splitting by underscores and selecting the second last element
        # /appsrc/machine/test_trained_models/picklefiles/EXP3-Comb_Real_Syn_BN_SYN_500-1000/Real_SYN_588/model_667c39585ba83f0038e02150_GBC_Real_SYN_588
        # for example SVC_SYN_5000 is the model_name.
        model_name = '_'.join(model_name.split('_')[-4:])
        print("Model Name:", model_name)



    dataset = '/appsrc/machine/test_trained_models/DATAPOOL_z_tr/newz_test.csv'  # Adjusted for Colab file path
    # dataset = '/appsrc/machine/test_trained_models/DATAPOOL_z_tr/newz_train.csv'  # Adjusted for Colab file path


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
    # 

    input_data = pd.read_csv(dataset, sep=None, engine='python')
    input_data = input_data.dropna()


    # reproducing training score and testing score from Aliro
    features = input_data.drop(target_column, axis=1).values
    target = input_data[target_column].values
    # predict using features
    predicted = model.predict(features)

    # instead of predict, use predict_proba
    predicted_proba = model.predict_proba(features)


    # make classification using predictied_proba using threshold 0.5



    print("predicted",predicted)
    print("target",target)
    print("predicted_proba",predicted_proba)
    # print("predicted_proba_threshold",predicted_proba_threshold)

    proba_th=1
    while proba_th >= 0:
        predicted_proba_threshold = np.where(predicted_proba[:,0] > proba_th, 0, 1)
        print("predicted_proba_threshold",predicted_proba_threshold)
        proba_th = proba_th - 0.005

    # predicted_proba_threshold = np.where(predicted_proba[:,0] > 0.4, 1, 0)


    # print pickle_file
    print(pickle_file)

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
        "model_name": model_name,
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

    # file_name = f'results_{pickle_file_data}.csv'
    # file_name = f'results_Syn_data_BN_Origin_data.csv'
    # file_name = f'results_Comb_data_BN_Origin_data.csv'
    file_name = f'results_temp.csv'

    if os.path.exists(file_name):
        # File exists, append without writing the header
        results_df.to_csv(file_name, mode='a', index=False, header=False)
    else:
        # File doesn't exist, write with header
        results_df.to_csv(file_name, mode='w', index=False, header=True)