###############################################################################
# Copyright (c) 2021, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory
# Written by Andre Goncalves <andre@llnl.gov>
#
# All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###############################################################################

import random
import scipy
import numpy as np
import pandas as pd

from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestNeighbors
import sklearn.metrics as metrics

from sklearn.neighbors import KernelDensity

from scipy.stats import entropy

# Linear regression
from sklearn.linear_model import LinearRegression
# MLP Regressor
from sklearn.neural_network import MLPRegressor


np.random.seed(1234)


#################################################
# HELPER FUNCTIONS
#################################################
def align_columns(df_r, df_s):
    """ Helper function to make sure two 
        dataFrames have their columns aligned. """
    df_r.sort_index(axis=1, inplace=True)
    df_s.sort_index(axis=1, inplace=True)

    def checkEqual(L1, L2):
        return len(L1) == len(L2) and sorted(L1) == sorted(L2)
    assert checkEqual(df_r.columns.tolist(), df_s.columns.tolist())
    return df_r, df_s


#################################################
# DISCLOSURE METRICS
#################################################
def membership_disclosure(data_a, data_b, r=np.inf, delta=0.1):
    """ Membership disclosure metric. It occurs when an attacker
        can determine that the statistical/ML method that generated
        the synthetic data was trained with a dataset including the
        record from patient x.
        
        Choi E, Biswal S, Malin B, Duke J, Stewart WF, Sun J. 
        Generating multi-label discrete patient records using generative
        adversarial networks. In: Machine Learning for Healthcare Conference;
        2017. p. 286–305: http://proceedings.mlr.press/v68/choi17a.html

    Args:
        data_a (pd.DataFrame): Real dataset.
        data_b (pd.DataFrame): Synthetic dataset.
        r (int, optional): Number of selected records to compute 
                           membership disclosure. 
                           Defaults to np.inf (all samples).
        delta (float, optional): Hamming distance threshold to 
                                 determine if synthetic sample 
                                 is "too close" to a real sample.
                                 Defaults to 0.1.

    Returns:
        float: Membership disclosure value.
    """
    if not np.isinf(r):
        idx = np.random.choice(np.arange(data_a.shape[0]), r)
        df_real_subset = data_a.iloc[idx, :]
        df_syn_subset = data_b.iloc[idx, :]
    else:
        df_real_subset = data_a
        df_syn_subset = data_b

    neigh = NearestNeighbors(radius=delta, metric='hamming')
    neigh.fit(df_syn_subset.values.astype(int))
    _, indices = neigh.radius_neighbors(df_real_subset.values.astype(int))
    revealed = 0

    for i in range(df_real_subset.shape[0]):
        if indices[i].shape[0] > 0:
            revealed += 1
    revealed = (revealed/df_real_subset.shape[0])
    return {'-': revealed}


def percentage_revealed(data_a, data_b):
    """ Helper function: """
    df_r, df_s = align_columns(data_a, data_b)
    df_all = df_r.merge(df_s.drop_duplicates(),
                        on=df_r.columns.tolist(),
                        how='left', indicator=True)
    df_all['_merge'] == 'left_only'
    both = df_all[df_all['_merge'] == 'both'].shape[0]
    revealed = (both / df_all['_merge'].shape[0])

    return {'-': revealed}


def attribute_disclosure(data_a, data_b, delta=0.1, k=5):
    """Attribute disclosure refers to the risk of an attacker correctly
       inferring sensitive attributes of a patient record (e.g., results of
       medical tests, medications, and diagnoses) based on a subset of 
       attributes known to the attacker.

    Args:
        data_a (pd.DataFrame): Real dataset.
        data_b (pd.DataFrame): Synthetic dataset.
        delta (float, optional): Hamming distance threshold to 
                                 determine if synthetic sample 
                                 is "too close" to a real sample.
                                 Defaults to 0.1.
        k (int): Number of attributes known by the attacker.

    Returns:
        float: Attribute disclosure value.
    """    

    assert delta >= 0 and delta <=1, "Delta should be in [0, 1] interval."
    
    data_real, data_synth = align_columns(data_a, data_b)

    columns = data_real.columns.tolist()
    sample = data_real.drop_duplicates()

    inferred = pd.DataFrame(data=np.nan*np.ones_like(sample),
                            columns=sample.columns)
    inferred.reset_index(drop=True, inplace=True)
    
    # TODO: run with many many different splits or
    #       use a fixed split, as used in the paper
    known_atts = random.sample(columns, k)
    unknown_atts = list(set(columns) - set(known_atts))

    neigh = NearestNeighbors(radius=delta, metric='hamming')        
    neigh.fit(data_synth[known_atts].values)
    _, indices = neigh.radius_neighbors(sample[known_atts].values)

    assert len(indices) == sample.shape[0], "It should return one for each sample."
    inferred[known_atts] = sample[known_atts].values

    for index, indice in enumerate(indices):
        nb_neighbors = np.minimum(k, indice.shape[0])
        if nb_neighbors > 0:
            idx = indice[:nb_neighbors]
            pred = data_synth.loc[idx, unknown_atts].mode()
            pred.reset_index(drop=True, inplace=True)
            inferred.loc[index, unknown_atts] = pred.values[0]
        else:
            inferred.loc[index, unknown_atts] = -1
    prctrev = percentage_revealed(sample.astype(int), inferred.astype(int))
    return {'-': prctrev}

#################################################
# UTILITY METRICS
#################################################
def cross_classification(data_a, data_b, 
                         base_metric='accuracy',
                         class_method='DT'):
    """ Classification error normalized by the error
        in the hold-out (real) set. """

    data_a, data_b = align_columns(data_a, data_b)

    # data_a: real dataset
    # data_b: synthetic dataset
    shfl_ids = np.random.permutation(range(data_a.shape[0]))
    n_train = int(data_a.shape[0] * 0.6)

    accs = dict()  # stores performances for all columns

    for col_k in data_a.columns:
        x_train = data_a.loc[shfl_ids[:n_train],
                             data_a.columns != col_k].values
        y_train = data_a.loc[shfl_ids[:n_train],
                             [col_k]].values.ravel()

        x_test = data_a.loc[shfl_ids[n_train:],
                            data_a.columns != col_k].values
        y_test = data_a.loc[shfl_ids[n_train:],
                            [col_k]].values.ravel()

        x_synth = data_b.loc[:, data_b.columns != col_k].values
        y_synth = data_b.loc[:, [col_k]].values.ravel()

        if class_method == 'DT':
            #
            # Tree-based model classifier
            #
            try:
                clf = tree.DecisionTreeClassifier(max_depth=10, min_samples_leaf=5)
                clf = clf.fit(x_train, y_train)

                if base_metric == 'accuracy':
                    on_holdout = clf.score(x_test, y_test)
                    on_synth = clf.score(x_synth, y_synth)
                elif base_metric == 'balanced_accuracy':
                    yhat = clf.predict(x_test).astype(int)
                    on_holdout = metrics.balanced_accuracy_score(y_test, yhat)
                    yhat = clf.predict(x_synth).astype(int)
                    on_synth = metrics.balanced_accuracy_score(y_synth, yhat)
                elif base_metric == 'f1_score':
                    yhat = clf.predict(x_test).astype(int)
                    on_holdout = metrics.f1_score(y_test, yhat, average='macro')
                    yhat = clf.predict(x_synth).astype(int)
                    on_synth = metrics.f1_score(y_synth, yhat, average='macro')
                elif base_metric == 'auc':
                    yhat = clf.predict(x_test).astype(int)
                    fpr, tpr, thresholds = metrics.roc_curve(y_test, yhat)
                    on_holdout = metrics.auc(fpr, tpr)
                    yhat = clf.predict(x_synth).astype(int)
                    fpr, tpr, thresholds = metrics.roc_curve(y_synth, yhat)
                    on_synth = metrics.auc(fpr, tpr)
                else:
                    raise NotImplementedError('Unknown base metric {}.'.format(base_metric))

            except ValueError:
                # metric can not be computed
                on_holdout = np.NaN
                on_synth = np.NaN

        elif class_method == 'LR':
            #
            # Logistic Regression classifier
            #
            try:
                clf = LogisticRegression(solver='lbfgs', multi_class='multinomial')
                clf = clf.fit(x_train, y_train.ravel())

                if base_metric == 'accuracy':
                    on_holdout = clf.score(x_test, y_test)
                    on_synth = clf.score(x_synth, y_synth)
                elif base_metric == 'f1_score':
                    yhat = clf.predict(x_test).astype(int)
                    on_holdout = metrics.f1_score(y_test, yhat, average='macro')
                    yhat = clf.predict(x_synth).astype(int)
                    on_synth = metrics.f1_score(y_synth, yhat, average='macro')
                elif base_metric == 'auc':
                    yhat = clf.predict(x_test).astype(int)
                    fpr, tpr, thresholds = metrics.roc_curve(y_test, yhat)
                    on_holdout = metrics.auc(fpr, tpr)
                    yhat = clf.predict(x_synth).astype(int)
                    fpr, tpr, thresholds = metrics.roc_curve(y_synth, yhat)
                    on_synth = metrics.auc(fpr, tpr)

            except ValueError:
                # metric can not be computed
                on_holdout = np.NaN
                on_synth = np.NaN

        elif class_method == 'MLP':
            #
            # Multilayer Perceptron Classifier
            #
            try:
                clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                                    hidden_layer_sizes=(64, 32), random_state=1)
                clf = clf.fit(x_train, y_train.ravel())

                if base_metric == 'accuracy':
                    on_holdout = clf.score(x_test, y_test)
                    on_synth = clf.score(x_synth, y_synth)
                elif base_metric == 'f1_score':
                    yhat = clf.predict(x_test).astype(int)
                    on_holdout = metrics.f1_score(y_test, yhat, average='macro')
                    yhat = clf.predict(x_synth).astype(int)
                    on_synth = metrics.f1_score(y_synth, yhat, average='macro')
                elif base_metric == 'auc':
                    yhat = clf.predict(x_test).astype(int)
                    fpr, tpr, thresholds = metrics.roc_curve(y_test, yhat)
                    on_holdout = metrics.auc(fpr, tpr)
                    yhat = clf.predict(x_synth).astype(int)
                    fpr, tpr, thresholds = metrics.roc_curve(y_synth, yhat)
                    on_synth = metrics.auc(fpr, tpr)

            except ValueError:
                # metric can not be computed
                on_holdout = np.NaN
                on_synth = np.NaN
        else:
            raise('Unknown classification method')
        
        # if on_holdout is not nan and on_synth is not nan
        if not np.isnan(on_holdout) and not np.isnan(on_synth):
            print("on_holdout and on_synth are not nan")
        accs[col_k] = on_synth / on_holdout

    # if want to return only mean, then use this. If one wants
    # to return crcl for every variable, just return 'accs'
    avg_crcl = {'Avg-CrCl': sum([accs[k] for k in accs.keys()])/len(accs)}

    return accs


def cross_classification_edited(data_a, data_b, 
                         base_metric='accuracy',
                         class_method='DT'):
    """ Classification error normalized by the error
        in the hold-out (real) set. """

    data_a, data_b = align_columns(data_a, data_b)

    # data_a: real dataset
    # data_b: synthetic dataset
    shfl_ids = np.random.permutation(range(data_a.shape[0]))
    n_train = int(data_a.shape[0] * 0.6)

    accs = dict()  # stores performances for all columns

    for col_k in data_a.columns:
        x_train = data_a.loc[shfl_ids[:n_train],
                             data_a.columns != col_k].values
        y_train = data_a.loc[shfl_ids[:n_train],
                             [col_k]].values.ravel()

        x_test = data_a.loc[shfl_ids[n_train:],
                            data_a.columns != col_k].values
        y_test = data_a.loc[shfl_ids[n_train:],
                            [col_k]].values.ravel()

        x_synth = data_b.loc[:, data_b.columns != col_k].values
        y_synth = data_b.loc[:, [col_k]].values.ravel()

        if class_method == 'DT':
            #
            # Tree-based model classifier
            #
            try:
                clf = tree.DecisionTreeClassifier(max_depth=10, min_samples_leaf=5)
                clf = clf.fit(x_train, y_train)

                if base_metric == 'accuracy':
                    on_holdout = clf.score(x_test, y_test)
                    on_synth = clf.score(x_synth, y_synth)
                elif base_metric == 'balanced_accuracy':
                    yhat = clf.predict(x_test).astype(int)
                    on_holdout = metrics.balanced_accuracy_score(y_test, yhat)
                    yhat = clf.predict(x_synth).astype(int)
                    on_synth = metrics.balanced_accuracy_score(y_synth, yhat)
                elif base_metric == 'f1_score':
                    yhat = clf.predict(x_test).astype(int)
                    on_holdout = metrics.f1_score(y_test, yhat, average='macro')
                    yhat = clf.predict(x_synth).astype(int)
                    on_synth = metrics.f1_score(y_synth, yhat, average='macro')
                elif base_metric == 'auc':
                    yhat = clf.predict(x_test).astype(int)
                    fpr, tpr, thresholds = metrics.roc_curve(y_test, yhat)
                    on_holdout = metrics.auc(fpr, tpr)
                    yhat = clf.predict(x_synth).astype(int)
                    fpr, tpr, thresholds = metrics.roc_curve(y_synth, yhat)
                    on_synth = metrics.auc(fpr, tpr)
                else:
                    raise NotImplementedError('Unknown base metric {}.'.format(base_metric))

            except ValueError:
                # metric can not be computed
                on_holdout = np.NaN
                on_synth = np.NaN

        elif class_method == 'LR':
            #
            # Logistic Regression classifier
            #
            try:
                clf = LogisticRegression(solver='lbfgs', multi_class='multinomial')
                clf = clf.fit(x_train, y_train.ravel())

                if base_metric == 'accuracy':
                    on_holdout = clf.score(x_test, y_test)
                    on_synth = clf.score(x_synth, y_synth)
                elif base_metric == 'balanced_accuracy':
                    yhat = clf.predict(x_test).astype(int)
                    on_holdout = metrics.balanced_accuracy_score(y_test, yhat)
                    yhat = clf.predict(x_synth).astype(int)
                    on_synth = metrics.balanced_accuracy_score(y_synth, yhat)
                elif base_metric == 'f1_score':
                    yhat = clf.predict(x_test).astype(int)
                    on_holdout = metrics.f1_score(y_test, yhat, average='macro')
                    yhat = clf.predict(x_synth).astype(int)
                    on_synth = metrics.f1_score(y_synth, yhat, average='macro')
                elif base_metric == 'auc':
                    yhat = clf.predict(x_test).astype(int)
                    fpr, tpr, thresholds = metrics.roc_curve(y_test, yhat)
                    on_holdout = metrics.auc(fpr, tpr)
                    yhat = clf.predict(x_synth).astype(int)
                    fpr, tpr, thresholds = metrics.roc_curve(y_synth, yhat)
                    on_synth = metrics.auc(fpr, tpr)

            except ValueError:
                # metric can not be computed
                on_holdout = np.NaN
                on_synth = np.NaN

        elif class_method == 'MLP':
            #
            # Multilayer Perceptron Classifier
            #
            try:
                clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                                    hidden_layer_sizes=(64, 32), random_state=1)
                clf = clf.fit(x_train, y_train.ravel())

                if base_metric == 'accuracy':
                    on_holdout = clf.score(x_test, y_test)
                    on_synth = clf.score(x_synth, y_synth)
                elif base_metric == 'balanced_accuracy':
                    yhat = clf.predict(x_test).astype(int)
                    on_holdout = metrics.balanced_accuracy_score(y_test, yhat)
                    yhat = clf.predict(x_synth).astype(int)
                    on_synth = metrics.balanced_accuracy_score(y_synth, yhat)
                elif base_metric == 'f1_score':
                    yhat = clf.predict(x_test).astype(int)
                    on_holdout = metrics.f1_score(y_test, yhat, average='macro')
                    yhat = clf.predict(x_synth).astype(int)
                    on_synth = metrics.f1_score(y_synth, yhat, average='macro')
                elif base_metric == 'auc':
                    yhat = clf.predict(x_test).astype(int)
                    fpr, tpr, thresholds = metrics.roc_curve(y_test, yhat)
                    on_holdout = metrics.auc(fpr, tpr)
                    yhat = clf.predict(x_synth).astype(int)
                    fpr, tpr, thresholds = metrics.roc_curve(y_synth, yhat)
                    on_synth = metrics.auc(fpr, tpr)

            except ValueError:
                # metric can not be computed
                on_holdout = np.NaN
                on_synth = np.NaN
        else:
            raise('Unknown classification method')
        
        # if on_holdout is not nan and on_synth is not nan
        if not np.isnan(on_holdout) and not np.isnan(on_synth):
            print("on_holdout: ", on_holdout)
            print("on_synth: ", on_synth)
        accs[col_k] = on_synth / on_holdout

    # if want to return only mean, then use this. If one wants
    # to return crcl for every variable, just return 'accs'
    avg_crcl = {'Avg-CrCl': sum([accs[k] for k in accs.keys()])/len(accs)}

    return accs

def cross_regression_edited(data_a, data_b, 
                            base_metric='mean_squared_error',
                            reg_method='DT'):
    """ Regression error normalized by the error
        in the hold-out (real) set. """

    data_a, data_b = align_columns(data_a, data_b)

    # data_a: real dataset
    # data_b: synthetic dataset
    shfl_ids = np.random.permutation(range(data_a.shape[0]))
    n_train = int(data_a.shape[0] * 0.6)

    metrics_dict = dict()  # stores performances for all columns

    for col_k in data_a.columns:
        x_train = data_a.loc[shfl_ids[:n_train],
                             data_a.columns != col_k].values
        y_train = data_a.loc[shfl_ids[:n_train],
                             [col_k]].values.ravel()

        x_test = data_a.loc[shfl_ids[n_train:],
                            data_a.columns != col_k].values
        y_test = data_a.loc[shfl_ids[n_train:],
                            [col_k]].values.ravel()

        x_synth = data_b.loc[:, data_b.columns != col_k].values
        y_synth = data_b.loc[:, [col_k]].values.ravel()

        if reg_method == 'DT':
            #
            # Tree-based model regressor
            #
            try:
                reg = tree.DecisionTreeRegressor(max_depth=10, min_samples_leaf=5)
                reg = reg.fit(x_train, y_train)

                if base_metric == 'mean_squared_error':
                    on_holdout = metrics.mean_squared_error(y_test, reg.predict(x_test))
                    on_synth = metrics.mean_squared_error(y_synth, reg.predict(x_synth))
                elif base_metric == 'mean_absolute_error':
                    on_holdout = metrics.mean_absolute_error(y_test, reg.predict(x_test))
                    on_synth = metrics.mean_absolute_error(y_synth, reg.predict(x_synth))
                elif base_metric == 'r2_score':
                    on_holdout = metrics.r2_score(y_test, reg.predict(x_test))
                    on_synth = metrics.r2_score(y_synth, reg.predict(x_synth))
                    if on_holdout !=0:
                        # if on_holdout > 1 or on_synth > 1:
                        if on_synth / on_holdout >10:
                            print(on_synth / on_holdout)
                            print("on_holdout: ", on_holdout)
                            print("on_synth: ", on_synth)
                else:
                    raise NotImplementedError('Unknown base metric {}.'.format(base_metric))

            except ValueError:
                # metric can not be computed
                on_holdout = np.NaN
                on_synth = np.NaN

        elif reg_method == 'LR':
            #
            # Linear Regression
            #
            try:
                reg = LinearRegression()
                reg = reg.fit(x_train, y_train.ravel())

                if base_metric == 'mean_squared_error':
                    on_holdout = metrics.mean_squared_error(y_test, reg.predict(x_test))
                    on_synth = metrics.mean_squared_error(y_synth, reg.predict(x_synth))
                elif base_metric == 'mean_absolute_error':
                    on_holdout = metrics.mean_absolute_error(y_test, reg.predict(x_test))
                    on_synth = metrics.mean_absolute_error(y_synth, reg.predict(x_synth))
                elif base_metric == 'r2_score':
                    on_holdout = metrics.r2_score(y_test, reg.predict(x_test))
                    on_synth = metrics.r2_score(y_synth, reg.predict(x_synth))

                     # if on_holdout > 1 or on_synth > 1:
                    if on_holdout > 1 or on_synth > 1:
                        print("on_holdout: ", on_holdout)
                        print("on_synth: ", on_synth)
                else:
                    raise NotImplementedError('Unknown base metric {}.'.format(base_metric))

            except ValueError:
                # metric can not be computed
                on_holdout = np.NaN
                on_synth = np.NaN

        elif reg_method == 'MLP':
            #
            # Multilayer Perceptron Regressor
            #
            try:
                reg = MLPRegressor(solver='lbfgs', alpha=1e-5,
                                   hidden_layer_sizes=(64, 32), random_state=1)
                reg = reg.fit(x_train, y_train.ravel())

                if base_metric == 'mean_squared_error':
                    on_holdout = metrics.mean_squared_error(y_test, reg.predict(x_test))
                    on_synth = metrics.mean_squared_error(y_synth, reg.predict(x_synth))
                elif base_metric == 'mean_absolute_error':
                    on_holdout = metrics.mean_absolute_error(y_test, reg.predict(x_test))
                    on_synth = metrics.mean_absolute_error(y_synth, reg.predict(x_synth))
                elif base_metric == 'r2_score':
                    on_holdout = metrics.r2_score(y_test, reg.predict(x_test))
                    on_synth = metrics.r2_score(y_synth, reg.predict(x_synth))

                    # if on_holdout > 1 or on_synth > 1:
                    if on_holdout > 1 or on_synth > 1:
                        print("on_holdout: ", on_holdout)
                        print("on_synth: ", on_synth)
                else:
                    raise NotImplementedError('Unknown base metric {}.'.format(base_metric))

            except ValueError:
                # metric can not be computed
                on_holdout = np.NaN
                on_synth = np.NaN
        else:
            raise('Unknown regression method')
        
        # if on_holdout is not nan and on_synth is not nan
        if not np.isnan(on_holdout) and not np.isnan(on_synth):
            print("on_holdout and on_synth are not nan")
        # metrics_dict[col_k] = on_synth / on_holdout
        try:
            metrics_dict[col_k] = on_synth / on_holdout
        except ZeroDivisionError:
            metrics_dict[col_k] = np.nan

    # if want to return only mean, then use this. If one wants
    # to return crcl for every variable, just return 'metrics_dict'
    avg_crcl = {'Avg-CrCl': sum([metrics_dict[k] for k in metrics_dict.keys()])/len(metrics_dict)}

    return metrics_dict

def cc_accuracy(**kwargs):
    """ Compute cross_classification metric using accuracy: train on data_a and test data_b"""
    base_metric = 'accuracy'
    return cross_classification(kwargs['data_a'], kwargs['data_b'],
                                base_metric, kwargs['class_method'])


def cc_f1score(**kwargs):
    """ Compute cross classification metric with f1score: train on data_a and test data_b"""
    base_metric = 'f1_score'
    return cross_classification(kwargs['data_a'], kwargs['data_b'],
                                base_metric, kwargs['class_method'])


def cca_accuracy(**kwargs):
    """ Compute cross classification metric with acc: train on synt and test on real"""
    data_real = kwargs['data_a']  # real data
    data_synt = kwargs['data_b']  # synthetic data
    # base_metric = 'accuracy'
    base_metric = 'balanced_accuracy'
    class_method = 'DT'
    prfs_rs = cross_classification(data_real, data_synt, base_metric, class_method)  # train: real, test: synth
    prfs_sr = cross_classification(data_synt, data_real, base_metric, class_method)  # train: synth, test: real
    prfs = {}
    for col in prfs_rs.keys():
        prfs[col] = 0.5 * (prfs_rs[col] + prfs_sr[col])
    #     avg = avg.groupby('Metric')['Value'].mean().reset_index()
    return prfs

def cca_accuracy_edited(**kwargs):
    """ Compute cross classification metric with acc: train on synt and test on real"""
    data_real = kwargs['data_a']  # real data
    data_synt = kwargs['data_b']  # synthetic data
    # base_metric = 'accuracy'
    base_metric = kwargs['base_metric']
    class_method = 'DT'
    prfs_rs = cross_classification_edited(data_real, data_synt, base_metric, class_method)  # train: real, test: synth
    prfs_sr = cross_classification_edited(data_synt, data_real, base_metric, class_method)  # train: synth, test: real
    prfs = {}
    for col in prfs_rs.keys():
        prfs[col] = 0.5 * (prfs_rs[col] + prfs_sr[col])
    #     avg = avg.groupby('Metric')['Value'].mean().reset_index()
    return prfs


def ccr_diff_edited(**kwargs):
    """ Compute cross regression metric squared error: train on synt and test on real"""
    data_real = kwargs['data_a']  # real data
    data_synt = kwargs['data_b']  # synthetic data
    # base_metric = 'accuracy'
    base_metric = kwargs['base_metric']
    class_method = 'DT'
    prfs_rs = cross_regression_edited(data_real, data_synt, base_metric, class_method)  # train: real, test: synth
    prfs_sr = cross_regression_edited(data_synt, data_real, base_metric, class_method)  # train: synth, test: real
    prfs = {}
    for col in prfs_rs.keys():
        prfs[col] = 0.5 * (prfs_rs[col] + prfs_sr[col])
    #     avg = avg.groupby('Metric')['Value'].mean().reset_index()
    return prfs


def cca_f1score(**kwargs):
    """ Compute average of cross classification metric with acc in both ways
        real -> synth and synth -> real"""
    data_real = kwargs['data_a']  # real data
    data_synt = kwargs['data_b']  # synthetic data
    base_metric = 'f1_score'
    class_method = 'DT'
    prfs_rs = cross_classification(data_real, data_synt, base_metric, class_method)  # train: real, test: synth
    prfs_sr = cross_classification(data_synt, data_real, base_metric, class_method)  # train: synth, test: real
    prfs = {}
    for col in prfs_rs.keys():
        prfs[col] = 0.5 * (prfs_rs[col] + prfs_sr[col])
    return prfs


def kl(p, q):
    """Kullback-Leibler divergence D(P || Q) for discrete distributions

    Args:
        p (array): discrete probability distribution p
        q (array): discrete probability distribution q

    Returns:
        float: kl divergence between the two probability distributions
    """
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)

    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def kl_divergence(**kwargs):
    """KL divergence of marginal multinomial distributions in dataframes.

    Args:
        data_a (DataFrame): dataset A
        data_b (DataFrame): dataset B

    Returns:
        float: SRMSE between datasets A and B.
    """
    data_a = kwargs['data_a']
    data_b = kwargs['data_b']
    data_a, data_b = align_columns(data_a, data_b)

    kl_div = dict()
    for var in data_a.columns:
        uniq_vals = list(data_a[var].unique()) + list(data_b[var].unique())
        uniq_vals = set(uniq_vals)
        p = compute_freqs(data_a[var], uniq_vals)  # original dataset
        q = compute_freqs(data_b[var], uniq_vals)  # synthetic dataset
        # avoid Inf KL
        p[p == 0] = np.finfo(float).eps
        q[q == 0] = np.finfo(float).eps
        p = p / p.sum()
        q = q / q.sum()

        # non-linear transformation of KL-Divergence to ease of show
        kl_div[var] = np.log10(1 + scipy.special.kl_div(p, q).sum())
    return kl_div

def compute_freqs(serie, uniq_vals):
    """Compute frequency of the values in categorical variable.

    Args:
        serie (Series): categorical variable (pandas' serie)
        uniq_vals (list): list of possible values in the categorical variable

    Returns:
        array: values frequency
    """
    counts = dict(serie.value_counts())
    p = np.zeros((len(uniq_vals), 1))
    for i, val in enumerate(uniq_vals):
        if val in counts:
            p[i] = counts[val]

    return p / p.sum()


# def kl_divergence_edited_v1(**kwargs):
#     """KL divergence for both categorical and continuous data in dataframes.
    
#     Args:
#         data_a (DataFrame): dataset A
#         data_b (DataFrame): dataset B
#         bandwidth (float): Bandwidth for KDE, used for continuous data.
    
#     Returns:
#         dict: KL divergence values for each variable in the datasets.
#     """
    

#     data_a = kwargs['data_a']
#     data_b = kwargs['data_b']
#     bandwidth = kwargs['bandwidth']

    
#     data_a, data_b = align_columns(data_a, data_b)
#     kl_div = dict()

#     for var in data_a.columns:
#         # Continuous data

#         if data_a[var].dtype == 'float64':
            
#             density_a = compute_kde(data_a[var].values, bandwidth=bandwidth)
#             density_b = compute_kde(data_b[var].values, bandwidth=bandwidth)

#             # Avoid Inf KL by adding small value to zero densities
#             density_a[density_a == 0] = np.finfo(float).eps
#             density_b[density_b == 0] = np.finfo(float).eps

#             density_a = density_a / density_a.sum()
#             density_b = density_b / density_b.sum()

#             # Non-linear transformation of KL-Divergence to ease of show
#             kl_div[var] = np.log10(1 + scipy.special.kl_div(density_a, density_b).sum())
        
#         # Categorical data
#         # We assume that the categorical data is already encoded
#         elif data_a[var].dtype == 'int64':
            
#             uniq_vals = list(data_a[var].unique()) + list(data_b[var].unique())
#             uniq_vals = set(uniq_vals)
#             p = compute_freqs(data_a[var], uniq_vals)  # original dataset
#             q = compute_freqs(data_b[var], uniq_vals)  # synthetic dataset
#             # avoid Inf KL
#             p[p == 0] = np.finfo(float).eps
#             q[q == 0] = np.finfo(float).eps
#             p = p / p.sum()
#             q = q / q.sum()

#             # Non-linear transformation of KL-Divergence to ease of show
#             kl_div[var] = np.log10(1 + scipy.special.kl_div(p, q).sum())
    
#     return kl_div


def calculate_kl_divergence_continuous(p_data, q_data, num_bins=100):
    # Find the minimum and maximum values between the two datasets
    min_val = min(p_data.min(), q_data.min())
    max_val = max(p_data.max(), q_data.max())
    
    # Create histograms for both datasets
    p_hist, bin_edges = np.histogram(p_data, bins=num_bins, range=(min_val, max_val), density=True)
    q_hist, _ = np.histogram(q_data, bins=bin_edges, density=True)
    
    # Add a small value to zero densities to avoid infinite KL divergence
    p_hist = np.where(p_hist == 0, 1e-10, p_hist)
    q_hist = np.where(q_hist == 0, 1e-10, q_hist)
    
    # Normalize the histograms to sum to 1
    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()
    
    # Calculate the KL divergence
    kl_div = entropy(p_hist, q_hist)
    return kl_div


def kl_divergence_edited_v2(**kwargs):
    """KL divergence for both categorical and continuous data in dataframes.
    
    Args:
        data_a (DataFrame): dataset A
        data_b (DataFrame): dataset B
        bandwidth (float): Bandwidth for KDE, used for continuous data.
    
    Returns:
        dict: KL divergence values for each variable in the datasets.
    """
    

    data_a = kwargs['data_a']
    data_b = kwargs['data_b']

    data_a, data_b = align_columns(data_a, data_b)
    kl_div = dict()

    for var in data_a.columns:
        # Continuous data

        if data_a[var].dtype == 'float64':
            
            kl_div[var]=calculate_kl_divergence_continuous(data_a[var].values, data_b[var].values)
            print(kl_div[var])
            
        
        # Categorical data
        # We assume that the categorical data is already encoded
        elif data_a[var].dtype == 'int64':
            
            uniq_vals = list(data_a[var].unique()) + list(data_b[var].unique())
            uniq_vals = set(uniq_vals)
            p = compute_freqs(data_a[var], uniq_vals)  # original dataset
            q = compute_freqs(data_b[var], uniq_vals)  # synthetic dataset
            # avoid Inf KL
            p[p == 0] = np.finfo(float).eps
            q[q == 0] = np.finfo(float).eps
            p = p / p.sum()
            q = q / q.sum()

            # Non-linear transformation of KL-Divergence to ease of show
            kl_div[var] = np.log10(1 + scipy.special.kl_div(p, q).sum())
    
    return kl_div


# def compute_kde(data, bandwidth=0.1):
#     """Compute the kernel density estimate for continuous data."""
#     kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(data[:, None])
#     log_density = kde.score_samples(data[:, None])
#     density = np.exp(log_density)
#     return density


def cluster_measure(data_a, data_b, **kwargs):
    """Cluster Measure.

    Args:
        data_a (DataFrame): dataset A (Real)
        data_b (DataFrame): dataset B (Synthetic)

    Returns:
        float: Cluster measure between datasets A and B.
    """
    data_a, data_b = align_columns(data_a, data_b)

    nb_clusters = 20
    # merge both datasets
    merged_data = np.vstack((data_a, data_b))
    # real data: zero
    # synth data: one
    labels = np.vstack((np.zeros((data_a.shape[0], 1)),
                        np.ones((data_b.shape[0], 1))))
    merged_data = np.hstack((labels, merged_data))

    # train k-means clustering
    kmeans = KMeans(n_clusters=nb_clusters, n_init=3,
                    max_iter=200, random_state=0).fit(merged_data[:, 1:])

    # relative number of real data
    c = data_a.shape[0] / float(data_a.shape[0] + data_b.shape[0])
    # clustering metric goes here: Uc
    Uc = 0
    for j in range(nb_clusters):
        # get ids of the merged_data which are in the j-th cluster
        ids = np.where(kmeans.labels_ == j)[0]
        # get the number of real data in j-th cluster
        njO = np.where(merged_data[ids, 0] == 0)[0].size
        # number of total samples in the j-th cluster
        nj = ids.size
        # cluster measure for the j-th cluster
        Uc += (njO / float(nj) - c)**2
    return {'-': Uc / float(nb_clusters)}


def loglik_ratio(**kwargs):
    """Loglikelihood ratio between dataset A and B considering a GMM
    model trained on dataset A.

    llk = log(llh_gmmA(B) / llh_gmmA(A))

    Args:
        data_a (DataFrame): dataset A
        data_b (DataFrame): dataset B

    Returns:
        float: Llk ratio of dataset A and dataset B
    """

    data_a = kwargs['data_a'].copy()  # real data
    data_b = kwargs['data_b'].copy()  # synthetic data
    data_a, data_b = align_columns(data_a, data_b)

    nb_components = 20
    gmm = GaussianMixture(n_components=nb_components)
    gmm.fit(kwargs['data_a'])

    return {'-': np.log10(gmm.score(data_b)) -
                   np.log10(gmm.score(data_a))}


def pairwise_correlation_difference(**kwargs):
    """Compare pairwise correlations of the two dataframes.

    Args:
        data_a (DataFrame): dataset A
        data_b (DataFrame): dataset B

    Returns:
        float: SRMSE between datasets A and B.
    """
    data_a = kwargs['data_a'].copy()  # real data
    data_b = kwargs['data_b'].copy()  # synthetic data
    data_a, data_b = align_columns(data_a, data_b)

    # remove columns with only one possible value: this makes the correlation
    # matrices full of nans
    for col in data_a.columns:
        if len(data_a[col].unique()) == 1 and len(data_b[col].unique()) == 1:
            data_a.drop(col, axis=1, inplace=True)
            data_b.drop(col, axis=1, inplace=True)

    cmtx_a = np.corrcoef(data_a, rowvar=False)
    cmtx_b = np.corrcoef(data_b, rowvar=False)
    diff_a_b = np.linalg.norm(cmtx_a - cmtx_b, 'fro')


    # if there is nan in cmtx_a, 
    if np.isnan(cmtx_a).any() or np.isnan(cmtx_b).any():
        print("cmtx_a has nan")

    # if np.isnan(diff_a_b):
    #     print("Nan diff_a_b")

    return {'-': diff_a_b}


def coverage(**kwargs):
    data_a = kwargs['data_a'].copy()  # real data
    data_b = kwargs['data_b'].copy()  # synthetic data
    coverage = dict()
    for col in data_a.columns:
        coverage[col] = len(data_b[col].unique()) / len(data_a[col].unique())
    return coverage

def coverage_categorical_continous(**kwargs):
    data_a = kwargs['data_a'].copy()  # real data
    data_b = kwargs['data_b'].copy()  # synthetic data
    coverage = dict()
    column_types = dict()

    for col in data_a.columns:
        if data_a[col].dtype == 'int64':
            column_types[col] = 'integer'
            coverage[col] = len(data_b[col].unique()) / len(data_a[col].unique())
        elif data_a[col].dtype == 'float64':
            column_types[col] = 'float'
            range_a = data_a[col].max() - data_a[col].min()
            range_b = data_b[col].max() - data_b[col].min()
            coverage[col] = range_b / range_a
        else:
            column_types[col] = 'other'
    
    return coverage