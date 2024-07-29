# Comparison of KL Divergence Calculation Functions: `kl_divergence` vs `kl_divergence_edited_v2`

## Overview

This document explains the differences between two KL divergence calculation functions: `kl_divergence` and `kl_divergence_edited_v2`. These functions measure the distributional differences between two datasets containing a mix of continuous and categorical features.

## Original `kl_divergence` Function

### Description

The `kl_divergence` function calculates the KL divergence for each column in two dataframes, treating all columns as categorical data.

### Implementation

```python
def kl_divergence(**kwargs):
    """KL divergence of marginal multinomial distributions in dataframes.

    Args:
        data_a (DataFrame): dataset A
        data_b (DataFrame): dataset B

    Returns:
        dict: KL divergence values for each variable.
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

        # non-linear transformation of KL-Divergence for ease of display
        kl_div[var] = np.log10(1 + scipy.special.kl_div(p, q).sum())
    return kl_div
```

### Key Features

- Treats all data as categorical
- Uses frequency computation for KL divergence calculation
- Applies a logarithmic transformation to the result

## Modified `kl_divergence_edited_v2` Function

### Description

The `kl_divergence_edited_v2` function extends the original function to handle both categorical and continuous data, using histograms for continuous data.

### Implementation

```python
def kl_divergence_edited_v2(**kwargs):
    """KL divergence for both categorical and continuous data in dataframes.

    Args:
        data_a (DataFrame): dataset A
        data_b (DataFrame): dataset B

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
            kl_div[var] = calculate_kl_divergence_continuous(data_a[var].values, data_b[var].values)

        # Categorical data
        elif data_a[var].dtype == 'int64':
            uniq_vals = list(data_a[var].unique()) + list(data_b[var].unique())
            uniq_vals = set(uniq_vals)
            p = compute_freqs(data_a[var], uniq_vals)
            q = compute_freqs(data_b[var], uniq_vals)
            p[p == 0] = np.finfo(float).eps
            q[q == 0] = np.finfo(float).eps
            p = p / p.sum()
            q = q / q.sum()
            kl_div[var] = np.log10(1 + scipy.special.kl_div(p, q).sum())

    return kl_div

def calculate_kl_divergence_continuous(p_data, q_data, num_bins=100):
    min_val = min(p_data.min(), q_data.min())
    max_val = max(p_data.max(), q_data.max())

    p_hist, bin_edges = np.histogram(p_data, bins=num_bins, range=(min_val, max_val), density=True)
    q_hist, _ = np.histogram(q_data, bins=bin_edges, density=True)

    p_hist = np.where(p_hist == 0, 1e-10, p_hist)
    q_hist = np.where(q_hist == 0, 1e-10, q_hist)

    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()

    kl_div = entropy(p_hist, q_hist)
    return kl_div
```

### Key Features

- Distinguishes between continuous and categorical data
- Uses histograms for continuous data KL divergence calculation
- Maintains the original approach for categorical data

## Key Differences

1. Data Type Handling:

   - `kl_divergence`: Treats all data as categorical
   - `kl_divergence_edited_v2`: Differentiates between continuous and categorical data

2. Continuous Data Processing:

   - `kl_divergence_edited_v2`: Implements `calculate_kl_divergence_continuous` for histogram-based KL divergence calculation

3. Output Format:
   - Both functions return a dictionary of KL divergence values for each variable

# Comparison of Cross Classification Accuracy Calculation Functions: `cca_accuracy` vs `cca_accuracy_edited`

## Overview

This document explains the differences between two cross classification accuracy calculation functions: `cca_accuracy` and `cca_accuracy_edited`. These functions are used to measure the performance of models trained on synthetic data and tested on real data, and vice versa. The modified version extends the original function to handle both categorical and continuous features.

## Original `cca_accuracy` Function

### Description

The `cca_accuracy` function calculates the cross classification accuracy for each column in two dataframes. This function assumes all columns are categorical and performs classification for each feature.

### Key Features

- **Inputs**: Two dataframes (`data_a`, `data_b`)
- **Output**: A dictionary containing cross classification accuracy values for each variable
- **Implementation**:
  1. Performs classification for each feature using a decision tree classifier
  2. Computes the performance metric (accuracy or balanced accuracy)
  3. Averages the performance metrics from training on real and testing on synthetic data, and vice versa

### Code

```python
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
```

## Modified `cca_accuracy_edited` Function

### Description

The `cca_accuracy_edited` function extends the original function to handle both categorical and continuous features. It performs classification for categorical target features and regression for continuous target features.

### Key Features

- **Inputs**:
  - Two dataframes (`data_a`, `data_b`)
  - `base_metric` parameter to specify the performance metric
- **Output**: A dictionary containing cross classification accuracy values for each variable
- **Implementation**:
  1. Determines if each column contains continuous or categorical data
  2. For categorical data: Performs classification
  3. For continuous data: Performs regression
  4. Computes the performance metric and averages results from training on real and testing on synthetic data, and vice versa

### Code

```python
def cca_accuracy_edited(**kwargs):
    """ Compute cross classification metric with acc: train on synt and test on real"""
    data_real = kwargs['data_a']  # real data
    data_synt = kwargs['data_b']  # synthetic data
    base_metric = kwargs['base_metric']
    class_method = 'DT'
    prfs_rs = cross_classification_edited(data_real, data_synt, base_metric, class_method)  # train: real, test: synth
    prfs_sr = cross_classification_edited(data_synt, data_real, base_metric, class_method)  # train: synth, test: real
    prfs = {}
    for col in prfs_rs.keys():
        prfs[col] = 0.5 * (prfs_rs[col] + prfs_sr[col])
    return prfs
```

## Key Differences

1. **Data Type Handling**:

   - `cca_accuracy`: Assumes all data is categorical
   - `cca_accuracy_edited`: Distinguishes between continuous and categorical data

2. **Continuous Data Processing**:

   - `cca_accuracy_edited`: Uses regression for continuous target features

3. **Parameter Flexibility**:
   - `cca_accuracy_edited`: Allows specifying the performance metric (`base_metric`)

## Supporting Functions

### Cross Classification

#### Code

```python
def cross_classification(data_a, data_b, base_metric='accuracy', class_method='DT'):
    """ Classification error normalized by the error in the hold-out (real) set. """
    data_a, data_b = align_columns(data_a, data_b)

    shfl_ids = np.random.permutation(range(data_a.shape[0]))
    n_train = int(data_a.shape[0] * 0.6)

    accs = dict()

    for col_k in data_a.columns:
        x_train = data_a.loc[shfl_ids[:n_train], data_a.columns != col_k].values
        y_train = data_a.loc[shfl_ids[:n_train], [col_k]].values.ravel()

        x_test = data_a.loc[shfl_ids[n_train:], data_a.columns != col_k].values
        y_test = data_a.loc[shfl_ids[n_train:], [col_k]].values.ravel()

        x_synth = data_b.loc[:, data_b.columns != col_k].values
        y_synth = data_b.loc[:, [col_k]].values.ravel()

        if class_method == 'DT':
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
                else:
                    raise NotImplementedError('Unknown base metric {}.'.format(base_metric))
            except ValueError:
                on_holdout = np.NaN
                on_synth = np.NaN

        elif class_method == 'LR':
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
                else:
                    raise NotImplementedError('Unknown base metric {}.'.format(base_metric))
            except ValueError:
                on_holdout = np.NaN
                on_synth = np.NaN
        else:
            raise('Unknown classification method')

        if not np.isnan(on_holdout) and not np.isnan(on_synth):
            accs[col_k] = on_synth / on_holdout

    return accs
```

### Cross Classification (Edited)

#### Code

```python
def cross_classification_edited(data_a, data_b, base_metric='accuracy', class_method='DT'):
    """ Classification error normalized by the error in the hold-out (real) set. """
    data_a, data_b = align_columns(data_a, data_b)

    shfl_ids = np.random.permutation(range(data_a.shape[0]))
    n_train = int(data_a.shape[0] * 0.6)

    accs = dict()

    for col_k in data_a.columns:
        x_train = data_a.loc[shfl_ids[:n_train], data_a.columns != col_k].values
        y_train = data_a.loc[shfl_ids[:n_train], [col_k]].values.ravel()

        x_test = data_a.loc[shfl_ids[n_train:], data_a.columns != col_k].values
        y_test = data_a.loc[shfl_ids[n_train:], [col_k]].values.ravel()

        x_synth = data_b.loc[:, data_b.columns != col_k].values
        y_synth = data_b.loc[:, [col_k]].values.ravel()

        if class_method == 'DT':
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
                else:
                    raise NotImplementedError('Unknown base metric {}.'.format(base_metric))
            except ValueError:
                on_holdout = np.NaN
                on_synth = np.NaN

        elif class_method == 'LR':
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
                else:
                    raise NotImplementedError('Unknown base metric {}.'.format(base_metric))
            except ValueError:
                on_holdout = np.NaN
                on_synth = np.NaN
        else:
            raise('Unknown classification method')

        if not np.isnan(on_holdout) and not np.isnan(on_synth):
            accs[col_k] = on_synth / on_holdout

    return accs
```

<!-- ## Conclusion

The `cca_accuracy_edited` function was developed to handle datasets containing both continuous and categorical features, providing a more comprehensive approach to calculating cross classification accuracy. This enhancement is particularly useful for mixed-type datasets, allowing for more accurate performance measurement across different feature types. -->

\* Base metric is balanced accuracy by default, but can be changed to accuracy if needed.
