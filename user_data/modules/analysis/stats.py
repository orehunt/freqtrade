import numpy as np
import pandas as pd
import scipy.stats as ss
from collections import Counter
from pyitlib.discrete_random_variable import entropy_conditional
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA


def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def theils_u(x, y):
    s_xy = entropy_conditional(x, y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n / total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x


# also check:
# - intra class correlation
# - cron bach reliability score
# - scipy.stats.sem (standard error of measure)
def correlation_ratio(categories, measurements):
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = measurements[:, np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.nanmean(cat_measures)
    y_total_avg = np.nansum(np.multiply(y_avg_array, n_array)) / np.nansum(n_array)
    numerator = np.nansum(
        np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2))
    )
    denominator = np.nansum(np.power(np.subtract(measurements, y_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator / denominator)
    return eta


def icc(data, targets, normalize=True, zero_fill=False):
    """
    :data: ndarray, each column represent the sample of observations for a specific metric
    :targets: ndarray, each column represent one element of the group for correlating the metrics
    A group is composed of one column from data, with targets, eg.
    (concat([data[:, n], targets], axis=1) for n in data.shape[0])
    """
    N = data.shape[0]
    assert N == targets.shape[0]

    targets = targets.values
    data = data.values
    if zero_fill:
        targets[np.isnan(targets)] = 0
        data[np.isnan(data)] = 0

    if len(targets.shape) == 1:
        K = 2
        targets = targets[:, None]
    else:
        K = targets.shape[1] + 1
    if len(data.shape) == 1:
        data = data[:, None]
    if zero_fill:
        targets_sum = targets.sum(axis=1)[:, None]
    else:
        targets_sum = np.nansum(targets, axis=1)[:, None]

    groups_sum = data + targets_sum
    d = 1 / K * N
    total_mean = d * groups_sum.sum(axis=0)

    member_diff = np.empty((targets.shape[0], targets.shape[1], total_mean.shape[0]))
    for t in range(K - 1):
        member_diff[:, t, :] = targets[:, t][:, None] - total_mean

    if zero_fill:
        targets_E = np.power(member_diff, 2).sum(axis=0).sum(axis=0)
        data_E = np.power(data - total_mean, 2).sum(axis=0)
    else:
        targets_E = np.nansum(np.nansum(np.power(member_diff, 2), axis=0), axis=0)
        data_E = np.nansum(np.power(data - total_mean, 2), axis=0)

    s2 = d * (targets_E + data_E)
    r = K / (K - 1) * (
        (np.power(groups_sum.mean(axis=0) - total_mean, 2).sum(axis=0) / N) / s2
    ) - (1 * K - 1)

    if normalize:
        r = (r - r.min()) / (r.max() / r.min())
    return r


def find_correlation(df, thresh=0.9):
    """
    Given a numeric pd.DataFrame, this will find highly correlated features,
    and return a list of features to remove
    params:
    - df : pd.DataFrame
    - thresh : correlation threshold, will remove one of pairs of features with
               a correlation greater than this value
    """

    corrMatrix = df.corr()
    corrMatrix.loc[:, :] = np.tril(corrMatrix, k=-1)

    already_in = set()
    result = []

    for col in corrMatrix:
        perfect_corr = corrMatrix[col][corrMatrix[col] > thresh].index.tolist()
        if perfect_corr and col not in already_in:
            already_in.update(set(perfect_corr))
            perfect_corr.append(col)
            result.append(perfect_corr)

    select_nested = [f[1:] for f in result]
    select_flat = [i for j in select_nested for i in j]
    return select_flat


# <===>

# Entropy
def entropy(Y):
    """
    Also known as Shanon Entropy
    Reference: https://en.wikipedia.org/wiki/Entropy_(information_theory)
    """
    _, count = np.unique(Y, return_counts=True, axis=0)
    prob = count / len(Y)
    en = np.sum((-1) * prob * np.log2(prob))
    return en


# Joint Entropy
def jEntropy(Y, X):
    """
    H(Y;X)
    Reference: https://en.wikipedia.org/wiki/Joint_entropy
    """
    YX = np.c_[Y, X]
    return entropy(YX)


# Conditional Entropy
def cEntropy(Y, X):
    """
    conditional entropy = Joint Entropy - Entropy of X
    H(Y|X) = H(Y;X) - H(X)
    Reference: https://en.wikipedia.org/wiki/Conditional_entropy
    """
    return jEntropy(Y, X) - entropy(X)


# Information Gain
def gain(Y, X):
    """
    Information Gain, I(Y;X) = H(Y) - H(Y|X)
    Reference: https://en.wikipedia.org/wiki/Information_gain_in_decision_trees#Formal_definition
    """
    return entropy(Y) - cEntropy(Y, X)


# <==>

# Coehn


def cohens_d(x, y):
    return (np.mean(x) - np.mean(y)) / (
        np.sqrt((np.std(x, ddof=1) ** 2 + np.std(y, ddof=1) ** 2) / 2)
    )
