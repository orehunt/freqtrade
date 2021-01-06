import numpy as np
import pandas as pd
import talib as ta
from scipy.stats import norm
from statsmodels.tsa import stattools as st
from ..nb.polyfit import fit_poly
from ..nb.types import njit
from freqtrade.optimize.backtest_nb import shift_nb


def iqr(data):
    q25 = np.percentile(data, q=25)
    q75 = np.percentile(data, q=75)
    return abs(q75 - q25)


def scaling(data, win):
    return data / data.rolling(window=win).apply(iqr)


def centering(data, win):
    return data - data.rolling(win).median()


def Normalization(data, length=100):
    v = np.zeros(data.shape)

    for i in range(len(data)):
        if i < length:
            continue

        data_in = data[(i - length + 1) : i]
        i_start = i - length + 1
        delta = data[i] - np.median(data_in)
        iq = iqr(data_in)
        v[i] = 100 * norm.cdf(0.5 * delta / iq) - 50
    return v


def RegularNormalization(data, length=100):
    max = ta.MAX(data, length)
    min = ta.MIN(data, length)
    return (data - min) / (max - min) * 100 - 50


def ZScore(data, length=100):
    me = ta.SMA(data, length)
    st = ta.STDDEV(data, length)
    return (data - me) / st


def adf(x):
    """
    adf(x) - Calculate adf stat, p-value and half-life of the given list. This test will tell us if a
             time series is mean reverted.
             e.g. adf(prices)
    :param x: A numpy ndarray of data.
    :return: (stat, p_value, half_life)
    Reference
        http://statsmodels.sourceforge.net/devel/generated/statsmodels.tsa.stattools.adfuller.html
        http://en.wikipedia.org/wiki/Augmented_Dickey%E2%80%93Fuller_test
    """
    # Calculate ADF to get ADF stats and p-value
    result = st.adfuller(x)

    # Calculate the half-life of reversion
    x_s1 = shift_nb(x)
    x_s1[0] = x_s1[1]
    delta = x - x_s1
    beta = fit_poly(x_s1, delta, 1)[0]
    # beta = np.polyfit(lagged_price, delta, 1)[0]  # Use price(t-1) to predicate delta.
    half_life = -1 * np.log(2) / beta
    return result[0], result[1], half_life
