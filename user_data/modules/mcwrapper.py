from typing import List

import numba as nb
import numpy as np
import vectorbt as vbt
from vectorbt.returns import nb as rnb

from freqtrade.optimize.vbt import BacktestResultTupleType


@nb.njit(cache=True)
def MCW(func: str, returns: np.ndarray, n_samples: int):
    if not len(returns):
        return np.nan
    if func == "min_calmar_ratio":
        return min_calmar_ratio(returns, n_samples)
    elif func == "avg_calmar_ratio":
        return avg_calmar_ratio(returns, n_samples)
    else:
        return np.nan


@nb.njit(cache=True)
def min_calmar_ratio(returns: np.ndarray, n_samples: int):
    sims = np.empty(n_samples)
    for n in range(n_samples):
        sims[n] = calmar_ratio(returns)
        np.random.shuffle(returns)
    return np.nanmin(sims)


@nb.njit(cache=True)
def avg_calmar_ratio(returns: np.ndarray, n_samples: int):
    sims = np.empty(n_samples)
    for n in range(n_samples):
        sims[n] = calmar_ratio_dd(returns)
        np.random.shuffle(returns)
    return np.nanmean(sims)


@nb.njit(cache=True)
def qtr_calmar_ratio(returns: np.ndarray, n_samples: int):
    sims = np.empty(n_samples)
    for n in range(n_samples):
        sims[n] = calmar_ratio(returns)
        np.random.shuffle(returns)
    bot, top, mid = get_quantile(sims)
    return quantile_ratio(bot, top, mid)


@nb.njit(cache=True)
def calmar_ratio(returns):
    dd = np.abs(rnb.max_drawdown_1d_nb(returns))
    return np.nan if dd == 0.0 else returns.sum() / np.abs(dd)


@nb.njit(cache=True)
def calmar_ratio_dd(returns):
    # drawdowns are always <= 0
    drawdowns = vbt.returns.nb.drawdown_1d_nb(returns)
    avg_drawdown = drawdowns.mean() or np.nan
    # higher is better
    return returns.mean() / avg_drawdown


@nb.njit(cache=True)
def sortino_ratio(returns, mar=0.0):
    if returns.shape[0] < 2:
        return np.nan
    adj_returns = returns - mar
    expected_returns = adj_returns.mean()

    adj_returns[adj_returns > 0.0] = 0.0
    downside_risk = (adj_returns ** 2).sum() / len(adj_returns)
    if downside_risk == 0.0:
        return np.inf
    return expected_returns / downside_risk


@nb.njit(cache=True)
def upside_ratio(returns: np.ndarray):
    """ The deviation of profits """
    return returns.mean() / (returns[returns > 0].std() or np.nan)


@nb.njit(cache=True)
def expectancy(returns, with_returns=False):
    wins = returns[returns > 0.0]
    n_wins = len(wins)
    n_trades = len(returns)
    total = returns.sum() if with_returns else 1.0
    if n_wins == 0:
        return total if with_returns else -1.0
    elif n_wins == n_trades:
        return total
    losses = returns[returns <= 0.0]
    risk_reward = abs(wins.mean() / losses.mean())
    wr = n_wins / n_trades
    return (risk_reward * wr - (1.0 - wr)) * total


@nb.njit(cache=True)
def get_quantile(arr, q=0.2):
    """ The average of the bottom, top and middle sets """
    arr = arr[np.isfinite(arr)]
    if len(arr):
        arr.sort()
        ln = len(arr)
        size = int(ln * q)
        if size:
            bot = np.mean(arr[:size])
            top = np.mean(arr[-size:])
            mid = np.mean(arr[size : ln - size])
        else:
            bot = np.mean(arr)
            top = bot
            mid = bot
        return bot, top, mid
    else:
        return np.nan, np.nan, np.nan


@nb.njit(cache=True)
def quantile_ratio(bot: float, top: float, mid: float):
    tpm = top + mid
    return np.sign(tpm) * tpm ** 2 / ((top - bot) or np.nan)


@nb.njit(cache=True)
def pairs_ratio(returns, pairs, n_pairs, moment=2):
    if returns.shape[0] < 2:
        return np.nan
    # pairs array has to match returns at index
    p_profits = np.zeros(n_pairs)
    for n, r in enumerate(returns):
        p_profits[pairs[n]] += r
    return p_profits.mean() / (p_profits.std() ** moment)


@nb.njit(cache=True)
def get_results_col(results, col):
    return np.asarray([v[col] for v in results])


def generate_jit_mc_calmar(name, max_staked, n_samples, col_idx):
    def loss_f(results: List[BacktestResultTupleType]):
        returns = get_results_col(results, col_idx) / max_staked
        ratio = -MCW("min_calmar_ratio", returns, n_samples)
        return ((name, ratio),)

    return nb.njit(loss_f, cache=True)


def generate_jit_mc_tasp(metrics, max_staked, n_samples, col_idx):
    def loss_f(results: List[BacktestResultTupleType]):
        returns = get_results_col(results, col_idx) / max_staked
        return (
            (metrics[0], -rnb.conditional_value_at_risk_1d_nb(returns, cutoff=0.33)),
            (metrics[1], -rnb.tail_ratio_1d_nb(returns)),
            (metrics[2], -MCW("min_calmar_ratio", returns, n_samples)),
        )

    return nb.njit(loss_f, cache=False)


def generate_jit_stc(metrics, max_staked, n_samples, ann_factor, col_idx):
    def loss_f(results: List[BacktestResultTupleType]):
        returns = get_results_col(results, col_idx) / max_staked
        return (
            (metrics[0], -rnb.sortino_ratio_1d_nb(returns, ann_factor)),
            (metrics[1], -rnb.tail_ratio_1d_nb(returns)),
            (metrics[2], -MCW("min_calmar_ratio", returns, n_samples)),
        )

    return nb.njit(loss_f, cache=True)


def generate_jit_socv(metrics, max_staked, n_samples, col_idx, p_idx, n_pairs):
    def loss_f(results: List[BacktestResultTupleType]):
        returns = get_results_col(results, col_idx) / max_staked
        pairs = get_results_col(results, p_idx)
        return (
            (metrics[0], -sortino_ratio(returns)),
            (metrics[1], -rnb.conditional_value_at_risk_1d_nb(returns, cutoff=0.5)),
            (metrics[2], -MCW("avg_calmar_ratio", returns, n_samples)),
            (metrics[3], -pairs_ratio(returns, pairs, n_pairs)),
        )

    return nb.njit(loss_f, cache=True)


def generate_jit_ppdev(metrics, max_staked, col_idx, p_idx, n_pairs):
    def loss_f(results: List[BacktestResultTupleType]):
        returns = get_results_col(results, col_idx) / max_staked
        pairs = get_results_col(results, p_idx)
        return ((metrics[0], -pairs_ratio(returns, pairs, n_pairs)),)

    return nb.njit(loss_f, cache=True)
