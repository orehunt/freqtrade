from typing import List

import numba as nb
import numpy as np
import vectorbt as vbt
from vectorbt.returns import nb as rnb

from freqtrade.optimize.vbt import BacktestResultTupleType
from user_data.modules.metrics_nb import upside_volatility


@nb.njit(cache=True)
def MCW(func: str, returns: np.ndarray, n_samples: int):
    if not len(returns):
        return np.nan
    sims = np.empty(n_samples)
    if func == "min_calmar_ratio":
        for n in range(n_samples):
            sims[n] = calmar_ratio(returns)
            np.random.shuffle(returns)
        return np.nanmin(sims)
    elif func == "avg_calmar_ratio":
        for n in range(n_samples):
            sims[n] = avg_calmar(returns)
            np.random.shuffle(returns)
        return np.nanmean(sims)
    else:
        return np.nan


@nb.njit(cache=True)
def calmar_ratio(returns):
    dd = np.abs(rnb.max_drawdown_1d_nb(returns))
    return np.nan if dd == 0.0 else returns.sum() / np.abs(dd)


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
def avg_calmar(returns):
    # drawdowns are always <= 0
    drawdowns = vbt.returns.nb.drawdown_1d_nb(returns)
    avg_drawdown = drawdowns.mean() or np.nan
    # higher is better
    return returns.sum() / avg_drawdown


@nb.njit(cache=True)
def pairs_deviation(returns, pairs, n_pairs, moment=2):
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
            (metrics[3], -pairs_deviation(returns, pairs, n_pairs)),
        )

    return nb.njit(loss_f, cache=True)


def generate_jit_ppdev(metrics, max_staked, col_idx, p_idx, n_pairs):
    def loss_f(results: List[BacktestResultTupleType]):
        returns = get_results_col(results, col_idx) / max_staked
        pairs = get_results_col(results, p_idx)
        return ((metrics[0], -pairs_deviation(returns, pairs, n_pairs)),)

    return nb.njit(loss_f, cache=True)
