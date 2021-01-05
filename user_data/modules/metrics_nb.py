import numba as nb
import numpy as np
import vectorbt as vbt
from vectorbt.returns import nb as rnb

from freqtrade.optimize.backtest_nb import calc_illiquidity, calc_liquidity, rolling_norm


i32mx = np.iinfo(np.int32).max
lmx = np.log10(i32mx)
lmn = -lmx


@nb.njit(cache=True)
def normalize(arr):
    mn = np.nanmin(arr)
    norm = (arr - mn) / (np.nanmax(arr) - mn)
    norm[~np.isfinite(norm)] = 0
    return norm


@nb.njit(cache=True)
def normalize_scalar_log10(arr, mn=lmn, mx=lmx):
    arr[np.isnan(arr) | (arr == 0)] = 1
    arr[arr == np.inf] = i32mx
    arr[arr == -np.inf] = -i32mx
    arr10 = np.log10(np.abs(arr)) * np.sign(arr)
    norm = (arr10 - mn) / (mx - mn)
    # shouldn't be needed
    # norm = np.fmax(np.fmin(norm, 1), 0)
    return norm.sum()


# @nb.njit(cache=True):
# def only_ratio_metrics(main_ratio: str, )


@nb.njit(cache=True)
def higher_is_better_metrics(rt, af):
    hib = np.empty(9)
    hib[0] = rnb.calmar_ratio_1d_nb(rt, af)
    hib[1] = rnb.max_drawdown_1d_nb(rt)
    hib[2] = rnb.sortino_ratio_1d_nb(rt, af)
    hib[3] = rnb.sharpe_ratio_1d_nb(rt, af)
    hib[4] = rnb.annualized_return_1d_nb(rt, af)
    hib[5] = rnb.tail_ratio_1d_nb(rt)
    hib[6] = rnb.omega_ratio_1d_nb(rt, af)
    # winrate
    hib[7] = rt[rt > 0].shape[0] / (rt[rt <= 0].shape[0] or 1)
    return normalize_scalar_log10(hib), hib


@nb.njit(cache=True)
def lower_is_better_metrics(
    rt: np.ndarray,
    af: float,
    ts: np.ndarray,
    trade_duration: np.ndarray,
    trade_start: np.ndarray,
):
    """
    :rt sparse array of returns
    :af annual factor
    :ts contiguous arr of dates
    """
    lib = np.empty(10)
    lib[0] = rnb.downside_risk_1d_nb(rt, af)
    lib[1] = rnb.conditional_value_at_risk_1d_nb(rt)
    lib[2] = rnb.annualized_volatility_1d_nb(rt, af)

    rt_1d = rt.reshape((rt.shape[0], 1))
    # dd_rec = vbt.records.nb.drawdown_records_nb(rt_1d)
    dd_rec = vbt.generic.drawdowns.nb.find_drawdowns_nb(rt_1d)
    # avg drawdown dur
    lib[3] = np.nanmean(ts[dd_rec["end_idx"]] - ts[dd_rec["start_idx"]])
    # avg drawdown
    lib[4] = np.nanmean(rnb.drawdown_1d_nb(rt))
    # min return
    lib[5] = abs(rt.min())
    # avg trades dur
    lib[6] = trade_duration.mean()
    # max trade duration
    lib[7] = trade_duration.max()
    # avg trades interval
    trade_diff = np.diff(trade_start)
    if len(trade_diff):
        lib[8] = trade_diff.mean()
        # max trade interval
        lib[9] = trade_diff.max()
    else:
        lib[8] = 0
        lib[9] = 0

    lib[np.isnan(lib)] = 0
    return normalize_scalar_log10(lib), lib


## reference normalization applied to portfolio class
def best_metric(pf):
    # higher is better metrics
    norm_sum = np.zeros(pf.returns.shape[1])
    for m in (
        "calmar_ratio",
        "max_drawdown",
        "sortino_ratio",
        "sharpe_ratio",
        "total_return",
        "annualized_return",
        "tail_ratio",
        "omega_ratio",
    ):
        norm_sum += np.nan_to_num(normalize(getattr(pf, m).values))
    # lower is better
    for m in ("downside_risk", "conditional_value_at_risk", "annualized_volatility"):
        norm_sum -= np.abs(normalize(getattr(pf, m).values))
    # post metrics
    norm_sum -= normalize(pf.drawdowns.avg_drawdown().values)
    norm_sum -= normalize(pf.drawdowns.avg_duration().values)

    norm_sum += normalize(pf.trades.win_rate().values)
    norm_sum += normalize(pf.returns.max().values)
    norm_sum -= normalize(pf.returns.min().abs().values)
    norm_sum -= normalize(pf.trades.duration.max().values)
    # average trades interval, higher is worse
    sum_mean_nb = nb.njit(lambda col, i, a: np.nanmean(np.diff(a)), cache=True)
    norm_sum -= normalize(
        records["entry_idx"]
        .vbt.groupby_apply(records["col"].values, sum_mean_nb)
        .values
    )
    # max trade interval
    sum_max_nb = nb.njit(lambda col, i, a: np.nanmax(np.diff(a)), cache=True)
    norm_sum -= normalize(
        records["entry_idx"].vbt.groupby_apply(records["col"].values, sum_max_nb).values
    )
    return norm_sum.argmax(), norm_sum


@nb.njit(cache=True)
def get_ilq(rw, iw, clo, vol):
    return rolling_norm(
        calc_illiquidity(
            clo,
            vol,
            window=iw,
            ofs=None,
        ),
        rw,
    )


@nb.njit(cache=True)
def get_lix(rw, vol, clo, hi, lo):
    return rolling_norm(
        calc_liquidity(vol, clo, hi, lo),
        rw,
    )


@nb.njit(cache=True)
def rolling_mean(arr, window):
    mu = np.empty(arr.shape[0], dtype=arr.dtype)
    mu[:window] = np.nan
    for n in range(window, arr.shape[0]):
        mu[n - 1] = np.mean(arr[n - window : n])
    return mu


@nb.njit(cache=True)
def mean_x1(arr):
    mu = np.empty(arr.shape[0])
    c = arr.shape[1]
    for n in range(arr.shape[0]):
        mu[n] = arr[n].sum() / c
    return mu


@nb.njit(cache=True)
def vprice(ohlcv, pw=240, vw=120):
    rol_v = rolling_mean(ohlcv[:, 4], window=vw)
    rol_v_1 = np.empty_like(rol_v)
    rol_v_1[0] = 0
    rol_v_1[1:] = rol_v[:-1]
    v_change = rol_v / rol_v_1
    mu_price = mean_x1(ohlcv[:, 0:4])
    rol_mu = rolling_mean(mu_price, window=pw)
    v = rol_mu * np.log(v_change)
    v[~np.isfinite(v)] = 0
    return v.cumsum()


@nb.njit(cache=True)
def get_vp(ohlcv, rw, pw, vw):
    return rolling_norm(vprice(ohlcv, pw, vw), rw)
