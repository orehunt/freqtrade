from freqtrade.optimize.backtest_constants import Float64Cols
from types import SimpleNamespace
import numba as nb
from numba import njit, int64, float64
from pandas import DataFrame
import numpy as np
from numpy import ndarray
from typing import NamedTuple, Any
from collections import namedtuple
from freqtrade.optimize.backtest_constants import *
from freqtrade.optimize.backtest_utils import *


@njit(cache=True, nogil=True)
def for_trail_idx(index, trig_idx, next_sold) -> int64[:]:
    last = -2
    col = [0] * len(index)
    for i in range(len(index)):
        # for the first buy of the buy/sell chunk
        if next_sold[i] != next_sold[i - 1]:
            # last == -1 means that a bought with no trigloss is pending
            last = trig_idx[i] if trig_idx[i] != -3 else -1
            col[i] = last
            continue
        # check that we are past the last trigloss index
        if index[i] > last and last != -1:
            # if the trigloss is not null (-3), update last trigloss
            if trig_idx[i] != -3:
                last = trig_idx[i]
            else:
                last = -1
        col[i] = last
    return col


@njit(fastmath=True, cache=True, nogil=True)
def ofs_cummax(data_ofs: ndarray, data: ndarray) -> ndarray:
    """ groupby like cumulative maximum """
    cumarr = np.empty(len(data))
    p = data_ofs[0]
    for i in data_ofs[1:]:
        cumarr[p] = data[p]
        for n, v in enumerate(data[p + 1 : i], p + 1):
            cumarr[n] = np.maximum(cumarr[n - 1], v)
        p = i
    return cumarr


@njit(fastmath=True, cache=True, nogil=True)
def iter_trg_cols(start, arr, trg):
    for rn, r in enumerate(arr):
        # each trigger
        for cn, e in enumerate(r):
            if e:
                trg[start + rn] = cn
                return


@njit(fastmath=True, cache=True, nogil=True)
def ofs_first_flat_true(data_ofs: ndarray, arr: ndarray):
    trg = np.full(arr.shape[0], -1)
    # each bought
    for start, stop in zip(data_ofs[::], data_ofs[1::]):
        iter_trg_cols(start, arr[start:stop, :], trg)
    return trg


@njit(fastmath=True, nogil=True, cache=True)
def copy_ranges(
    chunk_start,
    chunk_stop,
    bought_ofs,
    data_df,
    data_bought,
    ohlc_vals,
    bought_vals,
    bought_ranges,
):
    ofs = 0
    for n, i in enumerate(bought_ofs[chunk_start:chunk_stop], chunk_start):
        rn = bought_ranges[n]
        data_df[ofs : ofs + rn] = ohlc_vals[i : i + rn]
        # these vals are repeated for each range
        data_bought[ofs : ofs + rn] = bought_vals[n]
        ofs += rn


@njit(fastmath=True, nogil=True, cache=True)
def split_cumsum(max_size: int, arr: ndarray):
    cumsum = 0
    splits = []
    for i, e in enumerate(arr):
        cumsum += e
        if cumsum > max_size:
            splits.append(i)
            cumsum = e
    return splits


@njit(cache=True, nogil=True)
def round_8(arr):
    # pass profits_prc as out, https://github.com/numba/numba/issues/4439
    return np.round(arr, 8, arr)


@njit(cache=True, nogil=True)
def pct_change(arr, period, window=None):
    pct = np.empty_like(arr)
    pct[:period] = np.nan
    pct[period:] += arr[period:] / arr[:-period]
    if window is not None and window > 0:
        wnd = np.empty_like(arr)
        pw = abs(period) + window
        wnd[:pw] = np.nan
        for n in range(window):
            wnd[pw:] += pct[pw + n : -pw + n]
        return wnd / window
    return pct


# https://stackoverflow.com/a/62841583/2229761
@njit(cache=True, nogil=True)
def shift_nb(arr, num, fill_value=np.nan, ofs=Union[None, ndarray]):
    if num >= 0:
        if arr.ndim > 1:
            shifted_shape = (num, arr.shape[1])
        else:
            shifted_shape = (num,)
        shifted = np.concatenate((np.full(shifted_shape, fill_value), arr[:-num]))
    else:
        if arr.ndim > 1:
            shifted_shape = (-num, arr.shape[1])
        else:
            shifted_shape = (-num,)
        shifted = np.concatenate((arr[-num:], np.full(shifted_shape, fill_value)))
    if ofs is not None:
        # if an offset array is given, fill every index present in ofs
        # for every step of the window size, respecting direction (sign)
        for n in range(0, num, np.sign(num)):
            shifted[ofs + n] = fill_value
    return shifted


# A Simple Estimation of Bid Ask spread
@njit(cache=True, nogil=True)
def calc_spread(high, low, close, ofs):
    # calc mid price
    # mid_range = (np.log(high) + np.log(low)) / 2
    # forward mid price
    # mid_range_1 = (np.log(shift_nb(high, -1)) + np.log(shift_nb(low, -1))) / 2
    # log_close = np.log(close)

    # spread formula
    # return np.sqrt(
    #     np.maximum(4 * (log_close - mid_range) * (log_close - mid_range_1), 0)
    # )
    return np.sqrt(
        np.maximum(
            4
            * (np.log(close) - (np.log(high) + np.log(low)) / 2)
            * (
                np.log(close)
                - (
                    np.log(shift_nb(high, -1, np.nan, ofs))
                    + np.log(shift_nb(low, -1, np.nan, ofs))
                )
                / 2
            ),
            0,
        )
    )


# amihud illiquidity measure
@njit(cache=True, nogil=True, inline="always")
def calc_illiquidity(close, volume, window=120, ofs=None) -> ndarray:
    # volume in quote currency
    volume_curr = volume * close
    # returns are NOT a ratio
    returns_volume_ratio = np.abs(
        ((close - shift_nb(close, 1, np.nan, ofs))) / volume_curr
    )
    rolling_rvr_sum = rolling_sum(returns_volume_ratio, window, ofs)
    return rolling_rvr_sum / window * 1e6


@njit(cache=True, nogil=True)
def calc_skewed_spread(high, low, close, volume, wnd, ofs):
    spread = calc_spread(high, low, close, ofs)
    # min max liquidity statistics over a rolling window to use for interpolation
    lix_norm = rolling_norm(calc_liquidity(volume, close, high, low), wnd, ofs)
    ilq_norm = rolling_norm(calc_illiquidity(close, volume, wnd, ofs), wnd, ofs)
    return skew_rate_by_liq(ilq_norm, lix_norm, np.zeros(spread.shape), spread)


@njit(cache=True, nogil=True)
def sim_high_low(open, close):
    """ when True, high happens before low """
    return close <= open


@nb.njit(cache=True, nogil=True)
def null_ofs_ranges(rolled, window, ofs, null_v):
    for start in ofs:
        rolled[start : start + window] = null_v


@njit(cache=True, nogil=True)
def rolling_sum(arr, window, ofs=None, null_v=np.nan):
    rsum = np.empty_like(arr)
    c = 0
    for n in range(window - 1, arr.shape[0]):
        rsum[n] = 0
        for v in arr[c : c + window]:
            rsum[n] += v
        c += 1
    if ofs is not None:
        null_ofs_ranges(rsum, window, ofs, null_v)
    return rsum


@njit(cache=True, nogil=True)
def rolling_norm(arr, window, ofs=None, null_v=np.nan, static=0):
    rnorm = np.empty_like(arr)
    rnorm[:window] = np.nan
    c = 0
    for n in range(window - 1, arr.shape[0]):
        mn, mx = arr[c], arr[c]
        for v in arr[c + 1 : c + window]:
            if np.isnan(v):
                continue
            if np.isnan(mx) or v > mx:
                mx = v
            elif np.isnan(mn) or v < mn:
                mn = v
        rnorm[n] = ((arr[n] - mn) / (mx - mn)) if mx != mn else static
        c += 1
    if ofs is not None:
        null_ofs_ranges(rnorm, window, ofs, null_v)
    return rnorm


# LIX formula
# values between ~5..~10 higher is more liquid
@njit(cache=True, nogil=True)
def calc_liquidity(volume, close, high, low):
    return np.log10((volume * close) / (high - low))


@njit(cache=True, nogil=True)
def calc_profits(open_rate: ndarray, close_rate: ndarray, stake_amount, fee) -> ndarray:
    # am = stake_amount / open_rate
    # open_amount = am * open_rate
    # close_amount = am * close_rate
    # open_price = open_amount + open_amount * fee
    # close_price = close_amount - close_amount * fee
    # profits_prc = close_price / open_price - 1
    return round_8(
        (
            (
                ((stake_amount / open_rate) * close_rate)
                - ((stake_amount / open_rate) * close_rate) * fee
            )
            / (
                ((stake_amount / open_rate) * open_rate)
                + ((stake_amount / open_rate) * open_rate) * fee
            )
            - 1
        )
    )


@njit(cache=True, nogil=True)
def stoploss_triggered_col(data, low, rate):
    return data[:, low] <= data[:, rate]


@njit(cache=True, nogil=True)
def calc_roi_close_rate(
    open_rate: ndarray, min_rate: ndarray, roi: ndarray, high_low: ndarray, fee: float
):
    roi_rate = -(open_rate * roi + open_rate * (1 + fee)) / (fee - 1)
    return np.fmax(roi_rate, min_rate)


@njit(cache=True, nogil=True)
def skew_rate_by_liq(ilq_norm, lix_norm, good_rate, bad_rate):
    # rationale: if liquidity is low, the execution rate will be closer to the bad rate
    # (bad rate is high or low depending if buy or sell)
    liq = lix_norm - ilq_norm
    liq_above = liq > 0
    rate = np.empty_like(good_rate)
    # skew by liquidity indicator depending on which one is dominant
    rate[liq_above] = (
        good_rate[liq_above] * lix_norm[liq_above]
        + bad_rate[liq_above] * (1 - lix_norm[liq_above])
    ) / 2
    not_liq_above = ~liq_above
    rate[not_liq_above] = (
        good_rate[not_liq_above] * ilq_norm[not_liq_above]
        + bad_rate[not_liq_above] * (1 - ilq_norm[not_liq_above])
    ) / 2
    # skew a second time, reducing the liquidity adjustement based
    # on the intensity of the liquidity dominance
    mean_rate = (good_rate + bad_rate) / 2
    rate[:] = (rate * np.abs(liq) + mean_rate * (1 - np.abs(liq))) / 2
    return rate


@njit(fastmath=True, cache=True, nogil=True)
def find_first(vec, t):
    """return the index of the first occurence of item in vec"""
    for n, v in enumerate(vec):
        if v > t:
            return n
    return n + 1


@njit(fastmath=True, cache=True, nogil=True)
def cummax(arr) -> ndarray:
    cmax = arr[0]
    cumarr = np.empty_like(arr)
    for i, val in enumerate(arr):
        if val > cmax:
            cmax = val
        cumarr[i] = cmax
    return cumarr


@njit(fastmath=True, cache=True, nogil=True)
def next_bought(b, bofs, current_ofs):
    return b + np.searchsorted(bofs, current_ofs, "right")


def update_tpdict(keys, values, tpdict):
    [set_tpdict_item(v, k, tpdict) for k, v in zip(keys, values)]


@nb.njit(cache=True, nogil=True)
def set_tpdict_item(arr, key, tpdict):
    tpdict[key] = arr


@njit(cache=True, nogil=True)
def calc_high_profit(*args):
    pass


@njit(cache=True, nogil=True)
def calc_high_profit_op(
    open_rate: float, close_rate: float, stake_amount: float, fee: float
) -> float:
    am = stake_amount / open_rate
    open_amount = am * open_rate
    close_amount = am * close_rate
    open_price = open_amount + open_amount * fee
    close_price = close_amount - close_amount * fee
    profits_prc = close_price / open_price - 1
    # pass profits_prc as out, https://github.com/numba/numba/issues/4439
    return np.round(profits_prc, 8)


@njit(cache=True, nogil=True)
def calc_stoploss_static(*args):
    return 0.0, 0.0


@njit(cache=True, nogil=True)
def calc_stoploss_static_op(open_rate, stoploss):
    stoploss_static = open_rate * (1 - stoploss)
    return stoploss_static, stoploss_static


@njit(cache=True, nogil=True)
def stoploss_is_triggered(*args):
    return False


@njit(cache=True, nogil=True)
def stoploss_is_triggered_op(n, low, stoploss_rate, stoploss_static, fl_cols):
    if low <= stoploss_rate:
        if stoploss_rate != stoploss_static:
            fl_cols["col_trailing_rate"][n] = stoploss_rate
            fl_cols["col_trailing_triggered"][n] = True
        else:
            fl_cols["col_stoploss_rate"][n] = stoploss_rate
            fl_cols["col_stoploss_triggered"][n] = True
        return True
    return False


@njit(cache=True, nogil=True)
def roi_is_triggered(*args):
    return False


@njit(cache=True, nogil=True)
def roi_is_triggered_op(n, i, high_profit, inv_roi_timeouts, inv_roi_values, fl_cols):
    for t, tm in enumerate(inv_roi_timeouts):
        # i is the timeframe counter for the current trade at the current position
        # tm is the timeframe count after which a roi ratio is enabled
        # a roi with tm == 0 should be enabled at i == 0
        # a roi with tm 3 should not be enabled at i == 2
        if tm <= i:
            if high_profit > inv_roi_values[t]:
                fl_cols["col_roi_profit"][n] = inv_roi_values[t]
                fl_cols["col_roi_triggered"][n] = True
                return True
    return False


@njit(cache=True, nogil=True)
def calc_trailing_rate(stoploss_rate, *args):
    return stoploss_rate


@njit(cache=True, nogil=True)
def calc_trailing_rate_op(stoploss_rate, tf, bl, high_profit, fl_cols, fl):
    if not (bl["sl_only_offset"] and high_profit < fl["sl_offset"]):
        return max(
            # trailing only increases
            stoploss_rate,
            # use positive ratio if above positive offset (default > 0)
            fl_cols["ohlc_high"][tf] * (1 - fl["sl_positive"])
            if bl["sl_positive_not_null"] and high_profit > fl["sl_offset"]
            # otherwise trailing with stoploss ratio
            else fl_cols["ohlc_high"][tf] * (1 - fl["stoploss"]),
        )
    return stoploss_rate


@njit(cache=True, nogil=True)
def get_last_trigger(*args):
    pass


@njit(cache=True, nogil=True)
def get_last_trigger_op(n):
    return n


@njit(cache=True, nogil=True)
def set_last_trigger(*args):
    pass


@njit(cache=True, nogil=True)
def set_last_trigger_op(n, tf, last_trigger, fl_cols):
    fl_cols["col_last_trigger"][last_trigger:n] = tf


@njit(cache=True, nogil=True)
def trade_is_overlap(*args):
    return False


@njit(cache=True, nogil=True)
def trade_is_overlap_op(bl, b, tf):
    return bl["not_position_stacking"] and b <= tf


@njit(cache=True, nogil=True)
def copy_trigger_data(b, tf, n, fl_cols, it_cols):
    fl_cols["col_trigger_bought_ofs"][n] = b
    fl_cols["col_trigger_date"][n] = it_cols["ohlc_date"][tf]
    fl_cols["col_trigger_ofs"][n] = tf
    return get_last_trigger(n)


def define_callbacks(feat):
    global calc_stoploss_static, stoploss_is_triggered
    if feat["stoploss_enabled"] or not feat["trailing_enabled"]:
        calc_stoploss_static = calc_stoploss_static_op
        stoploss_is_triggered = stoploss_is_triggered_op

    global calc_trailing_rate
    if feat["trailing_enabled"]:
        calc_trailing_rate = calc_trailing_rate_op

    global calc_high_profit
    if feat["trailing_enabled"] or feat["roi_enabled"]:
        calc_high_profit = calc_high_profit_op

    global roi_is_triggered
    if feat["roi_enabled"]:
        roi_is_triggered = roi_is_triggered_op

    global get_last_trigger, set_last_trigger, trade_is_overlap
    if feat["not_position_stacking"]:
        get_last_trigger = get_last_trigger_op
        set_last_trigger = set_last_trigger_op
        trade_is_overlap = trade_is_overlap_op


@njit(cache=True, nogil=True)
def iter_triggers(
    fl_cols,
    it_cols,
    names,
    fl,
    it,
    bl,
    cmap,
    nan_early_idx,
    roi_timeouts,
    roi_values,
    trg_range_max,
    # NOTE: used to invalidate cache among different configs
    feat,
):

    b = 0
    n_bought = len(it_cols["bought_ranges"])
    bought_ofs = it_cols["bought_ranges"][b]
    s_trg_roi_idx = set(it_cols["trg_roi_idx"])
    inv_roi_timeouts = roi_timeouts[::-1]
    inv_roi_values = fl_cols["roi_vals"][::-1]

    last_trigger = -1
    triggered = False
    end_ofs = it["end_ofs"]

    tf = it_cols["bofs"][0] - 1
    for n, b in enumerate(it_cols["bofs"]):
        # skip if position stacking is NOT enabled
        if trade_is_overlap(bl, b, tf):
            continue
        # keep track of the last trigger and reset
        if triggered:
            set_last_trigger(n, tf, last_trigger, fl_cols)
            triggered = False
        # data for the current trade
        open_rate = fl_cols["bopen"][n]
        stoploss_static, stoploss_rate = calc_stoploss_static(open_rate, fl["stoploss"])
        # loop over the range of the current trade
        tf = b
        for i, low in enumerate(
            fl_cols["ohlc_low"][b : b + it_cols["bought_ranges"][n]]
        ):
            # if low happens first
            if not fl_cols["high_low"][tf]:
                # check stoploss against previous rate
                if triggered := stoploss_is_triggered(
                    n, low, stoploss_rate, stoploss_static, fl_cols
                ):
                    last_trigger = copy_trigger_data(b, tf, n, fl_cols, it_cols)
                    break

            # otherwise check against new high
            high_profit = calc_high_profit(
                open_rate, fl_cols["ohlc_high"][tf], fl["stake_amount"], fl["fee"]
            )
            stoploss_rate = calc_trailing_rate(
                stoploss_rate, tf, bl, high_profit, fl_cols, fl
            )
            if triggered := stoploss_is_triggered(
                n,
                low,
                stoploss_rate,
                stoploss_static,
                fl_cols
                # check roi only if updated trailing is not triggered
            ) or roi_is_triggered(
                n, i, high_profit, inv_roi_timeouts, inv_roi_values, fl_cols
            ):
                last_trigger = copy_trigger_data(b, tf, n, fl_cols, it_cols)
                break
            tf += 1
    # update the last last_trigger slice only if the last bought is sold with a trigger
    if triggered and bl["not_position_stacking"]:
        fl_cols["col_last_trigger"][last_trigger:n_bought] = tf
    return
