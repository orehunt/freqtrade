from freqtrade.optimize.backtest_constants import Float64Cols
from types import SimpleNamespace
import numba as nb
from numba import njit, int64, float64
from pandas import DataFrame
import numpy as np
from numpy import ndarray
from typing import NamedTuple
from collections import namedtuple
from freqtrade.optimize.backtest_constants import *


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
                trg[start+rn] = cn
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
    bought_ofs, data_ofs, data_df, data_bought, ohlc_vals, bought_vals, bought_ranges
):
    for n, i in enumerate(bought_ofs):
        start, stop = data_ofs[n], data_ofs[n + 1]
        data_df[start:stop] = ohlc_vals[i : i + bought_ranges[n]]
        # these vals are repeated for each range
        data_bought[start:stop] = bought_vals[n]

@njit(cache=True, nogil=True)
def round_8(arr):
    # pass profits_prc as out, https://github.com/numba/numba/issues/4439
    return np.round(arr, 8, arr)


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
    open_rate: ndarray, min_rate: ndarray, roi: ndarray, fee: float
):
    roi_rate = -(open_rate * roi + open_rate * (1 + fee)) / (fee - 1)
    return np.fmax(roi_rate, min_rate)


@njit(fastmath=True, cache=True, nogil=True)
def find_first(vec, t):
    """return the index of the first occurence of item in vec"""
    for n, v in enumerate(vec):
        if v > t:
            return n
    return n + 1

@njit(fastmath=True, cache=True, nogil=True)
def find_with_dups(haystack, needles):
    pos = np.full(needles.shape[0], -1)
    found = set()
    for i, d in enumerate(needles):
        for n, h in enumerate(haystack):
            if h == d and n not in found:
                pos[i] = n
                found.add(n)
                break
    return pos

@njit(fastmath=True, cache=True, nogil=True)
def increase(arr):
    out = np.empty(arr.shape[0], nb.int64)
    inc = 0
    for n, e in enumerate(arr):
        if e != arr[n-1]:
            out[n] = e
            inc = 0
        else:
            inc += 1
            out[n] = e + inc
    return out

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
        # tf is the duration of the current trade in timeframes
        # tm is the timeframe count after which a roi ratio is enabled
        # a roi with tm == 0 should be enabled at tf == 0
        # a roi with tm 3 should not be enabled at tf 2
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
    # if not (bl["sl_only_offset"] and high_profit < fl["sl_offset"]):
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
        # if bl["not_position_stacking"] and b <= tf:
        if trade_is_overlap(bl, b, tf):
            continue
        if triggered:
            set_last_trigger(n, tf, last_trigger, fl_cols)
            triggered = False
        open_rate = fl_cols["bopen"][n]
        stoploss_static, stoploss_rate = calc_stoploss_static(open_rate, fl["stoploss"])
        tf = b
        for i, low in enumerate(
            fl_cols["ohlc_low"][b : b + it_cols["bought_ranges"][n]]
        ):
            high_profit = calc_high_profit(
                open_rate, fl_cols["ohlc_high"][tf], fl["stake_amount"], fl["fee"]
            )
            stoploss_rate = calc_trailing_rate(
                stoploss_rate, tf, bl, high_profit, fl_cols, fl
            )
            if triggered := stoploss_is_triggered(
                n, low, stoploss_rate, stoploss_static, fl_cols
            ) or roi_is_triggered(
                n, i, high_profit, inv_roi_timeouts, inv_roi_values, fl_cols
            ):
                fl_cols["col_trigger_bought_ofs"][n] = b
                fl_cols["col_trigger_date"][n] = it_cols["ohlc_date"][tf]
                fl_cols["col_trigger_ofs"][n] = tf
                last_trigger = get_last_trigger(n)
                break
            tf += 1
    # update the last last_trigger slice if the last bought is sold with trigger
    if triggered and bl["not_position_stacking"]:
        fl_cols["col_last_trigger"][last_trigger:n_bought] = tf
    return
