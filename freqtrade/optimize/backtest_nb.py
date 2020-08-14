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


@njit(nogil=True, cache=True, inline="always")
def nan_early_roi(roi_triggers, nan_early_idx, br, n_timeouts, roi_timeouts):
    # roi_triggers[nan_early_idx[nan_early_idx <= br * n_timeouts]] = False
    for n, rt in enumerate(roi_timeouts):
        roi_triggers[:rt, n] = False
    return roi_triggers


@njit(fastmath=True, nogil=True, cache=True)
def check_profits(cur_profits, roi_vals):
    return cur_profits >= roi_vals


@njit(fastmath=False, nogil=True, cache=True)
def compare_roi_triggers(cur_profits, roi_vals, roi_timeouts, trg_range, trg_roi_idx):
    for n, rt in enumerate(roi_timeouts):
        trg_range[rt:, trg_roi_idx[n]] = cur_profits[rt:] >= roi_vals[n]


@njit(fastmath=True, nogil=True, cache=True)
def get_valid_cols(trg_first_idx, trg_range):
    return np.array([n for n, v in enumerate(trg_first_idx) if trg_range[v, n] != 0])


@njit(fastmath=True, nogil=True, cache=True)
def get_first_triggers(trg_first_idx, trg_range):
    # valid_cols = np.array(
    #         [n for n, v in enumerate(trg_first_idx) if trg_range[v, n] != 0]
    # )
    valid_cols = get_valid_cols(trg_first_idx, trg_range)
    if len(valid_cols):
        return np.where(trg_first_idx[valid_cols].min() == trg_first_idx)[0]
    else:
        return valid_cols


@njit(fastmath=True, nogil=True, cache=True)
def copy_ranges(
    range_vals, data_ofs, data_df, data_bought, ohlc_vals, bought_vals, bought_ranges
):
    for n, i in enumerate(range_vals):
        start, stop = data_ofs[n], data_ofs[n + 1]
        data_df[start:stop] = ohlc_vals[i : i + bought_ranges[n]]
        # these vals are repeated for each range
        data_bought[start:stop] = bought_vals[n]


@njit(cache=True, nogil=True)
def calc_profits(open_rate: ndarray, close_rate: ndarray, stake_amount, fee) -> ndarray:
    am = stake_amount / open_rate
    open_amount = am * open_rate
    close_amount = am * close_rate
    open_price = open_amount + open_amount * fee
    close_price = close_amount - close_amount * fee
    profits_prc = close_price / open_price - 1
    # pass profits_prc as out, https://github.com/numba/numba/issues/4439
    return np.round(profits_prc, 8, profits_prc)


@njit(cache=True, nogil=True)
def calc_roi_close_rate(
    open_rate: ndarray, min_rate: ndarray, roi: ndarray, fee: float
):
    roi_rate = -(open_rate * roi + open_rate * (1 + fee)) / (fee - 1)
    return np.fmax(roi_rate, min_rate)


@njit(cache=True, nogil=True)
def calc_stoploss_rate(
    open_rate, low_range, stoploss, trg_range, trg_col_stoploss_triggered,
):
    # calculate the rate from the bought candle
    stoploss_triggered_rate = open_rate * (1 - stoploss)
    trg_range[:, trg_col_stoploss_triggered] = low_range <= stoploss_triggered_rate
    return stoploss_triggered_rate


@njit(cache=True, nogil=True)
def calc_trailing_rate(
    calc_offset,
    sl_positive,
    sl_only_offset,
    sl_offset,
    cur_profits,
    high_range,
    stoploss,
    trg_range,
    low_range,
    trg_col_trailing_triggered,
):
    if calc_offset:
        trailing_offset_reached = cummax(cur_profits) >= sl_offset
        trailing_rate = (
            (
                np.where(
                    trailing_offset_reached,
                    cummax(high_range * (1 - sl_positive)),
                    cummax(high_range * (1 - stoploss))
                    if not sl_only_offset
                    else np.full(cur_profits.shape, np.nan),
                )
            )
            if sl_positive
            else (
                np.where(
                    trailing_offset_reached, cummax(high_range * (1 - stoploss)), np.nan
                )
            )
        )
    else:
        trailing_rate = cummax(high_range * (1 - stoploss))
    trg_range[:, trg_col_trailing_triggered] = low_range <= trailing_rate
    return trailing_rate


@njit(fastmath=True, cache=True, nogil=True)
def choose_trigger(
    trg_first,
    trg_first_idx,
    col_trailing_triggered,
    col_trailing_rate,
    col_stoploss_triggered,
    col_stoploss_rate,
    col_roi_triggered,
    col_roi_profit,
    stoploss_enabled,
    trailing_enabled,
    roi_enabled,
    trg_col_trailing_triggered,
    trg_col_stoploss_triggered,
    stoploss_triggered_rate,
    trailing_rate,
    b,
    bought_ofs,
    roi_values,
    trg_roi_pos,
    ohlc_date,
    col_trigger_ofs,
    col_trigger_date,
    col_trigger_bought_ofs,
    not_stop_over_trail,
    s_trg_roi_idx,
):
    # get the column index that triggered first row wise
    # NOTE: here we get all the first triggers that happened on the same candle
    # trg_first = where(amin(trg_first_idx[valid_cols]) == trg_first_idx)[0]
    trg_top = trg_first[0]
    # lastly get the trigger offset from the index of the first valid column
    # any trg_first index would return the same value here
    trg_ofs = trg_first_idx[trg_top]
    # check what trigger it is and copy related columns values
    # from left they are ordered stoploss, trailing, roi
    if (
        trailing_enabled
        and trg_top == trg_col_trailing_triggered
        and not_stop_over_trail
    ):
        col_trailing_triggered[b] = int64(True)
        col_trailing_rate[b] = trailing_rate[trg_ofs]
    elif stoploss_enabled and trg_top == trg_col_stoploss_triggered:
        col_stoploss_triggered[b] = int64(True)
        col_stoploss_rate[b] = stoploss_triggered_rate
    elif roi_enabled and trg_top in s_trg_roi_idx:
        col_roi_triggered[b] = int64(True)
        # NOTE: scale trg_first by how many preceding columns (stoploss,trailing)
        # there are before roi columns, in order to get the offset
        # relative to only the (ordered) roi columns
        # and pick the minimum value from the right (roi_values have to be pre osn.rdered desc)
        col_roi_profit[b] = roi_values[trg_first[-1] - trg_roi_pos]
    # trigger ofs is relative to the bought range, so just add it to the bought offset
    current_ofs = bought_ofs + trg_ofs
    # copy general trigger values shared by asn.ll trigger types
    col_trigger_ofs[b] = current_ofs
    col_trigger_date[b] = ohlc_date[current_ofs]
    col_trigger_bought_ofs[b] = bought_ofs
    return current_ofs


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


@njit(fastmath=True, cache=True, nogil=True)
def zero_arr(n, tp):
    return np.zeros(n, dtype=tp)


@njit(fastmath=True, cache=True, nogil=True)
def loop_max(arr, c):
    m = 0
    for r, v in enumerate(arr[1:, c], 1):
        if v > arr[m, c]:
            m = r
    return m


@njit(fastmath=True, cache=True, nogil=True)
def range_cols(maxes, cols, arr):
    for c in range(cols):
        maxes[c] = loop_max(arr, c)
    return maxes


@njit(fastmath=True, cache=True, nogil=True)
def rowargmax(arr) -> ndarray:
    cols = arr.shape[1]
    return range_cols(zero_arr(cols, nb.int64), cols, arr)


@njit(fastmath=True, cache=False, nogil=True)
def first_flat_true(arr):
    n_cols = arr.shape[1]
    for rn, r in enumerate(arr[:, :]):
        if r[1] == True:
            return rn, 1
        if r[0] == True:
            return rn, 0
        # for roi iterate in reverse
        for cn, e in zip(range(n_cols - 3, -1, -1), r[:1:-1]):
            if e == True:
                return rn, cn
    return -1, -1


def update_tpdict(keys, values, tpdict):
    [set_tpdict_item(v, k, tpdict) for k, v in zip(keys, values)]


@nb.njit(cache=True, nogil=True)
def set_tpdict_item(arr, key, tpdict):
    tpdict[key] = arr


@nb.njit(cache=True)
def select_triggers(
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
):
    b = 0
    n_bought = len(it_cols["bought_ranges"])
    bought_ofs = it_cols["bought_ranges"][b]
    s_trg_roi_idx = set(it_cols["trg_roi_idx"])

    end_ofs = it["end_ofs"]
    while bought_ofs < end_ofs:
        # check trigger for the range of the current bought
        triggered = False
        br = it_cols["bought_ranges"][b]
        bought_ofs_stop = bought_ofs + br
        trg_range = trg_range_max[:br]
        open_rate = fl_cols["bopen"][b]

        if bl["roi_or_trailing"]:
            high_range = fl_cols["ohlc_high"][bought_ofs:bought_ofs_stop]
            cur_profits = calc_profits(
                open_rate, high_range, fl["stake_amount"], fl["fee"]
            )
        if bl["stoploss_or_trailing"]:
            low_range = fl_cols["ohlc_low"][bought_ofs:bought_ofs_stop]
        if bl["roi_enabled"]:
            # get a view of the roi triggers because we need to nan indexes
            # relative to (flattened) roi triggers only
            # roi_triggers = (cur_profits >= roi_values).swapaxes(0, 1).flatten()
            roi_triggers = np.ravel(np.transpose(cur_profits >= roi_values))
            # NOTE: clip nan_early_idx to the length of the bought_range
            # NOTE: use False, not nan, since bool(nan) == True
            roi_triggers[nan_early_idx[nan_early_idx <= br * it["n_timeouts"]]] = False
            # roi_triggers.shape = (br, n_timeouts)
            # trg_range[:, it_cols["trg_roi_idx"]] = np.reshape(roi_triggers, (br, it["n_timeouts"]))
            compare_roi_triggers(
                cur_profits,
                fl_cols["roi_vals"],
                roi_timeouts,
                trg_range,
                it_cols["trg_roi_idx"],
            )
        if bl["stoploss_enabled"]:
            # calculate the rate from the bought candle
            stoploss_triggered_rate = calc_stoploss_rate(
                open_rate,
                low_range,
                fl["stoploss"],
                trg_range,
                it["trg_col_stoploss_triggered"],
            )
            # stoploss_triggered_rate = open_rate * (1 - fl["stoploss"])
            # trg_range[:, it["trg_col_stoploss_triggered"]] = (
            # low_range <= stoploss_triggered_rate
            # )
        if bl["trailing_enabled"]:
            trailing_rate = calc_trailing_rate(
                bl["calc_offset"],
                fl["sl_positive"],
                bl["sl_only_offset"],
                fl["sl_offset"],
                cur_profits,
                high_range,
                fl["stoploss"],
                trg_range,
                low_range,
                it["trg_col_trailing_triggered"],
            )
            # trailing_rate = cummax(high_range * (1 - fl["stoploss"]))
            #     trailing_offset_reached = cummax(cur_profits) >=
            # if fl["sl_positive"]:
            #     trailing_rate[trailing_offset_reached] = cummax(
            #         high_range[trailing_offset_reached] * (1 - fl["sl_positive"])
            #     )
            # if bl["sl_only_offset"]:
            #     trailing_rate[~trailing_offset_reached] = np.nan
            # trg_range[:, it["trg_col_trailing_triggered"]] = low_range <= trailing_rate
        # apply argmax over axis 0, such that we get the first timeframe
        # where a trigger happened (argmax of each column)
        trg_first_idx = rowargmax(trg_range)
        # filter out columns that have no true trigger
        # NOTE: the list is very small here (<10) so it might make sense to use python
        # but the big speed up seen here does not match outside testing of same lists lengths..
        # valid_cols = flatnonzero(trg_range[trg_first_idx, trg_idx])
        # valid_cols = [
        #     i for i, val in enumerate(trg_range[trg_first_idx, trg_idx]) if val != 0
        # ]
        # valid_cols = np.array(
        #     [n for n, v in enumerate(trg_first_idx) if trg_range[v, n] != 0]
        # )
        trg_first = get_first_triggers(trg_first_idx, trg_range)
        # check that there is at least one valid trigger
        if len(trg_first):
            # get the column index that triggered first row sie
            trg_top = trg_first[0]
            # lastly get the trigger offset from the index of the first valid column
            trg_ofs = trg_first_idx[trg_top]
            # check what trigger it is and copy related columns values
            if (
                bl["trailing_enabled"]
                and trg_top == it["trg_col_trailing_triggered"]
                and bl["not_stop_over_trail"]
            ):
                fl_cols["col_trailing_triggered"][b] = 1
                fl_cols["col_trailing_triggered"][b] = 1
                fl_cols["col_trailing_rate"][b] = trailing_rate[trg_ofs]
            elif bl["stoploss_enabled"] and trg_top == it["trg_col_stoploss_triggered"]:
                fl_cols["col_stoploss_triggered"][b] = 1
                fl_cols["col_stoploss_rate"][b] = stoploss_triggered_rate
            elif bl["roi_enabled"] and trg_top in s_trg_roi_idx:
                fl_cols["col_roi_triggered"][b] = 1
                # NOTE: scale trg_first by how many preceding columns (stoploss,trailing)
                # there are before roi columns, in order to get the offset
                # relative to only the (ordered) roi columns
                fl_cols["col_roi_profit"][b] = fl_cols["roi_vals"][
                    trg_first[-1] - it["trg_roi_pos"]
                ]
            # trigger ofs is relative to the bought range, so just add it to the bought offset
            current_ofs = bought_ofs + trg_ofs
            # copy general trigger values shared by all trigger types
            fl_cols["col_trigger_ofs"][b] = current_ofs
            fl_cols["col_trigger_date"][b] = it_cols["ohlc_date"][current_ofs]
            fl_cols["col_trigger_bought_ofs"][b] = bought_ofs
            triggered = True
        if bl["not_position_stacking"]:
            if triggered:
                last_trigger = b
                # get the first row where the bought index is
                # higher than the current stoploss index
                # b += np.searchsorted(bofs[b:], current_ofs, "right")
                # b += find_first(it_cols["bofs"][b:], current_ofs)
                b = next_bought(b, it_cols["bofs"][b:], current_ofs)
                # repeat the trigger index for the boughts in between the trigger
                # and the bought with higher idx
                fl_cols["col_last_trigger"][last_trigger:b] = current_ofs
                if b < n_bought:
                    bought_ofs = it_cols["bofs"][b]
                else:
                    break
            else:  # if no triggers executed, jump to the first bought after next sold idx
                # b += np.searchsorted(bofs[b:], bsold[b], "right")
                # b += find_first(it_cols["bofs"][b:], it_cols["bsold"][b])
                b = next_bought(b, it_cols["bofs"][b:], it_cols["bsold"][b])
                if b < n_bought:
                    bought_ofs = it_cols["bofs"][b]
                else:
                    break
        else:
            b += 1
            if b < n_bought:
                bought_ofs = it_cols["bofs"][b]
            else:
                break
    return


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
                break
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
