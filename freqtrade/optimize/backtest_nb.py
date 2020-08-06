import numba as nb
from numba import njit, int64, float64
from pandas import DataFrame
import numpy as np
from numpy import ndarray
from typing import NamedTuple

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
    cumarr = empty(len(data))
    p = data_ofs[0]
    for i in data_ofs[1:]:
        cumarr[p] = data[p]
        for n, v in enumerate(data[p + 1 : i], p + 1):
            cumarr[n] = maximum(cumarr[n - 1], v)
        p = i
    return cumarr

@njit(fastmath=True, nogil=True)
def copy_ranges(
    range_vals, data_ofs, data_df, data_bought, ohlc_vals, bought_vals, bought_ranges
):
    for n, i in enumerate(range_vals):
        start, stop = data_ofs[n], data_ofs[n + 1]
        data_df[start:stop] = ohlc_vals[i : i + bought_ranges[n]]
        # these vals are repeated for each range
        data_bought[start:stop] = bought_vals[n]

@njit(cache=True, nogil=True)
def calc_profits(
    open_rate: ndarray, close_rate: ndarray, stake_amount, fee
) -> ndarray:
    sa = stake_amount
    am = sa / open_rate
    open_amount = am * open_rate
    close_amount = am * close_rate
    open_price = open_amount + open_amount * fee
    close_price = close_amount - close_amount * fee
    profits_prc = close_price / open_price - 1
    # pass profits_prc as out, https://github.com/numba/numba/issues/4439
    return np.round(profits_prc, 8, profits_prc)

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
def rowargmax(arr) -> ndarray:
    cols = arr.shape[1]
    maxes = np.zeros(cols, dtype=nb.int64)
    for c in range(cols):
        m = 0
        for r, v in enumerate(arr[1:, c], 1):
            if v > arr[m, c]:
                m = r
        maxes[c] = m
    return maxes


class CharLists(NamedTuple):
    col_names: nb.types.unicode_type[:]
    trg_names: nb.types.unicode_type[:]
    stoploss_cols_names: nb.types.unicode_type[:]
    trailing_cols_names: nb.types.unicode_type[:]


class Flags(NamedTuple):
    roi_enabled: nb.types.boolean
    stoploss_enabled: nb.types.boolean
    trailing_enabled: nb.types.boolean
    sl_only_offset: nb.types.boolean


class Int32Vals(NamedTuple):
    n_timeouts: nb.int32
    trg_n_cols: nb.int32


class Float64Vals(NamedTuple):
    fee: nb.float64
    stake_amount: nb.float64
    stoploss: nb.float64
    sl_positive: nb.float64
    sl_offset: nb.float64


class Int64Cols(NamedTuple):
    bought_ohlc_ofs: nb.int64[:]
    bought_next_sold_ofs: nb.int64[:]
    df_date: nb.int64[:]
    df_ohlc_ofs: nb.int64[:]


class Float64Cols(NamedTuple):
    df_low: nb.float64[:]
    df_high: nb.float64[:]
    bought_open: nb.float64[:]
    roi_values: nb.float64[:]
    roi_vals: nb.float64[:]


@nb.njit(cache=True, nogil=True, nopython=True)
def select_triggers(
    fl_cols,
    it_cols,
    names,
    fl,
    it,
    bl,
    col,
    trg_col,
    roi_cols,
    trg_roi_cols,
    col_names,
    trg_names,
    nan_early_idx,
    roi_timeouts,
    bought_ranges,
):

    trg_n_cols = it.trg_n_cols
    roi_vals = fl_cols.roi_vals
    roi_values = fl_cols.roi_values
    stake_amount = fl.stake_amount
    fee = fl.fee
    n_timeouts = it.n_timeouts

    roi_enabled = bl.roi_enabled
    stoploss_enabled = bl.stoploss_enabled
    trailing_enabled = bl.trailing_enabled
    roi_or_trailing = roi_enabled or trailing_enabled
    stoploss_or_trailing = stoploss_enabled or trailing_enabled

    sl_positive = abs(fl.sl_positive)
    sl_offset = fl.sl_offset
    sl_only_offset = bl.sl_only_offset
    stoploss = abs(fl.stoploss)
    calc_offset = sl_positive or sl_only_offset

    # make views of each column for faster indexing
    bofs = it_cols.bought_ohlc_ofs
    n_bought = len(bofs)
    bsold = it_cols.bought_next_sold_ofs
    bopen = fl_cols.bought_open

    ohlc_low = fl_cols.df_low
    ohlc_high = fl_cols.df_high
    ohlc_date = it_cols.df_date

    b = 0
    last_trigger = -1
    bought_ofs = bofs[b]
    current_ofs = bought_ofs
    end_ofs = it_cols.df_ohlc_ofs[-1]

    roi_idx = roi_cols
    trg_roi_idx = np.array(trg_roi_cols)
    trg_roi_pos = len(trg_col)
    trg_idx = np.arange(trg_n_cols)

    triggers = np.zeros(shape=(len(bofs), len(names.col_names)))
    triggers[:, col["last_trigger"]] = -1
    triggers[:, col["trigger_ofs"]] = np.nan

    col_last_trigger = triggers[:, col["last_trigger"]]
    col_trigger_bought_ofs = triggers[:, col["trigger_bought_ofs"]]
    col_trigger_date = triggers[:, col["trigger_date"]]
    col_trigger_ofs = triggers[:, col["trigger_ofs"]]
    if trailing_enabled:
        col_trailing_rate = triggers[:, col["trailing_rate"]]
        idx_trailing_triggered = col["trailing_triggered"]
        col_trailing_triggered = triggers[:, idx_trailing_triggered]
        trg_col_trailing_triggered = trg_col["trailing_triggered"]
    if roi_enabled:
        col_roi_profit = triggers[:, col["roi_profit"]]
        col_roi_triggered = triggers[:, col["roi_triggered"]]
        s_trg_roi_idx = set(trg_roi_idx)
    else:
        s_trg_roi_idx = set([-1])

    if stoploss_enabled:
        col_stoploss_rate = triggers[:, col["stoploss_rate"]]
        col_stoploss_triggered = triggers[:, col["stoploss_triggered"]]
        trg_col_stoploss_triggered = trg_col["stoploss_triggered"]

    trg_range_max = np.empty(shape=(bought_ranges.max(), trg_n_cols))

    while bought_ofs < end_ofs:
        # check trigger for the range of the current bought
        br = bought_ranges[b]
        bought_ofs_stop = bought_ofs + br
        trg_range = trg_range_max[:br]

        if roi_or_trailing:
            high_range = ohlc_high[bought_ofs:bought_ofs_stop]
            cur_profits = calc_profits(bopen[b], high_range, stake_amount, fee)
        if stoploss_or_trailing:
            low_range = ohlc_low[bought_ofs:bought_ofs_stop]
        if roi_enabled:
            # get a view of the roi triggers because we need to nan indexes
            # relative to (flattened) roi triggers only
            # roi_triggers = (cur_profits >= roi_values).swapaxes(0, 1).flatten()
            roi_triggers = np.ravel(np.transpose(cur_profits >= roi_values))
            # NOTE: clip nan_early_idx to the length of the bought_range
            # NOTE: use False, not nan, since bool(nan) == True
            roi_triggers[nan_early_idx[nan_early_idx <= br * n_timeouts]] = False
            # roi_triggers.shape = (br, n_timeouts)
            trg_range[:, trg_roi_idx] = np.reshape(roi_triggers, (br, n_timeouts))
        if stoploss_enabled:
            # calculate the rate from the bought candle
            stoploss_triggered_rate = bopen[b] * (1 - stoploss)
            trg_range[:, trg_col_stoploss_triggered] = (
                low_range <= stoploss_triggered_rate
            )
        if trailing_enabled:
            trailing_rate = cummax(high_range * (1 - stoploss))
            if calc_offset:
                trailing_offset_reached = cummax(cur_profits) >= sl_offset
            if sl_positive:
                trailing_rate[trailing_offset_reached] = cummax(
                    high_range * (1 - sl_positive)
                )[trailing_offset_reached]
            if sl_only_offset:
                trailing_rate[~trailing_offset_reached] = np.nan
            trg_range[:, trg_col_trailing_triggered] = low_range <= trailing_rate
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
        valid_cols = np.array([n for n,v in enumerate(trg_first_idx) if trg_range[v, n] != 0])
        # check that there is at least one valid trigger
        if len(valid_cols):
            # get the column index that triggered first row sie
            trg_first = trg_first_idx[valid_cols].argmin()
            # lastly get the trigger offset from the index of the first valid column
            trg_ofs = trg_first_idx[valid_cols[trg_first]]
            # check what trigger it is and copy related columns values
            if trg_first == trg_col_stoploss_triggered:
                col_stoploss_triggered[b] = True
                col_stoploss_rate[b] = stoploss_triggered_rate
            elif trg_first in s_trg_roi_idx:
                col_roi_triggered[b] = True
                # NOTE: scale trg_first by how many preceding columns (stoploss,trailing)
                # there are before roi columns, in order to get the offset
                # relative to only the (ordered) roi columns
                col_roi_profit[b] = roi_vals[trg_first - trg_roi_pos]
            elif trg_first == idx_trailing_triggered:
                col_trailing_triggered[b] = True
                col_trailing_rate[b] = trailing_rate[trg_ofs]
            # trigger ofs is relative to the bought range, so just add it to the bought offset
            current_ofs = bought_ofs + trg_ofs
            # copy general trigger values shared by all trigger types
            col_trigger_ofs[b] = current_ofs
            col_trigger_date[b] = ohlc_date[current_ofs]
            col_trigger_bought_ofs[b] = bought_ofs
            last_trigger = b
            # get the first row where the bought index is
            # higher than the current stoploss index
            # b += np.searchsorted(bofs[b:], current_ofs, "right")
            b += find_first(bofs[b:], current_ofs)
            # repeat the trigger index for the boughts in between the trigger
            # and the bought with higher idx
            col_last_trigger[last_trigger:b] = current_ofs
            if b < n_bought:
                bought_ofs = bofs[b]
            else:
                break
        else:  # if no triggers executed, jump to the first bought after next sold idx
            # b += np.searchsorted(bofs[b:], bsold[b], "right")
            b += find_first(bofs[b:], bsold[b])
            if b < n_bought:
                bought_ofs = bofs[b]
            else:
                break
    return triggers

