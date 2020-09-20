from types import SimpleNamespace

from numpy import maximum, ndarray, where
from pandas import DataFrame
from freqtrade.optimize.backtest_nb import next_bought

np_cummax = maximum.accumulate


def _first_flat_true(arr: ndarray):
    n_cols = arr.shape[1]
    # each row
    for rn, r in enumerate(arr[:, :]):
        # col 1 is trailing, before stoploss
        if r[1]:
            return rn, 1
        # col 0 is stoploss
        if r[0]:
            return rn, 0
        # remaining cols are roi, iterate in reverse
        # so that always the latest interrupts
        for cn, e in zip(range(n_cols - 3, -1, -1), r[:1:-1]):
            if e:
                return rn, cn
    return -1, -1


def _calc_stoploss_rate(
    open_rate, low_range, stoploss, trg_range, trg_col_stoploss_triggered,
):
    # calculate the rate from the bought candle
    stoploss_triggered_rate = open_rate * (1 - stoploss)
    trg_range[:, trg_col_stoploss_triggered] = low_range <= stoploss_triggered_rate
    return stoploss_triggered_rate


def _calc_trailing_rate(
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
        # use cummax since once offset is reached it is set for the trade
        trailing_offset_reached = np_cummax(cur_profits) >= sl_offset
        trailing_rate = (
            (
                where(
                    trailing_offset_reached,
                    np_cummax(high_range * (1 - sl_positive)),
                    np_cummax(high_range * (1 - stoploss))
                    # where the offset is not reached, and the offset flag
                    # is set, trailing stop is not applied at all
                    if not sl_only_offset else nan,
                )
            )
            if sl_positive
            else (
                where(
                    trailing_offset_reached,
                    np_cummax(high_range * (1 - stoploss)),
                    nan,
                )
            )
        )
    else:
        trailing_rate = np_cummax(high_range * (1 - stoploss))
    trg_range[:, trg_col_trailing_triggered] = low_range <= trailing_rate
    return trailing_rate


def _compare_roi_triggers(cur_profits, roi_vals, roi_timeouts, trg_range, trg_roi_idx):
    # reset values in case timeouts (parameters) changed
    # so that the arr doesn't hold old truths (since we are partially filling)
    trg_range[:, trg_roi_idx] = 0
    # NOTE: only compare each roi from the start of its timeout
    for n, rt in enumerate(roi_timeouts):
        trg_range[rt:, trg_roi_idx[n]] = cur_profits[rt:] >= roi_vals[n]


def _loop_ranges_select_triggers(
    self, df: DataFrame, bought: ndarray, bought_ranges: ndarray, bts_vals: ndarray,
):
    v = self._get_vars(df, bought, bought_ranges)
    sn = SimpleNamespace(**v)

    bought_ofs = sn.bought_ofs
    current_ofs = bought_ofs
    b = sn.b

    while bought_ofs < sn.end_ofs:
        # check trigger for the range of the current bought
        triggered = False
        br = sn.bought_ranges[b]
        bought_ofs_stop = bought_ofs + br
        trg_range = sn.trg_range_max[:br, :]
        open_rate = sn.bopen[b]

        if sn.roi_or_trailing:
            high_range = sn.ohlc_high[bought_ofs:bought_ofs_stop]
            _, cur_profits = self._calc_profits_np(
                sn.stake_amount, sn.fee, open_rate, high_range, False
            )
        if sn.stoploss_or_trailing:
            low_range = sn.ohlc_low[bought_ofs:bought_ofs_stop]
        if sn.roi_enabled:
            # NOTE: use False, not nan, since bool(nan) == True
            _compare_roi_triggers(
                cur_profits, sn.roi_vals, sn.roi_timeouts, trg_range, sn.trg_roi_idx,
            )

        if sn.stoploss_enabled:
            # calculate the rate from the bought candle
            stoploss_triggered_rate = _calc_stoploss_rate(
                open_rate,
                low_range,
                sn.stoploss,
                trg_range,
                sn.trg_col.stoploss_triggered,
            )
        if sn.trailing_enabled:
            trailing_rate = _calc_trailing_rate(
                sn.calc_offset,
                sn.sl_positive,
                sn.sl_only_offset,
                sn.sl_offset,
                cur_profits,
                high_range,
                sn.stoploss,
                trg_range,
                low_range,
                sn.trg_col.trailing_triggered,
            )
        fft = _first_flat_true(trg_range)
        # check that there is at least one valid trigger
        if fft[1] != -1:
            # get the column index that triggered first row wise
            # NOTE: first_flat_true will return in order of precedence
            # - trailing_stop, static_stoploss, (latest) roi
            trg_top = fft[1]
            # lastly get the trigger offset from the index of the first valid column
            # any trg_first index would return the same value here
            trg_ofs = fft[0]
            # check what trigger it is and copy related columns values
            # from left they are ordered stoploss, trailing, roi
            if sn.trailing_enabled and trg_top == sn.trg_col.trailing_triggered:
                sn.col_trailing_triggered[b] = True
                sn.col_trailing_rate[b] = trailing_rate[trg_ofs]
            elif sn.stoploss_enabled and trg_top == sn.trg_col.stoploss_triggered:
                sn.col_stoploss_triggered[b] = True
                sn.col_stoploss_rate[b] = stoploss_triggered_rate
            elif sn.roi_enabled and trg_top in sn.s_trg_roi_idx:
                sn.col_roi_triggered[b] = True
                # NOTE: scale trg_first by how many preceding columns (stoploss,trailing)
                # there are before roi columns, in order to get the offset
                # relative to only the (ordered) roi columns
                # and pick the minimum value from the right (roi_values have to be pre osn.rdered desc)
                # sn.col_roi_profit[b] = sn.roi_vals[trg_first[-1] - sn.trg_roi_pos]
                sn.col_roi_profit[b] = sn.roi_vals[trg_top]

            # trigger ofs is relative to the bought range, so just add it to the bought offset
            current_ofs = bought_ofs + trg_ofs
            # copy general trigger values shared by asn.ll trigger types
            sn.col_trigger_ofs[b] = current_ofs
            sn.col_trigger_date[b] = sn.ohlc_date[current_ofs]
            sn.col_trigger_bought_ofs[b] = bought_ofs

            triggered = True
        if sn.not_position_stacking:
            if triggered:
                try:
                    last_trigger = b
                    # get the first row where the bought index is
                    # higher than the current stoploss insn.dex
                    b = next_bought(b, sn.bofs[b:], current_ofs)
                    # repeat  trigger index for the boughts in between the trigger
                    # and  bought with higher idx
                    sn.col_last_trigger[last_trigger:b] = current_ofs
                    bought_ofs = sn.bofs[b]
                except IndexError:
                    break
            else:  # if no trigg executed, jump to the first bought after next sold idx
                try:
                    b = next_bought(b, sn.bofs[b:], sn.bsold[b])
                    bought_ofs = sn.bofs[b]
                except IndexError:
                    break
        else:
            try:
                b += 1
                bought_ofs = sn.bofs[b]
            except IndexError:
                break

    return self._assign_triggers_vals(bts_vals, bought, sn.triggers, sn.col_names)
