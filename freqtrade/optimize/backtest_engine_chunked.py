from typing import Tuple, List, Dict
from freqtrade.optimize.backtest_constants import Candle
from freqtrade.optimize.backtest_utils import (
    add_columns,
    as_df,
    df_cols,
    np_fill,
    shift,
    without_cols,
)
from freqtrade.optimize.backtest_nb import (
    split_cumsum,
    copy_ranges,
    calc_profits,
    float64,
    ofs_cummax,
    ofs_first_flat_true,
    for_trail_idx,
)
from numpy import (
    arange,
    array,
    concatenate,
    flip,
    nan,
    ndarray,
    swapaxes,
    tile,
    where,
    maximum,
)
from pandas import DataFrame


def _np_calc_triggers(
    self, df_vals: ndarray, bought: ndarray, bought_ranges: ndarray,
) -> DataFrame:
    """ expand bought ranges into ohlc processed
        prefetch the columns of interest to avoid querying
        the index over the loop (avoid nd indexes) """
    df_loc = self.df_loc
    bts_loc = self.bts_loc
    df_cols = ["low", "high", "ohlc_ofs", "date"]
    # bought data rows will be repeated to match the bought_ranges
    bought_data = {"trigger_bought_ofs": bought[:, bts_loc["ohlc_ofs"]]}
    # post_cols are the empty columns that will be calculated after the expansion
    post_cols = []

    if self.roi_enabled:
        self.roi_col_names.extend(["roi_profit", "roi_triggered"])
        post_cols.extend(self.roi_col_names)
        # only the filtered roi couples will be used for calculation
        self.roi_timeouts, self.roi_values = self._filter_roi()
        roi_timeouts = array(list(self.roi_timeouts.keys()), dtype=int)
        n_timeouts = len(roi_timeouts)
        inv_roi_values = flip(self.roi_values)
        inv_roi_timeouts = flip(roi_timeouts)
    else:
        roi_timeouts = ndarray(0, dtype=int)

    stoploss = abs(self.strategy.stoploss)
    if self.stoploss_enabled:
        bought_data["stoploss_rate"] = bought[:, bts_loc["open"]] * (1 - stoploss)
        self.stoploss_col_names.extend(["stoploss_rate", "stoploss_triggered"])
        post_cols.append("stoploss_triggered")

    if self.trailing_enabled:
        sl_positive = self.strategy.trailing_stop_positive
        sl_offset = self.strategy.trailing_stop_positive_offset
        sl_only_offset = self.strategy.trailing_only_offset_is_reached
        calc_offset = sl_positive or sl_only_offset

        # calculate both rates
        trail_cols = ["high_rate"]
        if sl_positive:
            trail_cols.append("high_rate_positive")
        df_vals = add_columns(df_vals, df_loc, trail_cols)
        df_vals[:, df_loc["high_rate"]] = df_vals[:, df_loc["high"]] * (1 - stoploss)
        if sl_positive:
            df_vals[:, df_loc["high_rate_positive"]] = df_vals[:, df_loc["high"]] * (
                1 - sl_positive
            )
        df_cols.extend(trail_cols)

        self.trailing_col_names.extend(
            ["trailing_rate", "trailing_triggered",]
        )
        post_cols.extend(["trailing_rate", "trailing_triggered"])
    if self.roi_enabled or self.trailing_enabled:
        bought_data["bought_open"] = bought[:, bts_loc["open"]]

    df_cols_n = len(df_cols)
    # NOTE: order matters
    col_names = df_cols + list(bought_data.keys()) + post_cols
    expd_cols_n = df_cols_n + len(bought_data)
    col, _ = self._columns_indexes(col_names, roi_timeouts)
    # the triggers columns have to be continuous for a view of the ndarray
    tot_cols_n = expd_cols_n + len(post_cols)

    # get views of the data to expand
    ohlc_vals = df_vals[:, [df_loc[c] for c in df_cols]]
    bought_vals = swapaxes(array(list(bought_data.values())), 0, 1)

    # cap the bought range iteration to limit memory usage
    ranges_csum = bought_ranges.cumsum()
    split_idx = [0, *split_cumsum(self.max_ranges_size, bought_ranges), None]
    # will store the partial arrays for triggers
    data_chunks = []
    for chunk_start, chunk_stop in zip(split_idx, split_idx[1::1]):
        # the array storing the expanded data

        trigger_flags = []
        br_chunk = bought_ranges[chunk_start:chunk_stop]
        data_len = br_chunk.sum()
        data_ofs = ndarray(len(br_chunk) + 1, dtype=int)
        data_ofs[0] = 0
        # data_ofs[1:] = cumsum(br_chunk)
        data_ofs[1:] = ranges_csum[chunk_start:chunk_stop] - (
            ranges_csum[chunk_start - 1] if chunk_start > 0 else 0
        )
        data = ndarray(shape=(data_len, tot_cols_n))
        data_df = data[:, :df_cols_n]
        data_bought = data[:, df_cols_n:expd_cols_n]

        # copy the source data ranges over to their expanded indexes
        copy_ranges(
            chunk_start,
            chunk_stop,
            bought[:, bts_loc["ohlc_ofs"]],
            data_df,
            data_bought,
            ohlc_vals,
            bought_vals,
            bought_ranges,
        )

        if self.roi_enabled or self.trailing_enabled:
            cur_profits = calc_profits(
                data[:, col.bought_open],
                data[:, col.high],
                float64(self.config["stake_amount"]),
                float64(self.fee),
            )

        if self.roi_enabled:
            # the starting offset is already computed with data_ofs
            # exclude the last value since we only nan from the start
            # of each bought range
            bought_starts = data_ofs[:-1]
            # setup expanded required roi profit columns ordered by roi timeouts
            # NOTE: inversed because the latest one happens first
            inv_roi_vals = tile(inv_roi_values, (data_len, 1))

            # roi rates index null overrides based on timeouts
            # exclude indexes that exceed the maximum length of the expd array
            nan_early_roi_idx = {}
            for to in roi_timeouts:
                to_indexes = bought_starts + to
                nan_early_roi_idx[to] = to_indexes[to_indexes < data_len]
            # roi triggers store the truth of roi matching for each roi timeout
            roi_triggers = []
            for to, tv in zip(inv_roi_timeouts, inv_roi_values):
                roi_trg = cur_profits >= tv
                # null rows preceding the timeouts
                # NOTE: strictly below, bool arrays can't be nan, use False
                for t in roi_timeouts:
                    if t >= to:
                        break
                    roi_trg[nan_early_roi_idx[t]] = False
                roi_triggers.append(roi_trg)

            roi_triggers = swapaxes(roi_triggers, 0, 1)

            trigger_flags.append(roi_triggers)

        if self.stoploss_enabled:
            stoploss_triggered = data[:, col.low] <= data[:, col.stoploss_rate]
            # add new axis to concatenate all the triggers column wise
            trigger_flags.append(stoploss_triggered[:, None])

        if self.trailing_enabled:
            if calc_offset:
                trailing_offset_reached = ofs_cummax(data_ofs, cur_profits) >= sl_offset
                trailing_rate = (
                    where(
                        trailing_offset_reached,
                        # where the offset is reached, use the positive stoploss ratio
                        ofs_cummax(data_ofs, data[:, col.high_rate_positive]),
                        # else use the ratio from normal stoploss
                        ofs_cummax(data_ofs, data[:, col.high_rate])
                        if not sl_only_offset
                        else nan,
                    )
                    if sl_positive
                    # is a positive stoploss ratio is not configured, it is trailing
                    # only when the positive offset is reached, so use normal stoploss ratio
                    # and nan where the offset is not reached
                    else (
                        where(
                            trailing_offset_reached,
                            ofs_cummax(data_ofs, data[:, col.high_rate]),
                            nan,
                        )
                    )
                )
            else:
                # if not only offset, and no positive offset ratio,
                # always trail we normal stoploss ratio
                trailing_rate = ofs_cummax(data_ofs, data[:, col.high_rate])
            trailing_triggered = data[:, col.low] <= trailing_rate
            trigger_flags.append(trailing_triggered[:, None])

        triggered = concatenate(list(reversed(trigger_flags)), axis=1,)

        # find the first trigger for every bought
        trigger_max = ofs_first_flat_true(data_ofs, triggered)
        # mask to select only boughts with triggers
        any_trigger_mask = trigger_max != -1

        # reduce data to valid triggers
        data = data[any_trigger_mask]
        # go next if no triggers found
        if len(data) < 1:
            continue

        # NOTE: copying data before reduction appears to be faster, and less complex
        # not tested at what n_rows it starts being worth copying after reducing
        any_trigger = trigger_max[any_trigger_mask]
        if self.roi_enabled:
            # NOTE: order is trailing,stoploss, rev_roi1, rev_roi2, rev_roi3...
            # because triggers always end with roi, scalding down by the number of
            # roi instances (n_timeouts) will give the offset of the first roi
            trg_roi_pos = triggered.shape[1] - n_timeouts
            data[:, col.roi_triggered] = any_trigger >= trg_roi_pos
            data[:, col.roi_profit] = inv_roi_vals[any_trigger_mask][
                arange(len(any_trigger)), any_trigger - trg_roi_pos
            ]
        if self.stoploss_enabled:
            data[:, col.stoploss_triggered] = stoploss_triggered[any_trigger_mask]
        if self.trailing_enabled:
            data[:, col.trailing_triggered] = trailing_triggered[any_trigger_mask]
            data[:, col.trailing_rate] = trailing_rate[any_trigger_mask]

        # only where the trigger_bought_ofs is not the same as the previous
        # since we don't evaluate alternate universes
        data = data[
            data[:, col.trigger_bought_ofs] != shift(data[:, col.trigger_bought_ofs]),
        ]

        # exclude triggers that where bought past the max bought index of the triggers
        # since we don't travel back in time
        if not self.position_stacking:
            data = data[
                data[:, col.trigger_bought_ofs]
                >= maximum.accumulate(data[:, col.trigger_bought_ofs])
            ]
        data_chunks.append(data)

    if len(data_chunks) < 1:
        # keep shape since return value is accessed without reference
        return _triggers_return_df(self, col, full((0, data.shape[1]), nan))

    data = concatenate(data_chunks, axis=0)
    return _triggers_return_df(self, col, data)


def _triggers_return_df(self, col: Tuple, data: ndarray) -> DataFrame:
    col_dict = col.__dict__
    col_map = {
        "trigger_ofs": col.ohlc_ofs,
        "trigger_date": col.date,
        "trigger_bought_ofs": col.trigger_bought_ofs,
    }
    if self.stoploss_enabled:
        col_map.update({name: col_dict[name] for name in self.stoploss_col_names})
    if self.trailing_enabled:
        col_map.update({name: col_dict[name] for name in self.trailing_col_names})
    if self.roi_enabled:
        col_map.update({name: col_dict[name] for name in self.roi_col_names})
    # return DataFrame(
    #     data[:, list(col_map.values())], columns=list(col_map.keys()), copy=False
    # )
    return (
        data[:, list(col_map.values())],
        {k: n for n, k in enumerate(col_map.keys())},
    )


def _last_trigger_apply(df: DataFrame):
    """ Loop over each row of the dataframe and only select stoplosses for boughts that
    happened after the last set stoploss """
    # store the last stoploss [0] and the next sold [1]
    last = [-1, -1]
    df["trigger_ofs"].fillna(-3, inplace=True)

    def trail_idx(x, last: List[int]):
        if x.next_sold_ofs != last[1]:
            last[0] = x.trigger_ofs if x.trigger_ofs != -3 else -1
        elif x.name > last[0] and last[0] != -1:
            last[0] = x.trigger_ofs if x.trigger_ofs != -3 else -1
        last[1] = x.next_sold_ofs
        return last[0]

    return df.apply(trail_idx, axis=1, raw=True, args=[last]).values


def _last_trigger_numba(df: DataFrame) -> List[int]:
    """ pandas args; numba version of _last_trigger_apply """
    df["trigger_ofs"].fillna(-3, inplace=True)
    return for_trail_idx(
        df.index.values,
        df["trigger_ofs"].values.astype(int),
        df["next_sold_ofs"].values.astype(int),
    )


def _last_trigger_numba_arr(arr: ndarray, loc: Dict) -> ndarray:
    """ numpy args; numba version of _last_trigger_apply """
    np_fill(arr[:, loc["trigger_ofs"]], fill_value=-3, inplace=True)
    # arr[isnan(arr[:, loc["trigger_ofs"]]), loc["trigger_ofs"]] = -3
    return for_trail_idx(
        arr[:, loc["ohlc_ofs"]].astype(int),
        arr[:, loc["trigger_ofs"]].astype(int),
        arr[:, loc["next_sold_ofs"]].astype(int),
    )


# NOTE: results currently don't match, fix might be easy, haven't checked
def _chunked_select_triggers(
    self, df_vals: ndarray, bought: ndarray, bought_ranges: ndarray, bts_vals: ndarray,
) -> ndarray:

    # compute all the stoplosses for the buy signals and filter out clear invalids
    trg_vals, trg_loc = _np_calc_triggers(self, df_vals, bought, bought_ranges)
    bts_df = as_df(
        bts_vals, self.bts_loc, bts_vals[:, self.bts_loc["ohlc_ofs"]], int_idx=True
    )

    # align original index
    if not self.position_stacking:
        # exclude overlapping boughts
        # --> | BUY1 | BUY2..STOP2 | STOP1 | -->
        # -->      V    X      X       V     -->
        # preset indexes to merge (on indexes directly) without sorting
        # dbg.start_pyinst()

        outer_cols = ["trigger_ofs", "trigger_bought_ofs"]
        # can use assign here since all the trigger indices should be present in
        left_cols = [
            "trigger_ofs",
            "trigger_date",
            *self.roi_col_names,
            *self.stoploss_col_names,
            *self.trailing_col_names,
        ]

        # join from right, to not clobber indexes
        bts_loc = self.bts_loc
        without_bought_names = without_cols(trg_loc, ["trigger_bought_ofs"])
        without_bought_cols = [trg_loc[c] for c in without_bought_names]
        bts_df = as_df(
            trg_vals[:, without_bought_cols],
            without_bought_names,
            idx=trg_vals[:, trg_loc["trigger_bought_ofs"]],
            int_idx=False,
        ).merge(bts_df, left_index=True, right_index=True, how="right", copy=False)

        # bts_vals = np_left_join(
        #     bts_vals,
        #     trg_vals,
        #     bts_loc,
        #     trg_loc,
        #     "ohlc_ofs",
        #     "trigger_bought_ofs",
        #     fill_value=nan,
        # )

        # now add the trigger new rows with an outer join
        bts_df = bts_df.merge(
            as_df(
                trg_vals[:, [trg_loc[c] for c in outer_cols]],
                outer_cols,
                trg_vals[:, trg_loc["trigger_ofs"]],
                int_idx=True,
            ),
            how="outer",
            left_index=True,
            right_index=True,
            suffixes=("", "_b"),
            copy=False,
        )
        # dbg.stop_pyinst()

        # outer_cols_names = ("trigger_ofs_b", "trigger_bought_ofs")
        # outer_cols = [trg_loc[c] for c in outer_cols]

        # merged = merge_2d(
        #     bts_vals,
        #     trg_vals[:, outer_cols],
        #     bts_loc["ohlc_ofs"],
        #     trg_loc["trigger_ofs"],
        # )
        # for n, c in enumerate(outer_cols_names, len(bts_loc)):
        #     bts_loc[c] = n

        # dbg.stop_time()
        # mrg_idx = merged[:, bts_loc["ohlc_ofs"]].copy()

        # bts_df = as_df(merged, bts_loc, mrg_idx)

        # drop pandas once done with merging
        bts_vals = bts_df.values
        bts_loc = df_cols(bts_df)

        # both trigger_ofs_b and trigger_bought_ofs are on the correct
        # index where the stoploss is triggered, and backfill backwards
        np_fill(bts_vals[:, bts_loc["trigger_ofs_b"]], backfill=True, inplace=True)
        np_fill(bts_vals[:, bts_loc["trigger_bought_ofs"]], backfill=True, inplace=True)
        # this is a filter to remove obvious invalids
        bts_vals = bts_vals[
            ~(
                (bts_vals[:, bts_loc["bought_or_sold"]] == Candle.BOUGHT)
                & (shift(bts_vals[:, bts_loc["bought_or_sold"]]) == Candle.BOUGHT)
                & (
                    (
                        (
                            bts_vals[:, bts_loc["trigger_ofs_b"]]
                            == shift(bts_vals[:, bts_loc["trigger_ofs_b"]])
                        )
                        | (
                            bts_vals[:, bts_loc["trigger_bought_ofs"]]
                            == shift(bts_vals[:, bts_loc["trigger_bought_ofs"]])
                        )
                    )
                )
            )
        ]

        # loop over the filtered boughts to determine order of execution
        boughts = bts_vals[:, bts_loc["bought_or_sold"]] == Candle.BOUGHT
        bts_vals = add_columns(bts_vals, bts_loc, ["last_trigger"],)
        bts_vals[boughts, bts_loc["last_trigger"]] = _last_trigger_numba_arr(
            bts_vals[boughts], bts_loc
        )
        bts_vals[~boughts, bts_loc["last_trigger"]] = nan
        # fill the last_trigger column for non boughts
        np_fill(bts_vals[:, bts_loc["last_trigger"]], inplace=True)

        stale_boughts = ~(
            # last active stoploss matches the current stoploss, otherwise it's stale
            (
                bts_vals[:, bts_loc["trigger_ofs"]]
                == bts_vals[:, bts_loc["last_trigger"]]
            )
            # it must be the first bought matching that stoploss index,
            # in case of subsequent boughts that triggers on the same index
            # which wouldn't happen without position stacking
            & (
                bts_vals[:, bts_loc["last_trigger"]]
                != shift(bts_vals[:, bts_loc["last_trigger"]])
            )
        )
        # null stale boughts
        for c in ["trigger_ofs", *self.trigger_types]:
            bts_vals[stale_boughts, bts_loc[c]] = nan
        # outer merging doesn't include the pair column, so fill empty pair rows forward
        np_fill(bts_vals[:, bts_loc["pair"]], inplace=True)
    else:
        # add triggers data to the bought/sold dataframe
        # trigger.set_index("trigger_bought_ofs", inplace=True, drop=False)
        bts_df = bts_df.merge(
            as_df(trg_vals, trg_loc, trg_vals[:, trg_loc["trigger_bought_ofs"]]),
            # trigger,
            left_index=True,
            right_index=True,
            how="left",
            copy=False,
        )
        bts_vals = bts_df.values
        bts_loc = df_cols(bts_df)
        # don't apply triggers to sold candles
        bts_vals[
            bts_vals[:, bts_loc["bought_or_sold"]] == Candle.SOLD,
            bts_loc["trigger_ofs"],
        ] = nan
    # fill nan bool columns to False
    for t in self.trigger_types:
        np_fill(bts_vals[:, bts_loc[t]], fill_value=0, inplace=True)
    self.bts_loc = bts_loc
    return bts_vals
