import logging

from joblib import load
from functools import reduce
import arrow
import gc
from typing import Dict, Any, List, Tuple
from numba import njit
from numpy import (
    append,
    repeat,
    nan,
    concatenate,
    arange,
    ndarray,
    array,
    empty,
    memmap,
    where,
    transpose,
    maximum,
    full,
    split,
    unique,
    asarray,
    roll,
    insert,
    isnan,
)
from pandas import (
    Timedelta,
    to_timedelta,
    concat,
    Series,
    DataFrame,
    Index,
    CategoricalIndex,
    merge,
    MultiIndex,
    SparseArray,
    SparseDtype,
    Categorical,
    merge_ordered,
)
from pandas import IndexSlice as idx
import pandas as pd

pd.set_option("display.max_rows", 1000)
from memory_profiler import profile

from freqtrade.data.history import get_timerange
import freqtrade.optimize.hyperopt_backend as backend
from freqtrade.optimize.hyperopt_multi import HyperoptMulti
from freqtrade.resolvers import ExchangeResolver, StrategyResolver
from freqtrade.configuration import remove_credentials
from freqtrade.data.dataprovider import DataProvider
from freqtrade.optimize.backtesting import Backtesting, BacktestResult
from freqtrade.exceptions import OperationalException
from freqtrade.strategy.interface import SellType

from enum import IntEnum

logger = logging.getLogger(__name__)

import os
import psutil

process = psutil.Process(os.getpid())


class Candle(IntEnum):
    BOUGHT = 2
    SOLD = 5
    NOOP = 0
    END = 11  # references the last candle of a pair
    # STOPLOSS = 17


@njit  # fastmath=True ? there is no math involved here though..
def for_trail_idx(index, bos, rate, stop_idx):
    last = -2
    col = [0] * len(index)
    for i in range(len(index)):
        if bos[i] == Candle.BOUGHT:
            if index[i] > last and last != -1:
                if rate[i] > 0:
                    last = stop_idx[i]
                else:
                    last = -1
            col[i] = last
        else:
            last = -2
            col[i] = -1
    return col


def union_eq(arr: ndarray, vals: List) -> ndarray:
    res = arr == vals[0]
    for v in vals[1:]:
        res = res | (arr == v)
    return res


class HyperoptBacktesting(Backtesting):

    empty_results = DataFrame.from_records([], columns=BacktestResult._fields)
    debug = False

    td_zero = Timedelta(0)
    td_half_timeframe: Timedelta
    pairs_offset = []

    def __init__(self, config):
        if config.get("backtesting_engine") == "vectorized":
            self.backtest_stock = self.backtest
            self.backtest = (
                self._wrap_backtest if self.debug else self.vectorized_backtest
            )
            self.beacktesting_engine = "vectorized"
            self.td_half_timeframe = (
                Timedelta(config.get("timeframe", config["timeframe"])) / 2
            )
        super().__init__(config)

    def get_results(self, events_buy: DataFrame, events_sell: DataFrame) -> DataFrame:
        # choose sell rate depending on sell reason and set sell_reason
        events_sold = events_sell.loc[
            events_sell["bought_or_sold"].values == Candle.SOLD
        ]
        # add new columns with reindex to allow multi col assignments of new columns
        events_sell = events_sell.reindex(
            columns=[*events_sell.columns, "close_rate", "sell_reason"], copy=False,
        )
        events_sell.loc[events_sold.index, ["close_rate", "sell_reason", "ohlc"]] = [
            events_sold["open"].values,
            SellType.SELL_SIGNAL,
            events_sold["ohlc"].values,
        ]
        events_stoploss = events_sell.loc[events_sell["stoploss_idx"].notna().values]
        events_sell.loc[
            events_stoploss.index, ["close_rate", "sell_reason", "ohlc"]
        ] = [
            events_stoploss["stoploss_rate"].values,
            SellType.STOP_LOSS,
            events_stoploss["stoploss_idx"].values,
        ]

        open_rate = events_buy["open"].values
        close_rate = events_sell["close_rate"].values
        profits = (close_rate - close_rate * self.fee) / (
            open_rate + open_rate * self.fee
        ) - 1
        trade_duration = Series(events_sell["date"].values - events_buy["date"].values)
        # replace trade duration of same candle trades with half the timeframe reduce to minutes
        trade_duration.loc[trade_duration == self.td_zero] = self.td_half_timeframe

        return DataFrame(
            {
                "pair": events_buy["pair"].values,
                "profit_percent": profits,
                "profit_abs": self.config["stake_amount"] * profits,
                "open_time": events_buy["date"].values,
                "close_time": events_sell["date"].values,
                "open_index": events_buy["ohlc"].values,
                "close_index": events_sell["ohlc"].values,
                "trade_duration": trade_duration.dt.seconds / 60,
                "open_at_end": False,
                "open_rate": open_rate,
                "close_rate": close_rate,
                "sell_reason": events_sell["sell_reason"].values,
            }
        )

    def advise_pair_df(self, df: DataFrame, pair: str) -> DataFrame:
        """ Execute strategy signals and return df for given pair """
        meta = {"pair": pair}
        df = self.strategy.advise_buy(df, meta)
        df = self.strategy.advise_sell(df, meta)
        df.fillna({"buy": 0, "sell": 0}, inplace=True)
        return df

    @staticmethod
    def _get_multi_index(pairs: list, idx: ndarray) -> MultiIndex:
        # if a list of [idx, pairs] is passed to from_product , the df would infer
        # the counter as the columns, when we want it as the rows, so we have to pass
        # a swapped mi to the df, there surely is a better way for this...
        return MultiIndex.from_product([pairs, idx], names=["pair", "ohlc"]).swaplevel(
            0, 1
        )

    def merge_pairs_df(self, processed: Dict[str, DataFrame]) -> DataFrame:
        """ join all the pairs data into one concatenate df adding needed columns """
        advised = {}
        data = []
        max_len = 0
        pairs_end = []
        nan_data_pairs = []

        # get the df with the longest ohlc data since all the pairs will be padded to it
        max_df = max(processed.values(), key=len)
        max_len = len(max_df)
        for pair, df in processed.items():
            advised[pair] = self.advise_pair_df(df.copy(), pair)
            apv = advised[pair].values
            lapv = len(apv)
            pairs_end.append(lapv)
            if lapv < max_len:
                # pad shorter data, with an empty array of same shape (columns)
                data.extend(
                    concatenate([apv, full((max_len - lapv, apv.shape[1]), nan)])
                )
                nan_data_pairs.append(pair)
            else:
                data.extend(apv)

        self.pairs = {p: n for n, p in enumerate(advised.keys())}
        self.pairs_max_len = max_len
        self.n_pairs = len(self.pairs)
        self.pairs_end = array(pairs_end, dtype=int) - 1
        # the index shouldn't change after the advise call, so we can take the pre-advised index
        # to create the multiindex where each pair is indexed with max len
        self.n_rows = len(max_df.index.values)
        self.mi = self._get_multi_index(advised.keys(), max_df.index.values)
        # take a post advised df for the right columns count as the advise call
        # might have added new columns
        df = DataFrame(data, index=self.mi, columns=advised[pair].columns)
        # set startup offset from the first index (should be equal for all pairs)
        self.startup_offset = df.index.get_level_values(0)[0]
        # add a column for pairs offsets to make the index unique
        offsets_arr, self.pairs_offset = self._calc_pairs_offsets(df, return_ofs=True)
        self.pairs_ofs_end = self.pairs_offset + self.pairs_end
        # loop over the missing data pairs and calculate the point where data ends
        # plus the absolute offset
        self.nan_data_ends = [
            self.pairs_ofs_end[self.pairs[p]] + 1 for p in nan_data_pairs
        ]
        df["ofs"] = Categorical(offsets_arr, self.pairs_offset)
        # could as easily be arange(len(df)) ...
        df["ohlc_ofs"] = (
            df.index.get_level_values(0).values + offsets_arr - self.startup_offset
        )
        return df

    def bought_or_sold(self, df: DataFrame) -> Tuple[DataFrame, DataFrame, bool]:
        """ Set bought_or_sold columns according to buy and sell signals """
        # set bought candles
        df.loc[:, "bought_or_sold"] = asarray(  # us asarray in case of sparse data
            (df["buy"] - df["sell"]).groupby(level=1).shift().values
        )

        df.loc[df["bought_or_sold"].values == 1, "bought_or_sold"] = Candle.BOUGHT
        # skip if no valid bought candles are found
        if len(df.loc[df["bought_or_sold"].values == Candle.BOUGHT]) < 1:
            return None, True
        # set sold candles
        df.loc[df["bought_or_sold"].values == -1, "bought_or_sold"] = Candle.SOLD
        df["bought_or_sold"] = Categorical(
            df["bought_or_sold"].values, categories=list(map(int, Candle))
        )
        # set END candles as the last non nan candle of each pair data
        bos_loc = df.columns.get_loc("bought_or_sold")
        df.iloc[self.pairs_ofs_end, bos_loc] = Candle.END
        # Since bought_or_sold is shifted, null the row after the last non-nan one
        # as it doesn't have data, exclude pairs which data matches the max_len since
        # they have no nans
        df.iloc[self.nan_data_ends, bos_loc] = Candle.NOOP
        return df, False

    @staticmethod
    def boughts_to_sold(df: DataFrame) -> DataFrame:
        """
        reduce df such that there are many bought interleaved by one sold candle
        NOTE: does not modify input df
        """
        bos_df = df.loc[
            union_eq(
                df["bought_or_sold"].values, [Candle.BOUGHT, Candle.SOLD, Candle.END]
            )
        ]
        bos_df = bos_df.loc[
            # exclude duplicate sold
            ~(
                (bos_df["bought_or_sold"].values == Candle.SOLD)
                & (
                    bos_df["bought_or_sold"]
                    .groupby(level=1)
                    .shift(fill_value=Candle.SOLD)
                    .values
                    == Candle.SOLD
                )
            )
        ]
        return bos_df

    # @staticmethod
    # def fill_stub_sold(df: DataFrame, bts_df: DataFrame) -> DataFrame:
    #     """ Helper function to limit trades duration """
    #     sold = (
    #         df.loc[~df.index.isin(bts_df.set_index("index").index)]
    #         .iloc[::1000]
    #         .reset_index()
    #     )

    #     sold["bought_or_sold"] = Candle.SOLD
    #     bts_df = bts_df.merge(sold, how="outer", on=sold.columns.tolist()).sort_values(
    #         by="index"
    #     )
    #     bts_df.drop(
    #         bts_df.loc[
    #             (bts_df["bought_or_sold"].values == Candle.SOLD)
    #             & (bts_df["bought_or_sold"].shift().values == Candle.SOLD)
    #         ].index,
    #     )
    #     # ensure the latest candle is always sold
    #     if bts_df.iloc[-1]["bought_or_sold"] == Candle.BOUGHT:
    #         sold.iloc[len(sold)] = df.iloc[-1]
    #         sold.iloc[-1]["bought_or_sold"] = Candle.SOLD
    #     return (bts_df, sold)

    def set_stoploss(self, df: DataFrame) -> DataFrame:
        """
        returns the df of valid boughts where stoploss triggered, with matching stoploss
        index of each bought
        """
        # recompose the multi index swapping the ohlc count with a contiguous range
        bts_df = self.boughts_to_sold(df).reset_index()
        bts_df["bts_index"] = bts_df.index.values
        # don't drop the index column since groupby truncates the index to the specified level
        # and can't be accessed from the .index attribute
        bts_df = bts_df.set_index(["bts_index", "pair"], drop=False)
        # align sold to bought
        sold = bts_df.loc[
            union_eq(bts_df["bought_or_sold"].values, [Candle.SOLD, Candle.END])
        ]
        # if no sold signals, return
        if len(sold) < 1:
            return DataFrame(columns=[*bts_df.columns])
            # if no sell sig is provided a limit on the trade duration could be applied..
            # bts_df, sold = self.fill_stub_sold(df, bts_df)
        # skip remaining bought candles without sold candle
        bts_df = bts_df.groupby(level=1, group_keys=False).apply(
            # slice from the beginning (:) until the last ([-1]) pair-wise label (idx[:, x.name])
            # where x.name == "pair" of the sold index
            lambda x: x.loc[: sold.loc[idx[:, x.name], :].index[-1]]
        )
        first_boughts = bts_df.groupby(level=1).first()
        # index col is only needed by first_boughts
        bts_df.drop(columns="bts_index", inplace=True)

        def repeats(x, rep):
            vals = x.index.get_level_values(0).values
            # prepend the first range subtracting the index of the first bought
            rep.append(vals[0] - first_boughts.at[x.name, "bts_index"] + 1)
            rep.extend(vals[1:] - vals[:-1])

        sold_repeats = []
        sold.groupby(level=1).apply(repeats, rep=sold_repeats)
        # NOTE: use the "ohlc_ofs" col with offsetted original indexes
        # for stoploss calculation, consider the last candle of each pair as a sell,
        # even thought the bought will be valid only if an amount condition is triggered
        bts_df["next_sold_ofs"] = repeat(sold["ohlc_ofs"].values, sold_repeats)
        bought = bts_df.loc[bts_df["bought_or_sold"].values == Candle.BOUGHT]
        # get the index ranges of each bought->sold spans
        bought_ranges = bought["next_sold_ofs"].values - bought["ohlc_ofs"].values
        # could also just use the sum...
        if bought_ranges.mean() < 100:
            # intervals are short compute everything in one round
            bts_df = self._pd_select_triggered_stoploss(
                df, bought, bought_ranges, bts_df
            )
        else:
            # intervals are too long, jump over candles
            bts_df = self._pd_2_select_triggered_stoploss(
                df, bought, bought_ranges, sold, bts_df
            )
        return bts_df

    def _pd_2_select_triggered_stoploss(
        self,
        df: DataFrame,
        bought: DataFrame,
        bought_ranges: ndarray,
        sold: DataFrame,
        bts_df: DataFrame,
    ):
        stoploss_index = []
        stoploss_rate = []
        bought_stoploss_ofs = []
        last_stoploss_idx = []
        last_stoploss = []
        # copy cols for faster index accessing
        bofs = bought["ohlc_ofs"].values
        bohlc = bought["ohlc"].values
        bsold = bought["next_sold_ofs"].values
        bopen = bought["open"].values
        b = 0
        stoploss_bought_ohlc = bofs[b]

        ohlc_low = df["low"].values
        ohlc_ofs = df["ohlc_ofs"].values
        ohlc_ofs_start = 0
        ohlc_idx = df.index.get_level_values(0)
        current_ofs = stoploss_bought_ohlc
        end_ofs = ohlc_ofs[-1]

        while stoploss_bought_ohlc < end_ofs:
            # calculate the rate from the bought candle
            stoploss_triggered_rate = self._calc_stoploss_rate_value(bopen[b])
            # check trigger for the range of the current bought
            ohlc_ofs_start += ohlc_ofs[ohlc_ofs_start:].searchsorted(
                stoploss_bought_ohlc, "left"
            )
            stoploss_triggered = (
                ohlc_low[ohlc_ofs_start : ohlc_ofs_start + bought_ranges[b]]
                <= stoploss_triggered_rate
            )
            # get the position where stoploss triggered relative to the current bought slice
            stop_max_idx = stoploss_triggered.argmax()
            # check that the index returned by argmax is True
            if stoploss_triggered[stop_max_idx]:
                # set the offset of the triggered stoploss index
                current_ofs = stoploss_bought_ohlc + stop_max_idx
                stop_ohlc_idx = ohlc_idx[current_ofs]
                stoploss_index.append(stop_ohlc_idx)
                stoploss_rate.append(stoploss_triggered_rate)
                bought_stoploss_ofs.append(stoploss_bought_ohlc)
                try:
                    # get the first row where the bought index is higher than the current stoploss index
                    b += bofs[b:].searchsorted(current_ofs, "right")
                    # repeat the stoploss index for the boughts in between the stoploss
                    # and the bought with higher idx
                    last_stoploss_idx.extend(
                        [stop_ohlc_idx] * (b - len(last_stoploss_idx))
                    )
                    stoploss_bought_ohlc = bofs[b]
                except IndexError:
                    break
            else:  # if stoploss did not trigger, jump to the first bought after next sold idx
                try:
                    b += bofs[b:].searchsorted(bsold[b], "right")
                    last_stoploss_idx.extend([-1] * (b - len(last_stoploss_idx)))
                    stoploss_bought_ohlc = bofs[b]
                except IndexError:
                    break
        # pad the last stoploss array with the remaining boughts
        last_stoploss_idx.extend([-1] * (len(bought) - len(last_stoploss_idx)))
        # set the index to the offset and add the columns to set the stoploss
        # data points on the relevant boughts
        bts_df.set_index("ohlc_ofs", inplace=True)
        bts_df = bts_df.reindex(
            columns=[*bts_df.columns, "stoploss_idx", "stoploss_rate", "last_stoploss"]
        )
        bts_df.loc[bought["ohlc_ofs"], "last_stoploss"] = last_stoploss_idx
        bts_df.loc[
            bought_stoploss_ofs, ["stoploss_idx", "stoploss_rate", "last_stoploss"],
        ] = [
            [stoploss_index],
            [stoploss_rate],
            [stoploss_index],
        ]
        bts_df["last_stoploss"].fillna(-1, inplace=True)
        return bts_df

    def _calc_pairs_offsets(
        self, df: DataFrame, group=None, return_ofs=False
    ) -> ndarray:
        # all the pairs with df candles
        gb = df.groupby(group) if group else df.groupby(level=1)
        df_pairs = [self.pairs[p] for p in gb.indices.keys()]
        # since pairs are concatenated, their candles start at their ordered position
        pairs_offset = [self.n_rows * n for n in df_pairs]
        pairs_offset_arr = repeat(pairs_offset, gb.size().values)
        if return_ofs:
            return pairs_offset_arr, pairs_offset
        else:
            return pairs_offset_arr - self.startup_offset

    def _columns_indexes(self, df: DataFrame) -> Dict[str, int]:
        cols_idx = {}
        for col in ("open", "low", "ohlc", "pair"):
            cols_idx[col] = df.columns.get_loc(col)
        return cols_idx

    def _np_calc_triggered_stoploss(
        self, df: DataFrame, bought: DataFrame, bought_ranges: ndarray,
    ) -> (ndarray, ndarray):
        """ numpy equivalent of _pd_calc_triggered_stoploss that is more memory efficient """
        # clear up memory
        gc.collect()
        # expand bought ranges into ohlc processed
        r_df = df.reset_index()
        ohlc_cols = list(self._columns_indexes(r_df).values())
        # prefetch the columns of interest to avoid querying the index over the loop (avoid nd indexes)
        ohlc_vals = r_df.iloc[:, ohlc_cols].values

        # 0: open, 1: low, 2: stoploss_idx, 3: pair, 4: stoploss_bought_ohlc, 5: stoploss_rate
        data_expd = concatenate(
            [
                concatenate(
                    [
                        ohlc_vals[i : i + bought_ranges[n]]
                        # the array position of each bought row comes from the offset
                        # of each pair from the beginning (adjusted to the startup candles count)
                        # plus the ohlc (actual order of the initial df of concatenated pairs)
                        for n, i in enumerate(
                            bought["ohlc"].values
                            + bought["ofs"].values.tolist()
                            - self.startup_offset
                        )
                    ]
                ),
                # stoploss_bought_ohlc and stoploss_rate to the expanded columns
                transpose(
                    repeat(
                        [bought["ohlc"].values, self._calc_stoploss_rate(bought),],
                        bought_ranges,
                        axis=1,
                    )
                ),
            ],
            axis=1,
        )

        # low (1) <= stoploss_rate (5)
        triggered = data_expd[data_expd[:, 1] <= data_expd[:, 5], :]
        if len(triggered) < 1:
            # keep shape since return value is accessed without reference
            return full((0, data_expd.shape[1]), nan)
        # only where the stoploss_bought_ohlc (4) is not the same as the previous
        stoploss_bought_ohlc_triggered_s1 = insert(triggered[:-1, 4], 0, nan)
        pair_triggered_s1 = insert(triggered[:-1, 3], 0, nan)
        first_triggers = triggered[
            where(
                (triggered[:, 4] != stoploss_bought_ohlc_triggered_s1)
                # edge case when a pair has only one bought candle and the next one
                # first bought candle is at the same index
                | (triggered[:, 3] != pair_triggered_s1)
            )
        ]
        first_triggers_pairs_offset = unique(first_triggers[:, 3], return_index=True)[1]
        # index column is (2)
        stoploss = first_triggers[
            # exclude stoplosses that are triggered before the bought cumulative maximum index
            concatenate(
                [
                    # add the offset since where indexes are relative to each pairs array
                    first_triggers_pairs_offset[n]
                    + where(p[:, 4] >= maximum.accumulate(p[:, 4]))[0]
                    for n, p in enumerate(
                        split(
                            first_triggers,
                            # exclude the first since it's 0
                            first_triggers_pairs_offset[1:],
                        )
                    )
                ],
            )
        ]
        # mark objects for gc
        del (
            data_expd,
            triggered,
            stoploss_bought_ohlc_triggered_s1,
            first_triggers,
            r_df,
            ohlc_vals,
        )
        gc.collect()
        return stoploss

    def _pd_calc_triggered_stoploss(
        self, df: DataFrame, bought: DataFrame, bought_ranges: ndarray,
    ):
        """ Expand the ohlc dataframe for each bought candle to check if stoploss was triggered """
        gc.collect()

        # code you want to profile

        df["ohlc"] = df.index.get_level_values(0).values
        df["ohlc_ofs"] = (
            df["ohlc"].values + df["ofs"].values.tolist() - self.startup_offset
        )
        ohlc_vals = df["ohlc_ofs"].values

        # create a df with just the indexes to expand
        stoploss_idx_expd = DataFrame(
            (
                concatenate(
                    [
                        ohlc_vals[i : i + bought_ranges[n]]
                        # loop over the pair/offsetted indexes that will be used as merge key
                        for n, i in enumerate(
                            bought["ohlc"].values
                            + bought["ofs"].values.tolist()
                            - self.startup_offset
                        )
                    ]
                )
            ),
            columns=["ohlc_ofs"],
        )
        # add the row data to the expanded indexes
        bought_expd = stoploss_idx_expd.merge(
            # reset level 1 to preserve pair column
            df.reset_index(level=1),
            how="left",
            left_on="ohlc_ofs",
            right_on="ohlc_ofs",
        )
        # set bought idx for each bought timerange, so that we know to which bought candle
        # the row belongs to, and stoploss rates relative to each bought
        bought_expd["stoploss_bought_ohlc"], bought_expd["stoploss_rate"] = repeat(
            [bought["ohlc"].values, self._calc_stoploss_rate(bought),],
            bought_ranges,
            axis=1,
        )

        triggered = bought_expd.loc[
            bought_expd["low"].values <= bought_expd["stoploss_rate"].values
        ]
        # filter out duplicate subsequent triggers of the same bought candle as only the first ones matters
        first_triggers = triggered.loc[
            (
                triggered["stoploss_bought_ohlc"].values
                != triggered["stoploss_bought_ohlc"].shift().values
            )
            | (triggered["pair"].values != triggered["pair"].shift().values)
        ]
        # filter out "late" stoplosses that wouldn't be applied because a previous stoploss
        # would still be active at that time
        # since stoplosses are sorted by trigger date, any stoploss having a bought index older than
        # the ohlc index are invalid
        stoploss = first_triggers.loc[
            concatenate(
                first_triggers.groupby("pair")["stoploss_bought_ohlc"].apply(
                    lambda x: x.values >= x.cummax().values
                )
            )
        ]
        # select columns
        stoploss = stoploss[
            ["ohlc", "pair", "stoploss_bought_ohlc", "stoploss_rate"]
        ].rename({"ohlc": "stoploss_idx"}, axis=1)
        # mark objects for gc
        del (
            df,
            stoploss_idx_expd,
            bought_expd,
            triggered,
            first_triggers,
            ohlc_vals,
        )
        gc.collect()
        return stoploss

    @staticmethod
    def _last_stoploss_apply(df: DataFrame):
        """ Loop over each row of the dataframe and only select stoplosses for boughts that
        happened after the last set stoploss """
        last = [0]

        def trail_idx(x, last):
            if x.bought_or_sold == Candle.BOUGHT:
                # if a bought candle happens after the last active stoploss index
                if x.ohlc > last[0]:
                    # if stoploss is triggered
                    if x.stoploss_rate > 0:
                        # set the new active stoploss to the current stoploss index
                        last[0] = x.stoploss_idx
                    else:
                        last[0] = nan
                return last[0]
            else:
                # if the candle is sold, reset the last active stoploss
                last[0] = 0
                return nan

        return df.apply(trail_idx, axis=1, raw=True, args=[last]).values

    @staticmethod
    def _last_stoploss_numba(bts_df: DataFrame):
        """ numba version of _last_stoploss_apply """

        return for_trail_idx(
            bts_df["ohlc"].astype(int).values,
            bts_df["bought_or_sold"].astype(int).values,
            bts_df["stoploss_rate"].fillna(0).astype(float).values,
            bts_df["stoploss_idx"].fillna(-1).astype(int).values,
        )

    @staticmethod
    def start_pyinst():
        from pyinstrument import Profiler

        global profiler
        profiler = Profiler()
        profiler.start()

    @staticmethod
    def stop_pyinst():
        global profiler
        profiler.stop()
        print(profiler.output_text(unicode=True, color=True))
        exit()

    def _pd_select_triggered_stoploss(
        self,
        df: DataFrame,
        bought: DataFrame,
        bought_ranges: ndarray,
        bts_df: DataFrame,
    ) -> (Index, ndarray):

        bts_df.drop(columns="pair", inplace=True)
        # compute all the stoplosses for the buy signals and filter out clear invalids
        stoploss = DataFrame(
            self._np_calc_triggered_stoploss(df, bought, bought_ranges)[:, 2:],
            columns=["stoploss_idx", "pair", "stoploss_bought_ohlc", "stoploss_rate"],
            copy=False,
        )
        # stoploss = self._pd_calc_triggered_stoploss(df, bought, bought_ranges)

        # add stoploss data to the bought/sold dataframe
        bts_df = bts_df.merge(
            stoploss,
            left_on=["ohlc", "pair"],
            right_on=["stoploss_bought_ohlc", "pair"],
            how="left",
        ).set_index("ohlc_ofs")
        # don't apply stoploss to sold candles
        bts_df.loc[bts_df["bought_or_sold"].values == Candle.SOLD, "stoploss_idx"] = nan
        # exclude nested boughts
        # --> | BUY1 | BUY2..STOP2 | STOP1 | -->
        # -->      V       X      X       V  -->
        bts_df["last_stoploss"] = concatenate(
            bts_df.groupby("pair").apply(self._last_stoploss_numba).values
        )
        bts_df.loc[
            ~(  # last active stoploss matches the current stoploss, otherwise it's stale
                (bts_df["stoploss_idx"].values == bts_df["last_stoploss"].values)
                # it must be the first bought matching that stoploss index, in case of subsequent
                # boughts that triggers on the same index which wouldn't happen without position stacking
                & (
                    bts_df["last_stoploss"].values
                    != bts_df["last_stoploss"].shift().values
                )
            ),
            ["stoploss_idx", "stoploss_rate"],
        ] = [nan, nan]
        gc.collect()
        return bts_df

    def _set_stoploss_rate(self, df: DataFrame):
        """ Adds a column for the stoploss rate """
        df["stoploss_rate"] = self._calc_stoploss_rate(df)

    def _calc_stoploss_rate(self, df: DataFrame) -> ndarray:
        return df["open"].values * (1 + self.config["stoploss"])

    def _calc_stoploss_rate_value(self, open_price: float) -> float:
        return open_price * (1 + self.config["stoploss"])

    def vectorized_backtest_buy_sell(
        self,
        processed: Dict[str, DataFrame],
        start_date: arrow.Arrow,
        end_date: arrow.Arrow,
        **kwargs,
    ) -> DataFrame:
        return None

    def vectorized_backtest(
        self,
        processed: Dict[str, DataFrame],
        start_date: arrow.Arrow,
        end_date: arrow.Arrow,
        **kwargs,
    ) -> DataFrame:
        """ NOTE: can't have default values as arguments since it is an overridden function
        TODO: benchmark if rewriting without use of df masks for readability gives a worthwhile speedup
        """
        df = self.merge_pairs_df(processed)

        df, empty = self.bought_or_sold(df)

        if empty:  # if no bought signals
            return self.empty_results

        bts_df = self.set_stoploss(df)

        if len(bts_df) < 1:
            return self.empty_results

        bts_gb = bts_df.groupby("pair")
        bts_ls_s1 = bts_gb.last_stoploss.shift().values
        events_buy = bts_df.loc[
            (bts_df["bought_or_sold"].values == Candle.BOUGHT)
            & (
                (
                    bts_df["bought_or_sold"].shift(fill_value=Candle.SOLD).values
                    == Candle.SOLD
                )
                # last_stoploss is only valid if == shift(1)
                # if the previous candle is SOLD it is covered by the previous case
                | ((bts_df["last_stoploss"].values != bts_ls_s1))
            )
            # exclude the last boughts that are not stoploss and which next sold is
            # END sold candle
            & ~(
                (bts_df["stoploss_idx"].isna())
                & union_eq(bts_df["next_sold_ofs"], self.pairs_ofs_end)
            )
        ]
        events_sell = bts_df.loc[
            (
                (bts_df["bought_or_sold"].values == Candle.SOLD)
                # select only sold candles that are not preceded by a stoploss
                & (bts_ls_s1 == -1)
            )
            # and stoplosses (all candles with notna stoploss_idx should be valid)
            | ((bts_df["stoploss_idx"].notna().values))
        ]
        # if isnan(events_sell["open"].iat[-1]):
        # events_sell = events_sell.iloc[:-1]
        # remove
        self._validate_results(events_buy, events_sell)
        results = self.get_results(events_buy, events_sell)
        return results

    def _validate_results(self, events_buy: DataFrame, events_sell: DataFrame):
        try:
            assert len(events_buy) == len(events_sell)
        except AssertionError:
            print("Buy and sell events not matching", self.pairs_end)
            print(len(events_buy), len(events_sell))
            print(events_buy.iloc[-10:], events_sell.iloc[-10:])
            raise OperationalException

    def _wrap_backtest(
        self,
        processed: Dict[str, DataFrame],
        start_date: arrow.Arrow,
        end_date: arrow.Arrow,
        **kwargs,
    ) -> DataFrame:
        """ debugging """
        import pickle
        # results = self.backtest_stock(
        #     processed, self.config["stake_amount"], start_date, end_date
        # )
        results = self.vectorized_backtest(processed, start_date, end_date)
        with open("/tmp/backtest.pkl", "rb+") as fp:
            # pickle.dump(results, fp)
            saved_results: DataFrame = pickle.load(fp)
        to_print = []
        # for i in results["open_index"].values:
        #     if i not in saved_results["open_index"].values:
        #         to_print.append(i)
        for i in saved_results["open_index"].values:
            if i not in results["open_index"].values:
                to_print.append(i)
        # print(saved_results.sort_values(["pair", "open_time"]).iloc[:10])
        # print(
        #     "to_print count: ",
        #     len(to_print),
        #     "computed res: ",
        #     len(results),
        #     "saved res: ",
        #     len(saved_results),
        # )
        # print(to_print[:10])
        if to_print:
            print(saved_results.loc[saved_results["open_index"].isin(to_print)])
        return results
