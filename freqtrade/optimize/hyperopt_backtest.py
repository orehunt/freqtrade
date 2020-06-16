import logging

from joblib import load
import arrow
import gc
from typing import Dict, Any, List, Tuple
from numpy import (
    append,
    repeat,
    nan,
    concatenate,
    arange,
    ndarray,
    memmap,
    ma,
    where,
    transpose,
)
from pandas import Timedelta, to_timedelta, concat, Series, DataFrame, Index, merge
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
    BOUGHT_AND_STOPLOSS = 11
    STOPLOSS = 17
    NOOP = 0


class HyperoptBacktesting(Backtesting):

    empty_results = DataFrame.from_records([], columns=BacktestResult._fields)
    debug = False

    td_zero = Timedelta(0)
    td_half_timeframe: Timedelta

    def __init__(self, config):
        if config.get("backtesting_engine") == "vectorized":
            self.backtest = self.vectorized_backtest
            self.beacktesting_engine = "vectorized"
            self.td_half_timeframe = (
                Timedelta(config.get("timeframe", config["ticker_interval"])) / 2
            )
        super().__init__(config)

    # def set_stoploss(
    #     self, bought_index: Index, stoploss_rate: Series, pair_df: DataFrame
    # ) -> Series:
    #     """ Update dataframe stoploss col from Series of index matching stoploss rates """
    #     # reduce stoploss rate to bought_df index
    #     stoploss_rate = stoploss_rate.loc[bought_index]
    #     # calc the number of candles for which to apply stoploss using the index diff
    #     repeats = stoploss_rate.index.values[1:] - stoploss_rate.index.values[:-1]
    #     # repeats is one element short, append the last that will be repeated until pair_df max idx
    #     # add 1 to compensate for the shift of bought_or_sold
    #     repeats = append(repeats, pair_df.index[-1] - stoploss_rate.index[-1] + 1)
    #     stoploss_arr = repeat(stoploss_rate, repeats)
    #     # prepend the slice without signals
    #     no_sigs = Series(
    #         data=nan,
    #         index=pair_df.iloc[pair_df.index[0] : stoploss_rate.index[0]].index,
    #     )
    #     pair_df["stoploss"] = concat([no_sigs, stoploss_arr]).values

    def get_results(
        self, pair: str, events_buy: DataFrame, events_sell: DataFrame
    ) -> DataFrame:
        # choose sell rate depending on sell reason and set sell_reason
        events_sold = events_sell.loc[
            events_sell["bought_or_sold"].values == Candle.SOLD
        ]
        # add new columns with reindex to allow multi col assignments of new columns
        events_sell = events_sell.reindex(
            columns=[*events_sell.columns, "close_rate", "sell_reason"]
        )
        events_sell.loc[events_sold.index, ["close_rate", "sell_reason"]] = [
            events_sold["open"].values,
            SellType.SELL_SIGNAL,
        ]
        events_stoploss = events_sell.loc[events_sell["stoploss"] == Candle.STOPLOSS]
        events_sell.loc[events_stoploss.index, ["close_rate", "sell_reason"]] = [
            events_stoploss["stoploss_rate"].values,
            SellType.STOP_LOSS,
        ]

        open_rate = events_buy["open"].values
        close_rate = events_sell["close_rate"].values
        profits = 1 - open_rate / close_rate - 2 * self.fee
        trade_duration = Series(events_sell["date"].values - events_buy["date"].values)
        # replace trade duration of same candle trades with half the timeframe reduce to minutes
        trade_duration.loc[trade_duration == self.td_zero] = self.td_half_timeframe

        if self.debug:
            print("creating results df")
        return DataFrame(
            {
                "pair": pair,
                "profit_percent": profits,
                "profit_abs": profits * self.config["stake_amount"],
                "open_time": events_buy["date"].values,
                "close_time": events_sell["date"].values,
                "open_index": events_buy.index.values,
                "close_index": events_sell.index.values,
                "trade_duration": trade_duration.dt.seconds / 60,
                "open_at_end": False,
                "open_rate": open_rate,
                "close_rate": close_rate,
                "sell_reason": events_sell["sell_reason"].values,
            }
        )

    def get_pair_df(self, processed: Dict[str, DataFrame]) -> DataFrame:
        """ Execute strategy signals and return df """
        if len(processed) > 1:
            raise OperationalException(
                "Can only use vectorized backtest with one pair."
            )

        pair = next(iter(processed))
        pair_df = processed[pair].copy()
        metadata = {"pair": pair}
        pair_df = self.strategy.advise_buy(pair_df, metadata)
        pair_df = self.strategy.advise_sell(pair_df, metadata)
        pair_df.fillna({"buy": 0, "sell": 0}, inplace=True)
        return pair_df, pair

    def bought_or_sold(
        self, pair_df: DataFrame
    ) -> Tuple[DataFrame, DataFrame, BacktestResult]:
        """ Set bought_or_sold columns according to buy and sell signals """
        # set bought candles
        pair_df["bought_or_sold"] = pair_df["buy"].shift(-1) - pair_df["sell"].shift(-1)
        pair_df.loc[
            pair_df["bought_or_sold"].values == 1, "bought_or_sold"
        ] = Candle.BOUGHT
        # skip if no valid bought candles are found
        bought_df = pair_df.loc[pair_df["bought_or_sold"].values == Candle.BOUGHT]
        if len(bought_df) < 1:
            return None, None, self.empty_results
        # set sold candles
        pair_df.loc[
            pair_df["bought_or_sold"].values == -1, "bought_or_sold"
        ] = Candle.SOLD
        return pair_df, bought_df, None

    # def filter_adjacent_boughts(
    #     self, bought_df: DataFrame, pair_df: DataFrame
    # ) -> Tuple[DataFrame, DataFrame]:
    #     """ remove index adjacent bought candles where stoploss is not triggered """
    #     # update bought_df excluding bought candles with stoploss by index
    #     bought_df.drop(
    #         pair_df.loc[
    #             (pair_df["bought_or_sold"].values == Candle.BOUGHT)
    #             & (pair_df["stoploss"].values == Candle.STOPLOSS)
    #         ].index,
    #         inplace=True,
    #     )
    #     # 2; noop index adjacent bought candles if no position stacking, by subtracting indexes and matching against 1
    #     # (we already removed candles where stoploss is triggered on the same candle so the diff must at least be 2)
    #     bought_adjacent = bought_df.iloc[1:].loc[
    #         bought_df.index.values[1:] - bought_df.index.values[:-1] == 1
    #     ]
    #     pair_df.loc[bought_adjacent.index, "bought_or_sold"] = Candle.NOOP

    def remove_stale_boughts(self, pair_df: DataFrame):
        """
        remove subsequent bought candles as neither stoploss nor signal triggered
        or bought_and_stoploss candles preceded by a bought candles, as that bought
        would still be active
        """
        # select all possible boughts
        possible = pair_df.loc[
            (pair_df["bought_or_sold"].values != Candle.NOOP)
            | (pair_df["stoploss"].values == Candle.STOPLOSS)
        ]
        # can't buy if the previous candle did not close the trade
        stale_bought = possible.loc[
            (possible["bought_or_sold"].values == Candle.BOUGHT)
            & (
                possible["stoploss"].shift(fill_value=Candle.STOPLOSS).values
                != Candle.STOPLOSS
            )
            & (
                possible["bought_or_sold"].shift(fill_value=Candle.SOLD).values
                != Candle.SOLD
            )
        ].index
        pair_df.loc[stale_bought, ["bought_or_sold", "stoploss"]] = [
            Candle.NOOP,
            Candle.NOOP,
        ]

    @staticmethod
    def boughts_to_sold(pair_df: DataFrame) -> DataFrame:
        """
        reduce df such that there are many bought interleaved by one sold candle
        NOTE: does not modify input df
        """
        bos_df = pair_df.loc[
            (pair_df["bought_or_sold"].values == Candle.BOUGHT)
            | (pair_df["bought_or_sold"].values == Candle.SOLD)
        ]
        bos_df = bos_df.loc[
            # exclude duplicate sold
            ~(
                (bos_df["bought_or_sold"].values == Candle.SOLD)
                & (bos_df["bought_or_sold"].shift().values == Candle.SOLD)
            )
        ]
        # skip the first sold candle since it's useless without boughts
        if bos_df.iloc[0]["bought_or_sold"] == Candle.SOLD:
            bos_df = bos_df.iloc[1:]
        return bos_df

    # @staticmethod
    # def fill_stub_sold(pair_df: DataFrame, bts_df: DataFrame) -> DataFrame:
    #     sold = (
    #         pair_df.loc[~pair_df.index.isin(bts_df.set_index("index").index)]
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
    #         sold.iloc[len(sold)] = pair_df.iloc[-1]
    #         sold.iloc[-1]["bought_or_sold"] = Candle.SOLD
    #     return (bts_df, sold)

    def set_stoploss(self, pair_df: DataFrame) -> DataFrame:
        """
        returns the df of valid boughts where stoploss triggered, with matching stoploss
        index of each bought
        """
        pair_df = pair_df.reindex(
            copy=False, columns=[*pair_df.columns, "stoploss", "stoploss_rate"]
        )
        bts_df = self.boughts_to_sold(pair_df).reset_index()
        # align sold to bought
        sold = bts_df.loc[bts_df["bought_or_sold"].values == Candle.SOLD]
        # if no sold signals, return
        if len(sold) < 1:
            pair_df["bought_or_sold"] = Candle.NOOP
            return pair_df
            # bts_df, sold = self.fill_stub_sold(pair_df, bts_df)
        # skip remaining bought candles without sold candle
        bts_df = bts_df.loc[: sold.index[-1]]
        pair_df.loc[sold.index[-1] + 1 :, "bought_or_sold"] = Candle.NOOP
        sold_repeats = sold.index.values[1:] - sold.index.values[:-1]
        # prepend the first range subtracting the index of the first bought
        sold_repeats = [sold.index[0] + 1, *sold_repeats]
        # NOTE: use the "index" col with original indexes added by reset_index
        bts_df["next_sold_idx"] = repeat(sold["index"].values, sold_repeats)
        bought = bts_df.loc[bts_df["bought_or_sold"].values == Candle.BOUGHT]
        # get the index ranges of each bought->sold spans
        bought_ranges = bought["next_sold_idx"].values - bought["index"].values
        stoploss_index, stoploss_rate = self._np_select_triggered_stoploss(
            pair_df, bought, bought_ranges
        )

        # set stoploss and stoploss_rate columns
        pair_df.loc[stoploss_index, ["stoploss", "stoploss_rate"]] = [
            Candle.STOPLOSS,
            stoploss_rate,
        ]
        pair_df.fillna({"stoploss": Candle.NOOP}, inplace=True)
        return pair_df

    def _np_select_triggered_stoploss(
        self, pair_df: DataFrame, bought: DataFrame, bought_ranges: ndarray
    ) -> ndarray:
        # expand bought ranges into ohlc data
        r_pair_df = pair_df.reset_index()
        file_path = "/tmp/pair_df.mmap" + str(os.getpid())
        fp = memmap(
            file_path, dtype=r_pair_df.dtypes, mode="w+", shape=r_pair_df.shape,
        )
        fp[:] = r_pair_df.values ; del fp
        ohlc_vals = memmap(
            file_path, dtype=r_pair_df.dtypes, mode="r", shape=r_pair_df.shape,
        )
        # align the start of pair_df index in case it doesn't start from 0 (e.g. startup candles are trimmed
        offset_idx = pair_df.index[0]
        open_col_idx = r_pair_df.columns.get_loc("open")
        low_col_idx = r_pair_df.columns.get_loc("low")
        index_col_idx = r_pair_df.columns.get_loc("index")
        # 0: open, 1: low, 2: index, 3: bought_idx, 4: sold_idx
        data_expd = ma.concatenate(
            [
                ma.concatenate(
                    [
                        ohlc_vals[
                            i : i + bought_ranges[n],
                            [open_col_idx, low_col_idx, index_col_idx],
                        ]
                        for n, i in enumerate(bought["index"].values - offset_idx)
                    ]
                ),
                transpose(
                    repeat(
                        [bought["index"].values, bought["next_sold_idx"].values],
                        bought_ranges,
                        axis=1,
                    )
                ),
            ],
            axis=1,
        )
        stoploss_rate = data_expd[:, 0] * (1 + self.config["stoploss"] + 2 * self.fee)
        # add stoploss_rate column
        # 5: stoploss_rate
        data_expd = ma.concatenate([data_expd, transpose([stoploss_rate])], axis=1)
        stoploss_triggered = data_expd[:, 1] <= stoploss_rate
        data_triggered = data_expd[stoploss_triggered, :]
        # only where the bought_idx (3) is not the same as the previous
        bought_idx_triggered_s1 = ma.concatenate([[nan], data_triggered[:-1, 3]])
        # use where here because array is 1 element shorter
        first_triggers = data_triggered[
            where(data_triggered[:, 3] != bought_idx_triggered_s1)
        ]
        # index column is (2)
        stoploss_indexes_s1 = ma.concatenate([[0], first_triggers[:-1, 2]])
        stoploss = first_triggers[where(first_triggers[:, 3] > stoploss_indexes_s1)]
        del (
            data_expd,
            stoploss_rate,
            stoploss_triggered,
            data_triggered,
            bought_idx_triggered_s1,
            first_triggers,
            stoploss_indexes_s1,
            r_pair_df,
            ohlc_vals,
        )
        gc.collect()
        return stoploss[:, 4], stoploss[:, 5]

    def _pd_select_triggered_stoploss(
        self, pair_df: DataFrame, bought: DataFrame, bought_ranges: ndarray
    ) -> (Index, ndarray):
        idx = pair_df.index.values
        offset_idx = idx[0]
        offset = idx[0]
        ohlc_idx_expd = DataFrame(
            concatenate(
                [
                    idx[i : i + bought_ranges[n]]
                    for n, i in enumerate(bought["index"].values - offset_idx)
                ]
            ),
            columns=["ohlc_idx"],
        )
        bought_expd = ohlc_idx_expd.merge(
            pair_df, how="inner", left_on=["ohlc_idx"], right_index=True
        )
        # set bought idx for each bought timerange, so that we know to which bought candle
        # the row belongs to, and sold indexes
        bought_sold_idx = repeat(
            [bought["index"].values, bought["next_sold_idx"].values],
            bought_ranges,
            axis=1,
        )
        bought_expd["bought_idx"] = bought_sold_idx[0]
        bought_expd["next_sold_idx"] = bought_sold_idx[1]
        # mem = bought_expd.memory_usage(index=True, deep=True).sum() / 1048576
        # if mem > 500:
        #     self.debug = True
        # calc stoploss
        self._set_stoploss_rate(bought_expd)
        bought_expd["stoploss_triggered"] = (
            bought_expd["low"].values <= bought_expd["stoploss_rate"].values
        )
        triggered = bought_expd.loc[bought_expd["stoploss_triggered"]]
        # filter out duplicate subsequent triggers of the same bought candle as only the first matters
        first_triggers = triggered.loc[
            triggered["bought_idx"].values != triggered.shift()["bought_idx"].values
        ]
        # the only valid bought candles are the ones where the bought index is
        # STRICTLY higher than the previous ohlc_index (which is the index of the stoplosses)
        # fill to 0 (the first bought is always valid)
        stoploss = first_triggers.loc[
            first_triggers["bought_idx"].values
            > first_triggers["ohlc_idx"].shift(fill_value=0).values
        ]
        stoploss_index, stoploss_rate = (
            stoploss["ohlc_idx"].values,
            stoploss["stoploss_rate"].values,
        )

        del (
            pair_df,
            bought_expd,
            idx,
            bought_sold_idx,
            ohlc_idx_expd,
            triggered,
            first_triggers,
            stoploss,
        )
        gc.collect()
        return stoploss_index, stoploss_rate

    def _set_stoploss_rate(self, df: DataFrame):
        """ Adds a column for the stoploss rate """
        df["stoploss_rate"] = df["open"].values * (
            1 + self.config["stoploss"] + 2 * self.fee
        )

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
        pair_df, pair = self.get_pair_df(processed)

        pair_df, bought_df, no_bought = self.bought_or_sold(pair_df)
        if no_bought is not None:  # if no bought signals
            return no_bought

        pair_df = self.set_stoploss(pair_df)
        self.remove_stale_boughts(pair_df)

        # split buys and sell and calc results
        events = pair_df.loc[
            (pair_df["bought_or_sold"].values != Candle.NOOP)
            | (pair_df["stoploss"].values == Candle.STOPLOSS)
        ]
        # only bought candles preceded by a sold or a stoploss
        bos_s1 = events["bought_or_sold"].shift(fill_value=Candle.SOLD)
        stop_s1 = events["stoploss"].shift(fill_value=Candle.STOPLOSS)
        events_buy = events.loc[
            (events["bought_or_sold"].values == Candle.BOUGHT)
            & ((bos_s1.values == Candle.SOLD) | (stop_s1.values == Candle.STOPLOSS))
        ]
        # only sold candles preceded by bought candles without stoploss
        # or stoploss candles not sold and not bought and preceded by bought
        # or stoploss candles not sold and bought and not preceded by bought
        events_sell = events.loc[
            (
                (events["bought_or_sold"].values == Candle.SOLD)
                & (bos_s1 == Candle.BOUGHT)
                & (stop_s1 != Candle.STOPLOSS)
            )
            | (
                (events["stoploss"].values == Candle.STOPLOSS)
                & (events["bought_or_sold"].values != Candle.SOLD)
                & (
                    (
                        (events["bought_or_sold"].values != Candle.BOUGHT)
                        & (bos_s1.values == Candle.BOUGHT)
                        & (stop_s1 != Candle.STOPLOSS)
                    )
                    | (
                        (events["bought_or_sold"].values == Candle.BOUGHT)
                        & (
                            (bos_s1.values != Candle.BOUGHT)
                            | (stop_s1.values == Candle.STOPLOSS)
                        )
                    )
                )
            )
        ]
        try:
            assert len(events_buy) == len(events_sell)
        except AssertionError:
            print("Buy and sell events not matching")
            print(len(events_buy), len(events_sell))
            return self.empty_results
        results = self.get_results(pair, events_buy, events_sell)
        return results
