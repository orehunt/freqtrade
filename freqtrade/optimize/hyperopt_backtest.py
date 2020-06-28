import logging

from joblib import load
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
    memmap,
    ma,
    where,
    transpose,
    maximum,
)
from pandas import Timedelta, to_timedelta, concat, Series, DataFrame, Index, merge
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
                Timedelta(config.get("timeframe", config["timeframe"])) / 2
            )
        super().__init__(config)
        # self.config["stoploss"] = 0.01

    def get_results(
        self, pair: str, events_buy: DataFrame, events_sell: DataFrame
    ) -> DataFrame:
        # choose sell rate depending on sell reason and set sell_reason
        events_sold = events_sell.loc[
            events_sell["bought_or_sold"].values == Candle.SOLD
        ]
        # add new columns with reindex to allow multi col assignments of new columns
        events_sell = events_sell.reindex(
            columns=[*events_sell.columns, "close_rate", "sell_reason"], copy=False
        )
        events_sell.loc[events_sold.index, ["close_rate", "sell_reason"]] = [
            events_sold["open"].values,
            SellType.SELL_SIGNAL,
        ]
        events_stoploss = events_sell.loc[events_sell["stoploss_idx"].notna().values]
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

    def bought_or_sold(self, pair_df: DataFrame) -> Tuple[DataFrame, DataFrame, bool]:
        """ Set bought_or_sold columns according to buy and sell signals """
        # set bought candles
        pair_df["bought_or_sold"] = (pair_df["buy"] - pair_df["sell"]).shift(1).values
        pair_df.loc[
            pair_df["bought_or_sold"].values == 1, "bought_or_sold"
        ] = Candle.BOUGHT
        # skip if no valid bought candles are found
        bought_df = pair_df.loc[pair_df["bought_or_sold"].values == Candle.BOUGHT]
        if len(bought_df) < 1:
            return None, None, True
        # set sold candles
        pair_df.loc[
            pair_df["bought_or_sold"].values == -1, "bought_or_sold"
        ] = Candle.SOLD
        return pair_df, bought_df, False

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

    def set_stoploss_col(
        self, pair_df: DataFrame, stoploss_index, stoploss_rate: ndarray
    ):
        # set stoploss and stoploss_rate columns
        pair_df.loc[stoploss_index, ["stoploss", "stoploss_rate"]] = [
            Candle.STOPLOSS,
            stoploss_rate,
        ]

    def set_stoploss(self, pair_df: DataFrame) -> DataFrame:
        """
        returns the df of valid boughts where stoploss triggered, with matching stoploss
        index of each bought
        """
        pair_df = pair_df.reindex(
            copy=False, columns=[*pair_df.columns, "stoploss_idx", "stoploss_rate", "last_stoploss"],
        )
        bts_df = self.boughts_to_sold(pair_df).reset_index()
        # align sold to bought
        sold = bts_df.loc[bts_df["bought_or_sold"].values == Candle.SOLD]
        # if no sold signals, return
        if len(sold) < 1:
            return DataFrame(columns=[*bts_df.columns, "last_stoploss"])
            # if no sell sig is provided a limit on the trade duration could be applied..
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
        # bts_df = self._pd_select_triggered_stoploss(
        #     pair_df, bought, bought_ranges, bts_df
        # )
        bts_df = self._pd_2_select_triggered_stoploss(
            pair_df, bought, bought_ranges, sold, bts_df
        )

        return bts_df

    def _pd_2_select_triggered_stoploss(
        self,
        pair_df: DataFrame,
        bought: DataFrame,
        bought_ranges: ndarray,
        sold: DataFrame,
        bts_df: DataFrame,
    ):
        b = 0
        bought_idx = bought["index"].iat[b]
        current_index = bought_idx
        stoploss_index = []
        stoploss_rate = []
        active_boughts = []
        last_stoploss = []
        end_idx = pair_df.index[-1]
        # exclude sold candles from stoploss check
        ohlc_no_sold = pair_df.loc[~pair_df.index.isin(sold["index"].values)]

        while bought_idx < end_idx:
            active_boughts.append(bought_idx) 
            ohlc_range = ohlc_no_sold.loc[bought_idx : bought_idx + bought_ranges[b]]
            # calculate the rate only from the first candle
            stoploss_triggered_rate = self._calc_stoploss_rate_value(
                ohlc_range["open"].iat[0]
            )
            stoploss_triggered = ohlc_range.loc[
                ohlc_range["low"].values <= stoploss_triggered_rate
            ]
            if len(stoploss_triggered) > 0:
                # the first that triggers is the one that happens
                current_index = stoploss_triggered.index[0]
                stoploss_index.append(current_index)
                stoploss_rate.append(stoploss_triggered_rate)
                # print(
                #     "stoploss triggered for:",
                #     bought_idx,
                #     "at index: ",
                #     current_index,
                #     "before sell: ",
                #     bought["next_sold_idx"].iat[b],
                # )
                try:
                    # get the first row where the bought index is higher than the current stoploss index
                    b = list(bought["index"].values > current_index).index(True)
                    bought_idx = bought["index"].iat[b]
                except ValueError:
                    break
                # print("next bought after stoploss at:", bought_idx, b)
            else:  # if stoploss did not trigger, jump to the first bought after next sold idx
                try:
                    b = list(
                        bought["index"].values > bought["next_sold_idx"].iat[b]
                    ).index(True)
                    bought_idx = bought["index"].iat[b]
                except ValueError:
                    break
                # print("skipping to next bought", b, bought_idx)

        # print("returning stoploss ", bought_idx, pair_df.index[-1])
        bts_df.loc[bought_stoploss_idx, ["stoploss_idx", "stoploss_rate", "last_stoploss"]] = [
            stoploss_index,
            stoploss_rate,
            stoploss_index,
        ]
        return bts_df

    def _np_calc_triggered_stoploss(
        self, pair_df: DataFrame, bought: DataFrame, bought_ranges: ndarray,
    ) -> (ndarray, ndarray):
        """ numpy equivalent of _pd_calc_triggered_stoploss that is more memory efficient """
        # clear up memory
        gc.collect()
        # expand bought ranges into ohlc data
        r_pair_df = pair_df.reset_index()
        ohlc_vals = r_pair_df.values
        # align the start of pair_df index in case it doesn't start from 0 (e.g. startup candles are trimmed
        offset_idx = pair_df.index[0]
        open_col_idx = r_pair_df.columns.get_loc("open")
        low_col_idx = r_pair_df.columns.get_loc("low")
        index_col_idx = r_pair_df.columns.get_loc("index")
        # 0: open, 1: low, 2: stoploss_idx, 3: bought_idx, 4: stoploss_rate
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
                        [bought["index"].values, self._calc_stoploss_rate(bought),],
                        bought_ranges,
                        axis=1,
                    )
                ),
            ],
            axis=1,
        )
        triggered = data_expd[data_expd[:, 1] <= data_expd[:, 4], :]
        # # only where the bought_idx (3) is not the same as the previous
        bought_idx_triggered_s1 = ma.concatenate([[nan], triggered[:-1, 3]])
        # # use where here because array is 1 element shorter
        first_triggers = triggered[where(triggered[:, 3] != bought_idx_triggered_s1)]
        # # index column is (2)
        stoploss = first_triggers[
            where(first_triggers[:, 2] >= maximum.accumulate(first_triggers[:, 2]))
        ]
        # mark objects for gc
        del (
            data_expd,
            triggered,
            bought_idx_triggered_s1,
            first_triggers,
            r_pair_df,
            ohlc_vals,
        )
        gc.collect()
        return stoploss

    def _pd_calc_triggered_stoploss(
        self, pair_df: DataFrame, bought: DataFrame, bought_ranges: ndarray,
    ):
        """ Expand the ohlc dataframe for each bought candle to check if stoploss was triggered """
        idx = pair_df.index.values
        offset_idx = idx[0]
        offset = idx[0]
        stoploss_idx_expd = DataFrame(
            ma.concatenate(
                [
                    idx[i : i + bought_ranges[n]]
                    for n, i in enumerate(bought["index"].values - offset_idx)
                ]
            ),
            columns=["stoploss_idx"],
        )
        bought_expd = stoploss_idx_expd.merge(
            pair_df, how="left", left_on=["stoploss_idx"], right_index=True
        )
        # set bought idx for each bought timerange, so that we know to which bought candle
        # the row belongs to, and stoploss rates relative to each bought
        bought_stop_idx = repeat(
            [bought["index"].values, self._calc_stoploss_rate(bought),],
            bought_ranges,
            axis=1,
        )
        bought_expd["bought_idx"] = bought_stop_idx[0]
        bought_expd["stoploss_rate"] = bought_stop_idx[1]

        triggered = bought_expd.loc[
            bought_expd["low"].values <= bought_expd["stoploss_rate"].values
        ]
        # filter out duplicate subsequent triggers of the same bought candle as only the first ones matters
        first_triggers = triggered.loc[
            triggered["bought_idx"].values != triggered["bought_idx"].shift().values
        ]
        stoploss = first_triggers.loc[
            # if the stoploss_idx is lower than the cummax then it was a bought that
            # happened before the last stoploss so, exclude them
            (
                first_triggers["stoploss_idx"].values
                >= first_triggers["stoploss_idx"].cummax().values
            )
            # select only the relevant columns to speedup combine
        ].loc[:, ["stoploss_idx", "bought_idx", "stoploss_rate"],]
        return stoploss

    @staticmethod
    def _last_stoploss_apply(pair_df: DataFrame, bts_df: DataFrame):
        """ Loop over each row of the dataframe and only select stoplosses for boughts that
        happened after the last set stoploss """
        last = [0]

        def trail_idx(x, last):
            if x.bought_or_sold == Candle.BOUGHT:
                # if a bought candle happens after the last active stoploss index
                if x.name > last[0]:
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

        bts_df["last_stoploss"] = bts_df.apply(trail_idx, axis=1, raw=True, args=[last])

    @staticmethod
    def _last_stoploss_numba(pair_df: DataFrame, bts_df: DataFrame):
        """ numba version of _last_stoploss_apply """

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

        bts_df["last_stoploss"] = for_trail_idx(
            bts_df.index.astype(int).values,
            bts_df["bought_or_sold"].astype(int).values,
            bts_df["stoploss_rate"].fillna(0).astype(float).values,
            bts_df["stoploss_idx"].fillna(-1).astype(int).values,
        )

    def _pd_select_triggered_stoploss(
        self,
        pair_df: DataFrame,
        bought: DataFrame,
        bought_ranges: ndarray,
        bts_df: DataFrame,
    ) -> (Index, ndarray):

        stoploss = DataFrame(
            self._np_calc_triggered_stoploss(pair_df, bought, bought_ranges)[:, 2:],
            columns=["stoploss_idx", "bought_idx", "stoploss_rate"],
            copy=False,
        )

        bts_df = bts_df.set_index("index").combine_first(
            stoploss.set_index("bought_idx")
        )
        # don't apply stoploss to sold candles
        bts_df.loc[bts_df["bought_or_sold"].values == Candle.SOLD, "stoploss_idx"] = nan
        self._last_stoploss_numba(pair_df, bts_df)

        bts_df.loc[
            ~(  # last active stoploss matches the current stoploss, otherwise it's stale
                (bts_df["stoploss_idx"].values == bts_df["last_stoploss"].values)
                # it must be the first bought matching that stoploss index, in case of subsequent
                # boughts that triggers on the same index which wouldn't happen without position stacking
                & (bts_df["last_stoploss"].values != bts_df["last_stoploss"].shift().values)
            ),
            ["stoploss_idx", "stoploss_rate"],
        ] = [nan, nan]
        # del stoploss
        gc.collect()
        return bts_df

    def _set_stoploss_rate(self, df: DataFrame):
        """ Adds a column for the stoploss rate """
        df["stoploss_rate"] = self._calc_stoploss_rate(df)

    def _calc_stoploss_rate(self, df: DataFrame) -> ndarray:
        return df["open"].values * (1 + self.config["stoploss"] + 2 * self.fee)

    def _calc_stoploss_rate_value(self, open_price: float) -> float:
        return open_price * (1 + self.config["stoploss"] + 2 * self.fee)

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

        pair_df, bought_df, empty = self.bought_or_sold(pair_df)
        if empty:  # if no bought signals
            return self.empty_results

        bts_df = self.set_stoploss(pair_df)

        if len(bts_df) < 1:
            return self.empty_results

        bts_ls_s1 = bts_df["last_stoploss"].shift().values
        events_buy = bts_df.loc[
            (bts_df["bought_or_sold"].values == Candle.BOUGHT)
            & (
                (
                    bts_df["bought_or_sold"].shift(fill_value=Candle.SOLD).values
                    == Candle.SOLD
                )
                # happens if a stoploss was triggered on the previous bought
                # last_stoploss is only valid if == shift(1) if the previous candle is SOLD
                # which is covered by the previous case
                | (bts_df["last_stoploss"].values != bts_ls_s1)
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
        try:
            assert len(events_buy) == len(events_sell)
        except AssertionError:
            print("Buy and sell events not matching")
            print(len(events_buy), len(events_sell))
            print(events_buy.iloc[:10], events_sell.iloc[:10])
            raise OperationalException
        results = self.get_results(pair, events_buy, events_sell)
        return results
