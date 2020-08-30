import logging
import os
from typing import Dict, List, Tuple, Union
from types import SimpleNamespace
from enum import IntEnum
from collections import namedtuple
from functools import reduce

import talib as ta
from numba import types
from numba.typed import Dict as nb_Dict, List as nb_List
from numpy import (
    dtype,
    repeat,
    ones,
    nan,
    sum,
    concatenate,
    ndarray,
    recarray,
    array,
    split,
    where,
    hstack,
    argwhere,
    transpose,
    maximum,
    minimum,
    full,
    unique,
    insert,
    append,
    isfinite,
    isnan,
    isin,
    flip,
    in1d,
    sign,
    floor,
    roll,
    cumsum,
    arange,
    iinfo,
    int32,
    zeros,
    empty,
    nan_to_num,
    tile,
    sort,
    greater_equal,
    flatnonzero,
    bincount,
    swapaxes,
    ravel,
    empty_like,
    argmax,
    argmin,
    amin,
    searchsorted,
    argsort,
    argpartition,
    argsort,
    reshape,
    interp,
    amax,
)

np_cummax = maximum.accumulate

from pandas import (
    Timedelta,
    Series,
    DataFrame,
    Categorical,
    Index,
    MultiIndex,
    # SparseArray,
    set_option,
    to_timedelta,
    to_datetime,
    concat,
    Int64Index,
    factorize,
)

from freqtrade.optimize.backtesting import Backtesting, BacktestResult
from freqtrade.optimize.backtest_nb import *  # noqa ignore=F405
from freqtrade.optimize.backtest_utils import *  # noqa ignore=F405
from freqtrade.optimize.backtest_constants import *  # noqa ignore=F405
from freqtrade.optimize.debug import dbg  # noqa ignore=F405
from freqtrade.exceptions import OperationalException
from freqtrade.strategy.interface import SellType


logger = logging.getLogger(__name__)


class Candle(IntEnum):
    NOOP = 0
    BOUGHT = 1
    SOLD = 3
    END = 7  # references the last candle of a pair
    # STOPLOSS = 12


class OrderType(IntEnum):
    LIMIT = 0
    MARKET = 1


class HyperoptBacktesting(Backtesting):

    empty_results = DataFrame.from_records([], columns=BacktestResult._fields)
    events = None

    td_zero = Timedelta(0)
    td_timeframe: Timedelta
    td_half_timeframe: Timedelta
    pairs_offset: List[int]
    position_stacking: bool
    stoploss_enabled: bool
    sold_repeats: List[int]

    roi_col_names = []
    stoploss_col_names = []
    trailing_col_names = []
    conf_tuple = None

    df_loc = {}
    bts_loc = {}
    trigger_types = []
    # how many rows can an expanded array have
    max_ranges_size = 10e6
    # choose the backtesting function
    use_v1 = False
    use_v2 = False

    def __init__(self, config):
        if config.get("backtesting_engine") == "vectorized":
            self.backtest_vanilla = self.backtest
            if dbg:
                dbg._debug_opts()
                dbg.backtesting = self
                self.backtest = dbg._wrap_backtest
            else:
                self.backtest = self.vectorized_backtest
            self.beacktesting_engine = "vectorized"

        self.td_timeframe = Timedelta(config["timeframe"])
        self.td_half_timeframe = self.td_timeframe / 2
        self.timeframe_wnd = config.get("timeframe_window", TIMEFRAME_WND).get(
            config["timeframe"], DEFAULT_WND
        )

        backtesting_amounts = config.get("backtesting_amounts", {})
        self.stoploss_enabled = backtesting_amounts.get("stoploss", False)
        self.trailing_enabled = backtesting_amounts.get(
            "trailing", False
        ) and config.get("trailing_stop", False)
        self.roi_enabled = backtesting_amounts.get("roi", False)
        self.any_trigger = (
            self.stoploss_enabled or self.trailing_enabled or self.roi_enabled
        )
        self.trigger_types = [
            t
            for t, f in (
                ("stoploss_triggered", self.stoploss_enabled),
                ("trailing_triggered", self.trailing_enabled),
                ("roi_triggered", self.roi_enabled),
            )
            if f
        ]
        # null all config amounts for disabled ones (to compare against vanilla backtesting)
        if not self.roi_enabled:
            config["minimal_roi"] = {"0": 10}
        if not self.trailing_enabled:
            config["trailing_stop"] = False
        if not self.stoploss_enabled:
            config["stoploss"] = -100

        # parent init after config overrides
        super().__init__(config)

        self.position_stacking = self.config.get("position_stacking", False)
        if self.config.get("max_open_trades", 0) > 0:
            logger.warn("Ignoring max open trades...")

    def get_results(
        self, buy_vals: ndarray, sell_vals: ndarray, ohlc: DataFrame
    ) -> DataFrame:
        buy_cols = self.bts_loc
        sell_cols = buy_cols
        ohlc_vals = ohlc.values
        ohlc_cols = df_cols(ohlc)
        # choose sell rate depending on sell reason and set sell_reason
        # add new needed columns
        sell_vals = add_columns(sell_vals, sell_cols, ("close_rate", "trigger_ohlc"))
        # boolean masks can't query by slices and return a copy, so use where
        where_sold = where(sell_vals[:, sell_cols["bought_or_sold"]] == Candle.SOLD)[0]
        events_sold = sell_vals[where_sold, :]
        # use an external ndarray to store sell_type to avoid float conversion
        sell_reason = ndarray(shape=(sell_vals.shape[0]), dtype=IntEnum)

        sell_reason[where_sold] = SellType.SELL_SIGNAL
        sell_vals[where_sold, sell_cols["close_rate"]] = events_sold[
            :, sell_cols["open"]
        ]

        # adjust trigger_ofs to the startup offset, and the pairs offset
        # to match original ohlc index
        if self.any_trigger:
            sell_vals[:, sell_cols["trigger_ohlc"]] = (
                sell_vals[:, sell_cols["trigger_ofs"]]
                + self.startup_offset
                - sell_vals[:, sell_cols["ofs"]]
            )
            # list of columns that need data overwrite
            result_cols = [sell_cols[c] for c in ("date", "ohlc", "close_rate")]
            trigger_cols = [sell_cols[c] for c in ("trigger_date", "trigger_ohlc")]

        # order of events is stoploss -> trailing -> roi
        # since values are set cascading, the order is inverted
        # NOTE: using .astype(bool) converts nan to True
        if self.roi_enabled:
            roi_triggered = sell_vals[:, sell_cols["roi_triggered"]].astype(bool)
            where_roi = where(roi_triggered)[0]
            if len(where_roi):
                events_roi = sell_vals[roi_triggered]
                sell_reason[where_roi] = SellType.ROI
                for dst_col, src_col in zip(result_cols, trigger_cols):
                    sell_vals[where_roi, dst_col] = events_roi[:, src_col]
                # calc close rate from roi profit, using low (of the trigger candle) as the minimum rate
                roi_open_rate = buy_vals[where_roi, buy_cols["open"]]
                # cast as int since using as indexer
                roi_ofs = sell_vals[where_roi, sell_cols["trigger_ofs"]].astype(int)
                roi_low = ohlc_vals[roi_ofs, ohlc_cols["low"]]
                sell_vals[where_roi, sell_cols["close_rate"]] = calc_roi_close_rate(
                    roi_open_rate,
                    roi_low,
                    events_roi[:, sell_cols["roi_profit"]],
                    self.fee,
                )
        if self.trailing_enabled:
            trailing_triggered = sell_vals[:, sell_cols["trailing_triggered"]].astype(
                bool
            )
            where_trailing = where(trailing_triggered)[0]
            if len(where_trailing):
                events_trailing = sell_vals[trailing_triggered]
                sell_reason[where_trailing] = SellType.TRAILING_STOP_LOSS
                trailing_cols = trigger_cols + [sell_cols["trailing_rate"]]
                for dst_col, src_col in zip(result_cols, trailing_cols):
                    sell_vals[where_trailing, dst_col] = events_trailing[:, src_col]
        if self.stoploss_enabled:
            stoploss_triggered = sell_vals[:, sell_cols["stoploss_triggered"]].astype(
                bool
            )
            where_stoploss = (
                where(stoploss_triggered)
                if not self.trailing_enabled
                else where(stoploss_triggered & ~trailing_triggered)
            )[0]
            if len(where_stoploss):
                events_stoploss = sell_vals[where_stoploss, :]
                sell_reason[where_stoploss] = SellType.STOP_LOSS
                stoploss_cols = trigger_cols + [sell_cols["stoploss_rate"]]
                for dst_col, src_col in zip(result_cols, stoploss_cols):
                    sell_vals[where_stoploss, dst_col] = events_stoploss[:, src_col]

        open_rate = buy_vals[:, buy_cols["open"]]
        close_rate = sell_vals[:, sell_cols["close_rate"]]

        profits_abs, profits_prc = self._calc_profits(
            open_rate, close_rate, calc_abs=True
        )

        trade_duration = to_timedelta(
            Series(sell_vals[:, sell_cols["date"]] - buy_vals[:, buy_cols["date"]])
        )
        # replace trade duration of same candle trades with half the timeframe reduce to minutes
        trade_duration[trade_duration == self.td_zero].values[
            :
        ] = self.td_half_timeframe
        return DataFrame(
            {
                "pair": replace_values(
                    self.pairs_idx, self.pairs_name, buy_vals[:, buy_cols["pair"]]
                ),
                "profit_percent": profits_prc,
                "profit_abs": profits_abs,
                "open_date": to_datetime(buy_vals[:, buy_cols["date"]]),
                "close_date": to_datetime(sell_vals[:, sell_cols["date"]]),
                "open_index": buy_vals[:, buy_cols["ohlc"]].astype(int),
                "close_index": sell_vals[:, sell_cols["ohlc"]].astype(int),
                "trade_duration": trade_duration.dt.seconds / 60,
                "open_at_end": False,
                "open_rate": open_rate,
                "close_rate": close_rate,
                "sell_reason": sell_reason,
            }
        )

    def _calc_profits(
        self, open_rate: ndarray, close_rate: ndarray, dec=False, calc_abs=False
    ) -> ndarray:
        if dec:
            from decimal import Decimal

            sa, fee = Decimal(self.config["stake_amount"]), Decimal(self.fee)
            open_rate = array([Decimal(n) for n in open_rate], dtype=Decimal)
            close_rate = array([Decimal(n) for n in close_rate], dtype=Decimal)
        else:
            sa, fee = self.config["stake_amount"], self.fee
        profits_abs, profits_prc = self._calc_profits_np(
            sa, fee, open_rate, close_rate, calc_abs
        )
        if dec:
            profits_abs = profits_abs.astype(float) if profits_abs else None
            profits_prc = profits_prc.astype(float)
        if calc_abs:
            return profits_abs.round(8), profits_prc.round(8)
        else:
            return profits_prc.round(8)

    def _calc_profits_np(self, sa, fee, open_rate, close_rate, calc_abs) -> Tuple:
        am = sa / open_rate
        open_amount = am * open_rate
        close_amount = am * close_rate
        open_price = open_amount + open_amount * fee
        close_price = close_amount - close_amount * fee
        profits_prc = (close_price / open_price - 1).round(8)
        profits_abs = (close_price - open_price).round(8) if calc_abs else None
        return profits_abs, profits_prc

    def shifted_offsets(self, ofs: ndarray, period: int):
        s = sign(period)
        indexes = []
        if ofs is None:
            ofs = self.pairs_ofs_end
        elif len(ofs) < 1:
            return []
        for i in range(0, period, s):
            indexes.extend(ofs + i)
        return indexes

    def _shift_paw(
        self,
        data: ndarray,
        period=1,
        fill_v=nan,
        null_v=nan,
        diff_arr: Union[None, ndarray] = None,
    ) -> Union[DataFrame, Series]:
        """ pair aware shifting nulls rows that cross over the next pair data"""

        ofs = None if diff_arr is None or len(diff_arr) < 1 else diff_indexes(diff_arr)
        shifted = shift(data, period, fill=fill_v)
        shifted[self.shifted_offsets(ofs, period)] = null_v
        return shifted

    def advise_pair_df(self, df: DataFrame, pair: str, n_pair: float) -> DataFrame:
        """ Execute strategy signals and return df for given pair """
        meta = {"pair": pair}
        try:
            df["buy"].values[:] = 0
            df["sell"].values[:] = 0
            df["pair"].values[:] = pair
        # ignore if cols are not present
        except KeyError:
            df["buy"] = 0
            df["sell"] = 0
            df["pair"] = n_pair

        df = self.strategy.advise_buy(df, meta)
        df = self.strategy.advise_sell(df, meta)
        # strategy might be evil and nan set some  buy/sell rows
        # df.fillna({"buy": 0, "sell": 0}, inplace=True)
        # cast date as int to prevent time conversion when accessing values
        df["date"] = df["date"].values.astype(float)
        # only return required cols
        return df.iloc[:, where(union_eq(df.columns.values, MERGE_COLS))[0]]

    @staticmethod
    def _get_multi_index(pairs: list, idx: ndarray) -> MultiIndex:
        # if a list of [idx, pairs] is passed to from_product , the df would infer
        # the counter as the columns, when we want it as the rows, so we have to pass
        # a swapped mi to the df, there surely is a better way for this...
        return MultiIndex.from_product([pairs, idx], names=["pair", "ohlc"]).swaplevel(
            0, 1
        )

    def post_process(self, df_vals: ndarray, ofs=None):
        """
        Calculate estimates like spread and liquidity
        """
        loc = self.df_loc
        ofs = self.pairs_offset
        wnd = self.timeframe_wnd

        df_vals = add_columns(df_vals, loc, ["high_low", "spread", "illiq", "liq"])
        high = df_vals[:, loc["high"]]
        low = df_vals[:, loc["low"]]
        open = df_vals[:, loc["open"]]
        close = df_vals[:, loc["close"]]
        volume = df_vals[:, loc["volume"]]

        # spread is removed from profits
        if self.config.get("backtest_subtract_spread", True):
            df_vals[:, loc["spread"]] = calc_spread(high, low, close, ofs)
        # high low order determines open/close rate interpolation
        if self.config.get("backtest_skew_rates_by_volume", True):
            df_vals[:, loc["high_low"]] = sim_high_low(close, open)
            # min max liquidity statistics over a rolling window to use for interpolation
            df_vals[:, loc["liq"]] = rolling_norm(
                calc_liquidity(volume, close, high, low), wnd, ofs
            )
            df_vals[:, loc["illiq"]] = rolling_norm(
                calc_illiquidity(close, volume, wnd, ofs), wnd, ofs
            )
        return df_vals

    def merge_pairs_df(self, processed: Dict[str, DataFrame]) -> DataFrame:
        """ join all the pairs data into one concatenate df adding needed columns """
        advised = {}
        data = []
        max_len = 0
        pairs_end = []
        nan_data_pairs = []
        pairs = {}

        # get the df with the longest ohlc data since all the pairs will be padded to it
        # use float
        pair_counter = 0.0
        for pair, df in processed.items():
            # make sure to copy the df to not clobber the source data since it is accessed globally
            advised[pair] = self.advise_pair_df(df.copy(), pair, pair_counter)
            pairs[pair] = pair_counter
            pair_counter += 1
        self.pairs = pairs
        self.pairs_name = array(list(pairs.keys()))
        self.pairs_idx = array(list(pairs.values()), dtype=float)
        # the index shouldn't change after the advise call, so we can take the pre-advised index
        # to create the multiindex where each pair is indexed with max len
        df = concat(advised.values(), copy=False)
        # set startup offset from the first index (should be equal for all pairs)
        self.startup_offset = df.index[0]
        # add a column for pairs offsets to make the index unique
        offsets_arr, self.pairs_offset = self._calc_pairs_offsets(df, return_ofs=True)
        self.pairs_ofs_end = append(self.pairs_offset[1:] - 1, len(df) - 1)
        # loop over the missing data pairs and calculate the point where data ends
        # plus the absolute offset
        df["ofs"] = offsets_arr
        # could as easily be arange(len(df)) ...
        df["ohlc_ofs"] = df.index.values + offsets_arr - self.startup_offset
        df["ohlc"] = df.index.values
        # fill missing ohlc with open value index wise
        # df[isnan(df["low"].values), "low"] = df["open"]
        # df[isnan(df["high"].values), "high"] = df["open"]
        # df[isnan(df["close"].values), "close"] = df["open"]
        df.set_index("ohlc_ofs", inplace=True, drop=False)
        return df

    def bought_or_sold(self, df: DataFrame) -> Tuple[DataFrame, bool]:
        """ Set bought_or_sold columns according to buy and sell signals """
        df_vals = df.values
        loc = df_cols(df)
        # NOTE: add commas when using tuples, or use lists
        df_vals = add_columns(df.values, loc, ("bought_or_sold",))
        # set bought candles
        # skip if no valid bought candles are found
        # df["bought_or_sold"] = self._shift_paw(
        #     (df["buy"] - df["sell"]).values,
        #     fill_v=Candle.NOOP,
        #     null_v=Candle.NOOP,
        #     diff_arr=df["pair"],
        # )
        df_vals[:, loc["bought_or_sold"]] = self._shift_paw(
            df_vals[:, loc["buy"]] - df_vals[:, loc["sell"]],
            fill_v=Candle.NOOP,
            null_v=Candle.NOOP,
            diff_arr=df_vals[:, loc["pair"]],
        )

        # set sold candles
        bos = df_vals[:, loc["bought_or_sold"]]
        bos[bos == 1] = Candle.BOUGHT
        bos[bos == -1] = Candle.SOLD
        # set END candles as the last non nan candle of each pair data
        df_vals[self.pairs_ofs_end, loc["bought_or_sold"]] = Candle.END
        self.df_loc = loc

        return (
            df_vals,
            len(df_vals[df_vals[:, loc["bought_or_sold"]] == Candle.BOUGHT]) < 1,
        )

    def boughts_to_sold(self, df_vals: ndarray) -> ndarray:
        """
        reduce df such that there are many bought interleaved by one sold candle
        NOTE: does not modify input df
        """
        loc = self.df_loc.copy()
        bts_vals = df_vals[
            union_eq(
                df_vals[:, loc["bought_or_sold"]],
                [Candle.BOUGHT, Candle.SOLD, Candle.END,],
            )
        ]
        bts_vals = bts_vals[
            # exclude duplicate sold
            ~(
                (bts_vals[:, loc["bought_or_sold"]] == Candle.SOLD)
                & (
                    (
                        self._shift_paw(
                            (bts_vals[:, loc["bought_or_sold"]]),
                            fill_v=Candle.SOLD,
                            null_v=Candle.NOOP,
                            diff_arr=bts_vals[:, loc["pair"]],
                        )
                        == Candle.SOLD
                    )
                    # don't sell different pairs
                    | (bts_vals[:, loc["pair"]] != shift(bts_vals[:, loc["pair"]]))
                )
            )
        ]
        # add an index column
        bts_loc = loc
        bts_vals = add_columns(bts_vals, bts_loc, ("index",))
        bts_vals[:, bts_loc["index"]] = arange(len(bts_vals))
        self.bts_loc = bts_loc
        return bts_vals

    def _np_calc_sold_repeats(self, bts_vals: ndarray, sold: ndarray) -> ndarray:
        """ numpy version of the next_sold_ofs calculation """
        loc = self.bts_loc
        first_bought_idx = bts_vals[
            diff_indexes(bts_vals[:, loc["pair"]], with_start=True),
            # index calling is not needed because bts_df has the full index,
            # but keep it for clarity
            loc["index"],
        ]
        sold_idx = sold[:, loc["index"]]
        first_sold_loc = diff_indexes(sold[:, loc["pair"]], with_start=True)
        first_sold_idx = sold_idx[first_sold_loc]
        # the bulk of the repetitions, prepend an empty value
        sold_repeats = empty(len(sold_idx), dtype="int64")
        sold_repeats[1:] = sold_idx[1:] - sold_idx[:-1]
        # override the first repeats of each pair (will always override the value at idx 0)
        sold_repeats[first_sold_loc] = first_sold_idx - first_bought_idx + 1
        return sold_repeats

    def set_sold(self, df_vals: ndarray) -> DataFrame:
        # recompose the multi index swapping the ohlc count with a contiguous range
        bts_vals = self.boughts_to_sold(df_vals)
        loc = self.bts_loc
        # align sold to bought
        sold = bts_vals[
            union_eq(bts_vals[:, loc["bought_or_sold"]], [Candle.SOLD, Candle.END],)
        ]
        # if no sell sig is provided a limit on the trade duration could be applied..
        # if len(sold) < 1:
        # bts_df, sold = self.fill_stub_sold(df, bts_df)
        # calc the repetitions of each sell signal for each bought signal
        self.sold_repeats = self._np_calc_sold_repeats(bts_vals, sold)
        # NOTE: use the "ohlc_ofs" col with offsetted original indexes
        # for trigger , consider the last candle of each pair as a sell,
        # even thought the bought will be valid only if an amount condition is triggered
        bts_vals = add_columns(bts_vals, loc, ("next_sold_ofs",))
        # could also do this by setting by location (sold.index) and backfilling
        bts_vals[:, loc["next_sold_ofs"]] = repeat(
            sold[:, loc["ohlc_ofs"]], self.sold_repeats
        )
        return bts_vals, sold

    def set_triggers(self, df_vals: ndarray) -> ndarray:
        """
        returns the df of valid boughts where trigger happens, with matching trigger data
        points for each bought
        """
        bts_vals, sold = self.set_sold(df_vals)
        loc = self.bts_loc
        bought = bts_vals[bts_vals[:, loc["bought_or_sold"]] == Candle.BOUGHT]
        # get the index ranges of each bought->sold spans
        bought_ranges = (
            bought[:, loc["next_sold_ofs"]] - bought[:, loc["ohlc_ofs"]]
        ).astype("int64")
        # Use the first version only if the expanded array would take < ~500MB per col
        if self.use_v1 and bought_ranges.sum() < self.max_ranges_size:
            self.calc_type = True
            # intervals are short compute everything in one round
            bts_vals = self._v1_select_triggered_events(
                df_vals, bought, bought_ranges, bts_vals
            )
        else:
            # intervals are too long, jump over candles
            self.calc_type = False
            args = [df_vals, bought, bought_ranges, bts_vals]
            bts_vals = (
                self._v2_select_triggered_events(*args)
                if self.use_v2
                else self._nb_select_triggered_events(*args)
            )
        return bts_vals

    @staticmethod
    def _v2_compare_roi_triggers(
        cur_profits, roi_vals, roi_timeouts, trg_range, trg_roi_idx
    ):
        # reset values in case timeouts (parameters) changed
        # so that the arr doesn't hold old truths (since we are partially filling)
        trg_range[:, trg_roi_idx] = 0
        # NOTE: only compare each roi from the start of its timeout
        for n, rt in enumerate(roi_timeouts):
            trg_range[rt:, trg_roi_idx[n]] = cur_profits[rt:] >= roi_vals[n]

    @staticmethod
    def _v2_calc_stoploss_rate(
        open_rate, low_range, stoploss, trg_range, trg_col_stoploss_triggered,
    ):
        # calculate the rate from the bought candle
        stoploss_triggered_rate = open_rate * (1 - stoploss)
        trg_range[:, trg_col_stoploss_triggered] = low_range <= stoploss_triggered_rate
        return stoploss_triggered_rate

    @staticmethod
    def _v2_calc_trailing_rate(
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

    @staticmethod
    def _v2_first_flat_true(arr: ndarray):
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

    def _v2_vars(self, df_vals: ndarray, bought: ndarray, bought_ranges) -> Dict:
        v = {
            "roi_enabled": self.roi_enabled,
            "stoploss_enabled": self.stoploss_enabled,
            "trailing_enabled": self.trailing_enabled,
            "roi_or_trailing": self.roi_enabled or self.trailing_enabled,
            "stoploss_or_trailing": self.stoploss_enabled or self.trailing_enabled,
            "not_position_stacking": not self.position_stacking,
            "sl_positive": self.strategy.trailing_stop_positive or 0.0,
            "sl_positive_not_null": self.strategy.trailing_stop_positive is not None,
            "sl_offset": self.strategy.trailing_stop_positive_offset,
            "sl_only_offset": self.strategy.trailing_only_offset_is_reached,
            "stoploss": abs(self.strategy.stoploss),
            "stake_amount": self.config["stake_amount"],
            "fee": self.fee,
            # columns of the trigger array which stores all the calculations
            "col_names": ["trigger_ofs", "trigger_date", "trigger_bought_ofs",],
            # columns of the trg array which stores the calculation of each loop
            "trg_names": [],
            # the number of columns for the shape of the trigger range
            "trg_n_cols": 0,
            "bought_ranges": bought_ranges,
        }
        v["calc_offset"] = (v["sl_positive"] or v["sl_only_offset"],)

        # roi
        v["roi_cols_names"] = ("roi_profit", "roi_triggered")
        v["col_names"].extend(v["roi_cols_names"])
        self.roi_timeouts, self.roi_values = self._filter_roi()
        v["roi_timeouts"] = array(list(self.roi_timeouts.keys()), dtype="int32")
        # transpose roi values such that we can operate with profits ndarray
        v["n_timeouts"] = len(v["roi_timeouts"])
        v["roi_values"] = array(self.roi_values, dtype="float64").reshape(
            (v["n_timeouts"], 1)
        )
        v["roi_vals"] = v["roi_values"].reshape(v["roi_values"].shape[0])
        # nan indexes are relative, so can be pre calculated
        v["nan_early_idx"] = array(
            [
                v["n_timeouts"] * f + n
                for n, t in enumerate(v["roi_timeouts"])
                for f in range(t)
            ],
            dtype="int64",
        )
        v["trg_n_cols"] += v["n_timeouts"]

        # stoploss
        v["stoploss_cols_names"] = ("stoploss_rate", "stoploss_triggered")
        v["col_names"].extend(v["stoploss_cols_names"])
        v["trg_names"].append("stoploss_triggered")
        v["trg_n_cols"] += 1

        # tailing
        v["trailing_cols_names"] = ("trailing_rate", "trailing_triggered")
        v["col_names"].extend(v["trailing_cols_names"])
        v["trg_names"].append("trailing_triggered")
        v["trg_n_cols"] += 1

        # position stacking
        v["col_names"].append("last_trigger")
        v["last_trigger"] = -1

        bts_loc = self.bts_loc
        df_loc = self.df_loc
        v.update(
            {
                # make views of each column for faster indexing
                "bofs": bought[:, bts_loc["ohlc_ofs"]].astype("int64"),
                "bsold": bought[:, bts_loc["next_sold_ofs"]].astype("int64"),
                "bopen": bought[:, bts_loc["open"]],
                "ohlc_low": df_vals[:, df_loc["low"]],
                # reshaping __here__ is import for hinting broadcasting during loops
                "ohlc_high": df_vals[:, df_loc["high"]],
                "ohlc_date": df_vals[:, df_loc["date"]].astype("int64"),
                "b": 0,
                "end_ofs": int(df_vals[-1, df_loc["ohlc_ofs"]]),
            }
        )

        v["bought_ofs"] = v["bofs"][v["b"]]
        v["current_ofs"] = v["bought_ofs"]

        v["col"], _ = self._columns_indexes(v["col_names"], v["roi_timeouts"])
        v["trg_col"], v["trg_roi_cols"] = self._columns_indexes(
            v["trg_names"], v["roi_timeouts"], roi_as_array=True
        )
        v["trg_roi_idx"] = array([tp[1] for tp in v["trg_roi_cols"]])
        v["trg_roi_pos"] = len(v["trg_col"].__dict__)

        # NOTE: fill with zeros as some columns are booleans,
        # which would default to True if left empty or with nan
        triggers = zeros(shape=(len(bought), len(v["col_names"])))
        v["triggers"] = triggers
        # NOTE: this initializations are done because when splitting events we use ternary logic
        # to avoid edge cases where 0 would be considered an index
        # we check against -1 for last_trigger so initialize it to -1
        col = v["col"]
        if v["not_position_stacking"]:
            triggers[:, col.last_trigger] = -1
            v["col_last_trigger"] = triggers[:, col.last_trigger]
        # we check against nan/infinite for trigger ofs, so initialize it with nan
        triggers[:, col.trigger_ofs] = nan

        v["col_trigger_bought_ofs"] = triggers[:, col.trigger_bought_ofs]
        v["col_trigger_date"] = triggers[:, col.trigger_date]
        v["col_trigger_ofs"] = triggers[:, col.trigger_ofs]

        # trailing
        v["col_trailing_rate"] = triggers[:, col.trailing_rate]
        v["idx_trailing_triggered"] = col.trailing_triggered
        v["col_trailing_triggered"] = triggers[:, col.trailing_triggered]

        # roi
        v["col_roi_profit"] = triggers[:, col.roi_profit]
        v["col_roi_triggered"] = triggers[:, col.roi_triggered]
        v["s_trg_roi_idx"] = set(v["trg_roi_idx"])

        # stoploss
        v["col_stoploss_rate"] = triggers[:, col.stoploss_rate]
        v["col_stoploss_triggered"] = triggers[:, col.stoploss_triggered]

        # user the same arr at every iteration by slicing a subset of the max
        v["trg_range_max"] = ndarray(shape=(int(bought_ranges.max()), v["trg_n_cols"]))
        # initialize the roi columns to False, because we only set past the timeout
        # (instead of nulling pre timeout at every iteration)
        v["trg_range_max"][:, v["trg_roi_idx"]] = False
        return v

    def _nb_vars(self, v: Dict) -> Tuple:
        # columns of the trigger array which stores all the calculations

        fl_dict = {k: v[k] for k in ("ohlc_low", "ohlc_high", "bopen", "roi_vals")}
        fl_dict.update(
            {f"col_{k}": v["triggers"][:, n] for k, n in v["col"].__dict__.items()}
        )
        update_tpdict(fl_dict.keys(), fl_dict.values(), Float64Cols)

        it_dict = {
            k: v[k]
            for k in ("bofs", "bsold", "ohlc_date", "bought_ranges", "trg_roi_idx")
        }
        update_tpdict(it_dict.keys(), it_dict.values(), Int64Cols)

        fl = {
            k: v[k]
            for k in ("fee", "stake_amount", "stoploss", "sl_positive", "sl_offset")
        }
        update_tpdict(fl.keys(), fl.values(), Float64Vals)

        it = {
            k: v[k]
            for k in (
                "n_timeouts",
                "trg_n_cols",
                "trg_roi_pos",
                "bought_ofs",
                "end_ofs",
            )
        }
        update_tpdict(it.keys(), it.values(), Int64Vals)

        bl = {
            k: v[k]
            for k in (
                "roi_enabled",
                "stoploss_enabled",
                "trailing_enabled",
                "sl_only_offset",
                "sl_positive_not_null",
                "roi_or_trailing",
                "stoploss_or_trailing",
                "calc_offset",
                "not_position_stacking",
            )
        }
        update_tpdict(bl.keys(), bl.values(), Flags)

        names = {
            k: array(v[k], dtype="20U")
            for k in (
                "col_names",
                "trg_names",
                "stoploss_cols_names",
                "trailing_cols_names",
            )
        }
        update_tpdict(names.keys(), names.values(), NamesLists)

        trg_col = {f"trg_col_{c}": n for c, n in v["trg_col"].__dict__.items()}
        update_tpdict(trg_col.keys(), trg_col.values(), Int64Vals)

        update_tpdict(("trg_roi_cols",), (v["trg_roi_cols"],), ColsMap)

        return (
            Float64Cols,
            Int64Cols,
            NamesLists,
            Float64Vals,
            Int64Vals,
            Flags,
            ColsMap,
            v["nan_early_idx"],
            v["roi_timeouts"],
            v["roi_values"],
            v["trg_range_max"],
        )

    def _v2_select_triggered_events(
        self, df: DataFrame, bought: ndarray, bought_ranges: ndarray, bts_vals: ndarray,
    ):
        v = self._v2_vars(df, bought, bought_ranges)
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
                self._v2_compare_roi_triggers(
                    cur_profits,
                    sn.roi_vals,
                    sn.roi_timeouts,
                    trg_range,
                    sn.trg_roi_idx,
                )

            if sn.stoploss_enabled:
                # calculate the rate from the bought candle
                stoploss_triggered_rate = self._v2_calc_stoploss_rate(
                    open_rate,
                    low_range,
                    sn.stoploss,
                    trg_range,
                    sn.trg_col.stoploss_triggered,
                )
            if sn.trailing_enabled:
                trailing_rate = self._v2_calc_trailing_rate(
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
            fft = self._v2_first_flat_true(trg_range)
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

    def _assign_triggers(
        self, df: DataFrame, bought: DataFrame, triggers: ndarray, col_names: List[str]
    ) -> DataFrame:
        # set the index to the offset and add the columns to set the stoploss
        # data points on the relevant boughts
        df.set_index("ohlc_ofs", inplace=True, drop=False)
        df = df.reindex(columns=[*df.columns, *col_names], copy=False)
        # loop over col names and assign column by column because the df doesn't have the same
        # order as the triggers ndarray
        mask = isin(df.index.values, bought["ohlc_ofs"].values)
        for n, c in enumerate(col_names):
            df[c].values[mask] = triggers[:, n]
        # fill non bought candles
        if not self.position_stacking:
            df["last_trigger"].fillna(-1, inplace=True)
        # fill bool-like columns with 0
        df.fillna({tt: 0.0 for tt in self.trigger_types}, inplace=True)
        return df

    def _assign_triggers_vals(
        self,
        bts_vals: ndarray,
        bought: ndarray,
        triggers: ndarray,
        col_names: List[str],
    ) -> ndarray:
        # set the index to the offset and add the columns to set the stoploss
        # data points on the relevant boughts
        loc = self.bts_loc
        bts_vals = add_columns(bts_vals, loc, col_names)
        col_idx = [loc[c] for c in col_names]
        bts_vals[:, col_idx] = nan

        # loop over col names and assign column by column because the df doesn't have the same
        # order as the triggers ndarray
        mask = isin(bts_vals[:, loc["ohlc_ofs"]], bought[:, loc["ohlc_ofs"]])
        # can't copy over to multiple columns in one assignment
        for n, c in enumerate(col_names):
            bts_vals[mask, loc[c]] = triggers[:, n]
        # fill non bought candles
        if not self.position_stacking:
            nan_to_num(bts_vals[:, loc["last_trigger"]], copy=False, nan=-1)
        # fill bool-like columns with 0
        for tt in self.trigger_types:
            bts_vals[:, loc[tt]] = nan_to_num(bts_vals[:, loc[tt]], copy=False, nan=0.0)
        return bts_vals

    def _nb_select_triggered_events(
        self,
        df_vals: ndarray,
        bought: ndarray,
        bought_ranges: ndarray,
        bts_vals: ndarray,
    ) -> DataFrame:
        v = self._v2_vars(df_vals, bought, bought_ranges)
        (
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
        ) = self._nb_vars(v)

        # NOTE: functions are cached based on signatures, signatures are based on types
        # to load the correct cache when enabling or disabling features, we have to pass
        # True or None, (not False) such that the type of the tuple changes and and the sig/cache does too
        feat_dict = {
            k: v[k] or None
            for k in (
                "roi_enabled",
                "stoploss_enabled",
                "trailing_enabled",
                "not_position_stacking",
            )
        }
        define_callbacks(feat_dict)
        feat = Features(**feat_dict)

        iter_triggers(
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
            feat,
        )
        return self._assign_triggers_vals(
            bts_vals, bought, v["triggers"], v["col_names"]
        )

    def _remove_pairs_offsets(self, df: DataFrame, cols: List):
        ofs_vals = df["ofs"].values.tolist()
        for c in cols:
            # use to list in case of category
            df[c] = df[c].values - ofs_vals + self.startup_offset

    def _calc_pairs_offsets(
        self, df: DataFrame, group="pair", return_ofs=False
    ) -> ndarray:
        # since pairs are concatenated orderly diffing on the previous rows
        # gives the offset of each pair data
        pairs_offset = diff_indexes(df[group].values, with_start=True)
        # add the last span at the end of the repeats, since we are working with starting offsets
        pairs_ofs_repeats = append(
            pairs_offset[1:] - pairs_offset[:-1], len(df) - pairs_offset[-1]
        )
        # each pair queried at pairs_offset indexes should be unique
        pairs_offset_arr = repeat(pairs_offset, pairs_ofs_repeats)
        if return_ofs:
            return pairs_offset_arr, pairs_offset
        else:
            return pairs_offset_arr - self.startup_offset

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
            df_vals[:, df_loc["high_rate"]] = df_vals[:, df_loc["high"]] * (
                1 - stoploss
            )
            if sl_positive:
                df_vals[:, df_loc["high_rate_positive"]] = df_vals[
                    :, df_loc["high"]
                ] * (1 - sl_positive)
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
                    trailing_offset_reached = (
                        ofs_cummax(data_ofs, cur_profits) >= sl_offset
                    )
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
                data[:, col.trigger_bought_ofs]
                != shift(data[:, col.trigger_bought_ofs]),
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
            return self._triggers_return_df(col, full((0, data.shape[1]), nan))

        data = concatenate(data_chunks, axis=0)
        return self._triggers_return_df(col, data)

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

    @staticmethod
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

    @staticmethod
    def _last_trigger_numba(df: DataFrame) -> List[int]:
        """ numba version of _last_trigger_apply """
        df["trigger_ofs"].fillna(-3, inplace=True)
        return for_trail_idx(
            df.index.values,
            df["trigger_ofs"].values.astype(int),
            df["next_sold_ofs"].values.astype(int),
        )

    @staticmethod
    def _last_trigger_numba_arr(arr: ndarray, loc: Dict) -> ndarray:
        """ numba version of _last_trigger_apply """
        np_fill(arr[:, loc["trigger_ofs"]], fill_value=-3, inplace=True)
        # arr[isnan(arr[:, loc["trigger_ofs"]]), loc["trigger_ofs"]] = -3
        return for_trail_idx(
            arr[:, loc["ohlc_ofs"]].astype(int),
            arr[:, loc["trigger_ofs"]].astype(int),
            arr[:, loc["next_sold_ofs"]].astype(int),
        )

    def _v1_select_triggered_events(
        self,
        df_vals: ndarray,
        bought: ndarray,
        bought_ranges: ndarray,
        bts_vals: ndarray,
    ) -> ndarray:

        # compute all the stoplosses for the buy signals and filter out clear invalids
        trg_vals, trg_loc = self._np_calc_triggers(df_vals, bought, bought_ranges)
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
            np_fill(
                bts_vals[:, bts_loc["trigger_bought_ofs"]], backfill=True, inplace=True
            )
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
            bts_vals[boughts, bts_loc["last_trigger"]] = self._last_trigger_numba_arr(
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

    def _columns_indexes(
        self, cols: List[str], roi_timeouts: List[int], roi_as_array=False
    ) -> Tuple:
        """ maps list of col names into namespace, roi uses its own container (dict/array) """
        roi_cols = {to: n_c for n_c, to in enumerate(roi_timeouts, len(cols))}
        return (
            SimpleNamespace(**{c: n_c for n_c, c in enumerate(cols)}),
            array(tuple(zip(roi_cols.keys(), roi_cols.values())))
            if roi_as_array
            else roi_cols,
        )

    def _trg_to_columns(self) -> SimpleNamespace:
        """ returns namespace of trigger ndarray columns mapping according to config """
        trg = {}
        n_c = 0
        if self.trailing_enabled:
            trg["trailing"] = n_c
            n_c += 1
        if self.stoploss_enabled:
            trg["stoploss"] = n_c
            n_c += 1
        if self.roi_enabled:
            trg["roi"] = [n for n in range(n_c, len(self.roi_timeouts) + n_c)]
        return SimpleNamespace(**trg)

    def _filter_roi(self) -> Tuple[Dict[int, int], List[float]]:
        # ensure roi dict is sorted in order to always overwrite
        # with the latest duplicate value when rounding to timeframes
        # NOTE: make sure to sort numerically

        minimal_roi = self.config["minimal_roi"]
        sorted_minimal_roi = {k: minimal_roi[k] for k in sorted(minimal_roi, key=int)}
        roi_timeouts = self._round_roi_timeouts(list(minimal_roi.keys()))
        roi_values = [
            v for k, v in minimal_roi.items() if int(k) in roi_timeouts.values()
        ]
        return roi_timeouts, roi_values

    def _round_roi_timeouts(self, timeouts: List[float]) -> Dict[int, int]:
        """ rounds the timeouts to timeframe count that includes them, when
        different timeouts are included in the same timeframe count, the latest is
        used """
        return dict(
            zip(
                floor(
                    [Timedelta(f"{t}min") / self.td_timeframe for t in timeouts]
                ).astype(int),
                array(timeouts).astype(int),
            ),
        )

    def split_events(self, bts_vals: ndarray) -> Tuple[ndarray, ndarray]:
        bts_loc = self.bts_loc
        if self.any_trigger:
            bts_ls_s1 = self._shift_paw(
                bts_vals[:, bts_loc["last_trigger"]],
                diff_arr=bts_vals[:, bts_loc["pair"]],
                fill_v=-1,
                null_v=-1,
            )
            events_buy = bts_vals[
                (bts_vals[:, bts_loc["bought_or_sold"]] == Candle.BOUGHT)
                & (
                    (
                        shift(bts_vals[:, bts_loc["bought_or_sold"]], fill=Candle.SOLD)
                        == Candle.SOLD
                    )
                    # last_trigger is only valid if == shift(1)
                    # if the previous candle is SOLD it is covered by the previous case
                    # this also covers the case the previous candle == Candle.END
                    | (bts_vals[:, bts_loc["last_trigger"]] != bts_ls_s1)
                    | (
                        bts_vals[:, bts_loc["pair"]]
                        != shift(bts_vals[:, bts_loc["pair"]])
                    )
                )
                # exclude the last boughts that are not stoploss and which next sold is
                # END sold candle
                & ~(
                    isnan(bts_vals[:, bts_loc["trigger_ofs"]])
                    & isin(bts_vals[:, bts_loc["next_sold_ofs"]], self.pairs_ofs_end)
                )
            ]
            events_sell = bts_vals[
                (
                    (bts_vals[:, bts_loc["bought_or_sold"]] == Candle.SOLD)
                    # select only sold candles that are not preceded by a trigger
                    & (bts_ls_s1 == -1)
                )
                # and stoplosses (all candles with notna trigger_ofs should be valid)
                | isfinite(bts_vals[:, bts_loc["trigger_ofs"]])
            ]

        else:
            events_buy = bts_vals[
                (bts_vals[:, bts_loc["bought_or_sold"]] == Candle.BOUGHT)
                & (
                    union_eq(
                        shift(bts_vals[:, bts_loc["bought_or_sold"]], fill=Candle.SOLD),
                        # check for END too otherwise the first bought of mid-pairs
                        # wouldn't be included
                        [Candle.SOLD, Candle.END],
                    )
                )
                # exclude the last boughts that are not stoploss and which next sold is
                # END sold candle
                & isin(
                    bts_vals[:, bts_loc["next_sold_ofs"]],
                    self.pairs_ofs_end,
                    invert=True,
                )
            ]
            events_sell = bts_vals[
                bts_vals[:, bts_loc["bought_or_sold"]] == Candle.SOLD
            ]

        self.bts_loc = bts_loc
        return events_buy, events_sell

    def split_events_stack(self, bts_vals: ndarray):
        """"""
        bts_loc = self.bts_loc
        if self.any_trigger:
            events_buy = bts_vals[
                (bts_vals[:, bts_loc["bought_or_sold"]] == Candle.BOUGHT)
                # exclude the last boughts that are not stoploss and which next sold is
                # END sold candle
                & ~(
                    (isnan(bts_vals[:, bts_loc["trigger_ofs"]]))
                    & isin(bts_vals[:, bts_loc["next_sold_ofs"]], self.pairs_ofs_end)
                )
            ]
            # compute the number of sell repetitions for non triggered boughts
            nso, sell_repeats = unique(
                events_buy[isnan(events_buy[:, bts_loc["trigger_ofs"]])][
                    :, bts_loc["next_sold_ofs"]
                ],
                return_counts=True,
            )
            # need to check for membership against the bought candles next_sold_ofs here because
            # some sold candles can be void if all the preceding bought candles
            # (after the previous sold) are triggered
            # (otherwise would just be an eq check == Candle.SOLD)
            events_sell = bts_vals[
                isin(bts_vals[:, bts_loc["ohlc_ofs"]], nso)
                | isfinite(bts_vals[:, bts_loc["trigger_ofs"]])
            ]
            events_sell_repeats = ones(len(events_sell), dtype=int)
            events_sell_repeats[
                isin(events_sell[:, bts_loc["ohlc_ofs"]], nso)
            ] = sell_repeats
            events_sell = repeat(events_sell, events_sell_repeats, axis=0)
        else:
            events_buy = bts_vals[
                (bts_vals[:, bts_loc["bought_or_sold"]] == Candle.BOUGHT)
                # exclude the last boughts that are not stoploss and which next sold is
                # END sold candle
                & isin(
                    bts_vals[:, bts_loc["next_sold_ofs"]],
                    self.pairs_ofs_end,
                    invert=True,
                )
            ]
            events_sell = bts_vals[
                bts_vals[:, bts_loc["bought_or_sold"]] == Candle.SOLD
            ]
            _, sold_repeats = unique(
                events_buy[:, bts_loc["next_sold_ofs"]], return_counts=True
            )
            events_sell = repeat(events_sell, sold_repeats, axis=0)
        self.bts_loc = bts_loc
        return (events_buy, events_sell)

    def vectorized_backtest(
        self, processed: Dict[str, DataFrame], **kwargs,
    ) -> DataFrame:
        """ NOTE: can't have default values as arguments since it is an overridden function
        """
        df = self.merge_pairs_df(processed)

        df_vals, empty = self.bought_or_sold(df)

        # date = df_vals[:, self.df_loc["date"]]
        # high = df_vals[:, self.df_loc["high"]]
        # low = df_vals[:, self.df_loc["low"]]
        # close = df_vals[:, self.df_loc["close"]]
        # open = df_vals[:, self.df_loc["open"]]
        # volume = df_vals[:, self.df_loc["volume"]]
        # print(volume[:100])

        if empty:  # if no bought signals
            return self.empty_results

        df_vals = self.post_process(df_vals, self.pairs_offset)

        if self.any_trigger:
            bts_vals = self.set_triggers(df_vals)
        else:
            bts_vals, _ = self.set_sold(df_vals)

        if len(bts_vals) < 1:
            return self.empty_results

        # if dbg:
        #     self.events = as_df(
        #         bts_vals, self.bts_loc, bts_vals[:, self.bts_loc["ohlc_ofs"]]
        #     )

        events_buy, events_sell = (
            self.split_events(bts_vals)
            if not self.position_stacking
            else self.split_events_stack(bts_vals)
        )

        if dbg:
            dbg.bts_loc = self.bts_loc
            dbg._validate_events(events_buy, events_sell)
        return self.get_results(events_buy, events_sell, df)
