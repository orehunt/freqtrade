import logging
from typing import Dict, List, Tuple, Union
from enum import IntEnum
from collections import namedtuple
from functools import reduce

from numba import types
from numba.typed import Dict as nb_Dict
from numpy import (
    dtype,
    repeat,
    ones,
    nan,
    concatenate,
    ndarray,
    recarray,
    array,
    where,
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
    sign,
    floor,
    roll,
    cumsum,
    arange,
    iinfo,
    int32,
    zeros,
    empty,
    flatnonzero,
    bincount,
    swapaxes,
    ravel,
    empty_like,
    argmax,
    argmin,
    searchsorted,
    argsort,
    reshape,
    interp,
    amax,
)
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
from freqtrade.exceptions import OperationalException
from freqtrade.strategy.interface import SellType


logger = logging.getLogger(__name__)


class Candle(IntEnum):
    NOOP = 0
    BOUGHT = 1
    SOLD = 3
    END = 7  # references the last candle of a pair
    # STOPLOSS = 12


class HyperoptBacktesting(Backtesting):

    empty_results = DataFrame.from_records([], columns=BacktestResult._fields)
    debug = True

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

    bts_df_cols = {}
    merge_cols = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "buy",
        "sell",
        "pair",
    ]
    trigger_types = []
    sold_cols = [
        "bought_or_sold",
        "trigger_ofs",
        "trigger_bought_ofs",
        "last_trigger",
        "next_sold_ofs",
        "pair",
    ]

    def __init__(self, config):
        if config.get("backtesting_engine") == "vectorized":
            self.backtest_stock = self.backtest
            if self.debug:
                self._debug_opts()
                self.backtest = self._wrap_backtest
            else:
                self.backtest = self.vectorized_backtest
            self.beacktesting_engine = "vectorized"
            self.td_timeframe = Timedelta(config["timeframe"])
            self.td_half_timeframe = self.td_timeframe / 2
        super().__init__(config)

        backtesting_amounts = self.config.get("backtesting_amounts", {})
        self.stoploss_enabled = backtesting_amounts.get("stoploss", False)
        self.trailing_enabled = backtesting_amounts.get(
            "trailing", False
        ) and self.config.get("trailing_stop", False)
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

        self.position_stacking = self.config.get("position_stacking", False)
        if self.config.get("max_open_trades", 0) > 0:
            logger.warn("Ignoring max open trades...")

    def get_results(
        self, events_buy: DataFrame, events_sell: DataFrame, ohlc: DataFrame
    ) -> DataFrame:
        buy_cols = df_cols(events_buy)
        sell_cols = buy_cols
        buy_vals = events_buy.values
        sell_vals = events_sell.values
        ohlc_vals = ohlc.values
        ohlc_cols = df_cols(ohlc)
        # choose sell rate depending on sell reason and set sell_reason
        # add new needed columns
        sell_vals = add_columns(sell_vals, sell_cols, ("close_rate", "trigger_ohlc"))
        # boolean masks can't query by slices and return a copy, so use where
        where_sold = where(sell_vals[:, sell_cols["bought_or_sold"]] == Candle.SOLD)[0]
        events_sold = sell_vals[where_sold, :]
        # use an external ndarray to store sell_type to avoid float conversion
        sell_reason = ndarray(shape=(sell_vals.shape[0]), dtype="object")

        sell_reason[where_sold] = SellType.SELL_SIGNAL
        sell_vals[where_sold, sell_cols["close_rate"]] = events_sold[
            :, sell_cols["open"]
        ]

        # adjust trigger_ofs to the startup offset, and the pairs offset
        # to match original ohlc index
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
            events_roi = sell_vals[roi_triggered]
            sell_reason[where_roi] = SellType.ROI
            for dst_col, src_col in zip(result_cols, trigger_cols):
                sell_vals[where_roi, dst_col] = events_roi[:, src_col]
            # calc close rate from roi profit, using low (of the trigger candle) as the minimum rate
            roi_open_rate = buy_vals[where_roi, buy_cols["open"]]
            # cast as int since using as indexer
            roi_ofs = sell_vals[where_roi, sell_cols["trigger_ofs"]].astype(int)
            roi_low = ohlc_vals[roi_ofs, ohlc_cols["low"]]
            sell_vals[where_roi, sell_cols["close_rate"]] = self._calc_roi_close_rate(
                roi_open_rate, roi_low, events_roi[:, sell_cols["roi_profit"]]
            )
        if self.trailing_enabled:
            trailing_triggered = sell_vals[:, sell_cols["trailing_triggered"]].astype(
                bool
            )
            where_trailing = where(trailing_triggered)[0]
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
                if self.config.get("backtesting_stop_over_trail", False)
                or not self.trailing_enabled
                else where(stoploss_triggered & ~trailing_triggered)
            )[0]
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
                "open_time": to_datetime(buy_vals[:, buy_cols["date"]]),
                "close_time": to_datetime(sell_vals[:, sell_cols["date"]]),
                "open_index": buy_vals[:, buy_cols["ohlc"]].astype(int),
                "close_index": sell_vals[:, sell_cols["ohlc"]].astype(int),
                "trade_duration": trade_duration.dt.seconds / 60,
                "open_at_end": False,
                "open_rate": open_rate,
                "close_rate": close_rate,
                "sell_reason": sell_reason,
            }
        )

    def _calc_close_rate(self, open_rate, profits):
        return -(open_rate * profits + open_rate * (1 + self.fee)) / (self.fee - 1)

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
        am = sa / open_rate
        open_amount = am * open_rate
        close_amount = am * close_rate
        open_price = open_amount + open_amount * fee
        close_price = close_amount - close_amount * fee
        profits_prc = close_price / open_price - 1
        if calc_abs:
            profits_abs = close_price - open_price
        if dec:
            profits_abs = profits_abs.astype(float).round(8) if profits_abs else None
            profits_prc = profits_prc.astype(float).round(8)
        if calc_abs:
            return profits_abs.round(8), profits_prc.round(8)
        else:
            return profits_prc.round(8)

    def shifted_offsets(self, ofs: ndarray, period: int):
        s = sign(period)
        indexes = []
        if ofs is None:
            ofs = self.pairs_ofs_end
        # if s > 0:
        #     ofs = ofs[:-1]
        # else:  # ignore s == 0...
        #     ofs = ofs[1:]
        for i in range(0, period, s):
            indexes.extend(ofs + i)
        return indexes

    def _shift_paw(
        self,
        data: Union[DataFrame, Series],
        period=1,
        fill_v=nan,
        null_v=nan,
        diff_arr: Union[None, Series] = None,
    ) -> Union[DataFrame, Series]:
        """ pair aware shifting nulls rows that cross over the next pair data"""
        if diff_arr is None:
            ofs = None
        else:
            ofs = self._diff_indexes(diff_arr.values)
        shifted = shift(data.values, period, fill=fill_v)
        shifted[self.shifted_offsets(ofs, period)] = null_v
        return shifted

    @staticmethod
    def _diff_indexes(arr: ndarray, with_start=False, with_end=False) -> ndarray:
        """ returns the indexes where consecutive values are not equal,
        used for finding pairs ends """
        if with_start:
            if with_end:
                raise OperationalException("with_start and with_end are exclusive")
            return where(arr != shift(arr))[0]
        elif with_end:
            if with_start:
                raise OperationalException("with_end and with_start are exclusive")
            return where(arr != shift(arr, -1))[0]
        else:
            return where(arr != shift(arr, fill=arr[0]))[0]

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
        return df.iloc[:, where(union_eq(df.columns.values, self.merge_cols))[0]]

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
        # set bought candles
        # skip if no valid bought candles are found
        df["bought_or_sold"] = self._shift_paw(
            df["buy"] - df["sell"],
            fill_v=Candle.NOOP,
            null_v=Candle.NOOP,
            diff_arr=df["pair"],
        )

        # set sold candles
        bos = df["bought_or_sold"].values
        bos[bos == 1] = Candle.BOUGHT
        bos[bos == -1] = Candle.SOLD
        # set END candles as the last non nan candle of each pair data
        df["bought_or_sold"].values[self.pairs_ofs_end] = Candle.END
        return df, len(df.loc[df["bought_or_sold"].values == Candle.BOUGHT]) < 1

    def boughts_to_sold(self, df: DataFrame) -> DataFrame:
        """
        reduce df such that there are many bought interleaved by one sold candle
        NOTE: does not modify input df
        """
        bos_df = df[
            union_eq(
                df["bought_or_sold"].values, [Candle.BOUGHT, Candle.SOLD, Candle.END,],
            )
        ]
        bos_df = bos_df.loc[
            # exclude duplicate sold
            ~(
                (bos_df["bought_or_sold"].values == Candle.SOLD)
                & (
                    (
                        self._shift_paw(
                            bos_df["bought_or_sold"],
                            fill_v=Candle.SOLD,
                            null_v=Candle.NOOP,
                            diff_arr=bos_df["pair"],
                        )
                        == Candle.SOLD
                    )
                    # don't sell different pairs
                    | (bos_df["pair"].values != bos_df["pair"].shift().values)
                )
            )
        ]
        bos_df.reset_index(inplace=True, drop=True)
        return bos_df

    def _np_calc_sold_repeats(self, bts_df: DataFrame, sold: DataFrame) -> list:
        """ numpy version of the next_sold_ofs calculation """
        first_bought_idx = bts_df.iloc[
            self._diff_indexes(bts_df["pair"].values, with_start=True),
            # index calling is not needed because bts_df has the full index,
            # but keep it for clarity
        ].index.values
        sold_idx = sold.index.values
        first_sold_loc = self._diff_indexes(sold["pair"].values, with_start=True)
        first_sold_idx = sold_idx[first_sold_loc]
        # the bulk of the repetitions, prepend an empty value
        sold_repeats = concatenate([[0], sold_idx[1:] - sold_idx[:-1]])
        # override the first repeats of each pair (will always override the value at idx 0)
        sold_repeats[first_sold_loc] = first_sold_idx - first_bought_idx + 1
        return sold_repeats

    def set_sold(self, df: DataFrame) -> DataFrame:
        # recompose the multi index swapping the ohlc count with a contiguous range
        bts_df = self.boughts_to_sold(df)
        # align sold to bought
        sold = bts_df.loc[
            union_eq(bts_df["bought_or_sold"].values, [Candle.SOLD, Candle.END],)
        ]
        # if no sell sig is provided a limit on the trade duration could be applied..
        # if len(sold) < 1:
        # bts_df, sold = self.fill_stub_sold(df, bts_df)
        # calc the repetitions of each sell signal for each bought signal
        self.sold_repeats = self._np_calc_sold_repeats(bts_df, sold)
        # NOTE: use the "ohlc_ofs" col with offsetted original indexes
        # for stoploss calculation, consider the last candle of each pair as a sell,
        # even thought the bought will be valid only if an amount condition is triggered
        bts_df["next_sold_ofs"] = repeat(sold["ohlc_ofs"].values, self.sold_repeats)
        # could also do this by setting by location (sold.index) and backfilling, but this is faster
        return bts_df, sold

    def set_triggers(self, df: DataFrame) -> DataFrame:
        """
        returns the df of valid boughts where stoploss triggered, with matching stoploss
        index of each bought
        """
        bts_df, sold = self.set_sold(df)
        bought = bts_df[bts_df["bought_or_sold"].values == Candle.BOUGHT]
        # get the index ranges of each bought->sold spans
        bought_ranges = bought["next_sold_ofs"].values - bought["ohlc_ofs"].values
        # Use the first version only if the expanded array would take < ~500MB per col
        # if bought_ranges.sum() < 10e6:
        if False:
            self.calc_type = True
            # intervals are short compute everything in one round
            bts_df = self._v1_select_triggered_events(df, bought, bought_ranges, bts_df)
        else:
            # intervals are too long, jump over candles
            self.calc_type = False
            args = [df, bought, bought_ranges, bts_df]
            bts_df = (
                # self._v2_select_triggered_events(*args)
                self._nb_select_triggered_events(*args)
                if not self.position_stacking
                else self._pd_2_select_triggered_events_stack(*args)
            )
        return bts_df

    def _v2_vars(self, df: DataFrame, bought: DataFrame) -> Tuple:
        df_dict = {col: df[col].values for col in ("low", "high", "date", "ohlc_ofs")}
        bought_dict = {
            col: bought[col].values for col in ("ohlc_ofs", "next_sold_ofs", "open")
        }
        # columns of the trigger array which stores all the calculations
        col_names = [
            "trigger_ofs",
            "trigger_date",
            "trigger_bought_ofs",
            "last_trigger",
        ]
        # columns of the trg array which stores the calculation of each loop
        trg_names = []
        # the number of columns for the shape of the trigger range
        trg_n_cols = 0
        if self.roi_enabled:
            roi_cols_names = ("roi_profit", "roi_triggered")
            col_names.extend(roi_cols_names)
            self.roi_timeouts, self.roi_values = self._filter_roi()
            roi_timeouts = array(list(self.roi_timeouts.keys()))
            # transpose roi values such that we can operate with profits ndarray
            n_timeouts = len(roi_timeouts)
            roi_values = array(self.roi_values).reshape(n_timeouts, 1)
            # nan indexes are relative, so can be pre calculated
            nan_early_idx = array(
                [
                    n_timeouts * f + n
                    for n, t in enumerate(roi_timeouts)
                    for f in range(t)
                ],
                dtype="int64",
            )
            trg_n_cols += n_timeouts
        else:
            roi_timeouts = array([])
            roi_values = array([])
            n_timeouts = 0
            nan_early_idx = array([], dtype=int)

        if self.stoploss_enabled:
            stoploss_cols_names = ("stoploss_rate", "stoploss_triggered")
            col_names.extend(stoploss_cols_names)
            trg_names.append("stoploss_triggered")
            trg_n_cols += 1

        if self.trailing_enabled:
            trailing_cols_names = ("trailing_rate", "trailing_triggered")
            col_names.extend(trailing_cols_names)
            trg_names.append("trailing_triggered")
            trg_n_cols += 1
        else:
            trailing_cols_names = []

        col, roi_cols = self._columns_indexes(
            col_names, roi_timeouts, roi_as_tuple=True, col_as_dict=True
        )
        trg_col, trg_roi_cols = self._columns_indexes(
            trg_names, roi_timeouts, roi_as_tuple=True, col_as_dict=True
        )

        fl_cols = Float64Cols(
            df_low=df["low"].values,
            df_high=df["high"].values,
            bought_open=bought["open"].values,
            roi_values=roi_values,
            roi_vals=roi_values.reshape(roi_values.shape[0]),
        )
        it_cols = Int64Cols(
            bought_ohlc_ofs=bought["ohlc_ofs"].values,
            bought_next_sold_ofs=bought["next_sold_ofs"].values,
            df_date=df["date"].values,
            df_ohlc_ofs=df["ohlc_ofs"].values,
        )
        fl = Float64Vals(
            fee=float(self.fee),
            stake_amount=float(self.config["stake_amount"]),
            stoploss=self.strategy.stoploss,
            sl_positive=(self.strategy.trailing_stop_positive or 0),
            sl_offset=self.strategy.trailing_stop_positive_offset,
        )
        it = Int32Vals(n_timeouts=n_timeouts, trg_n_cols=trg_n_cols)
        bl = Flags(
            roi_enabled=self.roi_enabled,
            stoploss_enabled=self.stoploss_enabled,
            trailing_enabled=self.trailing_enabled,
            sl_only_offset=self.strategy.trailing_only_offset_is_reached,
            stop_over_trail=self.config.get("backtesting_stop_over_trail", False)
        )
        col_names = tuple(col_names)
        trg_names = tuple(trg_names)
        stoploss_cols_names = tuple(stoploss_cols_names)
        trailing_cols_names = tuple(trailing_cols_names)
        names = CharLists(
            col_names=col_names,
            trg_names=trg_names,
            stoploss_cols_names=stoploss_cols_names,
            trailing_cols_names=trailing_cols_names,
        )

        return (
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
        )

    def _pd_2_select_triggered_events_stack(
        self,
        df: DataFrame,
        bought: DataFrame,
        bought_ranges: ndarray,
        sold: DataFrame,
        bts_df: DataFrame,
    ):
        """ loop version of stoploss selection for position stacking, simply loops
        over all the bought candles of the bts dataframe """
        stoploss_index = []
        stoploss_rate = []
        trigger_date = []
        bought_trigger_ofs = []
        # copy cols for faster index accessing
        bofs = bought["ohlc_ofs"].values
        bopen = bought["open"].values
        b = 0
        trigger_bought_ofs = bofs[b]

        ohlc_low = df["low"].values
        ohlc_ofs = df["ohlc_ofs"].values
        ohlc_date = df["date"].values
        ofs_start = 0
        ohlc_idx = df.index.get_level_values(0)
        end_ofs = ohlc_ofs[-1]

        while trigger_bought_ofs < end_ofs:
            # calculate the rate from the bought candle
            stoploss_triggered_rate = bopen[b] * (1 - stoploss)
            # check trigger for the range of the current bought
            ofs_start += ohlc_ofs[ofs_start:].searchsorted(trigger_bought_ofs, "left")
            stoploss_triggered = (
                ohlc_low[ofs_start : ofs_start + bought_ranges[b]]
                <= stoploss_triggered_rate
            )
            # get the position where stoploss triggered relative to the current bought slice
            stop_max_idx = stoploss_triggered.argmax()
            # check that the index returned by argmax is True
            if stoploss_triggered[stop_max_idx]:
                # set the offset of the triggered stoploss index
                current_ofs = trigger_bought_ofs + stop_max_idx
                stoploss_index.append(ohlc_idx[current_ofs])
                trigger_date.append(ohlc_date[current_ofs])
                stoploss_rate.append(stoploss_triggered_rate)
                bought_trigger_ofs.append(trigger_bought_ofs)
            try:
                b += 1
                trigger_bought_ofs = bofs[b]
            except IndexError:
                break
        # set the index to the offset and add the columns to set the stoploss
        # data points on the relevant boughts
        bts_df.set_index("ohlc_ofs", inplace=True)
        stoploss_cols = ["trigger_ofs", "stoploss_rate", "trigger_date"]
        bts_df.assign(**{c: nan for c in stoploss_cols})
        bts_df = bts_df.reindex(columns=[*bts_df.columns, *stoploss_cols], copy=False)
        bts_df.loc[bought_trigger_ofs, stoploss_cols] = [
            [stoploss_index],
            [stoploss_rate],
            [trigger_date],
        ]
        return bts_df

    def _v2_select_triggered_events(
        self,
        df: DataFrame,
        bought: DataFrame,
        bought_ranges: ndarray,
        bts_df: DataFrame,
    ):
        roi_enabled = self.roi_enabled
        stoploss_enabled = self.stoploss_enabled
        trailing_enabled = self.trailing_enabled
        roi_or_trailing = roi_enabled or trailing_enabled
        stoploss_or_trailing = stoploss_enabled or trailing_enabled
        not_stop_over_trail = not self.config.get("backtesting_stop_over_trail", False)

        sl_positive = self.strategy.trailing_stop_positive
        sl_offset = self.strategy.trailing_stop_positive_offset
        sl_only_offset = self.strategy.trailing_only_offset_is_reached
        stoploss = abs(self.strategy.stoploss)
        calc_offset = sl_positive or sl_only_offset
        stake_amount = self.config["stake_amount"]
        fee = self.fee

        # columns of the trigger array which stores all the calculations
        col_names = [
            "trigger_ofs",
            "trigger_date",
            "trigger_bought_ofs",
            "last_trigger",
        ]
        # columns of the trg array which stores the calculation of each loop
        trg_names = []
        # the number of columns for the shape of the trigger range
        trg_n_cols = 0

        if roi_enabled:
            roi_cols_names = ("roi_profit", "roi_triggered")
            col_names.extend(roi_cols_names)
            self.roi_timeouts, self.roi_values = self._filter_roi()
            roi_timeouts = array(list(self.roi_timeouts.keys()))
            # transpose roi values such that we can operate with profits ndarray
            n_timeouts = len(roi_timeouts)
            roi_values = array(self.roi_values).reshape(n_timeouts, 1)
            # nan indexes are relative, so can be pre calculated
            nan_early_idx = array(
                [
                    n_timeouts * f + n
                    for n, t in enumerate(roi_timeouts)
                    for f in range(t)
                ],
                dtype="int64",
            )
            trg_n_cols += n_timeouts
        else:
            roi_timeouts = []

        if stoploss_enabled:
            stoploss_cols_names = ("stoploss_rate", "stoploss_triggered")
            col_names.extend(stoploss_cols_names)
            trg_names.append("stoploss_triggered")
            trg_n_cols += 1

        if trailing_enabled:
            trailing_cols_names = ("trailing_rate", "trailing_triggered")
            col_names.extend(trailing_cols_names)
            trg_names.append("trailing_triggered")
            trg_n_cols += 1

        # make views of each column for faster indexing
        bofs = bought["ohlc_ofs"].values
        bsold = bought["next_sold_ofs"].values
        bopen = bought["open"].values

        ohlc_low = df["low"].values
        ohlc_high = df["high"].values
        ohlc_date = df["date"].values

        b = 0
        last_trigger = -1
        bought_ofs = bofs[b]
        current_ofs = bought_ofs
        end_ofs = df["ohlc_ofs"].values[-1]

        col, roi_cols = self._columns_indexes(col_names, roi_timeouts)
        col_dict = col._asdict()
        trg_col, trg_roi_cols = self._columns_indexes(trg_names, roi_timeouts)
        roi_idx = tuple(roi_cols.values())
        trg_roi_idx = tuple(trg_roi_cols.values())
        trg_roi_pos = len(trg_col)
        trg_idx = tuple(arange(trg_n_cols))

        # NOTE: fill with zeros as some columns are booleans,
        # which would default to True if left empty or with nan
        triggers = zeros(shape=(len(bought), len(col_names)))
        # NOTE: this initializations are done because when splitting events we use ternary logic
        # to avoid edge cases where 0 would be considered an index
        # we check against -1 for last_trigger so initialize it to -1
        triggers[:, col.last_trigger] = -1
        # we check against nan/infinite for trigger ofs, so initialize it with nan
        triggers[:, col.trigger_ofs] = nan

        # NOTE: replacing col. lookups would be too verbose, speed up is <0.1x
        col_last_trigger = triggers[:, col.last_trigger]
        col_trigger_bought_ofs = triggers[:, col.trigger_bought_ofs]
        col_trigger_date = triggers[:, col.trigger_date]
        col_trigger_ofs = triggers[:, col.trigger_ofs]

        if trailing_enabled:
            col_trailing_rate = triggers[:, col.trailing_rate]
            idx_trailing_triggered = col.trailing_triggered
            col_trailing_triggered = triggers[:, idx_trailing_triggered]
        if roi_enabled:
            col_roi_profit = triggers[:, col.roi_profit]
            col_roi_triggered = triggers[:, col.roi_triggered]
            s_trg_roi_idx = set(trg_roi_idx)
        else:
            s_trg_roi_idx = set([])
        if stoploss_enabled:
            col_stoploss_rate = triggers[:, col.stoploss_rate]
            col_stoploss_triggered = triggers[:, col.stoploss_triggered]

        # user the same arr at every iteration by slicing a subset of the max
        trg_range_max = ndarray(shape=(bought_ranges.max(), trg_n_cols))

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
                # relative to (flattened) roi triggers only (flatten is faster than ravel here)
                roi_triggers = (cur_profits >= roi_values).swapaxes(0, 1).flatten()
                # NOTE: clip nan_early_idx to the length of the bought_range
                # NOTE: use False, not nan, since bool(nan) == True
                roi_triggers[nan_early_idx[nan_early_idx <= br * n_timeouts]] = False
                roi_triggers.shape = (br, n_timeouts)
                trg_range[:, trg_roi_idx] = roi_triggers
            if stoploss_enabled:
                # calculate the rate from the bought candle
                stoploss_triggered_rate = bopen[b] * (1 - stoploss)
                trg_range[:, trg_col.stoploss_triggered] = (
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
                    trailing_rate[~trailing_offset_reached] = nan
                trg_range[:, trg_col.trailing_triggered] = low_range <= trailing_rate
            # apply argmax over axis 0, such that we get the first timeframe
            # where a trigger happened (argmax of each column)
            trg_first_idx = rowargmax(trg_range)
            # filter out columns that have no true trigger
            # NOTE: the list is very small here (<10) so it might make sense to use python
            # but the big speed up seen here does not match outside testing of same lists lengths..
            # valid_cols = flatnonzero(trg_range[trg_first_idx, trg_idx])
            valid_cols = [
                i for i, val in enumerate(trg_range[trg_first_idx, trg_idx]) if val != 0
            ]
            # check that there is at least one valid trigger
            if len(valid_cols):
                # get the column index that triggered first row wise
                # NOTE: here we get all the first triggers that happened on the same candle
                trg_first = where(trg_first_idx[valid_cols].min() == trg_first_idx)[0]
                trg_top = trg_first[0]
                # lastly get the trigger offset from the index of the first valid column
                # any trg_first index would return the same value here
                trg_ofs = trg_first_idx[trg_top]
                # check what trigger it is and copy related columns values
                # from left they are ordered stoploss, trailing, roi
                if (
                    trailing_enabled
                    and trg_top == trg_col.trailing_triggered
                    and not_stop_over_trail
                ):
                    col_trailing_triggered[b] = True
                    col_trailing_rate[b] = trailing_rate[trg_ofs]
                elif stoploss_enabled and trg_top == trg_col.stoploss_triggered:
                    col_stoploss_triggered[b] = True
                    col_stoploss_rate[b] = stoploss_triggered_rate
                elif roi_enabled and trg_top in trg_roi_idx:
                    col_roi_triggered[b] = True
                    # NOTE: scale trg_first by how many preceding columns (stoploss,trailing)
                    # there are before roi columns, in order to get the offset
                    # relative to only the (ordered) roi columns
                    # and pick the minimum value from the right (roi_values have to be pre ordered desc)
                    col_roi_profit[b] = roi_values[trg_first[-1] - trg_roi_pos]
                # trigger ofs is relative to the bought range, so just add it to the bought offset
                current_ofs = bought_ofs + trg_ofs
                # copy general trigger values shared by all trigger types
                # if trigg:
                col_trigger_ofs[b] = current_ofs
                col_trigger_date[b] = ohlc_date[current_ofs]
                col_trigger_bought_ofs[b] = bought_ofs
                try:
                    last_trigger = b
                    # get the first row where the bought index is
                    # higher than the current stoploss index
                    b += bofs[b:].searchsorted(current_ofs, "right")
                    # repeat the trigger index for the boughts in between the trigger
                    # and the bought with higher idx
                    col_last_trigger[last_trigger:b] = current_ofs
                    bought_ofs = bofs[b]
                except IndexError:
                    break
            else:  # if no triggers executed, jump to the first bought after next sold idx
                try:
                    b += bofs[b:].searchsorted(bsold[b], "right")
                    bought_ofs = bofs[b]
                except IndexError:
                    break
        return self._assign_triggers(bts_df, bought, triggers, col_names)

    def _assign_triggers(
        self, df: DataFrame, bought: DataFrame, triggers: ndarray, col_names: List[str]
    ) -> DataFrame:
        # set the index to the offset and add the columns to set the stoploss
        # data points on the relevant boughts
        df.set_index("ohlc_ofs", inplace=True)
        df = df.reindex(columns=[*df.columns, *col_names], copy=False)
        # loop over col names and assign column by column because the df doesn't have the same
        # order as the triggers ndarray
        mask = isin(df.index.values, bought["ohlc_ofs"].values)
        for n, c in enumerate(col_names):
            df[c].values[mask] = triggers[:, n]
        # fill non bought candles
        df["last_trigger"].fillna(-1, inplace=True)
        # fill bool-like columns with 0
        df.fillna({tt: 0.0 for tt in self.trigger_types}, inplace=True)
        return df

    def _nb_select_triggered_events(
        self, df, bought, bought_ranges, bts_df
    ) -> DataFrame:
        (
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
        ) = self._v2_vars(df, bought)

        triggers = select_triggers(
            # triggers = iter_triggers(
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
        )
        # print(self._assign_triggers(bts_df, bought, triggers, col_names).iloc[:10][["stoploss_triggered", "roi_triggered", "trailing_triggered", "stoploss_rate", "roi_profit", "trailing_rate", "last_trigger"]])
        # exit()
        return self._assign_triggers(bts_df, bought, triggers, col_names)

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
        pairs_offset = self._diff_indexes(df[group].values, with_start=True)
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
        self, df: DataFrame, bought: DataFrame, bought_ranges: ndarray,
    ) -> Tuple[ndarray, Tuple]:
        """ expand bought ranges into ohlc processed
         prefetch the columns of interest to avoid querying
         the index over the loop (avoid nd indexes) """
        df_cols = ["low", "high", "ohlc_ofs", "date"]
        # bought data rows will be repeated to match the bought_ranges
        bought_data = {"trigger_bought_ofs": bought["ohlc_ofs"].values}
        # post_cols are the empty columns that will be calculated after the expansion
        post_cols = []
        trigger_flags = []

        if self.roi_enabled:
            self.roi_col_names.extend(["roi_profit", "roi_triggered"])
            post_cols.extend(self.roi_col_names)
            # only the filtered roi couples will be used for calculation
            self.roi_timeouts, self.roi_values = self._filter_roi()
            roi_timeouts = list(self.roi_timeouts.keys())
        else:
            roi_timeouts = []

        if self.stoploss_enabled or self.trailing_enabled:
            stoploss = self.strategy.stoploss

        if self.stoploss_enabled:
            bought_data["stoploss_rate"] = self._calc_stoploss_rate(bought, stoploss)
            post_cols.append("stoploss_triggered")
            self.stoploss_col_names.extend(["stoploss_rate", "stoploss_triggered"])

        if self.trailing_enabled:
            sl_positive = self.strategy.trailing_stop_positive
            sl_offset = self.strategy.trailing_stop_positive_offset
            sl_only_offset = self.strategy.trailing_only_offset_is_reached

            # calculate both rates
            df["high_rate"] = self._calc_trailing_rate(df["high"].values, stoploss)
            df_cols.append("high_rate")
            if sl_positive:
                df["high_rate_positive"] = self._calc_trailing_rate(
                    df["high"].values, sl_positive
                )
                df_cols.append("high_rate_positive")

            self.trailing_col_names.extend(
                ["trailing_rate", "trailing_triggered",]
            )
            post_cols.extend(self.trailing_col_names)
        if self.roi_enabled or self.trailing_enabled:
            bought_data["bought_open"] = bought["open"].values

        df_cols_n = len(df_cols)
        # order matters
        col_names = df_cols + list(bought_data.keys()) + post_cols
        expd_cols_n = df_cols_n + len(bought_data)
        col, roi_cols = self._columns_indexes(col_names, roi_timeouts)
        tot_cols_n = expd_cols_n + len(post_cols)

        # get views of the data to expand
        ohlc_vals = df[df_cols].values
        bought_vals = array(list(bought_data.values())).swapaxes(0, 1)
        data_len = bought_ranges.sum()
        # the array storing the expanded data
        data = ndarray(shape=(data_len, tot_cols_n))

        # copy the source data ranges over to their expanded indexes
        data_ofs = ndarray(shape=(len(bought) + 1), dtype=int)
        data_ofs[0] = 0
        data_ofs[1:] = bought_ranges.cumsum()
        data_df = data[:, :df_cols_n]
        data_bought = data[:, df_cols_n:expd_cols_n]
        copy_ranges(
            bought["ohlc_ofs"].values,
            data_ofs,
            data_df,
            data_bought,
            ohlc_vals,
            bought_vals,
            bought_ranges,
        )

        if self.roi_enabled or self.trailing_enabled:
            cur_profits = self._calc_profits(
                data[:, col.bought_open], data[:, col.high]
            )
        if self.roi_enabled:
            # the starting offset is already computed with data_ofs
            # exclude the last value since we only nan from the start
            # of each bought range
            bought_starts = data_ofs[:-1]
            # roi triggers store the truth of roi matching for each roi timeout
            roi_triggers = []
            # setup expanded required roi profit columns ordered by roi timeouts
            roi_vals = array([full(data_len, prf) for prf in self.roi_values]).swapaxes(
                0, 1
            )
            # roi rates index null overrides based on timeouts
            # exclude indexes that exceed the maximum length of the expd array
            nan_early_roi_idx = {}
            for to in roi_timeouts:
                to_indexes = bought_starts + to
                nan_early_roi_idx[to] = to_indexes[to_indexes < data_len]
            for n, to in enumerate(roi_timeouts):
                # roi triggered if candle profit are above the required ratio
                roi_trg = cur_profits >= self.roi_values[n]
                # null rows preceding the timeouts
                # NOTE: strictly below, bool arrays can't be nan, use False
                for t in filter(lambda x: x < to, roi_timeouts):
                    roi_trg[nan_early_roi_idx[t]] = False
                roi_triggers.append(roi_trg)
            roi_triggers = array(roi_triggers).swapaxes(0, 1)
            # get the first True index where roi triggered column wise
            roi_trigger_max = (roi_triggers == True).argmax(axis=1)
            # Now get a 1d array from the Truth value of the roi trigger index
            # check again for truth since argmax does not discern 0 from None

            data[:, col.roi_triggered] = (
                roi_triggers[arange(data_len), roi_trigger_max] == True
            )
            # and a 1d array of the roi rate from the roi trigger index
            # get a view of the roi rates to align roi_trigger_max indexes
            # (otherwise would have to offset it)
            data[:, col.roi_profit] = roi_vals[arange(data_len), roi_trigger_max]
            # at this point valid roi should be roi_profit[roi_triggered]
            trigger_flags.append(data[:, col.roi_triggered].astype(bool))

        if self.trailing_enabled:
            data[:, col.trailing_rate] = ofs_cummax(data_ofs, data[:, col.high_rate])
            if sl_positive or sl_only_offset:
                trailing_offset_reached = ofs_cummax(data_ofs, cur_profits) >= sl_offset
            if sl_positive:
                # and set it only where the offset is reached (offset can't be None)
                data[trailing_offset_reached, col.trailing_rate] = ofs_cummax(
                    data_ofs, data[:, col.high_rate_positive]
                )[trailing_offset_reached]
            if sl_only_offset:
                # if trailing only with offset, nan trailing rates where offset is not reached
                # (static stoploss would have triggered in that case, doesn't need to be covered here)
                data[~trailing_offset_reached, col.trailing_rate] = nan
            # where trailing_rate is nan, it is False
            data[:, col.trailing_triggered] = (
                data[:, col.low] <= data[:, col.trailing_rate]
            )
            trigger_flags.append(data[:, col.trailing_triggered].astype(bool))
        if self.stoploss_enabled:
            data[:, col.stoploss_triggered] = (
                data[:, col.low] <= data[:, col.stoploss_rate]
            )
            trigger_flags.append(data[:, col.stoploss_triggered].astype(bool))

        # select only candles with trigger
        data = data[reduce(lambda x, y: x | y, trigger_flags), :]

        if len(data) < 1:
            # keep shape since return value is accessed without reference
            return self._triggers_return_df(col, full((0, data.shape[1]), nan))

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
        return self._triggers_return_df(col, data)

    def _triggers_return_df(self, col: Tuple, data: ndarray) -> DataFrame:
        col_dict = col._asdict()
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
        return DataFrame(
            data[:, list(col_map.values())], columns=list(col_map.keys()), copy=False
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

    def _v1_select_triggered_events(
        self,
        df: DataFrame,
        bought: DataFrame,
        bought_ranges: ndarray,
        bts_df: DataFrame,
    ) -> DataFrame:

        # compute all the stoplosses for the buy signals and filter out clear invalids
        trigger = self._np_calc_triggers(df, bought, bought_ranges)

        # align original index
        if not self.position_stacking:
            # exclude overlapping boughts
            # --> | BUY1 | BUY2..STOP2 | STOP1 | -->
            # -->      V    X      X       V     -->
            # preset indexes to merge (on indexes directly) without sorting
            bts_df.set_index("ohlc_ofs", drop=True, inplace=True)
            trigger.set_index("trigger_bought_ofs", inplace=True, drop=False)
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
            bts_df = trigger.join(bts_df, how="right")
            # drop bought_ofs because is outer merged afterwards
            bts_df.drop(columns="trigger_bought_ofs", inplace=True)

            # now add the trigger new rows with an outer join
            trigger.set_index("trigger_ofs", inplace=True, drop=False)
            bts_df = bts_df.merge(
                trigger[outer_cols],
                how="outer",
                left_index=True,
                right_index=True,
                suffixes=("", "_b"),
                copy=False,
            )
            # both trigger_ofs_b and trigger_bought_ofs are on the correct
            # index where the stoploss is triggered, and backfill backwards
            bts_df["trigger_ofs_b"].fillna(method="backfill", inplace=True)
            bts_df["trigger_bought_ofs"].fillna(method="backfill", inplace=True)
            # this is a filter to remove obvious invalids
            # self.start_pyinst()
            bts_df = bts_df[
                ~(
                    (bts_df["bought_or_sold"].values == Candle.BOUGHT)
                    & (bts_df["bought_or_sold"].shift().values == Candle.BOUGHT)
                    & (
                        (
                            (
                                bts_df["trigger_ofs_b"].values
                                == bts_df["trigger_ofs_b"].shift().values
                            )
                            | (
                                bts_df["trigger_bought_ofs"].values
                                == bts_df["trigger_bought_ofs"].shift().values
                            )
                        )
                    )
                )
            ]

            # loop over the filtered boughts to determine order of execution
            boughts = bts_df["bought_or_sold"].values == Candle.BOUGHT
            bts_df.loc[boughts, "last_trigger"] = array(
                self._last_trigger_numba(bts_df[boughts])
            )
            # fill the last_trigger column for non boughts
            bts_df["last_trigger"].fillna(method="pad", inplace=True)

            self.bts_df_cols = {col: n for n, col in enumerate(bts_df.columns.values)}
            stale_boughts = ~(
                # last active stoploss matches the current stoploss, otherwise it's stale
                (bts_df["trigger_ofs"].values == bts_df["last_trigger"].values)
                # it must be the first bought matching that stoploss index,
                # in case of subsequent boughts that triggers on the same index
                # which wouldn't happen without position stacking
                & (
                    bts_df["last_trigger"].values
                    != shift(bts_df["last_trigger"].values)
                )
            )
            # null stale boughts
            for c in ["trigger_ofs", *self.trigger_types]:
                bts_df.values[stale_boughts, self.bts_df_cols[c]] = nan
            # merging doesn't include the pair column, so fill empty pair rows forward
            bts_df["pair"].fillna(method="pad", inplace=True)
            # fill nan bool columns to False
            bts_df.fillna({t: 0 for t in trigger_types}, inplace=True)
        else:
            # add stoploss data to the bought/sold dataframe
            bts_df = bts_df.merge(
                stoploss,
                left_on="ohlc_ofs",
                right_on="trigger_bought_ofs",
                how="left",
                copy=False,
            )
            bts_df.set_index("ohlc_ofs", inplace=True)
            # don't apply stoploss to sold candles
            bts_df.loc[
                (bts_df["bought_or_sold"].values == Candle.SOLD), "trigger_ofs",
            ] = nan
        return bts_df

    def _columns_indexes(
        self,
        cols: List[str],
        roi_timeouts: List[int],
        roi_as_tuple=False,
        col_as_dict=False,
    ) -> Tuple:
        col = {}
        n_c = 0
        col = {c: n_c for n_c, c in enumerate(cols)}
        roi_cols = {to: n_c for n_c, to in enumerate(roi_timeouts, len(cols))}
        if col_as_dict:
            ret_col = nb_Dict.empty(
                key_type=types.unicode_type, value_type=types.int32,
            )
            for c, v in col.items():
                ret_col[c] = v
        else:
            ret_col = namedtuple("columns", col.keys())(**col)
        return (
            ret_col,
            roi_cols if not roi_as_tuple else tuple(roi_cols.values()),
        )

    def _calc_stoploss_rate(self, df: DataFrame, stoploss: float) -> ndarray:
        return df["open"].values * (1 - abs(stoploss))

    def _calc_trailing_rate(self, high: ndarray, stoploss: float) -> ndarray:
        return high * (1 - abs(stoploss))

    def _calc_stoploss_rate_value(self, open_price: float) -> float:
        return open_price * (1 + self.config["stoploss"])

    def _set_stoploss_rate(self, df: DataFrame):
        """ Adds a column for the stoploss rate """
        df["stoploss_rate"] = self._calc_stoploss_rate(df)

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

    def _calc_roi_close_rate(
        self, open_rate: ndarray, min_rate: ndarray, roi: ndarray
    ) -> ndarray:
        roi_rate = -(open_rate * roi + open_rate * (1 + self.fee)) / (self.fee - 1)
        return maximum(roi_rate, min_rate)

    def split_events(self, bts_df: DataFrame) -> Tuple[DataFrame, DataFrame]:
        ## debugging
        if self.debug:
            self.events = bts_df

        if self.any_trigger:
            bts_ls_s1 = self._shift_paw(
                bts_df["last_trigger"], diff_arr=bts_df["pair"], fill_v=-1, null_v=-1,
            )
            events_buy = bts_df.loc[
                (bts_df["bought_or_sold"].values == Candle.BOUGHT)
                & (
                    (
                        bts_df["bought_or_sold"].shift(fill_value=Candle.SOLD).values
                        == Candle.SOLD
                    )
                    # last_trigger is only valid if == shift(1)
                    # if the previous candle is SOLD it is covered by the previous case
                    # this also covers the case the previous candle == Candle.END
                    | (bts_df["last_trigger"].values != bts_ls_s1)
                    | (bts_df["pair"].values != bts_df["pair"].shift().values)
                )
                # exclude the last boughts that are not stoploss and which next sold is
                # END sold candle
                & ~(
                    (isnan(bts_df["trigger_ofs"].values))
                    & isin(bts_df["next_sold_ofs"].values, self.pairs_ofs_end)
                )
            ]
            events_sell = bts_df.loc[
                (
                    (bts_df["bought_or_sold"].values == Candle.SOLD)
                    # select only sold candles that are not preceded by a stoploss
                    & (bts_ls_s1 == -1)
                )
                # and stoplosses (all candles with notna trigger_ofs should be valid)
                | isfinite(bts_df["trigger_ofs"].values)
            ]
        else:
            events_buy = bts_df.loc[
                (bts_df["bought_or_sold"].values == Candle.BOUGHT)
                & (
                    union_eq(
                        bts_df["bought_or_sold"].shift(fill_value=Candle.SOLD)
                        # check for END too otherwise the first bought of mid-pairs
                        # wouldn't be included
                        .values,
                        [Candle.SOLD, Candle.END],
                    )
                )
                # exclude the last boughts that are not stoploss and which next sold is
                # END sold candle
                & isin(bts_df["next_sold_ofs"].values, self.pairs_ofs_end, invert=True)
            ]
            events_sell = bts_df.loc[(bts_df["bought_or_sold"].values == Candle.SOLD)]

        return events_buy, events_sell

    def split_events_stack(self, bts_df: DataFrame):
        """"""
        if self.stoploss_enabled:
            events_buy = bts_df.loc[
                (bts_df["bought_or_sold"].values == Candle.BOUGHT)
                # exclude the last boughts that are not stoploss and which next sold is
                # END sold candle
                & ~(
                    (isnan(bts_df["trigger_ofs"].values))
                    & isin(bts_df["next_sold_ofs"].values, self.pairs_ofs_end)
                )
            ]
            # compute the number of sell repetitions for non stoplossed boughts
            nso, sell_repeats = unique(
                events_buy.loc[isnan(events_buy["trigger_ofs"].values)][
                    "next_sold_ofs"
                ],
                return_counts=True,
            )
            # need to check for membership against the bought candles next_sold_ofs here because
            # some sold candles can be void if all the preceding bought candles
            # (after the previous sold) are triggered by a stoploss
            # (otherwise would just be an eq check == Candle.SOLD)
            events_sell = bts_df.loc[
                bts_df.index.isin(nso) | isfinite(bts_df["trigger_ofs"].values)
            ]
            events_sell_repeats = ones(len(events_sell))
            events_sell_repeats[events_sell.index.isin(nso)] = sell_repeats
            events_sell = events_sell.reindex(
                events_sell.index.repeat(events_sell_repeats)
            )
        else:
            events_buy = bts_df.loc[
                (bts_df["bought_or_sold"].values == Candle.BOUGHT)
                # exclude the last boughts that are not stoploss and which next sold is
                # END sold candle
                & isin(bts_df["next_sold_ofs"].values, self.pairs_ofs_end, invert=True)
            ]
            events_sell = bts_df.loc[bts_df["bought_or_sold"].values == Candle.SOLD]
            _, sold_repeats = unique(
                events_buy["next_sold_ofs"].values, return_counts=True
            )
            events_sell = events_sell.reindex(events_sell.index.repeat(sold_repeats))
        return (events_buy, events_sell)

    def vectorized_backtest(
        self, processed: Dict[str, DataFrame], **kwargs,
    ) -> DataFrame:
        """ NOTE: can't have default values as arguments since it is an overridden function
        TODO: benchmark if rewriting without use of df masks for
        readability gives a worthwhile speedup
        """
        df = self.merge_pairs_df(processed)

        df, empty = self.bought_or_sold(df)

        if empty:  # if no bought signals
            return self.empty_results
        if self.any_trigger:
            bts_df = self.set_triggers(df)
        else:
            bts_df, _ = self.set_sold(df)

        if len(bts_df) < 1:
            return self.empty_results

        events_buy, events_sell = (
            self.split_events(bts_df)
            if not self.position_stacking
            else self.split_events_stack(bts_df)
        )

        self._validate_events(events_buy, events_sell)
        return self.get_results(events_buy, events_sell, df)

    def _validate_events(self, events_buy: DataFrame, events_sell: DataFrame):
        try:
            assert len(events_buy) == len(events_sell)
        except AssertionError:
            # find locations where a sell is after two or more buys
            print("buy:", len(events_buy), "sell:", len(events_sell))
            for n, i in enumerate(events_buy.index.values[1:], 1):
                nxt = (events_sell.iloc[n].name >= i) & (
                    events_sell.iloc[n - 1].name < i
                )
                if not nxt:
                    print(events_buy.iloc[n])
                    print(events_buy.iloc[n - 1 : n + 1])
                    print(events_sell.iloc[n - 1 : n + 1], end="\n")
                    raise OperationalException("Buy and sell events not matching")
                    return
            events_buy, events_sell = (
                events_buy[self.merge_cols],
                events_sell[self.merge_cols],
            )
            print(events_buy.iloc[:3], "\n", events_sell.iloc[:3])
            print(events_buy.iloc[-3:], "\n", events_sell.iloc[-3:])
            raise OperationalException("Buy and sell events not matching")

    @staticmethod
    def start_pyinst(interval=0.001, skip=0):
        from pyinstrument import Profiler

        global counter
        if skip:
            if "counter" not in globals():
                counter = 0
            if counter > skip:
                skip = 0
                counter = -1
            else:
                counter += 1
        if not skip and "profiler" not in globals():
            global profiler
            profiler = Profiler(interval=interval)
            profiler.start()

    @staticmethod
    def stop_pyinst(ex=True, keep=False):
        global profiler
        is_skipping = "counter" in globals() and counter != -1
        if not is_skipping:
            profiler.stop()
        if not is_skipping or (is_skipping and counter == -1):
            print(profiler.output_text(unicode=True, color=True))
            if not keep:
                exit()
            else:
                del profiler

    @staticmethod
    def print_pyinst():
        global profiler
        print(profiler.output_text(unicode=True, color=True))
        exit()

    def _debug_opts(self):
        # import os
        # import psutil
        # pid = psutil.Process(os.getpid())
        set_option("display.max_rows", 1000)
        self.cols = [
            "buy",
            "sell",
            "ohlc",
            "pair",
            "bought_or_sold",
            "next_sold_ofs",
            "trigger_bought_ofs",
            "trigger_ofs",
            "trigger_ofs_b",
            "stoploss_triggered",
            "stoploss_rate",
            "low",
            "roi_profit",
            "roi_triggered",
            "roi_triggered_b",
            "roi_profit_b",
            "date_trigger",
            "trigger_ofs_max",
            "last_trigger",
            "ohlc_ofs",
        ]
        self.counter = 0

    def _cols(self, df: DataFrame):
        columns = df.columns.values
        flt_cols = []
        for col in self.cols:
            if col in columns:
                flt_cols.append(col)
        return flt_cols

    def _load_results(self) -> DataFrame:
        import pickle

        with open("/tmp/backtest.pkl", "rb+") as fp:
            return pickle.load(fp)

    def _dump_results(self, results: DataFrame):
        import pickle

        with open("/tmp/backtest.pkl", "rb+") as fp:
            pickle.dump(results, fp)

    def _cmp_indexes(
        self,
        where: Tuple[str, str],
        results: DataFrame,
        saved_results: DataFrame,
        ex=False,
        print_data=False,
        filter_fsell=True,
        print_inc=True,
    ):
        """ find all the non matching indexes between results, differentiate between not present (to include)
        and invalid (to exclude) """
        to_inc, to_exc = [], []
        key_0 = where[0]
        key_1 = where[1]
        key_pair_0 = f"pair_{key_0}"
        key_pair_1 = f"pair_{key_1}"

        if len(results) == 0 and len(saved_results) == 0:
            return
        # we don't consider missing force sells as wrong
        if filter_fsell:
            if "events" in dir(self):
                end_candles = self.events.loc[
                    self.events["next_sold_ofs"].isin(self.pairs_ofs_end)
                ]["ohlc"].values
            else:
                end_candles = []
            results = results.loc[
                (results["sell_reason"].values != SellType.FORCE_SELL)
                & ~(results[key_0].isin(end_candles))
            ]
            saved_results = saved_results.loc[
                (saved_results["sell_reason"].values != SellType.FORCE_SELL)
                & ~(saved_results[key_1].isin(end_candles).values)
            ]

        # stock results are sorted, so align indexes
        if results["pair"].dtype == float:
            results["pair"] = replace_values(
                self.pairs_idx, self.pairs_name, results["pair"].values
            )
        results = results.sort_values(by=["pair", key_0])
        saved_results = saved_results.sort_values(by=["pair", key_1])
        # have to match indexes to the correct pairs, so make sets of (index, pair) tuples
        where_0 = list(
            zip(
                results[key_0]
                .fillna(method="pad")
                .astype(saved_results[key_1].dtype)
                .values,
                results["pair"].values,
            )
        )
        where_1 = list(zip(saved_results[key_1].values, saved_results["pair"].values,))
        results[key_pair_0], saved_results[key_pair_1] = where_0, where_1
        where_0_set = set(where_0)
        where_1_set = set(where_1)

        for i in results[key_pair_0].values:
            if i not in where_1_set:
                to_exc.append(i)
        for i in saved_results[key_pair_1]:
            if i not in where_0_set:
                to_inc.append(i)
        if print_data:
            print(to_inc, to_exc)
            print(saved_results.set_index(key_pair_1).loc[to_inc[:10]])
            print(results.set_index(key_pair_0).loc[to_exc[:10]])
        if to_inc:
            print(
                "TO INCLUDE (total): ",
                to_inc[0] if len(to_inc) > 0 else None,
                f"({len(to_inc)})",
            )
        if to_exc:
            print(
                "TO EXCLUDE (total): ",
                to_exc[0] if len(to_exc) > 0 else None,
                f"({len(to_exc)})",
            )
        # print the first event that is wrong and the range of the
        # boughts_to_sold df (saved in self.events) that includes it
        if to_inc and print_inc:
            first = to_inc[0]
        elif to_exc:
            first = to_exc[0]
        else:
            first = None

        if first is not None:
            idx = (
                (self.events["ohlc"].values == int(first[0]))
                & (self.events["pair"].values == first[1])
            ).argmax()
            print(
                self.events.iloc[max(0, idx - 50) : min(idx + 100, len(self.events))][
                    self._cols(self.events)
                ]
            )
            s_idx = (
                (saved_results["pair"].values == first[1])
                & (saved_results[key_1].values == int(first[0]))
            ).argmax()
            print(
                saved_results.iloc[
                    max(0, s_idx - 10) : min(s_idx + 10, len(saved_results))
                ][saved_results.columns.difference(["pair_open_index"])]
            )
            print("idx:", idx, ", calc_type:", self.calc_type, ", count:", self.counter)
            if ex:
                exit()

    def _check_counter(self, at=0) -> bool:
        self.counter += 1
        return self.counter < at

    def _wrap_backtest(self, processed: Dict[str, DataFrame], **kwargs,) -> DataFrame:
        """ debugging """
        # results to debug
        results = None
        # results to compare against
        saved_results = None
        # if some epoch far down the (reproducible) iteration needs debugging set it here
        check_at = 0
        if check_at and self._check_counter(check_at):
            return self.empty_results
        # if testing only one epoch use "store" once then set it to "load"
        cache = ""
        if cache == "load":
            results = self.vectorized_backtest(processed)
            saved_results = self._load_results()
        elif cache == "store":
            self._dump_results(self.backtest_stock(processed, **kwargs,))
            exit()
        else:
            results = self.vectorized_backtest(processed)
            saved_results = self.backtest_stock(processed, **kwargs,)
        idx_name = "close_index"
        self._cmp_indexes((idx_name, idx_name), results, saved_results)
        # print(
        #     results.iloc[:10],
        #     "\n",
        #     saved_results.sort_values(by=["pair", "open_index"]).iloc[:10],
        # )
        # return saved_results
        return results

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


def padfill(arr: ndarray):
    mask = isnan(arr)
    arr[mask] = interp(flatnonzero(mask), flatnonzero(~mask), arr[~mask])


def union_eq(arr: ndarray, vals: List) -> List[bool]:
    """ union of equalities from a starting value and a list of values to compare """
    res = arr == vals[0]
    for v in vals[1:]:
        res = res | (arr == v)
    return res


def shift(arr: ndarray, period=1, fill=nan) -> ndarray:
    """ shift ndarray """
    moved: ndarray = ndarray(shape=arr.shape, dtype=arr.dtype)
    if period < 0:
        moved[:period] = arr[-period:]
        moved[period:] = fill
    else:
        moved[period:] = arr[:-period]
        moved[:period] = fill
    return moved


def df_cols(df) -> Dict[str, int]:
    return {col: n for n, col in enumerate(df.columns.values)}


def add_columns(arr: ndarray, cols_dict: Dict, columns: Tuple) -> ndarray:
    tail = len(cols_dict)
    for c in columns:
        cols_dict[c] = tail
        tail += 1
    return concatenate((arr, ndarray(shape=(arr.shape[0], len(columns)))), axis=1)


def replace_values(v: ndarray, k: ndarray, arr: ndarray) -> ndarray:
    """ replace values in a 1D array """
    # searchsorted returns len(arr) where each element is the index of v
    # make sure types match
    idx = searchsorted(v, arr)
    # this is not needed if we assume that no element in arr is > the values in v
    # idx[idx == len(v)] = nan
    mask = v[idx] == arr
    return where(mask, k[idx], nan)
