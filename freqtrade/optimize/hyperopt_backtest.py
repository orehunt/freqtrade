import logging

import arrow
import gc
from typing import Dict, List, Tuple, Union
from enum import IntEnum
from collections import namedtuple
from freqtrade.vendor.search import search

from numba import njit
from numpy import (
    dtype,
    repeat,
    ones,
    nan,
    concatenate,
    ndarray,
    array,
    where,
    argwhere,
    transpose,
    maximum,
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
)

from freqtrade.optimize.backtesting import Backtesting, BacktestResult
from freqtrade.exceptions import OperationalException
from freqtrade.strategy.interface import SellType


logger = logging.getLogger(__name__)


class Candle(IntEnum):
    NOOP = 0
    BOUGHT = 1
    SOLD = 3
    END = 7  # references the last candle of a pair
    # STOPLOSS = 12


@njit  # fastmath=True ? there is no math involved here though..
def for_trail_idx(index, trig_idx, next_sold) -> List[int]:
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


def union_eq(arr: ndarray, vals: List) -> ndarray:
    """ union of equalities from a starting value and a list of values to compare """
    res = arr == vals[0]
    for v in vals[1:]:
        res = res | (arr == v)
    return res


def shift(arr: ndarray, period=1) -> ndarray:
    """ shift ndarray """
    moved = ndarray(shape=arr.shape, dtype=arr.dtype)
    if period < 0:
        moved[:period] = arr[-period:]
        moved[period:] = nan
    else:
        moved[period:] = arr[:-period]
        moved[:period] = nan
    return moved


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

    # TODO: cast columns for indexing to int (since they are inferred as floats)
    df_cols = ["date", "open", "high", "low", "close", "volume", "ofs"]
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
        self.trailing_enabled = backtesting_amounts.get("trailing", False)
        self.roi_enabled = backtesting_amounts.get("roi", False)

        self.position_stacking = self.config.get("position_stacking", False)
        if self.config.get("max_open_trades", 0) > 0:
            logger.warn("Ignoring max open trades...")

    def get_results(self, events_buy: DataFrame, events_sell: DataFrame) -> DataFrame:
        # choose sell rate depending on sell reason and set sell_reason
        events_sell = events_sell.reindex(
            [*events_sell.columns, "close_rate", "sell_reason"], axis=1, copy=False
        )
        events_sold = events_sell.loc[
            events_sell["bought_or_sold"].values == Candle.SOLD
        ]
        # add new columns to allow multi col assignments of new columns
        result_cols = ["close_rate", "sell_reason", "ohlc"]
        # can't pass the index here because indexes are duplicated with position_stacking,
        # would have to reindex beforehand
        events_sell.loc[
            events_sold.index
            if not self.position_stacking
            else events_sell.index.isin(unique(events_sold.index.values)),
            result_cols,
        ] = [
            events_sold["open"].values,
            SellType.SELL_SIGNAL,
            events_sold["ohlc"].values,
        ]
        # order of events is stoploss -> trailing -> roi
        # since values are set cascading, the order is inverted
        # NOTE: using .astype(bool) converts nan to True
        if self.roi_enabled:
            events_roi = events_sell.loc[
                # (events_sell["roi_triggered"].values == True)
                events_sell["roi_triggered"]
                .astype("boolean")
                .values
            ]
            events_sell.loc[events_roi.index, [*result_cols, "date"]] = [
                self._calc_close_rate(
                    events_roi["open"].values, events_roi["roi_profit"].values
                ),
                SellType.ROI,
                events_roi["trigger_ofs"].values,
                events_roi["trigger_date"].values,
            ]
        if self.stoploss_enabled:
            events_stoploss = events_sell.loc[
                # (events_sell["stoploss_triggered"].values == True)
                events_sell["stoploss_triggered"]
                .astype("boolean")
                .values
            ]
            events_sell.loc[events_stoploss.index, [*result_cols, "date"]] = [
                events_stoploss["stoploss_rate"].values,
                SellType.STOP_LOSS,
                events_stoploss["trigger_ofs"].values,
                events_stoploss["trigger_date"].values,
            ]

        open_rate = events_buy["open"].values
        close_rate = events_sell["close_rate"].values

        profits_abs, profits_prc = self._calc_profits(
            open_rate, close_rate, calc_abs=True
        )

        trade_duration = to_timedelta(
            Series(events_sell["date"].values - events_buy["date"].values)
        )
        # replace trade duration of same candle trades with half the timeframe reduce to minutes
        trade_duration.loc[trade_duration == self.td_zero] = self.td_half_timeframe
        return DataFrame(
            {
                "pair": events_buy["pair"].values,
                "profit_percent": profits_prc,
                "profit_abs": profits_abs,
                "open_time": to_datetime(events_buy["date"].values),
                "close_time": to_datetime(events_sell["date"].values),
                "open_index": events_buy["ohlc"].values,
                "close_index": events_sell["ohlc"].values,
                "trade_duration": trade_duration.dt.seconds / 60,
                "open_at_end": False,
                "open_rate": open_rate,
                "close_rate": close_rate,
                "sell_reason": events_sell["sell_reason"].values,
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
        shifted = data.shift(period, fill_value=fill_v)
        shifted.iloc[self.shifted_offsets(ofs, period)] = null_v
        return shifted

    @staticmethod
    def _diff_indexes(arr: ndarray, with_start=False, with_end=False) -> ndarray:
        """ returns the indexes where consecutive values are not equal,
        used for finding pairs ends """
        if with_start:
            if with_end:
                raise OperationalException("with_start and with_end are exclusive")
            return where(arr != insert(arr[:-1], 0, nan))[0]
        elif with_end:
            if with_start:
                raise OperationalException("with_end and with_start are exclusive")
            return where(arr != append(arr[1:], nan))[0]
        else:
            return where(arr != insert(arr[:-1], 0, arr[0]))[0]

    def advise_pair_df(self, df: DataFrame, pair: str) -> DataFrame:
        """ Execute strategy signals and return df for given pair """
        meta = {"pair": pair}
        df["buy"] = 0
        df["sell"] = 0
        df["pair"] = pair

        df = self.strategy.advise_buy(df, meta)
        df = self.strategy.advise_sell(df, meta)
        # strategy might be evil and nan set some  buy/sell rows
        # df.fillna({"buy": 0, "sell": 0}, inplace=True)
        # cast date as int to prevent time conversion when accessing values
        df["date"] = df["date"].astype(int, copy=False).values
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
        pairs = {}

        # get the df with the longest ohlc data since all the pairs will be padded to it
        pair_counter = 0
        for pair, df in processed.items():
            # make sure to copy the df to not clobber the source data since it is accessed globally
            advised[pair] = self.advise_pair_df(df.copy(), pair)
            pairs[pair] = pair_counter
            pair_counter += 1
        self.pairs = pairs
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
        df["ofs"] = Categorical(offsets_arr, self.pairs_offset)
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
        # df["bought_or_sold"] = (df["buy"] - df["sell"]).groupby(level=1).shift().values
        df["bought_or_sold"] = self._shift_paw(
            df["buy"] - df["sell"],
            fill_v=Candle.NOOP,
            null_v=Candle.NOOP,
            diff_arr=df["pair"],
        ).values

        df["bought_or_sold"].replace({1: Candle.BOUGHT, -1: Candle.SOLD}, inplace=True)
        # set sold candles
        df["bought_or_sold"] = Categorical(
            df["bought_or_sold"].values, categories=list(map(int, Candle))
        )
        # set END candles as the last non nan candle of each pair data
        bos_loc = df.columns.get_loc("bought_or_sold")
        # modify last signals to set the end
        df.iloc[self.pairs_ofs_end, bos_loc] = Candle.END
        # Since bought_or_sold is shifted, null the row after the last non-nan one
        # as it doesn't have data, exclude pairs which data matches the max_len since
        # they have no nans
        # df.iloc[self.nan_data_ends, bos_loc] = Candle.NOOP
        return df, len(df.loc[df["bought_or_sold"].values == Candle.BOUGHT]) < 1

    def boughts_to_sold(self, df: DataFrame) -> DataFrame:
        """
        reduce df such that there are many bought interleaved by one sold candle
        NOTE: does not modify input df
        """
        bos_df = df.loc[
            union_eq(
                df["bought_or_sold"].values, [Candle.BOUGHT, Candle.SOLD, Candle.END,],
            )
        ]
        bos_df = bos_df.loc[
            # exclude duplicate sold
            ~(
                (bos_df["bought_or_sold"].values == Candle.SOLD)
                & (
                    # bos_df["bought_or_sold"]
                    # .groupby(level=1)
                    # .shift(fill_value=Candle.SOLD)
                    # .values
                    (
                        self._shift_paw(
                            bos_df["bought_or_sold"],
                            fill_v=Candle.SOLD,
                            null_v=Candle.NOOP,
                            diff_arr=bos_df["pair"],
                        ).values
                        == Candle.SOLD
                    )
                    # don't sell different pairs
                    | (bos_df["pair"].values != bos_df["pair"].shift().values)
                )
            )
        ]
        bos_df.reset_index(inplace=True, drop=True)
        return bos_df

    def _pd_calc_sold_repeats(self, bts_df: DataFrame, sold: DataFrame) -> list:
        """ deprecated; pandas version of the next_sold_ofs calculation """
        first_bought = (
            bts_df.assign(bts_index=bts_df.index.values)
            .groupby("pair", sort=False)
            .first()
        )

        def repeats(x, rep):
            vals = x.index.values
            # prepend the first range subtracting the index of the first bought
            rep.append(vals[0] - first_bought.at[x.name, "bts_index"] + 1)
            rep.extend(vals[1:] - vals[:-1])

        sold_repeats: List = []
        sold.groupby("pair", sort=False).apply(repeats, rep=sold_repeats)
        return sold_repeats

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

    def set_stoploss(self, df: DataFrame) -> DataFrame:
        """
        returns the df of valid boughts where stoploss triggered, with matching stoploss
        index of each bought
        """
        bts_df, sold = self.set_sold(df)
        bought = bts_df.loc[bts_df["bought_or_sold"].values == Candle.BOUGHT]
        # get the index ranges of each bought->sold spans
        bought_ranges = bought["next_sold_ofs"].values - bought["ohlc_ofs"].values
        # Use the first version only if the expanded array would take < ~500MB per col
        # if bought_ranges.sum() < 10e6:
        if False:
            self.calc_type = True
            # intervals are short compute everything in one round
            bts_df = self._pd_select_triggered_events(df, bought, bought_ranges, bts_df)
        else:
            # intervals are too long, jump over candles
            self.calc_type = False
            args = [df, bought, bought_ranges, sold, bts_df]
            bts_df = (
                self._pd_2_select_triggered_events(*args)
                if not self.position_stacking
                else self._pd_2_select_triggered_events_stack(*args)
            )
        return bts_df

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
            stoploss_triggered_rate = self._calc_stoploss_rate_value(bopen[b])
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

    def _pd_2_select_triggered_events(
        self,
        df: DataFrame,
        bought: DataFrame,
        bought_ranges: ndarray,
        sold: DataFrame,
        bts_df: DataFrame,
    ):
        col_names = [
            "trigger_ofs",
            "trigger_date",
            "trigger_bought_ofs",
            "last_trigger_ofs",
        ]
        range_names = []
        # the number of columns for the shape of the trigger range
        trg_n_cols = 0
        if self.stoploss_enabled:
            col_names.extend(["stoploss_rate", "stoploss_triggered"])
            trg_stoploss_ofs = trg_n_cols
            trg_n_cols += 1
            range_names.append("stoploss_triggered")
        if self.roi_enabled:
            col_names.extend(["roi_profit", "roi_triggered"])
            self.roi_timeouts, self.roi_values = self._filter_roi({}, bought)
            roi_timeouts = array(list(self.roi_timeouts.keys()))
            n_timeouts = len(roi_timeouts)
            # nan indexes are relative, so can be pre calculated
            nan_early_idx = array(
                [
                    n_timeouts * f + n
                    for n, t in enumerate(roi_timeouts)
                    for f in range(t)
                ]
            )
            # transpose roi values such that we can operate with profits ndarray
            roi_values = array(self.roi_values).reshape((len(self.roi_values), 1))
            trg_roi_ofs = trg_n_cols
            trg_n_cols += n_timeouts
        # copy cols for faster index accessing
        bofs = bought["ohlc_ofs"].values
        bsold = bought["next_sold_ofs"].values
        bopen = bought["open"].values
        b = 0
        bought_ofs = bofs[b]

        ohlc_low = df["low"].values
        ohlc_high = df["high"].values
        ohlc_ofs = df["ohlc_ofs"].values
        ohlc_date = df["date"].values
        ofs_start = 0
        current_ofs = bought_ofs
        end_ofs = ohlc_ofs[-1]

        col, roi_cols = self._columns_indexes(col_names, [], [], roi_timeouts)
        roi_idx = tuple(roi_cols.values())
        trg_col, trg_roi_cols = self._columns_indexes(range_names, [], [], roi_timeouts)
        trg_roi_idx = tuple(trg_roi_cols.values())
        triggers = zeros(shape=(len(bought), len(col_names)))
        triggers[:, col.last_trigger_ofs] = -1
        triggers[:, col.trigger_ofs] = nan
        last_trigger = 0

        while bought_ofs < end_ofs:
            # check trigger for the range of the current bought
            br = bought_ranges[b]
            ofs_start += ohlc_ofs[ofs_start:].searchsorted(bought_ofs, "left")
            ofs_stop = ofs_start + br
            trg_range = ndarray(shape=(br, trg_n_cols))

            if self.stoploss_enabled:
                # calculate the rate from the bought candle
                stoploss_triggered_rate = self._calc_stoploss_rate_value(bopen[b])
                trg_range[:, trg_col.stoploss_triggered] = (
                    ohlc_low[ofs_start:ofs_stop] <= stoploss_triggered_rate
                )
            if self.roi_enabled:
                cur_profits = self._calc_profits(
                    bopen[b], ohlc_high[ofs_start:ofs_stop]
                )
                # trg_range[:, trg_roi_idx] = (cur_profits >= roi_values).swapaxes(0, 1)
                # get a view of the roi triggers because we need to nan indexes
                # relative to (flattened) roi triggers only
                roi_triggers = (cur_profits >= roi_values).swapaxes(0, 1).flatten()
                # NOTE: clip nan_early_idx to the length of the bought_range
                # NOTE: use False, not nan, since bool(nan) == True
                roi_triggers[nan_early_idx[nan_early_idx <= br * trg_n_cols]] = False
                trg_range[:, trg_roi_idx] = roi_triggers.reshape(br, n_timeouts)
            # apply argmax over axis 0, such that we get the first timeframe
            # where a trigger happened (argmax of each column)
            trg_first_idx = trg_range.argmax(axis=0)
            # filter out columns that have no true trigger, flatten because argwhere
            # preserve the shape, and truths are picked on different rows per each column
            valid_cols = argwhere(trg_range[trg_first_idx, arange(trg_n_cols)]).flatten()
            # check that it is actually True
            if len(valid_cols):
                # get the column index that triggered first row sie
                trg_first = trg_first_idx[valid_cols].argmin()
                # lastly get the trigger offset from the index of the first valid column
                trg_ofs = trg_first_idx[valid_cols[trg_first]]
                if trg_first == trg_col.stoploss_triggered:
                    # print("stoploss", bought_ofs, b)
                    # exit()
                    triggers[b, col.stoploss_triggered] = True
                    triggers[b, col.stoploss_rate] = stoploss_triggered_rate
                elif trg_first in trg_roi_idx:
                    triggers[b, col.roi_triggered] = True
                    # NOTE: scaletrg_first by how many preceding columns (stoploss,trailing)
                    # there are before roi columns, in order to get the offset
                    # relative to only the (ordered) roi columns
                    triggers[b, col.roi_profit] = roi_values[trg_first - 1]
                current_ofs = bought_ofs + trg_ofs
                trg_ohlc_idx = ohlc_ofs[current_ofs]
                triggers[b, col.trigger_ofs] = trg_ohlc_idx
                triggers[b, col.trigger_date] = ohlc_date[current_ofs]
                triggers[b, col.trigger_bought_ofs] = bought_ofs
                try:
                    last_trigger = b
                    # get the first row where the bought index is
                    # higher than the current stoploss index
                    b += bofs[b:].searchsorted(current_ofs, "right")
                    # repeat the trigger index for the boughts in between the trigger
                    # and the bought with higher idx
                    triggers[last_trigger:b, col.last_trigger_ofs] = trg_ohlc_idx
                    bought_ofs = bofs[b]
                except IndexError:
                    break
                trigger = False
            else:  # if stoploss did not trigger, jump to the first bought after next sold idx
                try:
                    # last_b = b
                    b += bofs[b:].searchsorted(bsold[b], "right")
                    bought_ofs = bofs[b]
                except IndexError:
                    break
        # set the index to the offset and add the columns to set the stoploss
        # data points on the relevant boughts
        bts_df.set_index("ohlc_ofs", inplace=True)
        trigger_names = [
            "trigger_ofs",
            "trigger_date",
            "trigger_bought_ofs",
            "last_trigger",
        ]
        trigger_vals = [
            col.trigger_ofs,
            col.trigger_date,
            col.trigger_bought_ofs,
            col.last_trigger_ofs,
        ]
        if self.stoploss_enabled:
            trigger_names.extend(["stoploss_rate", "stoploss_triggered"])
            trigger_vals.extend([col.stoploss_rate, col.stoploss_triggered])
        if self.roi_enabled:
            trigger_names.extend(["roi_profit", "roi_triggered"])
            trigger_vals.extend([col.roi_profit, col.roi_triggered])
        bts_df = bts_df.reindex(columns=[*bts_df.columns, *trigger_names], copy=False)
        bts_df.loc[bought["ohlc_ofs"], trigger_names] = triggers[:, trigger_vals]
        bts_df["last_trigger"].fillna(-1, inplace=True)
        return bts_df.astype(
            {
                "stoploss_triggered": "bool",
                "roi_triggered": "bool",
                "roi_profit": "float64",
            }
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
        # clear up memory
        # gc.collect()
        df_cols = ["low", "high", "ohlc_ofs", "date"]
        df_types = ["float64", "float64", "int64", "int64"]
        df_cols_n = len(df_cols)
        # bought data rows will be repeated to match the bought_ranges
        bought_data = {}
        bought_types = []
        # post_cols are the empty columns that will be calculated after the expansion
        post_cols = []
        post_types = []
        tot_cols_n = 0
        if self.stoploss_enabled:
            bought_data.update(
                {
                    "trigger_bought": bought["ohlc_ofs"].values,
                    "stoploss_rate": self._calc_stoploss_rate(bought),
                }
            )
            bought_types.extend(["int64", "float64"])
            post_cols.append("stoploss_triggered")
            post_types.append("float32")
        if self.roi_enabled:
            # only the filtered roi couples will be used for calculation
            self.roi_timeouts, self.roi_values = self._filter_roi({}, bought)
            bought_data["bought_open"] = bought["open"].values
            bought_types.append("float64")
            roi_timeouts = list(self.roi_timeouts.keys())
            post_cols.extend(["roi_profit", "roi_triggered"])
            post_types.extend(["float64", "float32"])
        expd_cols_n = df_cols_n + len(bought_data)
        col, roi_cols = self._columns_indexes(
            df_cols, bought_data.keys(), post_cols, roi_timeouts
        )
        tot_cols_n = expd_cols_n + len(post_cols)
        data_types = df_types + bought_types + post_types
        col_names = df_cols + list(bought_data.keys()) + post_cols

        ohlc_vals = df[df_cols].values
        bought_vals = array(list(bought_data.values())).swapaxes(0, 1)
        data_len = bought_ranges.sum()
        data = ndarray(shape=(data_len, tot_cols_n))

        data_ofs = ndarray(shape=(len(bought) + 1), dtype=int)
        data_ofs[0] = 0
        data_ofs[1:] = bought_ranges.cumsum()
        for n, i in enumerate(bought["ohlc_ofs"].values):
            start, stop = data_ofs[n], data_ofs[n + 1]
            data[start:stop, :df_cols_n] = ohlc_vals[i : i + bought_ranges[n]]
            # these vals are repeated
            data[start:stop, df_cols_n:expd_cols_n] = bought_vals[n]

        if self.roi_enabled:
            # the starting offset is already computed with data_ofs
            # exclude the last value since we only nan from the start
            # of each bought range
            bought_starts = data_ofs[:-1]
            # roi rates index null overrides based on timeouts
            nan_early_roi_idx = {to: bought_starts + to for to in roi_timeouts}
            # roi triggers store the truth of roi matching for each roi timeout
            roi_triggers = []
            # setup expanded roi columns ordered by roi timeouts
            roi_vals = array([full(data_len, to) for to in self.roi_values]).swapaxes(
                0, 1
            )
            cur_profits = self._calc_profits(
                data[:, col.bought_open], data[:, col.high]
            )
            for n, to in enumerate(roi_timeouts):
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
            # stoploss is triggered when low is below or equal
        # data = concatenate([data, roi_triggers, transpose([roi_trigger_max, cur_profits])], axis=1)
        # cols = list(col._fields)
        # cols.extend(self.roi_timeouts.values())
        # cols.extend(["roi_trigger_max", "cur_profit"])
        # df = DataFrame(data, columns=cols)
        # print(df.iloc[1700:1800])
        # exit()

        if self.stoploss_enabled:
            data[:, col.stoploss_triggered] = (
                data[:, col.low] <= data[:, col.stoploss_rate]
            )
        if self.stoploss_enabled and self.roi_enabled:
            data = data[
                data[:, col.stoploss_triggered].astype(bool)
                | data[:, col.roi_triggered].astype(bool)
            ]
        elif self.stoploss_enabled:
            data = data[data[:, col.stoploss_triggered].astype(bool)]
        # TODO: modify this to support roi/trailing
        if len(data) < 1:
            # keep shape since return value is accessed without reference
            return full((0, data.shape[1]), nan)
        # reduce expansion to only triggered columns
        # only where the trigger_bought_ofs is not the same as the previous
        # since we don't evaluate alternate universes
        data = data[
            data[:, col.trigger_bought] != shift(data[:, col.trigger_bought]),
        ]

        # exclude triggers that where bought past the max index of the triggers
        # since we don't travel back in time
        if not self.position_stacking:
            data = data[
                data[:, col.trigger_bought]
                >= maximum.accumulate(data[:, col.trigger_bought])
            ]

        # mark objects for gc
        # del (
        #     df,
        #     ohlc_vals,
        #     bought_vals
        # )
        # gc.collect()
        col_idx = [
            col.ohlc_ofs,
            col.date,
            col.trigger_bought,
        ]
        col_names = [
            "trigger_ofs",
            "trigger_date",
            "trigger_bought_ofs",
        ]
        if self.stoploss_enabled:
            col_idx.extend([col.stoploss_rate, col.stoploss_triggered])
            col_names.extend(["stoploss_rate", "stoploss_triggered"])
        if self.roi_enabled:
            col_idx.extend(
                [col.roi_profit, col.roi_triggered,]
            )
            col_names.extend(
                ["roi_profit", "roi_triggered",]
            )
        return DataFrame(data[:, col_idx], columns=col_names, copy=False).astype(
            {cn: tp for cn, tp in zip(col_names, data_types)}, copy=False
        )

    def _pd_calc_triggered_stoploss(
        self, df: DataFrame, bought: DataFrame, bought_ranges: ndarray,
    ):
        """ Expand the ohlc dataframe for each bought candle to check if stoploss was triggered """
        # gc.collect()

        ohlc_vals = df["ohlc_ofs"].values

        # create a df with just the indexes to expand
        trigger_ofs_expd = DataFrame(
            (
                concatenate(
                    [
                        ohlc_vals[i : i + bought_ranges[n]]
                        # loop over the pair/offsetted indexes that will be used as merge key
                        for n, i in enumerate(bought["ohlc_ofs"].values)
                    ]
                )
            ),
            columns=["trigger_ofs"],
        )
        # add the row data to the expanded indexes
        stoploss = trigger_ofs_expd.merge(
            # reset level 1 to preserve pair column
            df.reset_index(level=1),
            how="left",
            left_on="trigger_ofs",
            right_on="ohlc_ofs",
            copy=False,
        )
        # set bought idx for each bought timerange, so that we know to which bought candle
        # the row belongs to, and stoploss rates relative to each bought
        stoploss["trigger_bought_ofs"], stoploss["stoploss_rate"] = repeat(
            [bought["ohlc_ofs"].values, self._calc_stoploss_rate(bought),],
            bought_ranges,
            axis=1,
        )

        stoploss = stoploss.loc[
            stoploss["low"].values <= stoploss["stoploss_rate"].values
        ]
        # filter out duplicate subsequent triggers
        # of the same bought candle as only the first ones matters
        stoploss = stoploss.loc[
            (
                stoploss["trigger_bought_ofs"].values
                != stoploss["trigger_bought_ofs"].shift().values
            )
        ]
        if not self.position_stacking:
            # filter out "late" stoplosses that wouldn't be applied because a previous stoploss
            # would still be active at that time
            # since stoplosses are sorted by trigger date,
            # any stoploss having a bought index older than
            # the ohlc index are invalid
            stoploss = stoploss.loc[
                stoploss["trigger_bought_ofs"]
                >= stoploss["trigger_bought_ofs"].cummax().values
            ]
        # select columns
        stoploss = stoploss[
            ["trigger_ofs", "date", "trigger_bought_ofs", "stoploss_rate"]
        ]

        # mark objects for gc
        del (
            df,
            trigger_ofs_expd,
            ohlc_vals,
        )
        # gc.collect()
        return stoploss

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
            df["trigger_ofs"].astype(int, copy=False).values,
            df["next_sold_ofs"].astype(int, copy=False).values,
        )

    @staticmethod
    def start_pyinst(interval=0.001):
        from pyinstrument import Profiler

        global profiler
        profiler = Profiler(interval=interval)
        profiler.start()

    @staticmethod
    def stop_pyinst():
        global profiler
        profiler.stop()
        print(profiler.output_text(unicode=True, color=True))
        exit()

    def _pd_select_triggered_events(
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

            # can use assign here since all the trigger indices should be present in
            # bts_df since we set it as the boughts indices
            bts_df["trigger_ofs"] = trigger["trigger_ofs"]
            bts_df["trigger_date"] = trigger["trigger_date"]
            if self.stoploss_enabled:
                bts_df["stoploss_rate"] = trigger["stoploss_rate"]
                bts_df["stoploss_triggered"] = trigger["stoploss_triggered"]
            if self.roi_enabled:
                bts_df["roi_profit"] = trigger["roi_profit"]
                bts_df["roi_triggered"] = trigger["roi_triggered"]

            # now add the trigger new rows with an outer join
            trigger.set_index("trigger_ofs", inplace=True, drop=False)
            bts_df = bts_df.merge(
                trigger[["trigger_ofs", "trigger_bought_ofs", "roi_profit"]],
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
            bts_df = bts_df.loc[
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
                self._last_trigger_numba(bts_df.loc[boughts])
            )
            # fill the last_trigger column for non boughts
            bts_df["last_trigger"].fillna(method="pad", inplace=True)
            bts_df.loc[
                ~(
                    # last active stoploss matches the current stoploss, otherwise it's stale
                    (bts_df["trigger_ofs"].values == bts_df["last_trigger"].values)
                    # it must be the first bought matching that stoploss index,
                    # in case of subsequent boughts that triggers on the same index
                    # which wouldn't happen without position stacking
                    & (
                        bts_df["last_trigger"].values
                        != bts_df["last_trigger"].shift().values
                    )
                ),
                "trigger_ofs",
            ] = nan
            # merging doesn't include the pair column, so fill empty pair rows forward
            bts_df["pair"].fillna(method="pad", inplace=True)
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
        # gc.collect()
        return bts_df

    def _columns_indexes(
        self,
        df_cols: List[str],
        bought_cols: List[str],
        post_cols: List[str],
        roi_timeouts: List[int],
    ) -> Tuple:
        col = {}
        n_c = 0
        # ohlc columns
        for c in df_cols:
            col[c] = n_c
            n_c += 1
        # bought columns
        for c in bought_cols:
            col[c] = n_c
            n_c += 1
        for c in post_cols:
            col[c] = n_c
            n_c += 1
        # use unrounded numbers as keys
        roi_cols = {}
        for to in roi_timeouts:
            roi_cols[to] = n_c
            n_c += 1
        return namedtuple("columns", col.keys())(**col), roi_cols

    def _set_stoploss_rate(self, df: DataFrame):
        """ Adds a column for the stoploss rate """
        df["stoploss_rate"] = self._calc_stoploss_rate(df)

    def _calc_stoploss_rate(self, df: DataFrame) -> ndarray:
        return df["open"].values * (1 + self.config["stoploss"])

    def _calc_stoploss_rate_value(self, open_price: float) -> float:
        return open_price * (1 + self.config["stoploss"])

    def _filter_roi(
        self, minimal_roi: Dict, bought: DataFrame
    ) -> Tuple[Dict[int, int], List[float]]:
        # ensure roi dict is sorted in order to always overwrite
        # with the latest duplicate value when rounding to timeframes
        # NOTE: make sure to sort numerically
        minimal_roi = {
            "0": 0.05,
            "60": 0.025,
            "120": 0.01,
            "180": 0.0,
            # "240": -0.01,
            # "300": -0.025,
            # "360": -0.05
        }
        # minimal_roi = self.config['minimal_roi']
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

    def split_events(self, bts_df: DataFrame) -> Tuple[DataFrame, DataFrame]:
        ## debugging
        if self.debug:
            self.events = bts_df

        if self.stoploss_enabled:
            bts_ls_s1 = self._shift_paw(
                bts_df["last_trigger"].astype(int),
                diff_arr=bts_df["pair"],
                fill_v=-1,
                null_v=-1,
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
                | (isfinite(bts_df["trigger_ofs"].values))
            ]
            # self._cmp_indexes(("ohlc", "close_index"), events_sell, self._load_results(), filter_fsell=False, ex=True)
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
        if self.stoploss_enabled:
            bts_df = self.set_stoploss(df)
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
        return self.get_results(events_buy, events_sell)

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
                events_buy[self.df_cols],
                events_sell[self.df_cols],
            )
            print(events_buy.iloc[:3], "\n", events_sell.iloc[:3])
            print(events_buy.iloc[-3:], "\n", events_sell.iloc[-3:])
            raise OperationalException("Buy and sell events not matching")

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
        results = results.sort_values(by=["pair", key_0])
        saved_results = saved_results.sort_values(by=["pair", key_1])
        # have to match indexes to the correct pairs, so make sets of (index, pair) tuples
        where_0 = list(
            zip(
                results[key_0].fillna(method="pad").astype(int).values,
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
                self.events.iloc[max(0, idx - 20) : min(idx + 20, len(self.events))][
                    self._cols(self.events)
                ]
            )
            s_idx = (
                (saved_results["pair"].values == first[1])
                & (saved_results[key_1].values == int(first[0]))
            ).argmax()
            print(
                saved_results.iloc[
                    max(0, s_idx - 20) : min(s_idx + 20, len(saved_results))
                ]
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
        cache = "load"
        if cache == "load":
            results = self.vectorized_backtest(processed)
            saved_results = self._load_results()
        elif cache == "store":
            self._dump_results(self.backtest_stock(processed, **kwargs,))
            exit()
        else:
            results = self.vectorized_backtest(processed)
            saved_results = self.backtest_stock(processed, **kwargs,)
        self._cmp_indexes(("open_index", "open_index"), results, saved_results)
        # print(results.iloc[:10], '\n', saved_results.iloc[:10])
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
