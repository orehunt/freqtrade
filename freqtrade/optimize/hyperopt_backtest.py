from collections import namedtuple
import logging
import os
from joblib import dump, load
from enum import Enum, IntEnum
from types import SimpleNamespace
from typing import Dict, Iterable, List, Sequence, Tuple, Union

import pandas as pd
from freqtrade.exceptions import OperationalException
from freqtrade.optimize.backtest_constants import *  # noqa ignore=F405
from freqtrade.optimize.backtest_constants import Candle
from freqtrade.optimize.backtest_engine_chunked import _chunked_select_triggers
from freqtrade.optimize.backtest_engine_loop_candles import (
    _loop_candles_select_triggers,
)
from freqtrade.optimize.backtest_engine_loop_ranges import _loop_ranges_select_triggers
from freqtrade.optimize.backtest_nb import *  # noqa ignore=F405
from freqtrade.optimize.backtest_utils import *  # noqa ignore=F405
from freqtrade.optimize.backtesting import Backtesting, BacktestResult
from freqtrade.optimize.debug import dbg  # noqa ignore=F405
from freqtrade.strategy.interface import SellType, StoplossConfig
from numpy import (
    append,
    arange,
    array,
    empty,
    floor,
    isfinite,
    isin,
    isnan,
    maximum,
    nan,
    nan_to_num,
    ndarray,
    ones,
    repeat,
    sign,
    unique,
    where,
    zeros,
)
from pandas import DataFrame, Series, Timedelta, concat, to_datetime, to_timedelta
from freqtrade.optimize.vbt import (
    simulate_trades,
    IntSellType,
    StoplossConfigJit,
    Context,
)
from vectorbt import Portfolio

logger = logging.getLogger(__name__)


class OHLCV(NamedTuple):
    pairs: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    spread: np.ndarray


class HyperoptBacktesting(Backtesting):

    empty_results = DataFrame.from_records([], columns=BacktestResult._fields)
    events = None

    bt_ng = None
    td_zero = Timedelta(0, unit="m")
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

    merged_df: Union[None, DataFrame] = None
    df_loc = {}
    bts_loc = {}
    trigger_types = []
    # how many rows can an expanded array have
    max_ranges_size = 10e6

    def __init__(self, config):
        self.bt_ng = config.get("backtesting_engine")
        if self.bt_ng in ("chunked", "loop_ranges", "loop_candles", "vbt"):
            self.backtest_vanilla = self.backtest
            if self.bt_ng == "vbt":
                self.vectorized_backtest = self.vbt_backtest
            if dbg:
                dbg._debug_opts()
                dbg.backtesting = self
                self.backtest = dbg._wrap_backtest
            else:
                self.backtest = self.vectorized_backtest
        else:
            super().__init__(config)
            return

        assert (
            "pair",
            "profit_percent",
            "profit_abs",
            "open_date",
            "open_rate",
            "open_fee",
            "close_date",
            "close_rate",
            "close_fee",
            "amount",
            "trade_duration",
            "open_at_end",
            "sell_reason",
        ) == BacktestResult._fields

        ask_strat = config.get("ask_strategy", {})
        if ask_strat.get("sell_profit_only", False) or ask_strat.get(
            "ignore_roi_if_buy_signal", False
        ):
            raise OperationalException(
                "'sell_profit_only' and 'ignore_roi_if_buy_signal'"
                " are not implemented, disable them."
            )

        self.td_timeframe = Timedelta(config["timeframe"])
        self.td_half_timeframe = self.td_timeframe / 2
        self.timeframe_wnd = config.get("timeframe_window", TIMEFRAME_WND).get(
            config["timeframe"], DEFAULT_WND
        )

        backtesting_amounts = config.get("backtesting_amounts", {})
        self.stoploss_enabled = backtesting_amounts.get("stoploss", False)
        self.trailing_enabled = backtesting_amounts.get(
            "trailing",
            False
            # this is only useful for single backtesting
        ) and config.get("amounts", {}).get("trailing_stop", True)
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
        self.account_for_spread = backtesting_amounts.get("account_for_spread", True)
        self.static_stake = bool(backtesting_amounts.get("static_stake", True))
        self.ignore_volume_below = bool(
            backtesting_amounts.get("ignore_volume_below", True)
        )

        # parent init after config overrides
        super().__init__(config)

        # after having loaded the strategy
        # null all config amounts for disabled ones (to compare against vanilla backtesting)
        if not self.roi_enabled:
            self.strategy.minimal_roi = {0: 10.0}
            config["minimal_roi"] = {"0": 10.0}
            self.strategy.minimal_roi = {"0": 10.0}
        if not self.trailing_enabled:
            self.strategy.trailing_stop = False
            config["trailing_stop"] = False
            self.strategy.trailing_stop = False
        if not self.stoploss_enabled:
            self.strategy.stoploss = -100
            config["stoploss"] = -100
            self.strategy.stoploss = -100

        self.position_stacking = self.config.get("position_stacking", False)
        if self.config.get("max_open_trades", 0) > 0:
            logger.warn("Ignoring max open trades...")
        self.max_staked = self.config.get("max_staked", 0)
        self.min_stake = self.config.get("min_stake", 1e-8)
        self.force_sell_max_duration = self.config.get("signals", {}).get(
            "force_sell_max_duration", False
        )
        self.spaces = self.config.get("spaces", {})

        # VBT
        self.vbt_processed_file = (
            self.config["user_data_dir"] / "hyperopt_data" / "vbt_processed.pkl"
        )
        # make sure to delete previous stale files before run starts
        if os.path.exists(self.vbt_processed_file):
            os.remove(self.vbt_processed_file)
        # create int to type map for replacing
        self.sell_type_map = {}
        for a, b in zip(IntSellType, SellType):
            # make sure the IntSellType enum matches the SellType enum
            if a.name != b.name:
                raise OperationalException(
                    "SellType and IntSellType enums do not match"
                )
            # we must replace int numbers with selltype values
            self.sell_type_map[a.value] = b

    def get_trade_duration(
        self, duration: Optional[Sequence] = None, open_date=None, close_date=None
    ) -> np.ndarray:
        """ Convert difference from two datetime arrays into duration (minutes, float) """
        if duration is not None:
            trade_duration = to_timedelta(Series(duration), unit="ns")
        else:
            trade_duration = to_timedelta(Series(close_date - open_date), unit="ns",)
        # replace trade duration of same candle trades with half the timeframe reduce to minutes
        trade_duration[trade_duration == self.td_zero] = self.td_half_timeframe
        return trade_duration.dt.total_seconds() / 60

    def get_results(
        self, buy_vals: ndarray, sell_vals: ndarray, ohlc_vals: ndarray
    ) -> DataFrame:
        ohlc_cols = self.df_loc
        buy_cols = self.bts_loc
        sell_cols = buy_cols
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
        # NOTE: altough it shouldn't matter if the backtesting only computes
        # the events that truly happened
        # since values are set cascading, the order is inverted
        # NOTE: using .astype(bool) converts nan to True
        if self.roi_enabled:
            # assert np.isfinite(sell_vals[:, sell_cols["roi_triggered"]]).all()
            roi_triggered = sell_vals[:, sell_cols["roi_triggered"]].astype(bool)
            where_roi = where(roi_triggered)[0]
            if len(where_roi):
                # use this only for read operations, assign on sell_vals
                events_roi = sell_vals[roi_triggered]
                sell_reason[where_roi] = SellType.ROI
                for dst_col, src_col in zip(result_cols, trigger_cols):
                    sell_vals[where_roi, dst_col] = events_roi[:, src_col]
                # calc close rate from roi profit, using low (of the trigger candle) as the minimum rate
                roi_buy_rate = buy_vals[where_roi, buy_cols["open"]]
                # cast as int since using as indexer
                roi_ofs = sell_vals[where_roi, sell_cols["trigger_ofs"]].astype(int)
                # assert (roi_ofs == ohlc_vals[roi_ofs, self.df_loc["ohlc_ofs"]]).all()
                # the ohlc data of sell_events is not valid for triggers
                # (since the events are generated from the bts array, which
                # only includes ohlc data of buy/sell candles)
                # so take the correct data from the original ohlc array
                roi_low = ohlc_vals[roi_ofs, ohlc_cols["low"]]
                roi_close_rate = calc_roi_close_rate(
                    roi_buy_rate,
                    roi_low,
                    events_roi[:, sell_cols["roi_profit"]],
                    self.fee,
                )
                roi_buy_ofs = buy_vals[where_roi, buy_cols["ohlc_ofs"]].astype(int)
                roi_open = ohlc_vals[roi_ofs, ohlc_cols["open"]]
                # override roi close rate with open if
                roi_on_open_mask = (
                    # the trade lasts more than 1 candle
                    (roi_ofs - roi_buy_ofs > 0)
                    # and the open rate triggers roi
                    & (roi_open > roi_close_rate)
                )
                roi_close_rate[roi_on_open_mask] = roi_open[roi_on_open_mask]
                sell_vals[where_roi, sell_cols["close_rate"]] = roi_close_rate
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

        # use stake amount decided by the strategy
        if self.static_stake:
            self.strategy.stake = np.full(len(buy_vals), self.config["stake_amount"])
        else:
            self.strategy.stake = buy_vals[:, buy_cols["stake_amount"]]
        # assert (self.strategy.stake > 0.01).all()

        # if spread is included in the profits calculation
        # increase open_rate and decrease close_rate
        if self.account_for_spread:
            open_rate += open_rate * buy_vals[:, buy_cols["spread"]]
            close_rate -= close_rate * sell_vals[:, sell_cols["spread"]]

        profits_abs, profits_prc = self._calc_profits(
            open_rate, close_rate, calc_abs=True,
        )

        trade_duration = self.get_trade_duration(
            open_date=buy_vals[:, buy_cols["date"]],
            close_date=sell_vals[:, sell_cols["date"]],
        )

        results = DataFrame(
            {
                "pair": replace_values(
                    self.pairs_idx, self.pairs_name, buy_vals[:, buy_cols["pair"]]
                ),
                "profit_percent": profits_prc,
                "profit_abs": profits_abs,
                "open_date": to_datetime(buy_vals[:, buy_cols["date"]], utc=True),
                "open_rate": open_rate,
                "open_fee": self.fee,
                "close_date": to_datetime(sell_vals[:, sell_cols["date"]], utc=True),
                "close_rate": close_rate,
                "close_fee": self.fee,
                "amount": self.strategy.stake,
                "trade_duration": trade_duration,
                "open_at_end": False,
                "sell_reason": sell_reason,
                #
                # "open_index": buy_vals[:, buy_cols["ohlc"]].astype(int),
                # "close_index": sell_vals[:, sell_cols["ohlc"]].astype(int),
            }
        )
        results.sort_values(by=["open_date", 'pair'], inplace=True)
        if self.max_staked and not self.position_stacking:
            # NOTE: updates the amount and profit_abs/percent inplace
            can_buy = dont_buy_over_max_stake(
                self.max_staked,
                self.min_stake,
                results["open_date"].values,
                results["close_date"].values,
                results["amount"].values,
                open_rate,
                close_rate,
                self.fee,
                results["profit_abs"].values,
                results["profit_percent"].values,
            )
            results = results[can_buy]

        return results

    def _calc_profits(
        self, open_rate: ndarray, close_rate: ndarray, dec=False, calc_abs=False,
    ) -> ndarray:

        sa = self.strategy.stake
        fee = self.fee
        minus = 1

        if dec:
            from decimal import Decimal

            fee = Decimal(self.fee)
            if isinstance(sa, Iterable):
                sa = array([Decimal(a) for a in sa], dtype="object")
            else:
                sa = Decimal(sa)
            open_rate = array([Decimal(n) for n in open_rate], dtype="object")
            close_rate = array([Decimal(n) for n in close_rate], dtype="object")
            minus = Decimal(minus)

        profits_abs, profits_prc = self._calc_profits_np(
            sa, fee, open_rate, close_rate, calc_abs, minus=minus
        )

        if dec:
            profits_abs = profits_abs.astype(float) if profits_abs is not None else None
            profits_prc = profits_prc.astype(float)
        if calc_abs:
            return profits_abs.round(8), profits_prc.round(8)
        else:
            return profits_prc.round(8)

    @staticmethod
    def _calc_profits_np(sa, fee, open_rate, close_rate, calc_abs, minus=1) -> Tuple:
        am = sa / open_rate
        open_amount = am * open_rate
        close_amount = am * close_rate
        open_price = open_amount + open_amount * fee
        close_price = close_amount - close_amount * fee
        if open_price.dtype != "object":
            profits_prc = (close_price / open_price - minus).round(8)
            profits_abs = (close_price - open_price).round(8) if calc_abs else None
        else:
            profits_prc = close_price / open_price - minus
            profits_abs = (close_price - open_price) if calc_abs else None

        # can be 0, and 0/0 is nan
        # assert np.isfinite(profits_prc).all()
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
        assign = False
        try:
            df["buy"].values[:] = 0
            df["sell"].values[:] = 0
        # ignore if cols are not present
        except KeyError:
            df["buy"] = 0
            df["sell"] = 0
            assign = True

        df = self.strategy.advise_buy(df, meta)
        df = self.strategy.advise_sell(df, meta)
        # strategy might be evil and nan set some  buy/sell rows
        # df.fillna({"buy": 0, "sell": 0}, inplace=True)
        # cast date as int to prevent time conversion when accessing values
        df["date"] = df["date"].values.astype(float)
        if assign:
            df["pair"] = n_pair
        else:
            df["pair"].values[:] = n_pair
        # only return required cols
        return df[MERGE_COLS]

    def post_process(self, df_vals: ndarray, ofs=None):
        """
        Calculate estimates like spread and liquidity
        """
        loc = self.df_loc
        ofs = self.pairs_offset
        wnd = self.timeframe_wnd

        df_vals = add_columns(df_vals, loc, ["high_low", "spread"])
        # df_vals = df_vals.astype(float)

        high, low, close, open, volume = get_cols(
            df_vals, loc, ["high", "low", "close", "open", "volume"], dtype=float
        )

        # high low order determines trailing between old/new rate
        df_vals[:, loc["high_low"]] = sim_high_low(close, open)

        # spread is removed from profits, it's value is skewed by liquidity
        if self.account_for_spread:
            # there can be NaNs in the spread calculation
            df_vals[:, loc["spread"]] = np.nan_to_num(
                calc_skewed_spread(high, low, close, volume, wnd, ofs)
            )
        # override stake amount with stake amount from
        if not self.static_stake:
            if "stake_amount" not in loc:
                raise OperationalException(
                    "static stake enabled, but no stake amount columns",
                    "was given in the strategy",
                )
            # since the stake_amount is decided by the signal, and we work on forward shifted signals
            # shift the stake_amount accordingly
            self.strategy.stake = self._shift_paw(
                df_vals[:, loc["stake_amount"]],
                period=1,
                diff_arr=df_vals[:, loc["pair"]],
            )
            df_vals[:, loc["stake_amount"]] = self.strategy.stake
        else:
            self.strategy.stake = np.full(len(df_vals), self.config["stake_amount"])
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
        # set startup offset from the most common starting index
        # it is usually equal for all pairs unless pairs have missing data
        first_indexes = [processed[p].index[0] for p in processed]
        starts_idx, start_idx_counts = np.unique(first_indexes, return_counts=True)
        self.startup_offset = starts_idx[start_idx_counts.argmax()]
        for pair, df in processed.items():
            # make sure to copy the df to not clobber the source data since it is accessed globally
            advised[pair] = self.advise_pair_df(df.copy(), pair, pair_counter)
            # align index of pairs with possibly different startup offset
            if advised[pair].index[0] != self.startup_offset:
                advised[pair].index += self.startup_offset - advised[pair].index[0]
            pairs[pair] = pair_counter
            pair_counter += 1
        self.pairs = pairs
        self.pairs_name = array(list(pairs.keys()))
        self.pairs_idx = array(list(pairs.values()), dtype=float)
        # the index shouldn't change after the advise call, so we can take the pre-advised index
        # to create the multiindex where each pair is indexed with max len
        df = concat(advised.values(), copy=False)
        # add a column for pairs offsets to make the index unique
        # pairs_offset is the FIRST candle by contiguous index of each pair
        offsets_arr, self.pairs_offset = self._calc_pairs_offsets(df, return_ofs=True)
        self.pairs_ofs_end = append(self.pairs_offset[1:] - 1, len(df) - 1)
        # loop over the missing data pairs and calculate the point where data ends
        # plus the absolute offset
        df["ofs"] = offsets_arr
        # could as easily be arange(len(df)) ...
        df["ohlc_ofs"] = df.index.values + offsets_arr - self.startup_offset
        self.pairs_ohlc_ofs_end = df["ohlc_ofs"].iloc[self.pairs_ofs_end].values
        df["ohlc"] = df.index.values
        # fill missing ohlc with open value index wise
        # df[isnan(df["low"].values), "low"] = df["open"]
        # df[isnan(df["high"].values), "high"] = df["open"]
        # df[isnan(df["close"].values), "close"] = df["open"]
        df.set_index("ohlc_ofs", inplace=True, drop=False)
        if self.ignore_volume_below:
            zvol_mask = df["volume"].values > self.max_staked * 10
            df["buy"].values[:] = df["buy"].values.astype(bool) & zvol_mask
            df["sell"].values[:] = df["sell"].values.astype(bool) & zvol_mask
        return df

    def bought_or_sold(self, df: DataFrame) -> Tuple[DataFrame, bool]:
        """ Set bought_or_sold columns according to buy and sell signals """
        df_vals = df.values.astype(float)

        loc = df_cols(df)
        # NOTE: add commas when using tuples, or use lists
        df_vals = add_columns(df_vals, loc, ("bought_or_sold",))
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
            not (df_vals[:, loc["bought_or_sold"]] == Candle.BOUGHT).any(),
        )

    def boughts_to_sold(self, df_vals: ndarray) -> ndarray:
        """
        reduce df such that there are many bought interleaved by one sold candle
        NOTE: does not modify input df
        """
        loc = self.df_loc.copy()

        if self.force_sell_max_duration:
            max_trade_duration = FORCE_SELL_AFTER.get(self.timeframe, 300)
            df_vals[::max_trade_duration, loc["bought_or_sold"]] = Candle.SOLD

        # checks boughts again separately here because if force sell is applied
        # it could have override the only bought candles, so have to exit early
        bought_mask = df_vals[:, loc["bought_or_sold"]] == Candle.BOUGHT
        if not bought_mask.any():
            return np.zeros((0, df_vals.shape[1]))

        bts_vals = df_vals[
            bought_mask
            | isin(df_vals[:, loc["bought_or_sold"]], [Candle.SOLD, Candle.END])
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
        if not len(bts_vals):
            return bts_vals
        loc = self.bts_loc
        # align sold to bought
        sold = bts_vals[
            np.isin(bts_vals[:, loc["bought_or_sold"]], [Candle.SOLD, Candle.END])
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
        return bts_vals

    def set_triggers(self, df_vals: ndarray, bts_vals: ndarray) -> ndarray:
        """
        returns the df of valid boughts where trigger happens, with matching trigger data
        points for each bought
        """
        loc = self.bts_loc
        bought = bts_vals[bts_vals[:, loc["bought_or_sold"]] == Candle.BOUGHT]
        # get the index ranges of each bought->sold spans
        bought_ranges = (
            bought[:, loc["next_sold_ofs"]] - bought[:, loc["ohlc_ofs"]]
        ).astype("int64")
        if self.bt_ng == "loop_candles":
            try:
                args = [df_vals, bought, bought_ranges, bts_vals]
                bts_vals = _loop_candles_select_triggers(self, *args)
            except TypeError as e:
                raise OperationalException(e)
        elif self.bt_ng == "chunked":
            bts_vals = _chunked_select_triggers(
                self, df_vals, bought, bought_ranges, bts_vals
            )
        elif self.bt_ng == "loop_ranges":
            args = [df_vals, bought, bought_ranges, bts_vals]
            bts_vals = _loop_ranges_select_triggers(self, *args)

        return bts_vals

    def _get_vars(self, df_vals: ndarray, bought: ndarray, bought_ranges) -> Dict:
        stp = self.strategy.get_stoploss()
        v = {
            "roi_enabled": self.roi_enabled,
            "weighted_roi": self.strategy.time_weighted_roi,
            "stoploss_enabled": self.stoploss_enabled,
            "trailing_enabled": self.trailing_enabled,
            "roi_or_trailing": self.roi_enabled or self.trailing_enabled,
            "stoploss_or_trailing": self.stoploss_enabled or self.trailing_enabled,
            "not_position_stacking": not self.position_stacking,
            "sl_positive": stp.trailing_stop_positive or 0.0,
            "sl_positive_not_null": stp.trailing_stop_positive is not None,
            "sl_offset": stp.trailing_stop_positive_offset,
            "sl_only_offset": stp.trailing_only_offset_is_reached,
            "stoploss": abs(stp.stoploss),
            "stake_amount": self.strategy.stake,
            "fee": self.fee,
            # columns of the trigger array which stores all the calculations
            "col_names": ["trigger_ofs", "trigger_date", "trigger_bought_ofs",],
            # columns of the trg array which stores the calculation of each loop
            "trg_names": [],
            # the number of columns for the shape of the trigger range
            "trg_n_cols": 0,
            "bought_ranges": bought_ranges,
            "high_low": df_vals[:, self.df_loc["high_low"]],
        }
        v["calc_offset"] = v["sl_positive"] != 0.0 or v["sl_only_offset"]

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

        # trailing
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
                "ohlc_open": df_vals[:, df_loc["open"]],
                "ohlc_low": df_vals[:, df_loc["low"]],
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

        fl_dict = {
            k: v[k]
            for k in (
                "ohlc_open",
                "ohlc_low",
                "ohlc_high",
                "bopen",
                "roi_vals",
                "high_low",
                "stake_amount",
            )
        }
        fl_dict.update(
            {f"col_{k}": v["triggers"][:, n] for k, n in v["col"].__dict__.items()}
        )
        update_tpdict(fl_dict.keys(), fl_dict.values(), Float64Cols)

        it_dict = {
            k: v[k]
            for k in ("bofs", "bsold", "ohlc_date", "bought_ranges", "trg_roi_idx",)
        }
        update_tpdict(it_dict.keys(), it_dict.values(), Int64Cols)

        fl = {k: v[k] for k in ("fee", "stoploss", "sl_positive", "sl_offset",)}
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

    def _assign_triggers(
        self, df: DataFrame, bought: DataFrame, triggers: ndarray, col_names: List[str]
    ) -> DataFrame:
        """ Used by chunked and loop_ranges """
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
        """ Used by loop_candles engine """
        # set the index to the offset and add the columns to set the stoploss
        # data points on the relevant boughts
        loc = self.bts_loc
        bts_vals = add_columns(bts_vals, loc, col_names)
        col_idx = [loc[c] for c in col_names]
        bts_vals[:, col_idx] = nan

        # loop over col names and assign column by column because the df doesn't have the same
        # order as the triggers ndarray
        mask = bts_vals[:, loc["bought_or_sold"]] == Candle.BOUGHT
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

    def _calc_pairs_offsets(
        self, df: DataFrame, group="pair", return_ofs=False
    ) -> ndarray:
        """ The offset is calculated as the cumulative length of concatenated
        pairs, minus the startup period, such that the first candle of the
        merged df is always 0
        """
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

    def _filter_roi(self) -> Tuple[Dict[int, int], List[float]]:
        # ensure roi dict is sorted in order to always overwrite
        # with the latest duplicate value when rounding to timeframes
        # NOTE: make sure to sort numerically

        minimal_roi = self.strategy.minimal_roi
        try:
            assert (np.fromiter(minimal_roi, dtype=float) >= 0).all()
        except AssertionError:
            raise OperationalException(
                "ROI config has negative timeouts, can't sell in the past!"
            )
        sorted_minimal_roi = {k: minimal_roi[k] for k in sorted(minimal_roi, key=float)}
        roi_timeouts = self._round_roi_timeouts(list(sorted_minimal_roi.keys()))
        roi_values = []
        appended_roi_values = set()
        for k, v in sorted_minimal_roi.items():
            itk = int(k)
            # when we round, keys can overlap, so we only add the first in the loop
            if itk in roi_timeouts.values() and itk not in appended_roi_values:
                roi_values.append(v)
                appended_roi_values.add(itk)
        try:
            assert len(roi_timeouts) == len(roi_values)
        except AssertionError:
            print("minimal roi", minimal_roi)
            print("sorted roi", sorted_minimal_roi)
            print("roi_timeouts", roi_timeouts)
            print("roi values", roi_values)
            print("length of filtered roi timeouts did not match roi values")
            exit()
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
                    & isin(
                        bts_vals[:, bts_loc["next_sold_ofs"]], self.pairs_ohlc_ofs_end
                    )
                )
            ]
            # assert (events_buy[:, bts_loc["stake_amount"]] != 0).all()
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
                    np.isin(
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
                    self.pairs_ohlc_ofs_end,
                    invert=True,
                )
            ]
            events_sell = bts_vals[
                bts_vals[:, bts_loc["bought_or_sold"]] == Candle.SOLD
            ]
        self.bts_loc = bts_loc
        return events_buy, events_sell

    def split_events_stack(self, bts_vals: ndarray, ohlc_vals: ndarray):
        """"""
        bts_loc = self.bts_loc
        if self.any_trigger:
            events_buy = bts_vals[
                (bts_vals[:, bts_loc["bought_or_sold"]] == Candle.BOUGHT)
                # exclude the last boughts that are not stoploss and which next sold is
                # END sold candle
                & ~(
                    isnan(bts_vals[:, bts_loc["trigger_ofs"]])
                    & isin(
                        bts_vals[:, bts_loc["next_sold_ofs"]], self.pairs_ohlc_ofs_end,
                    )
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
                # a candle with trigger can never be a sold candle
                | isfinite(bts_vals[:, bts_loc["trigger_ofs"]])
            ]
            events_sell_repeats = ones(len(events_sell), dtype=int)
            # repeated next_sold indexes are always sequential, so can repeat them
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
                    self.pairs_ohlc_ofs_end,
                    invert=True,
                )
            ]
            nso, sold_repeats = unique(
                events_buy[:, bts_loc["next_sold_ofs"]], return_counts=True
            )
            events_sell = bts_vals[isin(bts_vals[:, bts_loc["ohlc_ofs"]], nso)]
            events_sell = repeat(events_sell, sold_repeats, axis=0)
        self.bts_loc = bts_loc
        return (events_buy, events_sell)

    @property
    def no_signals_optimization(self):
        return "buy" not in self.spaces and "sell" not in self.spaces

    def _vbt_get_ohlcv(self, processed: Dict[str, DataFrame]) -> OHLCV:
        vbt_processed = (
            load(self.vbt_processed_file)
            if os.path.exists(self.vbt_processed_file)
            else None
        )
        if vbt_processed is None:
            pairs = np.array(list(processed.keys()))
            window = self.timeframe_wnd
            op = []
            hi = []
            lo = []
            cl = []
            vol = []
            sp = []
            for df in processed.values():
                df.set_index("date", inplace=True, drop=False)
                op.append(df["open"])
                hi.append(df["high"])
                lo.append(df["low"])
                cl.append(df["close"])
                vol.append(df["volume"])
                spread = calc_skewed_spread(
                    hi[-1].values,
                    lo[-1].values,
                    cl[-1].values,
                    vol[-1].values,
                    window,
                    None,
                )
                sp.append(Series(spread, index=df.index))
            op = concat(op, axis=1)
            hi = concat(hi, axis=1)
            lo = concat(lo, axis=1)
            cl = concat(cl, axis=1)
            vol = concat(vol, axis=1)
            sp = concat(sp, axis=1)
            vbt_processed = OHLCV(pairs, op, hi, lo, cl, vol, sp)
            dump(vbt_processed, self.vbt_processed_file)
        return vbt_processed

    def vbt_backtest(self, processed: Dict[str, DataFrame], **kwargs,) -> DataFrame:
        self.min_date = kwargs["start_date"]
        self.max_date = kwargs["end_date"]
        pairs_map = {}
        buys = []
        sells = []
        amount = []
        for n, (pair, df) in enumerate(processed.items()):
            meta = {"pair": pair}
            try:
                df["buy"].values[:] = 0
                df["sell"].values[:] = 0
            # ignore if cols are not present
            except KeyError:
                df["buy"] = 0
                df["sell"] = 0
                assign = True
            df = self.strategy.advise_buy(df, meta)
            df = self.strategy.advise_sell(df, meta)
            idx = df.index
            df.set_index("date", inplace=True, drop=False)
            pairs_map[n] = pair
            buys.append(df["buy"])
            sells.append(df["sell"])
            amount.append(df["stake_amount"])
            df.set_index(idx, inplace=True)

        ohlcv = self._vbt_get_ohlcv(processed)
        buy_sigs = concat(buys, axis=1)
        sell_sigs = concat(sells, axis=1)
        amount = concat(amount, axis=1)

        assert all(tuple(pairs_map.values()) == ohlcv.pairs)
        roi_timeouts, roi_values = self._filter_roi()
        stp = self.strategy.get_stoploss().__dict__
        stp["stoploss"] = abs(stp["stoploss"])
        stp = StoplossConfigJit(**stp)

        logger.debug("creating context")
        # initialize sell context to
        ctx = Context(
            pairs=nb.typed.List(pairs_map.values()),
            date=ohlcv.open.index.values,
            buys=buy_sigs.values.astype(float),
            sells=sell_sigs.values.astype(float),
            op=ohlcv.open.values,
            hi=ohlcv.high.values,
            lo=ohlcv.low.values,
            cl=ohlcv.close.values,
            slippage=ohlcv.spread.values,
            slp_window=self.timeframe_wnd,
            fees=self.fee,
            stop_config=stp,
            amount=amount.values,
            min_buy_value=self.min_stake,
            min_sell_value=self.min_stake / 2,
            # the keys are the rounded timeframes
            inv_roi_timeouts=np.array(list(roi_timeouts.keys()), dtype=int)[::-1],
            inv_roi_values=np.array(roi_values, dtype=float)[::-1],
            cash_now=self.max_staked,
        )

        logger.debug("simulating trades")
        trades = simulate_trades(ctx)
        results = DataFrame.from_records(trades)

        logger.debug("casting tpes")
        # upcast types
        results["pair"].replace(pairs_map, inplace=True)
        results["open_date"] = results["open_date"].astype("datetime64[ns, UTC]")
        results["close_date"] = results["close_date"].astype("datetime64[ns, UTC]")
        results["sell_reason"].replace(self.sell_type_map, inplace=True)
        results["trade_duration"] = self.get_trade_duration(results["trade_duration"])

        logger.debug("returning results")
        # print(sell_reason_map)
        # print(trades["entry_idx"]
        #         .replace(sell_reason_map))
        # for k in results:
        #     assert np.isfinite(results[k].values).all()
        return results

    def vectorized_backtest(
        self, processed: Dict[str, DataFrame], **kwargs,
    ) -> DataFrame:
        """ NOTE: can't have default values as arguments since it is an overridden function
        """
        self.min_date = kwargs["start_date"]
        self.max_date = kwargs["end_date"]

        logger.debug("merging pairs signals")
        if self.no_signals_optimization:
            if self.merged_df is not None:
                df = self.merged_df.copy()
            else:
                df = self.merge_pairs_df(processed)
                self.merged_df = df.copy()
        else:
            df = self.merge_pairs_df(processed)

        logger.debug("setting bought and sold candles")
        df_vals, empty = self.bought_or_sold(df)

        # date = df_vals[:, self.df_loc["date"]]
        # high = df_vals[:, self.df_loc["high"]]
        # low = df_vals[:, self.df_loc["low"]]
        # close = df_vals[:, self.df_loc["close"]]
        # open = df_vals[:, self.df_loc["open"]]
        # volume = df_vals[:, self.df_loc["volume"]]
        # print(volume[:100])
        if empty:  # if no bought signals
            logger.debug("returning empty results, no bought signals")
            return self.empty_results

        logger.debug("applying post processing")
        df_vals = self.post_process(df_vals, self.pairs_offset)

        logger.debug("setting sold candles")
        bts_vals = self.set_sold(df_vals)

        if not len(bts_vals):
            logger.debug("returning empty results before triggers")
            return self.empty_results

        if self.any_trigger:
            logger.debug("setting triggers")
            bts_vals = self.set_triggers(df_vals, bts_vals)

        if len(bts_vals) < 1:
            logger.debug("returning empty results after triggers")
            return self.empty_results

        if dbg:
            self.events = as_df(
                bts_vals, self.bts_loc, bts_vals[:, self.bts_loc["ohlc_ofs"]]
            )

        logger.debug("splitting events")
        events_buy, events_sell = (
            self.split_events(bts_vals)
            if not self.position_stacking
            else self.split_events_stack(bts_vals, df_vals)
        )

        logger.debug("getting results")
        if dbg:
            dbg.bts_loc = self.bts_loc
            dbg._validate_events(events_buy, events_sell)
        return self.get_results(events_buy, events_sell, df_vals)
