import logging

from joblib import load
import arrow
from typing import Dict, Any, List
from numpy import append, repeat, nan
from pandas import Timedelta, to_timedelta, concat, Series, DataFrame, Index

from freqtrade.data.history import get_timerange
import freqtrade.optimize.hyperopt_backend as backend
from freqtrade.resolvers import ExchangeResolver, StrategyResolver
from freqtrade.configuration import remove_credentials
from freqtrade.data.dataprovider import DataProvider
from freqtrade.optimize.backtesting import Backtesting, BacktestResult
from freqtrade.exceptions import OperationalException
from freqtrade.strategy.interface import SellType

from enum import IntEnum

logger = logging.getLogger(__name__)


class Candle(IntEnum):
    BOUGHT = 2
    SOLD = 5
    BOUGHT_AND_STOPLOSS = 11
    STOPLOSS = 17
    NOOP = 0


class HyperoptBacktesting(Backtesting):

    empty_results = DataFrame.from_records([], columns=BacktestResult._fields)

    def __init__(self, config):
        if config.get("backtesting_engine") == "vectorized":
            self.backtest = self.vectorized_backtest
            self.beacktesting_engine = "vectorized"
        super().__init__(config)

    def set_stoploss(
        self, bought_index: Index, stoploss_rate: Series, pair_df: DataFrame
    ) -> Series:
        """ Update dataframe stoploss col from Series of index matching stoploss rates """
        # reduce stoploss rate to bought_df index
        stoploss_rate = stoploss_rate.loc[bought_index]
        # calc the number of candles for which to apply stoploss using the index diff
        repeats = stoploss_rate.index.values[1:] - stoploss_rate.index.values[:-1]
        # repeats is one element short, append the last that will be repeated until pair_df max idx
        # add 1 to compensate for the shift of bought_or_sold
        repeats = append(repeats, pair_df.index[-1] - stoploss_rate.index[-1] + 1)
        stoploss_arr = repeat(stoploss_rate, repeats)
        # prepend the slice without signals
        no_sigs = Series(
            data=nan,
            index=pair_df.iloc[pair_df.index[0] : stoploss_rate.index[0]].index,
        )
        pair_df["stoploss"] = concat([no_sigs, stoploss_arr]).values

    @staticmethod
    def get_events(pair_df: DataFrame) -> DataFrame:
        """ return the dataframe reduced to only actionable candles """
        # include rows where there is an action
        events = pair_df.loc[pair_df["bought_or_sold"].values != Candle.NOOP]
        bos = events["bought_or_sold"]
        bos_s1 = bos.shift(1)
        bos_s1_bought = bos_s1.values == Candle.BOUGHT
        # filter by
        events = events.loc[
            # all candles with bought_and_stoploss triggered
            (bos.values == Candle.BOUGHT_AND_STOPLOSS)
            |
            # sold and stoploss candles where the previous is bought
            (
                ((bos.values == Candle.SOLD) | (bos.values == Candle.STOPLOSS))
                & bos_s1_bought
                # bos.values - bos_s1.values == 3 # 5 - 2
            )
            | (
                # bought candles where the previous is not bought
                (bos.values == Candle.BOUGHT)
                # NOTE: every bought candle here is an actual event
                # and all duplicate actions should already have been filtered out
                # so it is not necessary to check prev candles to not be already bought
                # & ~bos_s1_bought
            )
        ]
        # exclude the first sell and the last buy
        if events["bought_or_sold"].iloc[0] == Candle.SOLD or \
           events["bought_or_sold"].iloc[0] == Candle.STOPLOSS:
            events = events.iloc[1:]
        if events["bought_or_sold"].iloc[-1] == Candle.BOUGHT:
            events = events.iloc[:-1]
        return events

    def get_results(self, pair: str, events_buy: DataFrame, events_sell: DataFrame) -> DataFrame:
        # choose sell rate depending on sell reason and set sell_reason
        bos_sell = events_sell["bought_or_sold"]
        events_sold = events_sell.loc[bos_sell.values == Candle.SOLD]
        # add new columns with reindex to allow multi col assignments of new columns
        events_sell = events_sell.reindex(
            columns=[*events_sell.columns, "close_rate", "sell_reason"]
        )
        events_sell.loc[events_sold.index, ["close_rate", "sell_reason"]] = [
            events_sold["open"].values,
            SellType.SELL_SIGNAL,
        ]
        events_stoploss = events_sell.loc[
            (bos_sell.values == Candle.STOPLOSS)
            | (bos_sell.values == Candle.BOUGHT_AND_STOPLOSS)
        ]
        events_sell.loc[events_stoploss.index, ["close_rate", "sell_reason"]] = [
            events_stoploss["stoploss"].values,
            SellType.STOP_LOSS,
        ]

        open_rate = events_buy["open"].values
        close_rate = events_sell["close_rate"].values
        profits = 1 - open_rate / close_rate - 2 * self.fee
        trade_duration = Series(events_sell["date"].values - events_buy["date"].values)
        # replace trade duration of same candle trades with half the timeframe reduce to minutes
        half_timeframe_td = Timedelta(self.config["timeframe"]) / 2
        trade_duration.loc[trade_duration == Timedelta(0)] = half_timeframe_td

        results = DataFrame(
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
        return results

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
        # set bought candles
        pair_df["bought_or_sold"] = pair_df["buy"].shift(-1) - pair_df["sell"].shift(-1)
        pair_df.loc[
            pair_df["bought_or_sold"].values == 1, "bought_or_sold"
        ] = Candle.BOUGHT
        # skip if no valid bought candles are found
        bought_df = pair_df.loc[pair_df["bought_or_sold"].values == Candle.BOUGHT]
        if len(bought_df) < 1:
            return self.empty_results
        # set sold candles
        pair_df.loc[
            pair_df["bought_or_sold"].values == -1, "bought_or_sold"
        ] = Candle.SOLD
        # add stoploss (it is a negative value) on all the bought candles
        stoploss_rate = bought_df["open"] * (1 + self.config["stoploss"] + 2 * self.fee)
        # 1; check for the case where stoploss is triggered on same bought candle
        bought_and_stoploss = bought_df.loc[
            bought_df["low"].values <= stoploss_rate
        ].index
        # switch those bought candle to bought and stoploss
        pair_df.loc[bought_and_stoploss, "bought_or_sold"] = Candle.BOUGHT_AND_STOPLOSS
        # update bought_df excluding bought and stoploss candles by index
        bought_df.drop(bought_and_stoploss, inplace=True)
        # 2; noop index adjacent bought candles if no position stacking, by subtracting indexes and matching against 1
        # (we already removed candles where stoploss is triggered on the same candle)
        bought_adjacent = (
            bought_df.iloc[1:]
            .loc[bought_df.index.values[1:] - bought_df.index.values[:-1] == 1]
            .index
        )
        pair_df.loc[bought_adjacent, "bought_or_sold"] = Candle.NOOP
        # update bought_df excluding adjacent bought candles by index
        bought_df.drop(bought_adjacent)

        # stoploss f bought candles will be repeated only for non adjacent boughts
        # and not bought_and_stoploss candles
        self.set_stoploss(bought_df.index, stoploss_rate, pair_df)

        # stoploss is triggered when low is below or equal
        stoploss_triggered = pair_df["low"].values <= pair_df["stoploss"].values
        # set the potential stoplosses as if no sell signal was triggered
        pair_df.loc[
            (pair_df["bought_or_sold"].values != Candle.SOLD) & stoploss_triggered,
            "bought_or_sold",
        ] = Candle.STOPLOSS
        # select all possible events
        possible = pair_df.loc[pair_df["bought_or_sold"].values != Candle.NOOP]
        # remove subsequent bought candles as neither stoploss nor signal triggered
        # or bought_and_stoploss candles preceded by a bought candles, as that bought
        # would still be active
        p_bos = possible["bought_or_sold"]
        stale_bought = possible.loc[
            (p_bos.shift(1).values == Candle.BOUGHT) &
            (
                (p_bos.values == Candle.BOUGHT) |
                (p_bos.values == Candle.BOUGHT_AND_STOPLOSS)
            )
        ].index
        pair_df.loc[stale_bought, "bought_or_sold"] = Candle.NOOP
        # after removing bought candles with no sell trigger reset stoploss rate
        self.set_stoploss(
            pair_df.loc[pair_df["bought_or_sold"].values == Candle.BOUGHT].index,
            stoploss_rate,
            pair_df,
        )

        # split buys and sell and calc results
        events = self.get_events(pair_df)
        bos = events["bought_or_sold"]
        bos_s1 = bos.shift(1)
        events_buy = events.loc[
            (bos_s1.values != Candle.BOUGHT)
            & (
                (bos.values == Candle.BOUGHT)
                | (bos.values == Candle.BOUGHT_AND_STOPLOSS)
            )
        ]
        events_sell = events.loc[
            (bos.values == Candle.SOLD)
            | (bos.values == Candle.STOPLOSS)
            | (bos.values == Candle.BOUGHT_AND_STOPLOSS)
        ]
        assert len(events_buy) == len(events_sell)
        return self.get_results(pair, events_buy, events_sell)
