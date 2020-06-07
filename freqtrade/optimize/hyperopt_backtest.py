import logging

from joblib import load
from numpy import append, repeat, nan
from pandas import Timedelta, to_timedelta, concat, Series, DataFrame

from freqtrade.data.history import get_timerange
import freqtrade.optimize.hyperopt_backend as backend
from freqtrade.resolvers import ExchangeResolver, StrategyResolver
from freqtrade.configuration import remove_credentials
from freqtrade.data.dataprovider import DataProvider
from freqtrade.optimize.backtesting import Backtesting
from freqtrade.exceptions import OperationalException

from enum import IntEnum

logger = logging.getLogger(__name__)


class Candle(IntEnum):
    BOUGHT = 2
    SOLD = 5
    BOUGHT_AND_STOPLOSS = 11
    STOPLOSS = 17
    NOOP = 0


class HyperoptBacktesting(Backtesting):
    def __init__(self, config):
        if self.config.get("backtesting_engine") == "vectorized":
            self.backtest = self.vectorized_backtest

    def set_params(self, params_dict: Dict[str, Any] = None):
        if self.has_space("buy"):
            self.backtesting.strategy.advise_buy = self.custom_hyperopt.buy_strategy_generator(
                params_dict
            )

        if self.has_space("sell"):
            self.backtesting.strategy.advise_sell = self.custom_hyperopt.sell_strategy_generator(
                params_dict
            )

        if self.has_space("stoploss"):
            self.backtesting.strategy.amounts["stoploss"] = params_dict["stoploss"]

        if self.has_space("trailing"):
            d = self.custom_hyperopt.generate_trailing_params(params_dict)
            self.backtesting.strategy.amounts["trailing_stop"] = d["trailing_stop"]
            self.backtesting.strategy.amounts["trailing_stop_positive"] = d[
                "trailing_stop_positive"
            ]
            self.backtesting.strategy.amounts["trailing_stop_positive_offset"] = d[
                "trailing_stop_positive_offset"
            ]
            self.backtesting.strategy.amounts["trailing_only_offset_is_reached"] = d[
                "trailing_only_offset_is_reached"
            ]

    def backtest_params(
        self,
        raw_params: List[Any] = None,
        iteration=None,
        params_dict: Dict[str, Any] = None,
    ) -> Dict:
        if not params_dict:
            if raw_params:
                params_dict = self._get_params_dict(raw_params)
            else:
                logger.debug("Epoch evaluation didn't receive any parameters")
                return {}
        params_details = self._get_params_details(params_dict)

        self.set_params(params_dict)

        if backend.data:
            processed = backend.data
        else:
            processed = load(self.data_pickle_file)
            backend.data = processed

        min_date, max_date = get_timerange(processed)

        backtesting_results = self.backtesting.backtest(
            processed=processed, start_date=min_date, end_date=max_date,
        )
        return self._get_result(
            backtesting_results,
            min_date,
            max_date,
            params_dict,
            params_details,
            processed,
        )

    def vectorized_backtest(
        self,
        processed: dict[str, DataFrame],
        start_date: arrow.Arrow,
        end_date: arrow.Arrow,
    ) -> DataFrame:
        """ NOTE: can't have default values as arguments since it is an overridden function """
        if len(processed) > 1:
            raise OperationalException(
                "Can only use vectorized backtest with one pair."
            )

        pair = next(processed)
        pair_df = processed[pair]
        pair_df["bought_or_sold"] = pair_df["buy"].shift(-1) - pair_df["sell"].shift(-1)
        pair_df.loc[bought_or_sold.values == 1, "bought_or_sold"] = Candle.BOUGHT
        pair_df.loc[bought_or_sold.values == -1, "bought_or_sold"] = Candle.SOLD
        # add stoploss
        bought = pair_df["bought_or_sold"].values == Candle.BOUGHT
        sold = pair_df["bought_or_sold"].values == Candle.SOLD
        not_bought_or_sold = pair_df["bought_or_sold"].values == Candle.NOOP
        stoploss_rate = pair_df.loc[bought, "open"] * (
            1 - self.config["stoploss"] + 2 * self.fee
        )
        # calc the number of candles after a buy order has been reached using the index diff
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
        # # stoploss is triggered when it is below or equal the low
        stoploss_triggered = pair_df["low"].values <= pair_df["stoploss"].values
        # set stoploss only on candles which are not sold, as those use the open_rate
        pair_df.loc[
            (bought & stoploss_triggered), "bought_or_sold"
        ] = Candle.BOUGHT_AND_STOPLOSS
        pair_df.loc[
            (not_bought_or_sold & stoploss_triggered), "bought_or_sold"
        ] = Candle.STOPLOSS

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
                & ~bos_s1_bought
            )
        ]
        # exclude the first sell and the last buy
        if events["bought_or_sold"].iloc[0] == Candle.SOLD:
            events = events.iloc[1:]
        if events["bought_or_sold"].iloc[-1] == Candle.BOUGHT:
            events = events.iloc[:-1]
        # adjust stoploss of bought_and_stoploss if the buy happens on the previous candle
        # and modify candle to stoploss
        bos = events["bought_or_sold"]
        bos_s1 = bos.shift(1)
        stoploss_after_bought = events.loc[
            (
                # (bos.values == Candle.BOUGHT_AND_STOPLOSS) &
                # (bos.shift(1).values == Candle.BOUGHT),
                bos.values - bos_s1.values
                == 9  # 11 - 2
            )
        ]
        bought_before_stoploss = events.loc[
            (
                # (bos.values == Candle.BOUGHT) &
                # (bos.shift(-1).values == Candle.BOUGHT_AND_STOPLOSS),
                bos.values - bos.shift(-1).values
                == -9  # 2 - 11
            )
        ]
        events.loc[stoploss_after_bought.index, ["bought_or_sold", "stoploss"]] = [
            Candle.STOPLOSS,
            events.loc[bought_before_stoploss.index, "stoploss"].values,
        ]
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
        # choose sell rate depending on sell reason
        bos_sell = events_sell["bought_or_sold"]
        events_sold = events_sell.loc[bos_sell.values == Candle.SOLD]
        events_sell.loc[events_sold.index, "sell_rate"] = events_sold["open"].values
        events_stoploss = events_sell.loc[
            (bos_sell.values == Candle.STOPLOSS)
            | (bos_sell.values == Candle.BOUGHT_AND_STOPLOSS)
        ]
        events_sell.loc[events_stoploss.index, "close_rate"] = events_stoploss[
            "stoploss"
        ].values

        open_rate = events_buy["open"].values
        close_rate = events_sell["sell_rate"].values
        profits = 1 - open_rate / close_rate - 2 * self.fee
        trade_duration = Series(events_sell["date"].values - events_buy["date"].values)
        # replace trade duration of same candle trades with half the timeframe
        half_timeframe_td = Timedelta(self.config["timeframe"]) / 2
        trade_duration.loc[trade_duration == pd.Timedelta(0)] = half_timeframe_td

        results = DataFrame(
            {
                "pair": pair,
                "profit_percent": profits,
                "profit_abs": profits * self.config["stake_amount"],
                "open_time": events_buy["date"].values,
                "close_time": events_sell["date"].values,
                "open_index": events_buy.index.values,
                "close_index": events_sell.index.values,
                "trade_duration": trade_duration.values,
                "open_at_end": False,
                "open_rate": open_rate,
                "close_rate": close_rate,
            }
        )
        # results["pair"] = pair
        # results["profit_percent"] = profits
        # results["profit_abs"] = profits * self.config["stake_amount"]
        # results["open_time"] = events_buy["date"].values
        # results["close_time"] = events_sell["date"].values
        # results["open_index"] = events_buy.index.values
        # results["close_index"] = events_sell.index.values
        # results["trade_duration"] = trade_duration.values
        # results["open_at_end"] = False
        # results["open_rate"] = open_rate
        # results["close_rate"] = close_rate

        avg_profit = profits.mean() * 100.0
        duration = trade_duration.dt.seconds.mean() / 60
        profits_sum = profits.values.sum()
        profit = profits_sum * 100.0
        total_profit = profits_sum * self.config["stake_amount"]
        trade_count = len(open_rate)
