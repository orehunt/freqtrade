from enum import Enum
from os import getpid
from typing import (
    Callable,
    Dict,
    List,
    MutableSequence,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numba as nb
import numpy as np
from vectorbt.portfolio.enums import (
    GroupContext,
    NoOrder,
    Order,
    OrderContext,
    SegmentContext,
    SizeType,
)

from freqtrade.optimize.backtest_nb import calc_roi_weight
from freqtrade.optimize.vbt_types import *
from logging import getLogger

logger = getLogger(__name__)
logger.name += f".{getpid()}"


class IntSellType(Enum):
    ROI = 0
    STOP_LOSS = 1
    STOPLOSS_ON_EXCHANGE = 2
    TRAILING_STOP_LOSS = 3
    SELL_SIGNAL = 4
    FORCE_SELL = 5
    EMERGENCY_SELL = 6
    NONE = 7


@njit()
def BacktestResultTuple(
    pair=0,
    profit_percent=0.0,
    profit_abs=0.0,
    open_date=0,
    open_rate=0.0,
    open_fee=0.0,
    close_date=0,
    close_rate=0.0,
    close_fee=0.0,
    amount=0.0,
    trade_duration=0,
    open_at_end=False,
    sell_reason=0,
):
    """ NOTE: this need to return the correct order """
    return (
        pair,
        profit_percent,
        profit_abs,
        open_date,
        open_rate,
        open_fee,
        close_date,
        close_rate,
        close_fee,
        amount,
        trade_duration,
        open_at_end,
        sell_reason,
    )


BacktestResultTupleType = nb.typeof(BacktestResultTuple())


class resultsMetrics(NamedTuple):
    win_ratio: nb.float64[:]
    med_profit: nb.float64[:]
    avg_profit: nb.float64[:]
    total_profit: nb.float64[:]
    trade_ratio: nb.float64[:]
    trade_count: nb.int64[:]
    trade_duration: nb.int64[:]


resultsMetricsFields = tuple(resultsMetrics._fields)


@jitclass(ContextTypeSig)
class Context(object):
    pairs_seq: np.ndarray

    def __init__(
        self,
        # pairs,
        pairs: nb.typed.List,
        date: nb.float64[:],
        # buy/sell signals
        buys: nb.float64[:, :],
        sells: nb.float64[:, :],
        # current ohlc for each col
        op: nb.float64[:, :],
        hi: nb.float64[:, :],
        lo: nb.float64[:, :],
        cl: nb.float64[:, :],
        # slippage for each row/col
        slippage: nb.float64[:, :],
        # lookback of slippage mean in case of nan slippage
        slp_window: nb.int64,
        # scalar fees to apply for each order
        fees: nb.float64,
        stop_config: StoplossConfigJit,
        # amount to use for buy orders
        amount: nb.float64[:, :],
        # inverted roi timeouts
        inv_roi_timeouts: nb.int64[:],
        # inverted roi values
        inv_roi_values: nb.float64[:],
        # minimum buy/sell amount to trigger orders
        min_buy_value: nb.float64,
        min_sell_value: nb.float64,
        # total cash on start
        cash_now: nb.float64,
    ):
        self.pairs = pairs
        self.pairs_seq = np.arange(len(pairs))
        self.orig_seq = self.pairs_seq.copy()

        self.date = date
        self.buys = buys
        self.sells = sells

        self.open = op
        self.high = hi
        self.low = lo
        self.close = cl

        self.slippage = slippage
        self.slp_window = slp_window
        self.fees = fees

        self.stop_config = stop_config

        self.amount = amount
        self.irt = inv_roi_timeouts
        self.irv = inv_roi_values

        self.min_bv = min_buy_value
        self.min_sv = min_sell_value
        self.cash_now = cash_now
        self.cash_start = cash_now

    def update_context(self, i, c, trades):
        # duration in number of candles for the current open order
        self.span = i - trades[c].open_idx
        # current ohlcv
        self.open_r = self.open[i, c]
        self.high_r = self.high[i, c]
        self.low_r = self.low[i, c]
        self.close_r = self.close[i, c]

    def reset_context(self, trades):
        self.cash_now = self.cash_start
        self.span = 0

        self.open_r = np.nan
        self.high_r = np.nan
        self.low_r = np.nan
        self.close_r = np.nan

        for t in trades:
            trades[t].status = 1

    def shuffle_context(self, seq=np.array([], dtype=np.int64)):
        """ Shuffle the order of columns """
        if not len(seq):
            # shuffle is inplace
            np.random.shuffle(self.pairs_seq)
            seq = self.pairs_seq
        self.orig_seq = self.unshuffle_seq(seq)[self.orig_seq]

        self.buys = self.buys[:, seq]
        self.sells = self.sells[:, seq]

        self.open = self.open[:, seq]
        self.high = self.high[:, seq]
        self.low = self.low[:, seq]
        self.close = self.close[:, seq]

        self.amount = self.amount[:, seq]
        self.slippage = self.slippage[:, seq]

    def unshuffle_seq(self, idx=None):
        if idx is None:
            idx = self.pairs_seq
        seq = np.empty_like(idx)
        for n, s in enumerate(idx):
            seq[s] = n
        return seq


@njit()
def clone_context(ctx):
    # NOTE: creating jitclasses from njitted code doesn't support named args
    return Context(
        ctx.pairs,
        ctx.date,
        ctx.buys,
        ctx.sells,
        ctx.open,
        ctx.high,
        ctx.low,
        ctx.close,
        ctx.slippage,
        ctx.slp_window,
        ctx.fees,
        ctx.stop_config,
        ctx.amount,
        ctx.irt,
        ctx.irv,
        ctx.min_bv,
        ctx.min_sv,
        ctx.cash_start,
    )


@jitclass(TradeJitTypeSig)
class TradeJit:
    """ A single trade instance """

    def __init__(self):
        self.status = 1

    def open(
        self,
        open_idx: nb.int64,
        open_price: nb.float64,
        cash: nb.float64,
        cash_now: nb.float64,
        min_cash: nb.float64,
        fees: nb.float64,
        slp: nb.float64,
        stoploss: nb.float64,
    ):
        """
        !!! both open_price and cash can't be 0 !!!
        The open function doesn't check for a minimum size
        """
        self.open_idx = open_idx
        self.open_price = open_price
        self.buy_price = open_price + open_price * slp

        # ensure the order size is at least min_cash
        cash_sized = max(min_cash, cash)
        self.shares_held = cash_sized / self.buy_price
        self.cash_spent = cash_sized + cash_sized * fees
        # skip orders where cash spent exceeds available cash
        if self.cash_spent > cash_now:
            return 0
        self.stoploss_price = open_price * (1.0 - stoploss)
        self.initial_stoploss_price = self.stoploss_price

        self.status = 0
        return self.cash_spent

    def close(
        self,
        close_idx: nb.int64,
        close_price: nb.float64,
        fees,
        slp: nb.float64,
        sell_reason: nb.int64,
        min_cash: nb.float64,
        # when trade is too low value to be closed, replace slippage with forfeit slippage
        forfeit_slp: nb.float64 = 0.33,
    ):
        self.sell_price = close_price - close_price * slp
        cash_value = self.shares_held * self.sell_price
        # don't sell if value is below minimum sell size
        if cash_value < min_cash:
            # print("can't sell, settling trade with adjusted slippage: ", cash_value, min_cash)
            self.sell_price = close_price - close_price * forfeit_slp
            cash_value = self.shares_held * self.sell_price

        self.close_idx = close_idx
        self.sell_reason = sell_reason
        self.cash_returned = cash_value - cash_value * fees

        self.profits = self.cash_returned / self.cash_spent - 1
        self.pnl = self.cash_returned - self.cash_spent

        self.status = 1
        return self.cash_returned

    def profit_at(self, price, fees):
        """ Can be called on open trades, does not use slippage """
        cash_value = self.shares_held * price
        cash_returned = cash_value - cash_value * fees
        return cash_returned / self.cash_spent - 1

    def price_at(self, roi, min_price, fees):
        """ Calc the close_price given a profit ratio """
        roi_rate = (self.buy_price * roi + self.buy_price * (1.0 + fees)) / (1.0 - fees)
        return np.fmax(roi_rate, min_price)

    def update_stoploss(self, price, profit, stp):
        self.stoploss_price = calc_trailing_rate(
            self.stoploss_price, price, profit, stp
        )
        return self.stoploss_price


TradeJitType = nb.typeof(TradeJit())


@njit()
def weighted_roi_is_triggered(i, profit, inv_roi_timeouts, inv_roi_values):
    for t, tm in enumerate(inv_roi_timeouts):
        if tm <= i:
            next_t = t - 1 if t > 0 else t
            roi_w = calc_roi_weight(
                # trade dur
                i,
                # current roi value
                inv_roi_values[t],
                # current roi timeout
                tm,
                # next roi value
                inv_roi_values[next_t],
                # next roi timeout
                inv_roi_timeouts[next_t],
            )
            if profit > roi_w:
                return True, roi_w
    return False, np.nan


@njit()
def calc_trailing_rate(stoploss_price, high_price, high_profit, stp):
    if stp.trailing_stop:
        if not (
            stp.trailing_only_offset_is_reached
            and high_profit < stp.trailing_stop_positive
        ):
            return max(
                # trailing only increases
                stoploss_price,
                # use positive ratio if above positive offset (default > 0) NOTE: strict > 0
                high_price * (1 - stp.trailing_stop_positive)
                if stp.trailing_stop_positive is not None
                and high_profit > stp.trailing_stop_positive_offset
                # otherwise trailing with stoploss ratio
                else high_price * (1 - stp.stoploss),
            )
        return stoploss_price
    return stoploss_price


@njit()
def get_slippage(idx, col, slippage, slp_window, base_slippage=0.033):
    slp = slippage[idx, col]
    if np.isfinite(slp):
        return slp
    else:
        slp = np.nanmean(slippage[idx - slp_window : idx])
        return slp if np.isfinite(slp) else base_slippage


@njit()
def check_for_buy(i, c, ctx, trades: Dict[int, TradeJit]):
    """ Should only be called when cash is available """
    if trades[c].status == 1:
        # buys and sell must not conflict
        if ctx.buys[i, c] and (not ctx.sells[i, c]):
            # index of bought, which is at start of next candle
            bi = i + 1
            # create trade for column
            ctx.cash_now -= trades[c].open(
                open_idx=bi,
                # price is open of next candle
                open_price=ctx.open[bi, c],
                # respect the minimum order size, buy if min_bv > cash_now, the order will fail
                cash=ctx.amount[i, c],
                cash_now=ctx.cash_now,
                min_cash=ctx.min_bv,
                fees=ctx.fees,
                slp=get_slippage(bi, c, ctx.slippage, ctx.slp_window),
                stoploss=ctx.stop_config.stoploss,
            )
            # if trades[c].status == 0:
            #     print("created trade: ", i, ctx.pairs[c])
        # always return true even if trade fails, because:
        # don't need to check sells if trade is closed (== there was no trade)
        # the trade is opened on the next candle, not the one with the signal
        return True
    return False


@njit()
def check_roi_on_open(i, c, ctx, trades):
    if ctx.span > 0:
        open_profit = trades[c].profit_at(ctx.open_r, ctx.fees)
        triggered, _ = weighted_roi_is_triggered(
            ctx.span, open_profit, ctx.irt, ctx.irv
        )
        # sell with roi on open
        if triggered:
            # assert i >= trades[c].open_idx
            # print("roi on open: ", i, ctx.pairs[c], trades[c].status)
            ctx.cash_now += trades[c].close(
                close_idx=i,
                close_price=ctx.open_r,
                fees=ctx.fees,
                slp=get_slippage(i, c, ctx.slippage, ctx.slp_window),
                sell_reason=IntSellType.ROI.value,
                min_cash=ctx.min_sv,
            )
            return trades[c].status
    return False


@njit()
def check_stop_on_open(i, c, ctx, trades):
    # if candle is green, likely sequence is low -> high
    if ctx.close_r > ctx.open_r:
        # check previous stoploss
        stoploss_price = trades[c].stoploss_price
        if ctx.low_r <= stoploss_price:
            # print("stoploss on open: ", i, ctx.pairs[c])
            # assert i >= trades[c].open_idx
            ctx.cash_now += trades[c].close(
                close_idx=i,
                close_price=stoploss_price,
                fees=ctx.fees,
                slp=get_slippage(i, c, ctx.slippage, ctx.slp_window),
                sell_reason=(
                    IntSellType.STOP_LOSS.value
                    if stoploss_price == trades[c].initial_stoploss_price
                    else IntSellType.TRAILING_STOP_LOSS.value
                ),
                min_cash=ctx.min_sv,
            )
            return trades[c].status
    return False


@njit()
def check_roi_on_high(i, c, high_profit, ctx, trades):
    triggered, roi = weighted_roi_is_triggered(ctx.span, high_profit, ctx.irt, ctx.irv)
    if triggered:
        # print("roi on high: ", i, ctx.pairs[c])
        # assert i >= trades[c].open_idx
        ctx.cash_now += trades[c].close(
            close_idx=i,
            # when roi is triggered on high, close_price depends on roi target
            close_price=trades[c].price_at(roi, ctx.low_r, ctx.fees),
            fees=ctx.fees,
            slp=get_slippage(i, c, ctx.slippage, ctx.slp_window),
            sell_reason=IntSellType.ROI.value,
            min_cash=ctx.min_sv,
        )
        return trades[c].status
    return False


@njit()
def check_stop_on_high(i, c, high_profit, ctx, trades):
    stoploss_price = trades[c].update_stoploss(ctx.high_r, high_profit, ctx.stop_config)
    if ctx.low_r <= stoploss_price:
        # print("stoploss on high: ", i, ctx.pairs[c])
        # assert i >= trades[c].open_idx
        ctx.cash_now += trades[c].close(
            close_idx=i,
            close_price=stoploss_price,
            fees=ctx.fees,
            slp=get_slippage(i, c, ctx.slippage, ctx.slp_window),
            sell_reason=(
                IntSellType.STOP_LOSS.value
                if stoploss_price == trades[c].initial_stoploss_price
                else IntSellType.TRAILING_STOP_LOSS.value
            ),
            min_cash=ctx.min_sv,
        )
        return trades[c].status
    return False


@njit()
def check_for_sell(i, c, ctx, trades):
    if ctx.sells[i, c] and not ctx.buys[i, c]:
        # print("sell signal: ", )
        next_i = i + 1
        ctx.cash_now += trades[c].close(
            close_idx=next_i,
            close_price=ctx.open[next_i, c],
            fees=ctx.fees,
            slp=get_slippage(next_i, c, ctx.slippage, ctx.slp_window),
            sell_reason=IntSellType.SELL_SIGNAL.value,
            min_cash=ctx.min_sv,
        )
        return trades[c].status
    return False


@njit()
def create_result(ctx, c, trade: TradeJit, results):
    open_date = ctx.date[trade.open_idx]
    close_date = ctx.date[trade.close_idx]
    # if trade.close_idx < trade.open_idx:
    #     print(ctx.pairs[c], trade.sell_reason)
    # assert trade.close_idx >= trade.open_idx
    # assert close_date > open_date
    results.append(
        BacktestResultTuple(
            pair=c,
            profit_percent=trade.profits,
            profit_abs=trade.pnl,
            open_date=open_date,
            open_rate=trade.buy_price,
            open_fee=ctx.fees,
            close_date=close_date,
            close_rate=trade.sell_price,
            close_fee=ctx.fees,
            amount=trade.shares_held,
            trade_duration=close_date - open_date,
            open_at_end=(not trade.status),
            sell_reason=trade.sell_reason,
        )
    )


@njit()
def shuffle_idx(idx: nb.int64[:], seq: nb.int64[:]):
    """ , seq holds the index to revert to the previous order """
    np.random.shuffle(idx)
    for n, s in enumerate(idx):
        seq[s] = n


@njit()
def iterate(ctx: Context, results, trades):
    open_trades = 0
    col_idx = np.arange(ctx.close.shape[1])
    # loop over dates
    for i, close in enumerate(ctx.close):
        # shuffle col index
        np.random.shuffle(col_idx)
        # loop over assets
        for c in col_idx:
            # terminate if out of cash with no trades open
            if ctx.cash_now < ctx.min_bv and not open_trades:
                break
            # skip unavailable data
            if np.isnan(close[c]):
                continue
            # the buy check returns true if a new buy order
            # is executed, or no buy order is open for the current date/col
            if check_for_buy(i, c, ctx, trades):
                open_trades += not trades[c].status
                continue
            ctx.update_context(i, c, trades)
            # check roi against open if trade is open for more than one candle
            if check_roi_on_open(i, c, ctx, trades):
                open_trades -= 1
                create_result(ctx, c, trades[c], results)
                continue
            if check_stop_on_open(i, c, ctx, trades):
                open_trades -= 1
                create_result(ctx, c, trades[c], results)
                continue
            # check against roi on high
            high_profit = trades[c].profit_at(ctx.high_r, ctx.fees)
            if check_roi_on_high(i, c, high_profit, ctx, trades):
                open_trades -= 1
                create_result(ctx, c, trades[c], results)
                continue
            if check_stop_on_high(i, c, high_profit, ctx, trades):
                open_trades -= 1
                create_result(ctx, c, trades[c], results)
                continue
            # at last check for sell signals
            if check_for_sell(i, c, ctx, trades):
                open_trades -= 1
                create_result(ctx, c, trades[c], results)


@njit(cache=False)
def run_iterations(
    n_samples: int,
    ctx: Context,
    o_func: Callable[[List[BacktestResultTupleType]], Tuple[Tuple[str, float], ...]],
    o_metrics: NamedTuple,
    res_metrics: resultsMetrics,
):
    # NOTE: This function can be trivially parallelized
    cash_start = ctx.cash_start
    results, trades, agg = init_containers(ctx)
    for n in range(n_samples):
        # reset
        del results[:]
        # do sim
        iterate(ctx, results, trades)

        # skip empty results
        if not len(results):
            continue

        # calc and store objective for current sim
        obj = o_func(results)
        for o_n, (_, v) in enumerate(obj):
            o_metrics[o_n][n] = v

        # calc and store metrics for current sim
        calc_sim_metrics(n, res_metrics, results, cash_start)

        # switch order
        # ctx.shuffle_context()
        ctx.reset_context(trades)
        append_results(agg, results)
    return agg


profit_abs_idx = BacktestResultDType.names.index("profit_abs")
profit_percent_idx = BacktestResultDType.names.index("profit_percent")
trade_duration_idx = BacktestResultDType.names.index("trade_duration")


@njit(cache=False)
def calc_sim_metrics(
    i, res_metrics, results: List[BacktestResultTupleType], cash_start
):
    # vars
    profit_abs = np.empty(len(results), dtype=nb.float64)
    profit_percent = np.empty(len(results), dtype=nb.float64)
    w, l, duration_sum = 0, 0, 0

    # process results
    for n, r in enumerate(results):
        if r[profit_abs_idx] > 0:
            w += 1
        else:
            l += 1
        profit_abs[n] = r[profit_abs_idx]
        profit_percent[n] = r[profit_percent_idx]
        duration_sum += r[trade_duration_idx]

    res_metrics.win_ratio[i] = w / (l or 1)
    # NOTE: median as trouble with 0d arrays, so results should be > 0
    res_metrics.med_profit[i] = np.median(profit_abs)
    res_metrics.avg_profit[i] = np.mean(profit_abs)
    res_metrics.total_profit[i] = np.sum(profit_abs)
    res_metrics.trade_ratio[i] = np.mean(profit_percent)
    res_metrics.trade_count[i] = len(results)
    res_metrics.trade_duration[i] = duration_sum / len(results)


@njit()
def float_dict():
    return nb.typed.Dict.empty(key_type=nb.types.unicode_type, value_type=nb.float64)


@njit()
def get_quantile(arr, q=1 / 3):
    """ The average of the bottom, top and middle sets """
    arr = arr[np.isfinite(arr)]
    if len(arr):
        arr.sort()
        ln = len(arr)
        size = int(ln * q)
        if size:
            bot = np.mean(arr[:size])
            top = np.mean(arr[-size:])
            mid = np.mean(arr[size : ln - size])
        else:
            bot = np.mean(arr)
            top = bot
            mid = bot
        return bot, top, mid
    else:
        return np.nan, np.nan, np.nan


@njit()
def obj_quantile(arr, q=1 / 3):
    """ The average of the top set (since objective is minimized, the average worst) """
    arr = arr[np.isfinite(arr)]
    if len(arr):
        arr.sort()
        ln = len(arr)
        size = int(ln * q)
        # the end has the highest (worst) values
        return np.mean(arr[-size:])
    else:
        return np.nan


@njit(cache=False)
def reduce_sims(o_names, o_metrics, res_metrics, qnt):
    red_obj = float_dict()
    red_res = float_dict()
    for n, arr in enumerate(o_metrics):
        red_obj[o_names[n]] = obj_quantile(arr, q=qnt)

    bot, top, mid = get_quantile(res_metrics.win_ratio, q=qnt)
    red_res["win_ratio_bot"] = bot
    red_res["win_ratio_top"] = top
    red_res["win_ratio_mid"] = mid

    bot, top, mid = get_quantile(res_metrics.med_profit, q=qnt)
    red_res["med_profit_bot"] = bot
    red_res["med_profit_top"] = top
    red_res["med_profit_mid"] = mid

    bot, top, mid = get_quantile(res_metrics.avg_profit, q=qnt)
    red_res["avg_profit_bot"] = bot
    red_res["avg_profit_top"] = top
    red_res["avg_profit_mid"] = mid

    bot, top, mid = get_quantile(res_metrics.total_profit, q=qnt)
    red_res["total_profit_bot"] = bot
    red_res["total_profit_top"] = top
    red_res["total_profit_mid"] = mid

    bot, top, mid = get_quantile(res_metrics.trade_ratio, q=qnt)
    red_res["trade_ratio_bot"] = bot
    red_res["trade_ratio_top"] = top
    red_res["trade_ratio_mid"] = mid

    bot, top, mid = get_quantile(res_metrics.trade_count, q=qnt)
    red_res["trade_count_bot"] = bot
    red_res["trade_count_top"] = top
    red_res["trade_count_mid"] = mid

    bot, top, mid = get_quantile(res_metrics.trade_duration, q=qnt)
    red_res["trade_duration_bot"] = bot
    red_res["trade_duration_top"] = top
    red_res["trade_duration_mid"] = mid
    return red_obj, red_res


@njit()
def init_trades(trades_dict, n_cols):
    for c in range(n_cols):
        trades_dict[c] = TradeJit()


@njit()
def init_containers(ctx):
    results = nb.typed.List.empty_list(item_type=BacktestResultTupleType)

    trades = nb.typed.Dict.empty(key_type=nb.int64, value_type=TradeJitType)
    init_trades(trades, ctx.close.shape[1])

    agg = nb.typed.List.empty_list(item_type=BacktestResultArr)
    return results, trades, agg


@njit()
def append_results(*args):
    return


def enable_aggregation():
    global append_results

    @njit()
    def append_results(agg: nb.typed.List, res: nb.types.Array):
        agg.append(np.asarray([v for v in res]))


def sample_simulations(
    ctx: Context,
    n_samples: int,
    o_func: Callable,
    o_names: Tuple[str, ...],
    quantile: float,
    aggregate=False,
):
    o_metrics = NamedTuple(
        typename="o_metrics", fields=((name, nb.float64[:]) for name in o_names)
    )(**{name: np.full(n_samples, np.nan, dtype=float) for name in o_names})
    res_metrics = resultsMetrics(
        **{
            name: np.full(n_samples, np.nan, dtype=float)
            for name in resultsMetricsFields
        }
    )

    if aggregate:
        logger.debug("enabling aggregation")
        enable_aggregation()

    logger.debug("running iterations...")
    agg = run_iterations(n_samples, ctx, o_func, o_metrics, res_metrics)

    logger.debug("reducing simulation metrics...")
    red_obj, red_res = reduce_sims(o_names, o_metrics, res_metrics, quantile)

    logger.debug("converting to dict and returning...")
    # convert typed dicts to python dicts
    return dict(red_obj), dict(red_res), agg


def simulate_trades(
    ctx,
) -> np.ndarray:
    results, trades, _ = init_containers(ctx)
    iterate(ctx, results, trades)

    return np.array(results, dtype=BacktestResultDType)
