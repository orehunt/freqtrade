from enum import Enum
from freqtrade.optimize.backtesting import BacktestResult
from functools import partial
from typing import (
    List,
    Callable,
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
from vectorbt.portfolio.nb import create_order_nb

from freqtrade.optimize.backtest_nb import calc_roi_close_rate, calc_roi_weight
from freqtrade.optimize.vbt_types import *


class IntSellType(Enum):
    ROI = 0
    STOP_LOSS = 1
    STOPLOSS_ON_EXCHANGE = 2
    TRAILING_STOP_LOSS = 3
    SELL_SIGNAL = 4
    FORCE_SELL = 5
    EMERGENCY_SELL = 6
    NONE = 7

@jitclass(BacktestResultTypeSig)
class BacktestResultJit(object):
    def __init__(self):
        pass

    def get(
        self,
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


BacktestResultJitType = nb.typeof(BacktestResultJit().get())


@jitclass(ContextTypeSig)
class Context(object):
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

    def update_context(self, i, c, trades):
        # duration in number of candles for the current open order
        self.span = i - trades[c].open_idx
        # current ohlcv
        self.open_r = self.open[i, c]
        self.high_r = self.high[i, c]
        self.low_r = self.low[i, c]
        self.close_r = self.close[i, c]


@jitclass(TradeJitTypeSig)
class TradeJit:
    """ A single trade instance """

    def __init__(self):
        self.status = 1
        pass

    def open(
        self,
        open_idx: nb.int64,
        open_price: nb.float64,
        cash: nb.float64,
        min_cash: nb.float64,
        fees: nb.float64,
        slp: nb.float64,
        stoploss: nb.float64,
    ):
        """ !!! both open_price and cash can't be 0 !!! """
        # skip orders that don't meet minimum order size
        if cash < min_cash:
            return 0
        self.open_idx = open_idx
        self.open_price = open_price
        self.buy_price = open_price + open_price * slp

        self.shares_held = cash / self.buy_price
        self.cash_spent = cash + cash * fees
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
    ):
        sell_price = close_price - close_price * slp
        cash_value = self.shares_held * sell_price
        # don't sell if value is below minimum sell size
        if cash_value < min_cash:
            return 0

        self.close_idx = close_idx
        self.sell_price = sell_price
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


@njit(cache=True)
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


@njit(cache=True)
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


@njit(cache=True)
def get_slippage(idx, col, slippage, slp_window):
    slp = slippage[idx, col]
    if np.isfinite(slp):
        return slp
    else:
        slp = np.nanmean(slippage[idx - slp_window : idx])
        return slp if np.isfinite(slp) else 0


@njit(cache=True)
def check_for_buy(i, c, ctx, trades):
    """ Should only be called when cash is available """
    trade_is_closed = trades[c].status == 1
    if (
        trade_is_closed
        # buys and sell must not conflict
        and ctx.buys[i, c]
        and (not ctx.sells[i, c])
        # cash at hand must be above minimum tradeable
        and ctx.cash_now > ctx.min_bv
    ):
        # index of bought, which is at start of next candle
        bi = i + 1
        # create trade for column
        ctx.cash_now -= trades[c].open(
            open_idx=bi,
            # price is open of next candle
            open_price=ctx.open[bi, c],
            # don't buy over available cash
            cash=min(ctx.cash_now, ctx.amount[i, c]),
            min_cash=ctx.min_bv,
            fees=ctx.fees,
            slp=get_slippage(bi, c, ctx.slippage, ctx.slp_window),
            stoploss=ctx.stop_config.stoploss,
        )
        return True
    return trade_is_closed


@njit(cache=True)
def check_roi_on_open(i, c, ctx, trades):
    if ctx.span > 0:
        open_profit = trades[c].profit_at(ctx.open_r, ctx.fees)
        triggered, _ = weighted_roi_is_triggered(
            ctx.span, open_profit, ctx.irt, ctx.irv
        )
        # sell with roi on open
        if triggered:
            # print('open roi, size: ', trades[c].shares_held)
            ctx.cash_now += trades[c].close(
                close_idx=i,
                close_price=ctx.open_r,
                fees=ctx.fees,
                slp=get_slippage(i, c, ctx.slippage, ctx.slp_window),
                sell_reason=IntSellType.ROI.value,
                min_cash=ctx.min_sv,
            )
            return True
    return False


@njit(cache=True)
def check_stop_on_open(i, c, ctx, trades):
    if ctx.close_r <= ctx.open_r:
        # check previous stoploss
        stoploss_price = trades[c].stoploss_price
        if ctx.low_r <= stoploss_price:
            # print('previous trailing: ', )
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
            return True
    return False


@njit(cache=True)
def check_roi_on_high(i, c, high_profit, ctx, trades):
    triggered, roi = weighted_roi_is_triggered(ctx.span, high_profit, ctx.irt, ctx.irv)
    if triggered:
        # print("high roi: ", )
        ctx.cash_now += trades[c].close(
            close_idx=i,
            # when roi is triggered on high, close_price depends on roi target
            close_price=trades[c].price_at(roi, ctx.low_r, ctx.fees),
            fees=ctx.fees,
            slp=get_slippage(i, c, ctx.slippage, ctx.slp_window),
            sell_reason=IntSellType.ROI.value,
            min_cash=ctx.min_sv,
        )
        return True
    return False


@njit(cache=True)
def check_stop_on_high(i, c, high_profit, ctx, trades):
    stoploss_price = trades[c].update_stoploss(ctx.high_r, high_profit, ctx.stop_config)
    if ctx.low_r <= stoploss_price:
        # print("updated trailing: ", )
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
        return True
    return False


@njit(cache=True)
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
        return True
    return False


@njit(cache=False)
def create_result(ctx, c, trade: TradeJit, results):
    open_date = ctx.date[trade.open_idx]
    close_date = ctx.date[trade.close_idx]
    res = BacktestResultJit()
    val = res.get(
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
    results.append(val)


@njit(cache=False)
def iterate(ctx, results, trades):
    for i, row in enumerate(ctx.close):
        for c, clo in enumerate(row):
            # skip unavailable data
            if np.isnan(clo):
                continue
            # the buy check returns true if a new buy order
            # is executed, or no buy order is open for the current date/col
            if check_for_buy(i, c, ctx, trades):
                continue
            ctx.update_context(i, c, trades)
            # check roi against open if trade is open for more than one candle
            if check_roi_on_open(i, c, ctx, trades):
                create_result(ctx, c, trades[c], results)
                continue
            if check_stop_on_open(i, c, ctx, trades):
                create_result(ctx, c, trades[c], results)
                continue
            # check against roi on high
            high_profit = trades[c].profit_at(ctx.high_r, ctx.fees)
            if check_roi_on_high(i, c, high_profit, ctx, trades):
                create_result(ctx, c, trades[c], results)
                continue
            if check_stop_on_high(i, c, high_profit, ctx, trades):
                create_result(ctx, c, trades[c], results)
                continue
            # at last check for sell signals
            if check_for_sell(i, c, ctx, trades):
                create_result(ctx, c, trades[c], results)


@njit(cache=True)
def init_trades(trades_dict, n_cols):
    for c in range(n_cols):
        trades_dict[c] = TradeJit()

def simulate_trades(ctx) -> np.ndarray:
    results = nb.typed.List.empty_list(item_type=BacktestResultJitType)
    trades = nb.typed.Dict.empty(key_type=nb.int64, value_type=TradeJitType)
    init_trades(trades, ctx.close.shape[1])
    iterate(ctx, results, trades)
    return np.array(results, dtype=BacktestResultDType)
