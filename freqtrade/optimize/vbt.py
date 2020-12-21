from enum import Enum
from functools import partial
from typing import Callable, MutableSequence, NamedTuple, Optional, Sequence, Union

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


def njit(cache: bool = False) -> Callable:
    return partial(nb.njit, cache=cache)


class IntSellType(Enum):
    ROI = 0
    STOP_LOSS = 1
    STOPLOSS_ON_EXCHANGE = 2
    TRAILING_STOP_LOSS = 3
    SELL_SIGNAL = 4
    FORCE_SELL = 5
    EMERGENCY_SELL = 6
    NONE = 7


@nb.experimental.jitclass(
    [
        ("stoploss", nb.float64),
        ("trailing_stop", nb.types.boolean),
        ("trailing_stop_positive", nb.types.Optional(nb.float64)),
        ("trailing_stop_positive_offset", nb.float64),
        ("trailing_only_offset_is_reached", nb.types.boolean),
    ]
)
class StoplossConfigJit(object):
    def __init__(
        self,
        stoploss: nb.float64,
        trailing_stop: bool = False,
        trailing_stop_positive: Optional[float] = None,
        trailing_stop_positive_offset: float = 0.0,
        trailing_only_offset_is_reached: bool = False,
    ):
        self.stoploss = stoploss
        self.trailing_stop = trailing_stop
        self.trailing_stop_positive = trailing_stop_positive
        self.trailing_stop_positive_offset = trailing_stop_positive_offset
        self.trailing_only_offset_is_reached = trailing_only_offset_is_reached


SellReason = nb.typeof(nb.typed.Dict.empty(key_type=nb.int64, value_type=nb.float64))


# @nb.experimental.jitclass([("sell_reason", SellReason)], inline=True)
@nb.experimental.jitclass(
    [
        ("sell_reason", SellReason),
        ("buys", nb.float64[:, :]),
        ("sells", nb.float64[:, :]),
        ("last_br", nb.float64[:]),
        ("last_bi", nb.int64[:]),
        ("slippage", nb.float64[:, :]),
        ("open", nb.float64[:, :]),
        ("high", nb.float64[:, :]),
        ("low", nb.float64[:, :]),
        ("close", nb.float64[:, :]),
        ("slip_window", nb.int64),
        ("fees", nb.float64),
        ("stop_rates", nb.float64[:]),
        ("stop_config", nb.typeof(StoplossConfigJit(0))),
        ("amount", nb.float64[:, :]),
        ("irt", nb.int64[:]),
        ("irv", nb.float64[:]),
        ("min_bv", nb.float64),
        ("min_sv", nb.float64),
        ("br", nb.float64),
        ("bi", nb.int64),
        ("span", nb.int64),
        ("open_r", nb.float64),
        ("high_r", nb.float64),
        ("low_r", nb.float64),
        ("close_r", nb.float64),
    ],
)
class Context(object):
    def __init__(
        self,
        # buy/sell signals
        buys: nb.float64[:, :],
        sells: nb.float64[:, :],
        # last buy rate for each col
        buy_rate: nb.float64[:],
        # buy index of the last buy order for each col
        buy_row: nb.int64,
        # current ohlc for each col
        op: nb.float64[:, :],
        hi: nb.float64[:, :],
        lo: nb.float64[:, :],
        cl: nb.float64[:, :],
        # slippage for each row/col
        slippage: nb.float64[:, :],
        # lookback of slippage mean in case of nan slippage
        slip_window: nb.int64,
        # scalar fees to apply for each order
        fees: nb.float64,
        # stoploss rate of the last order for each col
        stop_rates: nb.float64[:],
        stop_config: StoplossConfigJit,
        # amount to use for buy orders
        amount: nb.float64[:, :],
        # dict of buy order idx to IntSellType enum
        sell_reason: nb.typed.Dict,
        # inverted roi timeouts
        inv_roi_timeouts: nb.int64[:],
        # inverted roi values
        inv_roi_values: nb.float64[:],
        # minimum buy/sell amount to trigger orders
        min_buy_value: nb.float64,
        min_sell_value: nb.float64,
    ):
        self.buys: nb.float64[:, :] = buys
        self.sells = sells

        self.last_br = buy_rate
        self.last_bi = buy_row

        self.open = op
        self.high = hi
        self.low = lo
        self.close = cl

        self.slippage = slippage
        self.slip_window = slip_window
        self.fees = fees

        self.stop_rates = stop_rates
        self.stop_config = stop_config

        self.amount = amount
        self.sell_reason = sell_reason
        self.irt = inv_roi_timeouts
        self.irv = inv_roi_values

        self.min_bv = min_buy_value
        self.min_sv = min_sell_value

    def update_context(self, oc: OrderContext):
        # buy rate of the last buy order for the current col
        self.br: nb.float64 = self.last_br[oc.col]
        # index of the last buy order for the current col
        self.bi: nb.int64 = self.last_bi[oc.col]
        # duration in number of candles for the current open order
        self.span = oc.i - self.bi
        # current ohlcv
        self.open_r: nb.float64 = self.open[oc.i, oc.col]
        self.high_r: nb.float64 = self.high[oc.i, oc.col]
        self.low_r: nb.float64 = self.low[oc.i, oc.col]
        self.close_r: nb.float64 = self.close[oc.i, oc.col]


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
def calc_profit(open_rate: float, close_rate: float, qqty: float, fees: float) -> float:
    shares = qqty / open_rate
    buy_qqty = shares * open_rate
    sell_qqty = shares * close_rate
    spent_qqty = buy_qqty + buy_qqty * fees
    return_qqty = sell_qqty - sell_qqty * fees
    return_prc = return_qqty / spent_qqty - 1
    # pass profits_prc as out, https://github.com/numba/numba/issues/4439
    return np.round(return_prc, 8)


@njit(cache=True)
def calc_trailing_rate(previous_rate, high_rate, high_profit, stp):
    if stp.trailing_stop:
        if not (
            stp.trailing_only_offset_is_reached
            and high_profit < stp.trailing_stop_positive
        ):
            return max(
                # trailing only increases
                previous_rate,
                # use positive ratio if above positive offset (default > 0) NOTE: strict > 0
                high_rate * (1 - stp.trailing_stop_positive)
                if stp.trailing_stop_positive is not None
                and high_profit > stp.trailing_stop_positive_offset
                # otherwise trailing with stoploss ratio
                else high_rate * (1 - stp.stoploss),
            )
        return previous_rate
    return


@njit(cache=True)
def get_slippage(idx, col, slippage, slip_window):
    slip = slippage[idx, col]
    if np.isfinite(slip):
        return slip
    else:
        slip = np.nanmean(slippage[idx - slip_window : idx])
        return slip if np.isfinite(slip) else 0


@njit(cache=False)
def check_for_buy(oc, ctx):
    if (
        ctx.buys[oc.i, oc.col]
        and (not ctx.sells[oc.i, oc.col])
        and oc.cash_now > ctx.min_bv
    ):
        # index of bought
        ctx.last_bi[oc.col] = bi = oc.i + 1
        # price is open of next candle
        ctx.last_br[oc.col] = br = ctx.open[bi, oc.col]
        shares = ctx.amount[oc.i, oc.col] / br
        # skip unavailable data
        if np.isnan(shares):
            return NoOrder
        # initialize sell reason to initial stoploss
        # so we can check if stoploss was trailing or static
        ctx.stop_rates[oc.col] = ctx.sell_reason[bi] = br * (1 - ctx.stop_config.stoploss)
        slip = get_slippage(bi, oc.col, ctx.slippage, ctx.slip_window)
        return create_order_nb(
            size=shares,
            # price is open of next candle
            price=br,
            fees=ctx.fees,
            slippage=slip,
        )
    return NoOrder


@njit(cache=True)
def check_roi_on_open(oc, ctx):
    if ctx.span > 0:
        open_profit = calc_profit(ctx.br, ctx.open_r, oc.shares_now, ctx.fees)
        triggered, _ = weighted_roi_is_triggered(
            ctx.span, open_profit, ctx.irt, ctx.irv
        )
        # sell with roi on open
        if triggered:
            # print('open roi, size: ', oc.shares_now * open_rate)
            ctx.sell_reason[ctx.bi] = IntSellType.ROI.value
            slip = get_slippage(oc.i, oc.col, ctx.slippage, ctx.slip_window)
            return (
                True,
                create_order_nb(
                    size=-oc.shares_now,
                    # when roi is triggered on open,
                    # sell price matches the open rate
                    price=ctx.open_r,
                    fees=ctx.fees,
                    slippage=slip,
                ),
            )
    return False, NoOrder


@njit(cache=True)
def check_stop_on_open(oc, ctx):
    if ctx.close_r <= ctx.open_r:
        # check previous trailing stop
        if ctx.low_r <= ctx.stop_rates[oc.col]:
            # print('previous trailing: ', oc.shares_now * stop_rates[oc.col])
            ctx.sell_reason[ctx.bi] = (
                IntSellType.STOP_LOSS.value
                if ctx.sell_reason[ctx.bi] == ctx.stop_rates[oc.col]
                else IntSellType.TRAILING_STOP_LOSS.value
            )
            slip = get_slippage(oc.i, oc.col, ctx.slippage, ctx.slip_window)
            return (
                True,
                create_order_nb(
                    size=-oc.shares_now,
                    price=ctx.stop_rates[oc.col],
                    fees=ctx.fees,
                    slippage=slip,
                ),
            )
    return False, NoOrder


@njit(cache=True)
def check_roi_on_high(oc, high_profit, ctx):
    triggered, roi = weighted_roi_is_triggered(
        ctx.span, high_profit, ctx.irt, ctx.irv
    )
    if triggered:
        # print("high roi: ", oc.shares_now * calc_roi_close_rate(br, low_rate, roi, fees))
        ctx.sell_reason[ctx.bi] = IntSellType.ROI.value
        slip = get_slippage(oc.i, oc.col, ctx.slippage, ctx.slip_window)
        return (
            True,
            create_order_nb(
                size=-oc.shares_now,
                # when roi is triggered on high, sell price depends on roi target
                price=calc_roi_close_rate(ctx.br, ctx.low_r, roi, ctx.fees),
                # fees are included in roi rate calculation (although without slippage...)
                fees=0,
                slippage=slip,
            ),
        )
    return False, NoOrder


@njit(cache=True)
def check_stop_on_high(oc, ctx, high_profit):
    ctx.stop_rates[oc.col] = calc_trailing_rate(
        ctx.stop_rates[oc.col], ctx.high_r, high_profit, ctx.stop_config
    )
    if ctx.low_r <= ctx.stop_rates[oc.col]:
        # print("updated trailing: ", oc.shares_now * stop_rates[oc.col])
        ctx.sell_reason[ctx.bi] = (
            IntSellType.STOP_LOSS.value
            if ctx.sell_reason[ctx.bi] == ctx.stop_rates[oc.col]
            else IntSellType.TRAILING_STOP_LOSS.value
        )
        slip = get_slippage(oc.i, oc.col, ctx.slippage, ctx.slip_window)
        return (
            True,
            create_order_nb(
                size=-oc.shares_now,
                price=ctx.stop_rates[oc.col],
                fees=ctx.fees,
                slippage=slip,
            ),
        )
    return False, NoOrder


@njit(cache=True)
def check_for_sell(oc, ctx):
    if ctx.sells[oc.i, oc.col] and not ctx.buys[oc.i, oc.col]:
        # print("sell signal: ", oc.shares_now)
        ctx.sell_reason[ctx.bi] = IntSellType.SELL_SIGNAL.value
        next_i = oc.i + 1
        slip = get_slippage(next_i, oc.col, ctx.slippage, ctx.slip_window)
        return (
            True,
            create_order_nb(
                size=-oc.shares_now,
                # price from a sell signal is open of next candle
                price=ctx.open[next_i, oc.col],
                fees=ctx.fees,
                slippage=slip,
            ),
        )
    return False, NoOrder


@njit(cache=True)
def order_func_nb(
    oc: OrderContext, ctx,
):
    if oc.shares_now == 0:
        return check_for_buy(oc, ctx,)
    elif oc.val_price_now > ctx.min_sv:
        ctx.update_context(oc)
        # skip unavailable data
        if np.isnan(ctx.open_r):
            return NoOrder
        # check roi against open if trade is open for more than one candle
        res, order = check_roi_on_open(oc, ctx)
        if res:
            return order
        res, order = check_stop_on_open(oc, ctx)
        if res:
            return order
        # check against roi on high
        high_profit = calc_profit(ctx.br, ctx.high_r, oc.shares_now, ctx.fees)
        res, order = check_roi_on_high(oc, high_profit, ctx)
        if res:
            return order

        # then check against updated stoploss on high
        res, order = check_stop_on_high(oc, ctx, high_profit)
        if res:
            return order

        # at last check for sell signals
        # res, order = check_for_sell(oc, ctx)
        # if res:
        #     return order
    return NoOrder

OrdersList = nb.typed.List.empty_list(item_type=nb.typeof(Order))

@njit(cache=True)
def generate_orders(ctx):
    balance = 100
    for row in ctx.close:
        for clo in row:
            if np.isnan(clo):
                continue

        break
