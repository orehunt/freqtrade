from functools import partial
from enum import IntEnum
from typing import Callable, Optional
import inspect
from freqtrade.optimize.backtesting import BacktestResult

import numba as nb
import numpy as np


def njit(cache: bool = True, **kwargs) -> Callable:
    kwargs["cache"] = cache
    return partial(nb.njit, **kwargs)


def jitclass(spec) -> Callable:
    return partial(nb.experimental.jitclass, spec=spec)


StoplossConfigTypeSig = [
    ("stoploss", nb.float64),
    ("trailing_stop", nb.types.boolean),
    ("trailing_stop_positive", nb.types.Optional(nb.float64)),
    ("trailing_stop_positive_offset", nb.float64),
    ("trailing_only_offset_is_reached", nb.types.boolean),
]

# this is definied here because it is used in the Context type
@nb.experimental.jitclass(StoplossConfigTypeSig)
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


StoplossConfigJitType = nb.typeof(StoplossConfigJit(0))

BacktestResultTypeSig = [
    ("pair", nb.int64),
    ("profit_percent", nb.float64),
    ("profit_abs", nb.float64),
    ("open_date", nb.int64),
    ("open_rate", nb.float64),
    ("open_fee", nb.float64),
    ("close_date", nb.int64),
    ("close_rate", nb.float64),
    ("close_fee", nb.float64),
    ("amount", nb.float64),
    ("trade_duration", nb.int64),
    ("open_at_end", nb.types.boolean),
    ("sell_reason", nb.int64),
]
BacktestResultDType = np.dtype([(name, str(t)) for name, t in BacktestResultTypeSig])
# make sure backtest results parameters order match
assert all(
    a == b
    for a, b in zip(
        BacktestResultDType.names, inspect.signature(BacktestResult).parameters.keys()
    )
)


ContextTypeSig = [
    ("pairs", nb.typeof(nb.typed.List.empty_list(item_type=nb.types.unicode_type))),
    ("pairs_seq", nb.int64[:]),
    ("orig_seq", nb.int64[:]),
    ("date", nb.int64[:]),
    ("buys", nb.float64[:, :]),
    ("sells", nb.float64[:, :]),
    ("slippage", nb.float64[:, :]),
    ("open", nb.float64[:, :]),
    ("high", nb.float64[:, :]),
    ("low", nb.float64[:, :]),
    ("close", nb.float64[:, :]),
    ("slp_window", nb.int64),
    ("fees", nb.float64),
    ("stop_config", nb.typeof(StoplossConfigJit(0))),
    ("amount", nb.float64[:, :]),
    ("irt", nb.int64[:]),
    ("irv", nb.float64[:]),
    ("min_bv", nb.float64),
    ("min_sv", nb.float64),
    ("span", nb.int64),
    ("open_r", nb.float64),
    ("high_r", nb.float64),
    ("low_r", nb.float64),
    ("close_r", nb.float64),
    ("cash_start", nb.float64),
    ("cash_now", nb.float64),
]

TradeJitTypeSig = [
    ("open_idx", nb.int64),
    ("close_idx", nb.int64),
    ("open_price", nb.float64),
    ("buy_price", nb.float64),
    ("shares_held", nb.float64),
    ("cash_spent", nb.float64),
    ("sell_price", nb.float64),
    ("cash_returned", nb.float64),
    ("profits", nb.float64),
    ("pnl", nb.float64),
    ("status", nb.types.boolean),
    ("sell_reason", nb.int64),
    ("stoploss_price", nb.float64),
    ("initial_stoploss_price", nb.float64),
]

MetricsDictSpec = {
    "key_type": nb.types.unicode_type,
    "value_type": nb.float64[:],
}
