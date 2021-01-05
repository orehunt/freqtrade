from typing import NamedTuple, Union
from enum import IntEnum
from numba import float64, from_dtype, int64, types
from numba.typed import Dict as nb_Dict
from numpy import dtype

MERGE_COLS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "buy",
    "sell",
    "pair",
    "stake_amount",
]

SOLD_COLS = [
    "bought_or_sold",
    "trigger_ofs",
    "trigger_bought_ofs",
    "last_trigger",
    "next_sold_ofs",
    "pair",
]

DEFAULT_QUANTILE = 0.2
DEFAULT_PROBS = (0.8, 0.8)

float_array = float64[:]
int_array = int64[:]

# arrays of strings no longer than 20chars
NamesLists = nb_Dict.empty(
    key_type=types.unicode_type,
    value_type=types.Array(from_dtype(dtype("U20")), 1, "A"),
)

Flags = nb_Dict.empty(key_type=types.unicode_type, value_type=types.boolean)

Int64Vals = nb_Dict.empty(key_type=types.unicode_type, value_type=int64)

Float64Vals = nb_Dict.empty(key_type=types.unicode_type, value_type=float64)

Int64Cols = nb_Dict.empty(key_type=types.unicode_type, value_type=int64[:])

Float64Cols = nb_Dict.empty(key_type=types.unicode_type, value_type=float64[:])

ColsMap = nb_Dict.empty(
    key_type=types.unicode_type,
    value_type=types.Array(from_dtype(dtype("int64")), 2, "A"),
)


class DictArr(NamedTuple):
    fl_cols: nb_Dict
    it_cols: nb_Dict
    names: nb_Dict
    fl: nb_Dict
    it: nb_Dict
    bl: nb_Dict
    cmap: nb_Dict


class Features(NamedTuple):
    roi_enabled: Union[bool, None]
    weighted_roi: Union[bool, None]
    stoploss_enabled: Union[bool, None]
    trailing_enabled: Union[bool, None]
    not_position_stacking: Union[bool, None]


TIMEFRAME_WND = {
    "1m": 60,  # 1h
    "5m": 24,  # 2h
    "15m": 16,  # 4h
    "30m": 16,  # 8h
    "1h": 24,  # 24h
    "8h": 90,  # 30 days
    "1d": 90,  # 3 months
}
DEFAULT_WND = 48

FORCE_SELL_AFTER = {
    "1m": 240,  # 4h
    "5m": 144,  # 12h
    "15m": 96,  # 24h
    "30m": 72,  # 36h
    "1h": 48,  # 48h
    "8h": 36,  # 12 days
    "1d": 24,  # 24 days
}

# the weight of the next price point (in ohlc) compared to the current one
SLIPPAGE_BALANCE = {"1m": 0.5, "5m": 0.3, "15m": 0.2, "1h": 0.1, "8h": 0.05, "1d": 0.03}


class Candle(IntEnum):
    NOOP = 0
    BOUGHT = 1
    SOLD = 3
    END = 7  # references the last candle of a pair
    FORCE_SOLD = 12


class OrderType(IntEnum):
    LIMIT = 0
    MARKET = 1
