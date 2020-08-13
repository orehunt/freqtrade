from typing import NamedTuple, TypedDict
from numba import types, int32, int64, float64, from_dtype, optional
from numba.typed import Dict as nb_Dict, List as nb_List
from numpy import ndarray, dtype

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
]

SOLD_COLS = [
        "bought_or_sold",
        "trigger_ofs",
        "trigger_bought_ofs",
        "last_trigger",
        "next_sold_ofs",
        "pair",
    ]

float_array = float64[:]
int_array = int64[:]

# arrays of strings no longer than 20chars
NamesLists = nb_Dict.empty(key_type=types.unicode_type, value_type=types.Array(from_dtype(dtype('U20')), 1, 'A'))

Flags = nb_Dict.empty(key_type=types.unicode_type, value_type=types.boolean)

Int64Vals = nb_Dict.empty(key_type=types.unicode_type, value_type=int64)

Float64Vals = nb_Dict.empty(key_type=types.unicode_type, value_type=float64)

Int64Cols = nb_Dict.empty(key_type=types.unicode_type, value_type=int64[:])

Float64Cols = nb_Dict.empty(key_type=types.unicode_type, value_type=float64[:])

class DictArr(NamedTuple):
    fl_cols: nb_Dict
    it_cols: nb_Dict
    names: nb_Dict
    fl: nb_Dict
    it: nb_Dict
    bl: nb_Dict
    cmap: nb_Dict

ColsMap = nb_Dict.empty(key_type=types.unicode_type, value_type=types.Array(from_dtype(dtype('int64')), 2, 'A'))



