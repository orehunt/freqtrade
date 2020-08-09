from typing import Dict, List, Tuple
from numpy import (ndarray, flatnonzero, nan, concatenate, where, searchsorted, isnan, interp)

def padfill(arr: ndarray):
    mask = isnan(arr)
    arr[mask] = interp(flatnonzero(mask), flatnonzero(~mask), arr[~mask])


def union_eq(arr: ndarray, vals: List) -> List[bool]:
    """ union of equalities from a starting value and a list of values to compare """
    res = arr == vals[0]
    for v in vals[1:]:
        res = res | (arr == v)
    return res


def shift(arr: ndarray, period=1, fill=nan) -> ndarray:
    """ shift ndarray """
    moved: ndarray = ndarray(shape=arr.shape, dtype=arr.dtype)
    if period < 0:
        moved[:period] = arr[-period:]
        moved[period:] = fill
    else:
        moved[period:] = arr[:-period]
        moved[:period] = fill
    return moved


def df_cols(df) -> Dict[str, int]:
    return {col: n for n, col in enumerate(df.columns.values)}


def add_columns(arr: ndarray, cols_dict: Dict, columns: Tuple) -> ndarray:
    tail = len(cols_dict)
    for c in columns:
        cols_dict[c] = tail
        tail += 1
    return concatenate((arr, ndarray(shape=(arr.shape[0], len(columns)))), axis=1)


def replace_values(v: ndarray, k: ndarray, arr: ndarray) -> ndarray:
    """ replace values in a 1D array """
    # searchsorted returns len(arr) where each element is the index of v
    # make sure types match
    idx = searchsorted(v, arr)
    # this is not needed if we assume that no element in arr is > the values in v
    # idx[idx == len(v)] = nan
    mask = v[idx] == arr
    return where(mask, k[idx], nan)
