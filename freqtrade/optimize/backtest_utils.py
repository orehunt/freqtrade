from typing import Dict, List, Tuple, Union
from types import SimpleNamespace
from numpy import (
    ndarray,
    flatnonzero,
    nan,
    concatenate,
    where,
    searchsorted,
    isnan,
    interp,
    in1d,
    full,
    delete,
    isnan,
    arange,
    maximum,
    bincount,
    nonzero,
    r_,
    diff,
    sort,
)
import numpy as np
from pandas import DataFrame
from freqtrade.exceptions import OperationalException


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
    col_dict = {}
    for n, col in enumerate(df.columns.values):
        col_dict[col] = n
    return col_dict


def as_df(vals, cols, idx=None, int_idx=False, dtype=np.float64):
    return DataFrame(
        vals,
        columns=cols,
        index=idx.astype(np.int64) if int_idx else idx,
        dtype=dtype,
        copy=False,
    )


def add_columns(
    arr: ndarray, cols_dict: Dict, columns: Union[List, Dict, Tuple], data=None
) -> ndarray:
    # NOTE: arrays are empty, not filled with nan
    for n, c in enumerate(columns, len(cols_dict)):
        cols_dict[c] = n
    return concatenate(
        (arr, ndarray(shape=(arr.shape[0], len(columns)), dtype=arr.dtype) if data is None else data),
        axis=1,
    )


def get_cols(arr: ndarray, col_dict, col_names: List[int], dtype=None) -> List[ndarray]:
    if dtype:
        return [arr[:, col_dict[c]].astype(dtype) for c in col_names]
    else:
        return [arr[:, col_dict[c]] for c in col_names]


def without_cols(cols_dict: Dict, exclude: List) -> List:
    without = cols_dict.copy()
    for e in exclude:
        del without[e]
    return list(without.keys())


def replace_values(v: ndarray, k: ndarray, arr: ndarray) -> ndarray:
    """ replace values in a 1D array """
    # searchsorted returns len(arr) where each element is the index of v
    # make sure types match
    idx = searchsorted(v, arr)
    # this is not needed if we assume that no element in arr is > the values in v
    # idx[idx == len(v)] = nan
    mask = v[idx] == arr
    return where(mask, k[idx], nan)


def np_left_join(
    arr1: ndarray,
    arr2: ndarray,
    cols1: Dict,
    cols2: Dict,
    key1: str,
    key2: str,
    fill_value=nan,
    drop_key2=True,
):
    k1n, k2n = cols1[key1], cols2[key2]
    # keys are assumed to have unique values, and be ordered
    # make a mask of all the keys of arr2 that are in arr1
    arr2_in1 = in1d(arr2[:, k2n], arr1[:, k1n], assume_unique=True)
    arr1_in2 = in1d(arr1[:, k1n], arr2[arr2_in1, k2n], assume_unique=True)
    # don't join key from arr2 used for indexing
    if drop_key2:
        arr2 = delete(arr2, k2n, axis=1)
        # make a copy to not modify the original cols2
        cols2 = cols2.copy()
        del cols2[key2]
    # create an array of same length as arr1 to allow concatenation on axis1
    arr1_ext = full((arr1.shape[0], arr2.shape[1]), fill_value)
    # copy data from arr2 into the empty arr1 shaped array, but only for keys
    # of arr1 that are present in arr2
    arr1_ext[arr1_in2] = arr2[arr2_in1]
    # update columns names and concatenate with the new data
    return add_columns(arr1, cols1, cols2, arr1_ext)


# https://stackoverflow.com/q/41190852/2229761
def np_fill(
    arr_src, replace_value=None, fill_value=None, backfill=False, inplace=False
):
    """ forward fill array with nans or value """
    # NOTE: last value wraps to beginning value if nan
    if replace_value is not None:
        compare = lambda a: a == replace_value
    else:
        compare = lambda a: np.isnan(a)
    if fill_value is not None:
        if inplace:
            arr_src[compare(arr_src)] = fill_value
            return
        return np.where(compare(arr_src), fill_value, arr_src)
    arr = arr_src[::-1] if backfill else arr_src.view()
    prev = np.arange(len(arr))
    prev[compare(arr)] = -1
    prev = np.maximum.accumulate(prev)
    if inplace and backfill:
        arr_src[:] = arr[prev][::-1]
    elif inplace:
        arr_src[:] = arr[prev]
    elif backfill:
        return arr[prev][::-1]
    else:
        return arr[prev]


# https://stackoverflow.com/a/54136635/2229761
def count_in1d(arr1, arr2):
    """ count occurrences of arr1 in arr2 """
    idx = np.searchsorted(arr1, arr2)
    idx[idx == len(arr1)] = 1
    mask = arr1[idx] == arr2
    return np.bincount(idx[mask])


# https://stackoverflow.com/a/1044443/2229761
def first_unique(arr, is_sorted=False, kind="stable"):
    """ finds the first occurrence of elements in arr """
    return np.nonzero(np.r_[1, np.diff(arr if is_sorted else np.sort(arr, kind=kind))])[
        0
    ]


def idx_to_mask(idx: ndarray, shape: Tuple, invert=False) -> ndarray:
    if invert:
        mask = np.ones(shape, dtype=np.bool)
        mask[idx] = 0
    else:
        mask = np.zeros(shape, dtype=np.bool)
        mask[idx] = 1
    return mask


def merge_2d(arr1, arr2, k1, k2, sort=True, null_k2=True):
    """ Assumptions:
    - arr1 sorted, unique
    - arr2 force sorted, duplicated
    """
    # columns to merge
    arr2_to_merge = np.arange(arr2.shape[1])
    # sort arr2
    if sort:
        arr2 = arr2[np.argsort(arr2[:, k2], kind="stable")]
    arr2_not_in_arr1 = np.isin(
        arr2[:, k2], arr1[:, k1], assume_unique=False, invert=True,
    )

    arr1_ofs = arr1[:, k1]
    arr2_ofs = arr2[:, k2]
    unq_arr2_idx = first_unique(arr2_ofs)
    arr2_is_dup = idx_to_mask(unq_arr2_idx, arr2_ofs.shape[0], invert=True)
    arr1_arr2_mask = arr2_not_in_arr1 | arr2_is_dup
    arr2_ofs = arr2[arr1_arr2_mask, k2]
    arr2_append_idx = searchsorted(arr1_ofs, arr2_ofs, "right")

    merged = full(
        (arr1_ofs.shape[0] + arr2_ofs.shape[0], arr1.shape[1] + arr2.shape[1],), nan,
    )
    merged[:, k1] = np.insert(arr1_ofs, arr2_append_idx, arr2_ofs)
    merged_ofs = merged[:, k1]
    arr2_mrg_idx = arr2_append_idx + arange(arr2_append_idx.shape[0])
    # this includes the (dup) indexes from arr2 that will be repeated
    # with data from arr1
    arr1_mrg_mask = np.isin(merged[:, k1], arr1_ofs, assume_unique=False)
    arr2_mrg_mask = np.isin(merged[:, k1], arr2[:, k2], assume_unique=False,)
    arr1_shape = arr1.shape[1]

    dup_arr1 = count_in1d(arr1[:, k1], merged[:, k1])

    mrg_unq_idx = first_unique(merged_ofs)
    mrg_is_unq = idx_to_mask(mrg_unq_idx, merged_ofs.shape[0])

    merged[arr1_mrg_mask, :arr1_shape] = np.repeat(arr1, dup_arr1, axis=0)
    # copy the inserted rows from arr2 into arr1 into the merged arr
    not_arr1_arr2_mask = arr2_mrg_mask & arr1_mrg_mask & mrg_is_unq
    merged[not_arr1_arr2_mask, arr1_shape:] = arr2[~arr1_arr2_mask][:, arr2_to_merge]

    merged[arr2_mrg_idx, arr1_shape:] = arr2[arr1_arr2_mask][:, arr2_to_merge]
    if null_k2:
        merged[~arr1_mrg_mask, k1] = nan
    return merged


def diff_indexes(arr: ndarray, with_start=False, with_end=False) -> ndarray:
    """ returns the indexes where consecutive values are not equal,
    used for finding pairs ends """
    try:
        if with_start:
            if with_end:
                raise OperationalException("with_start and with_end are exclusive")
            return where(arr != shift(arr))[0]
        elif with_end:
            if with_start:
                raise OperationalException("with_end and with_start are exclusive")
            return where(arr != shift(arr, -1))[0]
        else:
            return where(arr != shift(arr, fill=arr[0]))[0]
    except:
        return []
