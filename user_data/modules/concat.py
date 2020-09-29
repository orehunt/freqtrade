import numba as nb
import numpy as np
from numba import njit
import pandas as pd
from typing import List, Tuple, Union, Iterable, Dict
from freqtrade.optimize.backtest_nb import cummax
from itertools import product


def pairs_timeframes(
    pairs: Tuple[Union[Tuple[str], Tuple[str, str]]],
    timeframes=None,
    only_timeframes=False,
    sort=False,
):
    pairlist = pairs if timeframes is None else tuple(product(pairs, timeframes))
    if only_timeframes:
        return pd.to_timedelta([t[1] for t in pairlist]).values.astype(int)
    pairs_tf = np.empty((len(pairlist), 3), dtype="O")
    pairs_tf[:, :2] = np.asarray(pairlist)
    # strings to timedelta
    pairs_tf[:, 2] = pd.to_timedelta(pairs_tf[:, 1])
    if sort:
        # from shorter to longer, this also inverts pairs order, but shouldn't matter
        pairs_tf = pairs_tf[np.argsort(pairs_tf[:, 1])]
    return pairs_tf


def merge_pairs_df(df_dict: Dict[Tuple[str, str], pd.DataFrame]):
    """"""

    indexes = nb.typed.List([d.index.values.astype(float) for d in df_dict.values()])
    data = nb.typed.List([d.values.astype(float) for d in df_dict.values()])

    # debug
    # df_list = list(df_dict.keys())
    # for n, _ in enumerate(data[1:], 1):
    #     k = df_list[n]
    #     if k[1] == "1d":
    #         print(df_dict[k].iloc[0])
    #         print(df_dict[k].iloc[-1])
    #     print(n, k)

    out, idx = concat_outer_timeframes(indexes, data)
    _, ref_df = next(iter(df_dict.items()))
    # dict keys are tuples, the index has to be 3 level
    columns = list(
        (pair, tf, col) for (pair, tf), col in product(df_dict.keys(), ref_df.columns)
    )

    # replace the columns of the reference df
    n_cols = len(ref_df.columns)

    df = pd.DataFrame(
        out, columns=columns, index=pd.to_datetime(idx, utc=True), copy=False,
    )
    # make a copy of the columns of the reference df for compatibility
    # if pandas supported aliases for columns, that would be nice
    df[ref_df.columns] = df[columns[:n_cols]]

    df.index.name = "date"
    df.reset_index(drop=False, inplace=True)
    return df


@njit(cache=False)
def zip_flat(ls):
    """ like zip, but flattens inner elements """
    out_ls = nb.typed.List([])
    n = 0
    max_len = len(ls[0])
    for n in range(max_len):
        new_tup = ()
        for l in ls:
            new_tup += l[n]
        out_ls.append(new_tup)
    return out_ls


@njit(cache=True)
def concat_outer_timeframes(
    indexes: nb.typed.List, data: nb.typed.List,
):
    """ outer concatenate on indexes with trimmed bounds """
    # first find the minimum and maximum of indexes
    # the first index is the reference index to which resampling is applied
    n_cols = data[0].shape[1]
    n_cc = len(indexes)
    ref = indexes[0]
    ref_tf = ref[1] - ref[0]
    out = np.full((ref.shape[0], n_cols * n_cc), np.nan)
    ref_min, ref_max = ref.min(), ref.max()
    out[:, :n_cols] = data[0]
    out_len = len(out)
    for n, d in enumerate(data[1:], 1):
        ix = indexes[n]
        # get the current timeframe from the index diff
        # and calc it's ratio to the reference timeframe
        mul, rem = divmod(ix[1] - ix[0], ref_tf)
        if rem != 0:
            print("Index at position", n, "is not a multiple of reference index (at 0)")
            return np.empty((0, 2)), ref
        # if timedelta is > 1 the current timeframe is longer than reference
        # so gets shifted according to its ratio
        # not used since dataframes should be already shifted
        # if mul > 1:
        #     ix[mul:] = ix[:-mul]
        #     ix[:mul] = np.nan
        # find the minimum index that is above reference
        for ni in range(len(ix)):
            t = ix[ni]
            if t >= ref_min:
                start = int(ni)
                out_start = int((t - ref_min) // ref_tf)
                break
        for ni in range(len(ix) - 1, -1, -1):
            t = ix[ni]
            if t <= ref_max:
                # the index at `ni` satisfies the condition
                # stop +1 because bounds are [inclusive, exclusive)
                stop = len(d) if ni == -1 else (int(ni) + 1)
                # out stop is not inclusive because it is calculated
                # from the difference of the index values (dates)
                # if the delta is 1, it means correctly that the slice
                # should stop at -1
                out_stop = -int((ref_max - t) // ref_tf) or out_len
                break

        # debug
        # o_s = out[out_start:out_stop:mul].shape[0]
        # i_s = d[start:stop].shape[0]
        # print(len(ref), out_stop, len(indexes[n]), stop)
        # print(ref[out_start], indexes[n][start])
        # print(ref[out_stop], indexes[n][stop])
        # if o_s != i_s:
        #     print(out_stop - out_start)
        #     print(stop - start, d.shape[0])

        repeat_rows(
            d[start:stop], mul, out[out_start:out_stop, n_cols * n : n_cols * (n + 1)]
        )

    return out, ref


@nb.njit(cache=True)
def repeat_rows(arr, repeats, out=None):
    if out is None:
        if len(arr.shape) > 1:
            new_shape = (int(arr.shape[0] * repeats), arr.shape[1])
        else:
            new_shape = (int(arr.shape[0] * repeats), 1)
        out = np.empty(new_shape)
    if repeats == 1:
        out[:] = arr
        return out
    n = 0
    for a in arr:
        out[n : n + repeats] = a
        n += repeats
    return out


# @njit(cache=True)
# def join_column_names(pairs, timeframes, columns):
#     new_names = nb.typed.List.empty_list(item_type=nb.types.unicode_type)
#     for p in pairs:
#         for t in timeframes:
#             for c in columns:
#                 new_names.append(t + "_" + p + "_" + c)
#     return new_names
