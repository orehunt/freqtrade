from freqtrade.strategy.interface import IStrategy
import numba as nb
import numpy as np
from numba import njit
import pandas as pd
from typing import Any, Dict, Iterable, List, Tuple, Union, Callable
from freqtrade.optimize.backtest_nb import cummax
from itertools import product

cache = {}


def concat_pairs_df(
    ref: Dict[str, Any],
    cc_pairs: Tuple[str, ...],
    cc_tf: Tuple[str, ...],
    strategy: IStrategy,
):
    ref_tf, ref_pair, ref_d = ref["timeframe"], ref["pair"], ref["data"]
    key = (
        ref_d["date"].iloc[-1],
        *cc_tf,
        *cc_pairs,
    )
    merge_dict = {(ref_pair, ref_tf): ref_d}

    if key in cache:
        ref_d.set_index("date", inplace=True)
        merge_dict.update(cache[key])
        return merge_pairs_df(df_dict=merge_dict)

    cc_dict = {}
    last_date = ref_d["date"].iloc[-1]
    ref_td = pd.Timedelta(ref_tf)
    tds = pd.to_timedelta(cc_tf)

    for pair in cc_pairs:
        for n, td in enumerate(tds):
            pair_df = strategy.get_pairs_data(
                pair=pair, timeframe=cc_tf[n], last_date=last_date
            )
            if td > ref_td:
                pair_df["date"].values[:] += td
            pair_df.set_index("date", inplace=True)
            cc_dict[(pair, cc_tf[n])] = pair_df

    ref_d.set_index("date", inplace=True)
    # get columns (after date is set as index)
    # to keep base ohlcv names for the ref pair
    ref_columns = ref_d.columns

    merge_dict.update(cc_dict)
    cc_df = pd.concat(merge_dict, axis=1, copy=False)

    cc_df.columns = cc_df.columns.to_flat_index()
    cc_df[ref_columns] = cc_df[((ref_pair, ref_tf, c) for c in ref_columns)]

    cc_df.fillna(method="pad", inplace=True)
    cc_df.reset_index(drop=False, inplace=True)
    cache[key] = cc_dict
    return cc_df


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
#


def pairs_timeframes(
    pairs: Union[Tuple[str, ...], Tuple[Tuple[str, str], ...], None],
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


def concat_timeframes_data(
    pairs: Union[Tuple[str, ...], Tuple[Tuple[str, str], ...], None],
    get_data: Union[Callable, None] = None,
    timeframes: Union[Tuple[str, ...], None] = None,
    sort=True,
    source_df: pd.DataFrame = None,
    df_list: List[pd.DataFrame] = [],
) -> pd.DataFrame:
    """
    Concatenate combinations of pairs and timeframes, using dates as index.
    The smaller timeframe is taken as base

    :param pairs: List of strings or tuples
        tuples in the form (pair, timeframe)
    :param get_data: callable accepting arguments 'pair', 'timeframe' and 'last_date'
        the fuction which retrieves the data for each combination
    :param timeframes: List of strings
        if provided the data will be the product of pairs and timeframes lists
    :param sort: order the resulting dataframe by length of timeframes
    :param source_df: the starting dataframe against which cache is checked
    :param df_list: a list of frames to concatenate, takes precedence over pairs list
    """
    if pairs is not None:
        pairs_tf = pairs_timeframes(pairs, timeframes, sort=sort)
        base_td = pairs_tf[-1, 2]
    elif not df_list:
        raise OperationalException("a list of pairs, or a list of frames is required")
    if source_df is not None:
        source_df.set_index("date", inplace=True)
        data = [source_df]
        last_date = source_df.index.max()
    else:
        data = []
        last_date = datetime.now()
    if not len(df_list):
        for pair, tf, td in pairs_tf:
            prefix = f"{tf}_{pair}_"
            prefixes.append(prefix)
            pair_df = get_data(pair=pair, timeframe=tf, last_date=last_date)
            pair_df.set_index("date", inplace=True)
            # longer timeframes have to be shifted, because longer candles
            # appear later in the timeline
            if td > base_td:
                pair_df.index = pair_df.index + td
            pair_df.columns = prefix + pair_df.columns
            data.append(pair_df)
    else:
        data.extend(df_list)

    cc_df = pd.concat(data, axis=1, join="outer", copy=False)
    cc_df.fillna(method="pad", inplace=True)
    # this should drop only starting rows if concatenated dfs start
    # from different dates
    cc_df.dropna(inplace=True)
    cc_df.reset_index(drop=False, inplace=True)
    # print(cc_df.iloc[:10])
    return cc_df, data if source_df is None else data[1:]
