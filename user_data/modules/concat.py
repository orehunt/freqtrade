import logging
import os
from itertools import product
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import numba as nb
import numpy as np
import pandas as pd
from numba import njit

from freqtrade.optimize.backtest_utils import df_cols
from freqtrade.strategy.strategy_helper import merge_informative_pair


JIT_ENABLED = os.environ.get("NUMBA_DISABLE_JIT", 0)
logger = logging.getLogger(__name__)



class PairKey(NamedTuple):
    pair: str
    tf: str


class RefPair(NamedTuple):
    pair: str
    tf: str
    df: pd.DataFrame


PairsDict = Optional[Dict[PairKey, pd.DataFrame]]
TfList = Tuple[str, ...]
PairsKeys = Union[TfList, Tuple[PairKey, ...], None]

cache: Dict[Tuple, PairsDict] = {}

@njit(cache=True)
def reorder_list(ls: List, order: Union[np.ndarray, List[int]]):
    return nb.typed.List([ls[n] for n in order])


def parse_pairs_timeframes(
    pairs: PairsKeys,
    timeframes=(),
    ref: Optional[RefPair] = None,
    only_timeframes=False,
    sort=True,
    # a 2d array of 3 columns (pair, tf_string, timedelta)
) -> Optional[np.ndarray]:
    if pairs is None:
        return
    pairlist = pairs if timeframes is () else tuple(product(pairs, timeframes))
    if ref:
        ref_pair, ref_tf, _ = ref
        ref_key = (ref_pair, ref_tf)
        pairlist = (ref_key,) + pairlist
    if only_timeframes:
        tfs = pd.to_timedelta([t[1] for t in pairlist]).values
        if sort:
            tfs.sort(kind="stable")
        return tfs.astype(int)
    pairs_tf = np.empty((len(pairlist), 3), dtype="O")
    pairs_tf[:, :2] = np.asarray(pairlist)
    # strings to timedelta
    pairs_tf[:, 2] = pd.to_timedelta(pairs_tf[:, 1])
    if sort:
        # from shorter to longer, this also inverts pairs order, but shouldn't matter
        pairs_tf = pairs_tf[np.argsort(pairs_tf[:, 2], kind="stable")]
    return pairs_tf


def concat_pairs(
    ref: RefPair,
    cc_pairs: PairsKeys = (),
    cc_tf: TfList = (),
    get_data: Callable = None,
):
    """
    Concatenate combinations of pairs and timeframes, using dates as index.
    The smaller timeframe is taken as base

    :param ref: the starting dataframe against which cache is checked
    :param cc_pairs: can be a list of pairs or a list of tuples (pairs, timeframe)
    :param cc_tf: if provided the data will be the product of pairs and timeframes lists
    :param get_data: callable accepting arguments 'pair', 'timeframe' and 'last_date'
    """
    ref_pair, ref_tf, ref_df = ref
    ref_key = (ref_pair, ref_tf)
    key = (
        ref_df["date"].iloc[-1],
        *cc_tf,
        *cc_pairs,
    )

    if key in cache:
        cc_dict = cache[key]
        # the pairslist INCLUDES the ref
        pairs_tf = parse_pairs_timeframes(cc_pairs, cc_tf, ref=ref, sort=True)
    else:
        # the pairslist EXCLUDES the ref
        pairs_tf = parse_pairs_timeframes(cc_pairs, cc_tf, sort=True)
        # construct the dict according to the ref, but WITHOUT it
        cc_dict = construct_cc_dict(pairs_tf, ref, get_data=get_data, date_index=False,)
        cache[key] = cc_dict
        # after having saved the dict insert the reference;
        # (this two times constructions allows to cache the dict)
        pairs_tf = parse_pairs_timeframes(cc_pairs, cc_tf, ref=ref, sort=True)
    src_dict = insert_cc_dict({ref_key: ref_df}, pairs_tf, cc_dict)
    cc_df = _concat_pairs_nb(ref, df_dict=src_dict)
    return cc_df




def _concat_pairs_nb(ref: RefPair, df_dict: PairsDict):
    """ NOTE: dict is supposed to be sorted (lowest tf to highest tf)
    and the reference is the first (the lowest) """
    # use lists because data doesn't have same shape
    ref_pair, ref_tf, ref_df = ref
    indexes = []
    data = []
    columns = []
    for (pair, tf), d in df_dict.items():
        loc = df_cols(d)
        # use iloc because numpy can't convert pandas timestamps to float
        indexes.append(d.iloc[:, loc["date"]].values.astype(float))
        del loc["date"]
        data.append(d.iloc[:, list(loc.values())].values.astype(float))
        columns.extend([(pair, tf, col) for col in loc.keys()])
    # take the first timeframe as base since it is the smallest
    (_, base_tf), _ = next(iter(df_dict.items()))
    base_td = pd.Timedelta(base_tf).value
    ref_columns = ref_df.columns.difference(["date"])

    # use numpy equivalent for keys since we don't pass them to nb
    # keys = np.array(list(df_dict.keys()), dtype="O")

    logger.debug("concatenating data with ref timeframe %s", base_tf)

    out, idx = _concat_arrays_axis1(base_td, indexes, data)

    logger.debug("concatenated data with shape %s", out.shape)

    # the index of the reference pair has
    df = pd.DataFrame(
        out,
        columns=columns,
        index=pd.to_datetime(idx, utc=True),
        copy=False,
    )
    # restore the reference df column names as they were
    ref_keys = ((ref_pair, ref_tf, c) for c in ref_columns)
    df[ref_columns] = df[ref_keys]

    df.reset_index(drop=False, inplace=True)
    df.rename(columns={"index": "date"}, inplace=True)
    return df


@njit(cache=True)
def _concat_arrays_axis1(
    ref_tf: float, date_indexes: nb.typed.List, data: nb.typed.List,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    outer concatenate on date_indexes with trimmed bounds
    NOTE: ref_tf HAS to be the SMALLEST timeframe
    :param tds: timedeltas of indexes, SORTED
    :param date_indexes: indexes ordered according to `tds`
    :param data: the data to concatenate, ALL the same shape
    """
    # first find the minimum and maximum of date_indexes
    # the first index is the reference index to which resampling is applied

    n_cols = data[0].shape[1]
    n_cc = len(date_indexes)
    # if JIT_ENABLED:
    #     logger.debug("concatenating %s date_indexes with %s columns", n_cc, n_cols)
    ref = date_indexes[0]
    # if JIT_ENABLED:
    #     logger.debug("reference timeframe %s", pd.Timedelta(ref_tf))
    out = np.full((ref.shape[0], n_cols * n_cc), np.nan)
    ref_min, ref_max = ref.min(), ref.max()
    # if JIT_ENABLED:
    #     logger.debug("start date: %s, stop date: %s", pd.Timestamp(ref_min), pd.Timestamp(ref_max))
    out[:, :n_cols] = data[0]
    out_len = len(out)
    for n, d in enumerate(data[1:], 1):
        ix = date_indexes[n]
        # get the current timeframe from the index diff
        # and calc it's ratio to the reference timeframe
        mul, rem = divmod(ix[1] - ix[0], ref_tf)
        if rem != 0:
            # if JIT_ENABLED:
            #     logger.debug("Index at position %s is not a multiple of reference index (at 0), %s", n, pd.Timestamp(rem))
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

        start_cols = n_cols * n
        stop_cols = n_cols * (n + 1)
        repeat_rows(d[start:stop], mul, out[out_start:out_stop, start_cols:stop_cols])
        # forward fill the end
        if out_stop != out_len:
            for nc in range(start_cols, stop_cols):
                out[out_stop:, nc] = out[out_stop - 1, nc]
    return out, ref


@nb.njit(cache=True)
def repeat_rows(arr, repeats, out=None):
    """ like numpy.repeat """
    # construct the out array if not provided
    if out is None:
        if len(arr.shape) > 1:
            new_shape = (int(arr.shape[0] * repeats), arr.shape[1])
        else:
            new_shape = (int(arr.shape[0] * repeats), 1)
        out = np.empty(new_shape)
    # copy the same array if nothing to repeat
    if repeats == 1:
        out[:] = arr
        return out
    n = 0
    for a in arr:
        out[n : n + repeats] = a
        n += repeats
    return out


def concat_informative_pairs(
    ref: Union[Tuple, RefPair],
    pairs: Optional[PairsKeys] = (),
    timeframes: TfList = (),
    get_data: Optional[Callable] = None,
    *args,
    **kwargs
):
    if not isinstance(ref, RefPair):
        ref = RefPair(*ref)
    pairs_tf = parse_pairs_timeframes(pairs, timeframes, sort=True)
    # return the reference dataframe is pairs list is empty
    if pairs_tf is None:
        return ref[2]
    last_date = ref[2]["date"].iloc[-1]
    df = ref.df
    for pair, tf, _ in pairs_tf:
        inf_df = get_data(pair=pair, timeframe=tf, last_date=last_date)
        # get he columns BEFORE the merge because date_merge col is added
        # during the merge and should not be added to the merged df
        inf_cols = inf_df.columns
        df = merge_informative_pair(df, inf_df, ref.tf, tf)
        new_col_names = {f"{col}_{tf}": (pair, tf, col) for col in inf_cols}
        # renaming only applies if ALL the columns in the new names dict
        # are present in df
        df.rename(columns=new_col_names, inplace=True)
    return df


def concat_pairs_pd(
    # (pair, tf, data)
    ref: RefPair,
    pairs: PairsKeys = (),
    timeframes: TfList = (),
    get_data: Optional[Callable] = None,
    sort=True,
    as_source=False,
) -> pd.DataFrame:
    pairs_tf = parse_pairs_timeframes(pairs, timeframes, sort=sort)
    # return the reference dataframe is pairs list is empty
    if pairs_tf is None:
        return ref[2]

    cc_dict = construct_cc_dict(pairs_tf, ref, get_data, date_index=True)
    cc_df = pd.concat(cc_dict, axis=1, join="outer", copy=False, sort=False)
    cc_df.columns = cc_df.columns.to_flat_index()

    ref_pair, ref_tf, ref_df = ref
    ref_columns = ref_df.columns.difference(['date'])
    # restore the reference df column names as they were
    ref_keys = ((ref_pair, ref_tf, c) for c in ref_columns)
    # this works like a rename (tuple col names are removed)
    cc_df[ref_columns] = cc_df[ref_keys]

    # this needs to be applied for the whole length
    cc_df.fillna(method="pad", inplace=True)

    # cc_df.dropna(subset='close', inplace=True)
    if as_source:
        cc_df = cc_df.loc[ref_df["date"].iat[0] : ref_df["date"].iat[-1]]
    cc_df.reset_index(drop=as_source, inplace=True)
    return cc_df


def insert_cc_dict(
    ins_dict: PairsDict, pairs_tf: Optional[np.ndarray], cc_dict: PairsDict
):
    """ Expects:
    - a dataframe to insert into an ordered dict
    - an ndarray with 3 columns (pair, timeframe, timedelta)
    - the dict to update
    Will only insert if key is MISSING from the cc_dict
    """
    new_cc_dict = {}
    for pair, tf, td in pairs_tf:
        key = (pair, tf)
        try:
            new_cc_dict[key] = cc_dict[key]
        except KeyError:
            logger.debug("inserting dataframe of pair %s in concat dict", key)
            # check for membership because there might be
            # missing pairs (from pairs_tf list) that didn't have enough data
            # and weren't included in the concat dict
            if key in ins_dict:
                new_cc_dict[key] = ins_dict[key]
                assert ins_dict[key]["date"].freq == td
    return new_cc_dict


def trim_by_date(
    df: pd.DataFrame, start: pd.Timestamp, stop: pd.Timestamp
) -> Optional[pd.DataFrame]:
    date = df["date"]
    sstart = date.searchsorted(start)
    # not enough data
    if date.iloc[sstart] > stop:
        return None
    sstop = date.searchsorted(stop) or len(df)
    return df.iloc[sstart:sstop]


def trim_by_len(
    df: pd.DataFrame, count: int, stop: pd.Timestamp
) -> Optional[pd.DataFrame]:
    sstop = df["date"].searchsorted(stop) or len(df)
    # not enough data
    if sstop < len(df) - count:
        return None
    return df.iloc[-count:sstop]


def trim(
    df: pd.DataFrame,
    start: int,
    stop: pd.Timestamp,
    count: int,
    td: pd.Timedelta,
    ref_td: pd.Timedelta,
) -> Optional[pd.DataFrame]:
    """ Trim a dataframe according to its frequency """
    # if the timedelta (of the NON ref_df) is bigger, trim by length
    # because on equal lengths the bigger timeframe
    # already includes the reference timerange
    if td > ref_td:
        trimmed = trim_by_len(df, count, stop)
        # longer timeframes have to be shifted, because longer candles
        # appear later in the timeline
        if trimmed is not None:
            # don't use values as that would loose the timezone
            trimmed["date"] = trimmed["date"] + td
        return trimmed
    # if the timedelta is smaller trim by date, to ensure that
    # the smaller frequency includes all the required (and available) dates
    else:
        return trim_by_date(df, start, stop)


def construct_cc_dict(
    pairs_tf: np.ndarray,
    ref: RefPair,
    get_data: Optional[Callable] = None,
    date_index=True,
    verify_bounds=False,
) -> PairsDict:
    """
    Construct a dictionary of dataframes from a list of (pair, timeframe)
    adjusting each dataframe to the provided reference frequency and timerange
    """
    if not len(pairs_tf):
        return {}
    # NOTE: the reference timeframe should be ALREADY included
    # in the `pairs_tf` array
    pair, tf, ref_df = ref
    first_date = ref_df["date"].iloc[0]
    last_date = ref_df["date"].iloc[-1]
    if verify_bounds:
        assert first_date == ref_df["date"].min()
        assert last_date == ref_df["date"].max()
    ref_td = pd.Timedelta(tf)
    count = len(ref_df)

    cc_dict = {}
    for pair, tf, td in pairs_tf:
        pair_df = trim(
            get_data(pair=pair, timeframe=tf, last_date=last_date),
            first_date,
            last_date,
            count,
            td,
            ref_td,
        )
        if pair_df is None:
            continue
        # date index should be used when concatenating with pandas
        if date_index:
            pair_df.set_index("date", inplace=True, drop=True)
        cc_dict[(pair, tf)] = pair_df
    return cc_dict


def _dbg_print_df_keys(df_dict):
    df_list = list(df_dict.keys())
    data = list(df_dict.values())
    for n, _ in enumerate(data[1:], 1):
        k = df_list[n]
        if k[1] == "1d":
            print(df_dict[k].iloc[0])
            print(df_dict[k].iloc[-1])
        print(n, k)
