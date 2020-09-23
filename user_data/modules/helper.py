import os
from typing import Callable, Dict, Union, Tuple, List
from itertools import product
from datetime import datetime
from joblib import cpu_count, wrap_non_picklable_objects
import pandas as pd
import numpy as np
from multiprocessing import Pool, Manager, Queue
from functools import partial
import json

from freqtrade.exceptions import OperationalException

mgr: Manager = None


def error_callback(e, q: Queue):
    print(e)
    q.put((None, None))


def get_all_signals(
    target: Callable, pairs_args: Dict, jobs=(cpu_count() // 2 or 1)
) -> Dict:
    """ Apply function over a dict where the values are the args of the function, parallelly """

    results = {}
    queue = mgr.Queue() if mgr else Manager().Queue()
    err = partial(error_callback, q=queue)

    def func_queue(func: Callable, queue: Queue, pair: str, *args) -> pd.DataFrame:
        res = func(*args)
        queue.put((pair, res))
        return res

    target = wrap_non_picklable_objects(target)
    func_queue = wrap_non_picklable_objects(func_queue)

    try:
        with Pool(jobs) as p:
            p.starmap_async(
                func_queue,
                [(target, queue, pair, *v) for pair, v in pairs_args.items()],
                error_callback=err,
            )
            for pair in pairs_args:
                proc_pair, res = queue.get()
                if proc_pair:
                    results[proc_pair] = res
                else:
                    break
        # preserve the dict order
        return {pair: results[pair] for pair in pairs_args}
    except KeyError:
        return {pair: target(*args) for pair, args in pairs_args.items()}


def read_json_file(file_path: str, key=""):
    if not os.path.exists(file_path):
        raise OperationalException(f"path: {file_path} does not exist")
    with open(file_path, "r") as fp:
        if key:
            data = json.load(fp)
            try:
                return data[key]
            except KeyError:
                return None
        else:
            return json.load(fp)


def concat_timeframes_data(
    pairs: Union[Tuple[str, ...], Tuple[Tuple[str, str], ...]],
    get_data: Callable,
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
    :param timeframes: List of strings
        if provided the data will be the product of pairs and timeframes lists
    :param get_data: callable accepting arguments 'pair', 'timeframe' and 'last_date'
        the fuction which retrieves the data for each combination
    """
    pairlist = pairs if timeframes is None else tuple(product(pairs, timeframes))
    pairs_tf = np.empty((len(pairlist), 3), dtype="O")
    pairs_tf[:, :2] = np.asarray(pairlist)
    # strings to timedelta
    pairs_tf[:, 2] = pd.to_timedelta(pairs_tf[:, 1])
    if sort:
        # from shorter to longer, this also inverts pairs order, but shouldn't matter
        pairs_tf = pairs_tf[np.argsort(pairs_tf[:, 1])]
    base_td = pairs_tf[-1, 2]
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

    cc_df = pd.concat(data, axis=1)
    cc_df.fillna(method="pad", inplace=True)
    # this should drop only starting rows if concatenated dfs start
    # from different dates
    cc_df.dropna(inplace=True)
    cc_df.reset_index(drop=False, inplace=True)
    # print(cc_df.iloc[:10])
    return cc_df, data if source_df is None else data[1:]
