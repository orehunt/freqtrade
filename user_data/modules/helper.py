import json
from datetime import datetime
import os
from functools import partial
from multiprocessing import Manager, Pool, Queue
from pathlib import Path
from typing import Callable, Dict, Union

import pandas as pd
from joblib import cpu_count, wrap_non_picklable_objects
from pandas import date_range

from freqtrade.exceptions import OperationalException


mgr: Manager = None


def error_callback(e, q: Queue):
    print(f"{__name__}.get_all_signals:", e)
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


def resample_sum(
    series: pd.DataFrame,
    min_date: datetime,
    max_date: datetime,
    freq: str,
    value_col="profit_percent",
    date_col="close_date",
    fill_value=0,
):
    t_index = date_range(start=min_date, end=max_date, freq=freq, normalize=True)
    return (
        series.resample(freq, on=date_col)
        .agg({value_col: sum})
        .reindex(t_index)
        .fillna(fill_value)
    )


def read_json_file(file_path: Union[Path, str], key=""):
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
