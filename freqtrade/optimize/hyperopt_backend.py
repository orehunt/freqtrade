from typing import Any, Dict, List, Tuple
from queue import Queue
from multiprocessing.managers import SyncManager
from pandas import DataFrame, concat, read_hdf, merge
from numpy import arange, isfinite
from pathlib import Path

from freqtrade.constants import HYPEROPT_LIST_STEP_VALUES
from freqtrade.exceptions import OperationalException

hyperopt: Any = None
manager: SyncManager
# stores the optimizers in multi opt mode
optimizers: Queue
# stores the results to share between optimizers
# in the form of key = Tuple[Xi], value = Tuple[float, int]
# where float is the loss and int is a decreasing counter of optimizers
# that have registered the result
results_shared: Dict[Tuple, Tuple]
# in single mode the results_list is used to pass the results to the optimizer
# to fit new models
results_list: List
# results_batch stores keeps results per batch that are eventually logged and stored
results_batch: Queue


class Trial:
    """
    Data representing one backtest sample
    """

    def __init__(
        self,
        loss: float,
        params_dict: Dict,
        params_details: Dict,
        results_metrics: Dict,
        results_explanation: str,
        total_profit: float,
    ):
        self.loss = loss
        self.params_dict = params_dict
        self.params_details = params_details
        self.results_metrics = results_metrics
        self.results_explanation = results_explanation
        self.total_profit = total_profit


def trials_to_df(trials: List, metrics: bool = False) -> Tuple[DataFrame, str]:
    df = DataFrame(trials)
    last_col = df.columns[-1]
    if metrics:
        df_metrics = DataFrame(t["results_metrics"] for t in trials)
        return concat([df, df_metrics], axis=1), last_col
    else:
        return df, last_col


def filter_options(config: Dict[str, Any]):
    """ parse filtering config options into dict """
    return {
        "best": config.get("hyperopt_list_best", "sum"),
        "n_best": config.get("hyperopt_list_n_best", 10),
        "ratio_best": config.get("hyperopt_list_ratio_best", 0.99),
        "profitable": config.get("hyperopt_list_profitable", False),
        "min_trades": config.get("hyperopt_list_min_trades", 0),
        "max_trades": config.get("hyperopt_list_max_trades", 0),
        "min_avg_time": config.get("hyperopt_list_min_avg_time", None),
        "max_avg_time": config.get("hyperopt_list_max_avg_time", None),
        "min_avg_profit": config.get("hyperopt_list_min_avg_profit", None),
        "max_avg_profit": config.get("hyperopt_list_max_avg_profit", None),
        "min_total_profit": config.get("hyperopt_list_min_total_profit", None),
        "max_total_profit": config.get("hyperopt_list_max_total_profit", None),
        "step_values": config.get("hyperopt_list_step_values", HYPEROPT_LIST_STEP_VALUES),
        "step_key": config.get("hyperopt_list_step_metric", None),
        "sort_key": config.get("hyperopt_list_sort_metric", "loss"),
    }


def filter_trials(trials: Any, config: Dict[str, Any]) -> List:
    """
    Filter our items from the list of hyperopt results
    """
    trials, trials_last_col = trials_to_df(trials, metrics=True)
    filters = filter_options(config)

    if filters["best"]:
        return norm_best(trials, trials_last_col, filters)
    if filters["profitable"]:
        trials = trials.loc[trials["profit"] > 0]
    if filters["min_trades"]:
        trials = trials.loc[trials["trade_count"] > filters["min_trades"]]
    if filters["max_trades"]:
        trials = trials.loc[trials["trade_count"] < filters["max_trades"]]

    with_trades = trials.loc[trials["trade_count"] > 0]
    with_trades_len = len(with_trades)
    if filters["min_avg_time"]:
        with_trades = with_trades.loc[with_trades["duration"] > filters["min_avg_time"]]
    if filters["max_avg_time"]:
        with_trades = with_trades.loc[with_trades["duration"] < filters["max_avg_time"]]
    if filters["min_avg_profit"]:
        with_trades = with_trades.loc[with_trades["avg_profit"] > filters["min_avg_profit"]]
    if filters["max_avg_profit"]:
        with_trades = with_trades.loc[with_trades["avg_profit"] < filters["max_avg_profit"]]
    if filters["min_total_profit"]:
        with_trades = with_trades.loc[with_trades["profit"] > filters["min_total_profit"]]
    if filters["max_total_profit"]:
        with_trades = with_trades.loc[with_trades["profit"] < filters["max_total_profit"]]
    if len(with_trades) != with_trades_len:
        trials = with_trades

    if len(trials) > 0:
        return sample_trials(trials, trials_last_col, filters)
    else:
        return trials.to_dict(orient="records")


def norm_best(trials: Any, trials_last_col, filters: dict) -> List:
    metrics = ("profit", "avg_profit", "duration", "trade_count", "loss")
    min_ratio = filters["ratio_best"]
    n_best = filters["n_best"]

    # invert loss and duration to simplify
    trials["loss"] = trials["loss"].mul(-1)
    trials["duration"] = trials["duration"].mul(-1)
    # calculate the normalized metrics as columns
    for m in metrics:
        m_col = trials[m].values
        m_min = m_col.min()
        m_max = m_col.max()
        trials[f"norm_{m}"] = (m_col - m_min) / (m_max - m_min)
    # re-invert
    trials["loss"] = trials["loss"].mul(-1)
    trials["duration"] = trials["duration"].mul(-1)

    # calc the norm ratio between metrics:
    # compare each normalized metric against the set minimum ratio;
    # also get a sum of all the normalized metrics
    trials["norm_sum"] = 0
    trials["norm_ratio"] = 0
    for m in metrics:
        norm_m = trials[f"norm_{m}"].values
        norm_m[~isfinite(norm_m)] = 0 # reset infs and nans
        trials["norm_ratio"] += (norm_m > min_ratio).astype(int)
        trials["norm_sum"] += trials[f"norm_{m}"].values

    # You're the best! Around!
    # trials["is_best"] = True

    best_trials = []
    if filters["best"] == "ratio":
        # filter the trials to the ones that meet the min_ratio for all the metrics
        m_best = (
            trials.sort_values("norm_ratio")
            .loc[:, :trials_last_col]
            .iloc[-n_best:]
            .to_dict("records")
        )
        best_trials.extend(m_best)
    else:
        m_best = (
            trials.sort_values("norm_sum")
            .loc[:, :trials_last_col]
            .iloc[-n_best:]
            .to_dict("records")
        )
        best_trials.extend(m_best)

    return best_trials

def sample_trials(trials: Any, trials_last_col: Any, filters: Dict) -> List:
    """ Pick one trial, every `step_value` of `step_metric`, sorted by `sort_metric` """
    if filters["step_key"]:
        step_k = filters["step_key"]
        step_v = filters["step_values"][step_k]
        step_start = trials[step_k].values.min()
        step_stop = trials[step_k].values.max()
        steps = arange(step_start, step_stop, step_v)
        flt_trials = []
        last_epoch = None
        sort_key = filters["sort_key"]
        ascending = sort_key in ("duration", "loss")
        if len(steps) > len(trials):
            min_step = step_v * (len(steps) / len(trials))
            raise OperationalException(
                f"Step value of {step_v} for metric {step_k} is too small. "
                f"Use a minimum of {min_step:.4f}, or choose closer bounds."
            )
        for n, s in enumerate(steps):
            try:
                t = (
                    # the trials between the current step
                    trials.loc[(trials[step_k].values >= s) & (trials[step_k].values <= s + step_v)]
                    # sorted according to the specified key
                    .sort_values(filters["sort_key"], ascending=ascending)
                    # select the columns of the trial, and return the first row
                    .loc[:, :trials_last_col]
                    .iloc[0]
                    .to_dict()
                )
                if t["current_epoch"] == last_epoch:
                    break
                else:
                    last_epoch = t["current_epoch"]
                    flt_trials.append(t)
            except IndexError:
                pass
    else:
        flt_trials = trials.loc[:, :trials_last_col].to_dict(orient="records")
    return flt_trials


def save_trials(trials: Any, path: Path, offset: int = 0):
    trials, _ = trials_to_df(trials)
    trials.to_hdf(path, key="trials", append=True, format="table", complevel=9, mode="a")


def load_trials(path: Path):
    trials = read_hdf(path, key="trials")
    return trials
