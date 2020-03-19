from typing import Any, Dict, List, Tuple
from queue import Queue
from multiprocessing.managers import SyncManager
from pandas import DataFrame, concat, read_hdf
from numpy import arange
from pathlib import Path

hyperopt: Any = None
manager: SyncManager
# stores the optimizers in multi opt mode
optimizers: Queue
# stores a list of the results to share between optimizers
# in the form of (set(Xi), set(yi))
results_board: Queue
# store the results in single opt mode
results: Queue


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
        "best": config.get("hyperopt_list_best", False),
        "profitable": config.get("hyperopt_list_profitable", False),
        "min_trades": config.get("hyperopt_list_min_trades", 0),
        "max_trades": config.get("hyperopt_list_max_trades", 0),
        "min_avg_time": config.get("hyperopt_list_min_avg_time", None),
        "max_avg_time": config.get("hyperopt_list_max_avg_time", None),
        "min_avg_profit": config.get("hyperopt_list_min_avg_profit", None),
        "max_avg_profit": config.get("hyperopt_list_max_avg_profit", None),
        "min_total_profit": config.get("hyperopt_list_min_total_profit", None),
        "max_total_profit": config.get("hyperopt_list_max_total_profit", None),
        "step_value": config.get("hyperopt_list_step_value", 0),
        "step_key": config.get("hyperopt_list_step_metric", None),
        "sort_key": config.get("hyperopt_list_sort_metric", "loss"),
        "sort_order": config.get("hyperopt_list_sort_order", "ascending") == "ascending",
    }


def filter_trials(trials: Any, config: Dict[str, Any]) -> List:
    """
    Filter our items from the list of hyperopt results
    """
    trials, trials_last_col = trials_to_df(trials, metrics=True)
    filters = filter_options(config)

    if filters["best"]:
        trials = trials.loc[trials["is_best"] is True]
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

    return sample_trials(trials, trials_last_col, filters)


def sample_trials(trials: Any, trials_last_col: Any, filters: Dict) -> List:
    if filters["step_value"]:
        step_k = filters["step_key"]
        step_v = filters["step_value"]
        step_start = trials[step_k].min()
        step_stop = trials[step_k].max()
        steps = arange(step_start, step_stop, step_v)
        flt_trials = []
        last_epoch = None
        for n, s in enumerate(steps):
            try:
                t = (
                    trials.loc[(trials[step_k].values > s) & (trials[step_k].values < s + step_v)]
                    .sort_values(filters["sort_key"], ascending=filters["sort_order"])
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
