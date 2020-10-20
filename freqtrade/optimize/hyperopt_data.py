import json
import logging
import os
import random
from datetime import datetime
from os import makedirs
from pathlib import Path
from shutil import rmtree
from time import sleep
from typing import Any, Dict, Iterable, List, Tuple, Union

import numcodecs
import numpy as np
import zarr as za
from numpy import array, iinfo, int32
from numpy import repeat as np_repeat
from operator import itemgetter


za.storage.default_compressor = za.Blosc(cname="zstd", clevel=2)


import io
from os import path

from numpy import arange, float64, isfinite, nanmean
from pandas import DataFrame, concat, isna, json_normalize

# Import IHyperOpt and IHyperOptLoss to allow unpickling classes from these modules
import freqtrade.optimize.hyperopt_backend as backend
from freqtrade.constants import HYPEROPT_LIST_STEP_VALUES
from freqtrade.exceptions import DependencyException, OperationalException
from freqtrade.misc import plural, round_dict
from freqtrade.optimize.hyperopt_backend import Epochs, TrialsState
from freqtrade.optimize.optimizer import IOptimizer, Parameter


logger = logging.getLogger(__name__)


class HyperoptData(backend.HyperoptBase):
    """
    Data and state for Hyperopt
    """

    # run modes
    mode: str
    multi: bool
    shared: bool
    cv: bool

    # total number of candles being backtested
    n_candles = 0

    # a guessed number extracted by the space dimensions
    search_space_size: int

    # evaluations
    min_epochs: int
    epochs_limit: int
    total_epochs: int
    trials: DataFrame
    space_reduction_interval: int
    space_reduction_config = None

    # identifier for HyperOpt class, set of parameters, loss function
    trials_instance: str
    # store the indentifier string here
    trials_instances_file: Path
    # when to save trials
    trials_timeout: float
    trials_maxout: int
    trials_max_empty: int
    # where to save trials
    trials_file: Path
    trials_dir: Path

    opt: IOptimizer
    # the maximum time to wait for the oracle
    # after which optimization is stopped
    opt_ask_timeout: float
    # in single mode, tell results every n points
    opt_tell_frequency: int
    # list of all the optimizers random states
    rngs: List[int]
    # path where the hyperopt state loaded by workers is dumped
    cls_file: Path
    # path used by CV to store parameters values loaded by workers
    Xi_path: Path
    # tune exploration/exploitation based on heuristics
    adjust_acquisition = True
    parameters: List[Parameter]
    n_rand = 10
    # True if optimization mode is shared or multi
    async_sched = False

    metrics = ("profit", "avg_profit", "duration", "trade_count", "loss")
    min_date: datetime
    max_date: datetime

    _store: za.storage.DirectoryStore
    _group: za.hierarchy.Group

    def __init__(self, config):
        self.config = config

        self.total_epochs = self.config.get("hyperopt_epochs", 0)

        self.hyperopt_dir = "hyperopt_data"

        self.trials_dir = self.config["user_data_dir"] / self.hyperopt_dir / "trials"

        if not os.path.exists(self.trials_dir):
            os.makedirs(self.trials_dir)

        self.trials_instances_file = (
            self.config["user_data_dir"] / self.hyperopt_dir / "trials_instances.json"
        )
        self.data_pickle_file = (
            self.config["user_data_dir"] / self.hyperopt_dir / "processed_data.pkl"
        )
        self.Xi_path = self.config["user_data_dir"] / self.hyperopt_dir / "Xi"
        self.cls_file = self.config["user_data_dir"] / self.hyperopt_dir / "cls.pkl"
        self.trials_file = self.get_trials_file(self.config, self.trials_dir)
        self._store = za.storage.DirectoryStore(self.trials_file)
        self._group = za.hierarchy.open_group(store=self._store)

    def clear_hyperopt(self) -> None:
        """
        Remove hyperopt pickle files to restart hyperopt.
        """
        if not self.trials_dir.is_dir():
            makedirs(self.trials_dir)
        for f in [self.trials_file]:
            p = Path(f)
            logger.info(f"Removing `{p}`.")
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                rmtree(p)

    @staticmethod
    def _space_params(params, space: str, r: int = None) -> Dict:
        d = params[space]
        # Round floats to `r` digits after the decimal point if requested
        return round_dict(d, r) if r else d

    @staticmethod
    def _params_update_for_json(result_dict, params, space: str) -> None:
        if space in params:
            space_params = HyperoptData._space_params(params, space)
            if space in ["buy", "sell"]:
                result_dict.setdefault("params", {}).update(space_params)
            else:  # 'stoploss', 'trailing'
                result_dict.update(space_params)

    @staticmethod
    def cast_trials_types(trials: DataFrame) -> DataFrame:
        """ Force types for ambiguous metrics """
        trials = trials.astype(
            dtype={
                "total_profit": float64,
                "loss": float64,
                "total_profit": float64,
                "avg_profit": float64,
                "duration": float64,
            },
            copy=False,
        ).fillna(0)
        return trials

    @staticmethod
    def save_trials(
        trials: DataFrame,
        trials_file: Path,
        instance_name: str,
        trials_state: TrialsState = TrialsState(),
        indexer: Tuple = (),
        fields: List = [],
        final: bool = False,
        backup: bool = False,
        append: bool = True,
    ) -> None:
        """
        Save hyperopt trials
        """
        num_trials = len(trials)
        interval = 0.5
        saved = num_trials < 1
        locked = False
        # this is needed because it is a string that can exceed
        # the size preset by pd, and in append mode it can't be changed
        # min_itemsize = {"results_explanation": 120}
        # if "roi" in trials.columns:  # roi stored as json
        #     min_itemsize["roi"] = 190
        # NOTE: make sure to only index the columns really used for indexing
        # since each parameter is stored in a different col, the col number can
        # be very high
        # data_columns = ["Xi_h", "random_state", "loss"]

        while not saved:
            try:
                logger.debug("Saving %s %s.", num_trials, plural(num_trials, "epoch"))
                # cast types
                trials = HyperoptData.cast_trials_types(trials)
                # save on storage, lock is blocking
                locked = backend.acquire_lock(trials_state)
                if locked:
                    try:

                        HyperoptData._to_storage(
                            trials[fields if fields else slice(None)],
                            trials_file,
                            instance_name,
                            indexer=indexer,
                            fields=fields,
                            append=append,
                        )

                    except ValueError as e:
                        logger.info(e)

                if not backup:
                    trials_state.num_saved += num_trials
                    if final:
                        logger.info(
                            f"{trials_state.num_saved} {plural(trials_state.num_saved, 'epoch')} "
                            f"saved to '{trials_file}'."
                        )
                saved = True
            except (IOError, OSError, AttributeError) as e:
                # If a lock on the file can't be acquired
                if isinstance(e, AttributeError):
                    # reset table as it has been corrupted
                    append = False
                logger.warn(
                    f"Couldn't access the trials storage, {e} retrying in {interval}.."
                )
                sleep(interval)
                interval += 0.5
            finally:
                if locked:
                    backend.release_lock(trials_state)
                    locked = False

    def _from_group(self, fields=[], indexer=slice(None)) -> DataFrame:
        z = self._group.get(self.trials_instance)
        if z is None:
            return DataFrame()
        columns = z.attrs.get("columns")
        if fields:
            idx = [columns[f] for f in fields]
            indexer = (indexer, idx)
        z = self._group.get(self.trials_instance)
        if z:
            data = z.get_orthogonal_selection(indexer)
        else:
            return DataFrame()
        # consider 1d arrays as 1 row
        if len(data.shape) == 1:
            data.shape = (1, data.shape[0])
        return DataFrame(data=data, columns=fields or columns)

    @staticmethod
    def _to_storage(
        trials: DataFrame,
        trials_location: Union[Path, za.storage.DirectoryStore],
        key: str,
        indexer=(),
        fields=[],
        chunks=None,
        append: bool = True,
    ):
        if isinstance(trials_location, Path):
            trials_location = za.storage.DirectoryStore(trials_location)
        g = za.open_group(store=trials_location)
        z = g.get(key)
        if z is not None and (append or indexer) and len(z):
            if append:
                z.append(trials[fields or slice(None)].values)
            else:
                if fields:
                    columns = z.attrs["columns"]
                    indexer = (indexer, [columns[f] for f in fields])
                if indexer:
                    z.set_coordinate_selection(indexer, value=trials.values)
        else:
            if key in g:
                del g[key]
            trials = trials[fields or slice(None)]
            g.create_dataset(
                name=key,
                data=trials.values,
                dtype=object,
                object_codec=numcodecs.Pickle(),
                shape=trials.shape,
            )
            g[key].attrs["columns"] = {col: n for n, col in enumerate(trials.columns)}

    @staticmethod
    def _from_storage(
        trials_location: Union[Path, za.storage.DirectoryStore],
        key: str,
        indexer=slice(None),
        fields: List = [],
    ) -> DataFrame:
        if isinstance(trials_location, Path):
            trials_location = za.storage.DirectoryStore(trials_location)
        g = za.group(store=trials_location)
        trials = g.get(key)
        if trials:
            columns = trials.attrs.get("columns", {})
            if fields:
                indexer = (indexer, [columns[f] for f in fields])
            else:
                if isinstance(columns, dict):
                    columns = dict(sorted(columns.items(), key=itemgetter(1)))
            trials = trials.get_orthogonal_selection(indexer)
        else:
            columns = []
        return DataFrame(trials, columns=fields or columns)

    @staticmethod
    def _read_trials(
        trials_location: Union[Path, za.storage.DirectoryStore],
        trials_instance: str,
        backup: bool,
        trials_state: TrialsState = backend.trials,
        fields: List = [],
        indexer=slice(None),
        use_backup=True,
    ) -> DataFrame:
        """
        Read hyperopt trials file
        """
        # Only log at the beginning
        if hasattr(backend.trials, "exit") and not backend.trials.exit:
            logger.info("Reading Trials from '%s'", trials_location)
        trials = DataFrame()
        locked = False
        try:
            locked = (
                backend.acquire_lock(trials_state)
                if hasattr(trials_state, "lock")
                else None
            )
            if locked is not None:
                while not locked:
                    logger.debug("Acquiring trials state lock for reading trials")
                    locked = backend.acquire_lock(trials_state)
            trials = HyperoptData._from_storage(
                trials_location, key=trials_instance, indexer=indexer, fields=fields
            )
            if locked:
                backend.release_lock(trials_state)
                locked = False
            # make a copy of the trials in case this optimization run corrupts it
            # (wrongful termination)
            if backup and len(trials) > 0:
                HyperoptData.save_trials(
                    trials,
                    trials_location,
                    instance_name=f"{trials_instance}_bak",
                    trials_state=trials_state,
                    backup=True,
                )
            elif len(trials) < 1 and use_backup:
                logger.warn(
                    f"Instance table {trials_instance} appears empty, using backup..."
                )
                trials = HyperoptData._from_storage(
                    trials_location, f"{trials_instance}_bak", indexer
                )
        except (
            KeyError,
            AttributeError,
        ) as e:  # trials instance is not in the database or corrupted
            # if corrupted
            if backup or not indexer:
                try:
                    logger.warning(
                        f"Instance table {trials_instance} either "
                        "empty or corrupted, trying backup..."
                    )
                    trials = HyperoptData._from_storage(
                        trials_location, f"{trials_instance}_bak", indexer
                    )
                except KeyError:
                    logger.warning(f"Backup not found...")
            else:
                logger.warning("trials not found at... %s", e)
        finally:
            if locked:
                backend.release_lock(trials_state)
        return trials

    @staticmethod
    def trials_to_csv_file(
        config: dict,
        trials: DataFrame,
        total_epochs: int,
        highlight_best: bool,
        csv_file: str,
    ) -> None:
        """
        Log result to csv-file
        """
        if len(trials) < 1:
            return

        # Verification for overwrite
        if path.isfile(csv_file):
            logger.error("CSV-File already exists!")
            return

        try:
            io.open(csv_file, "w+").close()
        except IOError:
            logger.error("Filed to create CSV-File!")
            return

        trials["Best"] = ""
        trials["Stake currency"] = config["stake_currency"]
        trials = trials[
            [
                "Best",
                "current_epoch",
                "results_metrics.trade_count",
                "results_metrics.avg_profit",
                "results_metrics.total_profit",
                "Stake currency",
                "results_metrics.profit",
                "results_metrics.duration",
                "loss",
                "is_initial_point",
                "is_best",
            ]
        ]
        trials.columns = [
            "Best",
            "Epoch",
            "Trades",
            "Avg profit",
            "Total profit",
            "Stake currency",
            "Profit",
            "Avg duration",
            "Objective",
            "is_initial_point",
            "is_best",
        ]
        trials["is_profit"] = False
        trials.loc[trials["is_initial_point"], "Best"] = "*"
        trials.loc[trials["is_best"], "Best"] = "Best"
        trials.loc[trials["Total profit"] > 0, "is_profit"] = True
        trials["Epoch"] = trials["Epoch"].astype(str)
        trials["Trades"] = trials["Trades"].astype(str)

        trials["Total profit"] = trials["Total profit"].apply(
            lambda x: "{:,.8f}".format(x) if x != 0.0 else ""
        )
        trials["Profit"] = trials["Profit"].apply(
            lambda x: "{:,.2f}".format(x) if not isna(x) else ""
        )
        trials["Avg profit"] = trials["Avg profit"].apply(
            lambda x: "{:,.2f}%".format(x) if not isna(x) else ""
        )
        trials["Avg duration"] = trials["Avg duration"].apply(
            lambda x: "{:,.1f} m".format(x) if not isna(x) else ""
        )
        trials["Objective"] = trials["Objective"].apply(
            lambda x: "{:,.5f}".format(x) if x != 100000 else ""
        )

        trials = trials.drop(columns=["is_initial_point", "is_best", "is_profit"])
        trials.to_csv(csv_file, index=False, header=True, mode="w", encoding="UTF-8")
        logger.info(f"CSV-File created at {csv_file} !")

    @staticmethod
    def _filter_options(config: Dict[str, Any]):
        """ Parse filtering config options into dict """
        return {
            "enabled": config.get("hyperopt_list_filter", True),
            "dedup": config.get("hyperopt_list_dedup", False),
            "best": config.get("hyperopt_list_best", []),
            "pct_best": config.get("hyperopt_list_pct_best", 0.1),
            "cutoff_best": config.get("hyperopt_list_cutoff_best", 0.99),
            "trail": config.get("hyperopt_list_trail_bounds", True),
            "no_trades": config.get("hyperopt_list_keep_no_trades", False),
            "min_trades": config.get("hyperopt_list_min_trades", None),
            "max_trades": config.get("hyperopt_list_max_trades", None),
            "min_avg_time": config.get("hyperopt_list_min_avg_time", None),
            "max_avg_time": config.get("hyperopt_list_max_avg_time", None),
            "min_avg_profit": config.get("hyperopt_list_min_avg_profit", None),
            "max_avg_profit": config.get("hyperopt_list_max_avg_profit", None),
            "min_total_profit": config.get("hyperopt_list_min_total_profit", None),
            "max_total_profit": config.get("hyperopt_list_max_total_profit", None),
            "step_values": config.get(
                "hyperopt_list_step_values", HYPEROPT_LIST_STEP_VALUES
            ),
            "step_keys": config.get("hyperopt_list_step_metric", []),
            "sort_keys": config.get("hyperopt_list_sort_metric", ["loss"]),
        }

    @staticmethod
    def list_or_df(d: DataFrame, return_list: bool) -> Any:
        if return_list:
            return d.to_dict("records")
        else:
            return d

    @staticmethod
    def params_Xi(v: dict):
        return list(v["params_dict"].values())

    @staticmethod
    def filter_trials(
        trials: Any, config: Dict[str, Any], return_list=False, intensity=1
    ) -> Any:
        """
        Filter our items from the list of hyperopt trials
        """
        hd = HyperoptData
        filters = hd._filter_options(config)
        trials = trials.infer_objects()

        if not filters["enabled"] or len(trials) < 1:
            return hd.list_or_df(trials, return_list)
        if filters["no_trades"]:
            no_trades = trials.loc[trials["trade_count"] < 1]
        else:
            no_trades = DataFrame()

        trials = trials.loc[trials["trade_count"] > 0]
        filters_col = {
            "trades": "trade_count",
            "avg_time": "duration",
            "avg_profit": "avg_profit",
            "total_profit": "profit",
        }
        for bound in ("min", "max"):
            for f, c in filters_col.items():
                if filters[f"{bound}_{f}"] is not None:
                    trials = HyperoptData.trim_bounds(
                        trials, filters["trail"], c, bound, filters[f"{bound}_{f}"],
                    )

        if len(trials) > 0:
            flt_trials = [no_trades]
            if filters["dedup"]:
                flt_trials.append(hd.dedup_trials(trials))
            if filters["best"]:
                flt_trials.append(hd.norm_best(trials, filters, intensity))
            flt_trials.append(hd.sample_trials(trials, filters, intensity))
            return hd.list_or_df(
                # filtering can overlap, drop dups
                concat(flt_trials).drop_duplicates(subset="current_epoch"),
                return_list,
            )
        else:
            return hd.list_or_df(concat([no_trades, trials]), return_list)

    @staticmethod
    def trim_bounds(
        trials: DataFrame, trail_enabled: Any, col: str, bound: str, val: Any
    ) -> DataFrame:
        """ range calculations require adjustments to the bounds of the trials metrics """
        logger.debug("trimming bounds for col %s", col)
        if bound not in ("min", "max"):
            raise OperationalException("Wrong min max choice")
        if len(trials) < 1:
            return trials
        if bound == "min":
            trail = lambda x, y: x - y  # noqa: E731
            flt = lambda x, y: x.loc[x[col] > y]  # noqa: E731
        else:
            trail = lambda x, y: x + y  # noqa: E731
            flt = lambda x, y: x.loc[x[col] < y]  # noqa: E731
        if trail_enabled:
            # use std to increase and decrease bounds
            val_step = trials[col].values.std() or val
            flt_trials = flt(trials, val)
            iters = 0
            while len(flt_trials) < 1:
                # use an exponential step in case val_step is 0
                # since we don't know the span of the metric
                val = trail(val, val_step or 2 ** iters)
                flt_trials = flt(trials, val)
                iters += 1
            return flt_trials
        else:
            return flt(trials, val)

    @staticmethod
    def norm_best(trials: Any, filters: dict, intensity=1) -> List:
        """ Normalize metrics and sort by sum or minimum score """
        metrics = ("profit", "avg_profit", "duration", "trade_count", "loss")

        # invert loss and duration to simplify
        trials["loss"] = trials["loss"].mul(-1)
        trials["duration"] = trials["duration"].mul(-1)
        # calculate the normalized metrics as columns
        for m in metrics:
            m_col = trials[m].values
            m_min = m_col.min()
            m_max = m_col.max()
            trials[f"norm_{m}"] = (m_col - m_min) / ((m_max - m_min) or 1)
        # re-invert
        trials["loss"] = trials["loss"].mul(-1)
        trials["duration"] = trials["duration"].mul(-1)

        # Calc cutoff percentage based on normalization
        types_best = filters["best"]
        if filters["cutoff_best"] == "std":
            min_ratio = lambda m: 1 - trials[m].values.std()  # noqa: E731
        elif filters["cutoff_best"] == "mean":
            min_ratio = lambda m: 1 - trials[m].values.mean()  # noqa: E731
        else:
            min_ratio = lambda m: filters["cutoff_best"]  # noqa: E731

        # calc the norm ratio between metrics:
        # compare each normalized metric against the set minimum ratio;
        # also get a sum of all the normalized metrics
        trials["norm_sum"] = 0
        trials["norm_ratio"] = 0
        for m in metrics:
            norm_m = trials[f"norm_{m}"].values
            norm_m[~isfinite(norm_m)] = 0  # reset infs and nans
            trials["norm_ratio"] += (norm_m > min_ratio(m)).astype(int)
            trials["norm_sum"] += trials[f"norm_{m}"].values

        # You're the best! Around!
        # trials["is_best"] = True

        # Calc number of trials to keep based on summed normalization
        if filters["pct_best"] == "std":
            pct_best = trials["norm_sum"].values.std()
        elif filters["pct_best"] == "mean":
            pct_best = trials["norm_sum"].values.mean()
        else:
            pct_best = filters["pct_best"]
        n_best = int(len(trials) * pct_best * intensity // len(types_best))
        n_types = len(types_best)
        if n_best < n_types:
            n_best = max(1, n_types)

        if "ratio" in types_best:
            # filter the trials to the ones that meet the min_ratio for all the metrics
            cutoff_best = trials.sort_values("norm_ratio").iloc[-n_best:]
        if "sum" in types_best:
            sum_best = trials.sort_values("norm_sum").iloc[-n_best:]

        return concat([cutoff_best, sum_best]).drop_duplicates(subset="current_epoch")

    @staticmethod
    def dedup_trials(trials: DataFrame) -> DataFrame:
        """ Filter out duplicate metrics, then filter duplicate epochs """
        metrics = HyperoptData.metrics
        dedup_metrics = []
        for m in metrics:
            if m in trials:
                dedup_metrics.append(trials.drop_duplicates(subset=m))
        return concat(dedup_metrics).drop_duplicates(subset="current_epoch")

    @staticmethod
    def sample_trials(trials: DataFrame, filters: Dict, intensity=1) -> DataFrame:
        """
        Pick one trial, every `step_value` of `step_metric`...
        or pick n == `range` trials for every `step_metric`...
        for every `step_metric`, sorted by `sort_metric` for every `sort_metric`...
        """
        metrics = HyperoptData.metrics
        if filters["step_keys"]:
            step_keys = (
                metrics if filters["step_keys"] == ["all"] else filters["step_keys"]
            )
            sort_keys = (
                metrics if filters["sort_keys"] == ["all"] else filters["sort_keys"]
            )
            step_values = filters["step_values"]
            flt_trials = []
            for step_k in step_keys:
                for sort_k in sort_keys:
                    flt_trials.extend(
                        HyperoptData.step_over_trials(
                            step_k, step_values, sort_k, trials, intensity
                        )
                    )
        else:
            flt_trials = [trials]
        if flt_trials:
            # stepping can overlap, dedup
            return concat(flt_trials).drop_duplicates(subset="current_epoch")
        else:
            return DataFrame()

    @staticmethod
    def find_steps(
        step_k: str, step_values: Dict, trials: DataFrame
    ) -> Tuple[List, Any]:
        """
        compute the range of steps to perform over the trials metrics
        """
        finite_k = trials[step_k].loc[isfinite(trials[step_k])]
        step_start = finite_k.values.min()
        step_stop = finite_k.values.max()
        # choose the value of each step automatically if
        # a number of steps is specified
        defined_range = step_values.get("range", "mean")
        if defined_range:
            if defined_range == "mean":
                step_v = nanmean((finite_k - finite_k.shift(1)))
            elif defined_range == "std":
                step_v = finite_k.values.std()
            else:
                step_v = (step_stop - step_start) / step_values["range"]
        else:
            step_v = step_values[step_k]
        if step_start == step_stop:
            steps = [step_start]
        else:
            try:
                steps = arange(step_start, step_stop, step_v)
            except (ValueError, MemoryError):
                steps = array([])
        n_trials = len(trials)
        if len(steps) > n_trials:
            min_step = step_v * (len(steps) / n_trials)
            if not defined_range:
                logger.warn(
                    f"Step value of {step_v} for metric {step_k} is too small. "
                    f"Using a minimum of {min_step:.4f}"
                )
            step_v = min_step
            steps = np_repeat(step_v, n_trials).cumsum()
        return steps, step_v

    @staticmethod
    def step_over_trials(
        step_k: str, step_values: Dict, sort_k: str, trials: DataFrame, intensity=1
    ) -> List:
        """ Apply the sampling of a metric_key:sort_key combination over the trials """
        # for duration and loss we sort by the minimum
        ascending = sort_k in ("duration", "loss")
        flt_trials = []
        last_epoch = None
        steps, step_v = HyperoptData.find_steps(step_k, step_values, trials)
        # adjust steps by the given intensity
        steps = steps[:: int(len(steps) / intensity) or 1]
        step_v /= intensity

        # print("looping over {steps} steps!")
        for _, s in enumerate(steps):
            try:
                t = (
                    # the trials between the current step
                    trials.loc[
                        (trials[step_k].values >= s)
                        & (trials[step_k].values <= s + step_v)
                    ]
                    # sorted according to the specified key
                    .sort_values(sort_k, ascending=ascending).iloc[
                        [-1]
                    ]  # use double brackets to return the dataframe
                )
                if t["current_epoch"].iat[-1] == last_epoch:
                    break
                else:
                    last_epoch = t["current_epoch"].iat[-1]
                    flt_trials.append(t)
            except IndexError:
                pass
        return flt_trials

    @staticmethod
    def load_trials(
        trials_file: Path,
        trials_instance: str,
        trials_state: TrialsState = backend.trials,
        indexer=slice(None),
        backup=False,
        use_backup=True,
        clear=False,
    ) -> DataFrame:
        """
        Load data for epochs from the file if we have one
        """
        trials: DataFrame = DataFrame()
        has_lock = hasattr(trials_state, "lock")
        # locked = False
        if (
            trials_file.is_dir() or trials_file.is_file()
        ) and trials_file.stat().st_size > 0:
            trials = HyperoptData._read_trials(
                trials_file,
                trials_instance,
                backup=backup,
                trials_state=trials_state,
                indexer=indexer,
                use_backup=use_backup,
            )
            # clear the table by replacing it with an empty df
            if clear:
                HyperoptData.clear_instance(
                    trials_file,
                    trials_instance,
                    trials_state=trials_state if has_lock else None,
                )
        return trials

    @staticmethod
    def get_last_instance(trials_instances_file: Path, cv=False) -> str:
        """
        When an instance is not specified get the last one saved,
        should be used by hyperopt related commands
        """
        with open(trials_instances_file, "r") as tif:
            instances = json.load(tif)
        if len(instances) < 1:
            raise OperationalException(
                f"No instances were found at {trials_instances_file}"
            )
        else:
            if cv:
                return "{}_cv".format(instances[-1])
            else:
                return instances[-1]

    @staticmethod
    def clear_instance(
        trials_file: Path, instance_name: str, trials_state=None, backup=False,
    ) -> bool:
        success = False
        locked = backend.acquire_lock(trials_state) if trials_state else False
        interval = 0.01
        while not ((trials_state and locked) or not trials_state):
            logger.debug("Acquiring trials state lock for clearing trials instance")
            sleep(interval)
            locked = backend.acquire_lock(trials_state)
            interval += 0.5
        try:
            with za.open(str(trials_file)) as store:
                del store["{}".format(instance_name)]
                if backup:
                    del store["{}_bak".format(instance_name)]
                success = True
        except (KeyError, IOError, OSError, AttributeError) as e:
            logger.debug("Failed clearing instance: %s", e)
            pass
        if locked:
            backend.release_lock(trials_state)
            locked = False
        return success

    @staticmethod
    def get_trials_file(config: dict, trials_dir: Path) -> Path:
        hyperopt = config["hyperopt"]
        strategy = config["strategy"]
        if not hyperopt or not strategy:
            raise DependencyException(
                "Strategy or Hyperopt name not specified, both are required."
            )
        trials_file = trials_dir / f"{hyperopt}_{strategy}"
        return trials_file

    def _setup_optimizers(self):
        """
        Setup the optimizers objects, applies random state from saved trials if present,
        adds a few attributes to determine logic of execution during trials evaluation
        """
        # try to load previous optimizers
        if self.async_sched:
            # on startup distribute reproduced optimizers
            backend.optimizers = backend.manager.Queue()
            max_opts = self.n_jobs
            rngs = []
            # generate as many optimizers as the job count
            if len(self.trials) > 0:
                rngs = self.trials["random_state"].drop_duplicates().values.tolist()
                # make sure to only load as many random states as the job count
                prev_rngs = rngs[-max_opts:]
            else:
                prev_rngs = []
            rngs = []
            random.seed(self.random_state)
            for _ in range(max_opts):  # generate optimizers
                # random state is preserved
                if len(prev_rngs) > 0:
                    rs = prev_rngs.pop(0)
                else:
                    rs = random.randint(0, np.int32(iinfo(int32).max))
                # in multi mode generate a new optimizer
                # to randomize the base estimator
                opt_copy = self.get_optimizer(
                    random_state=rs, parameters=self.parameters
                )
                rngs.append(rs)
                backend.optimizers.put(opt_copy)
            del opt_copy
            self.rngs = rngs
        else:
            self.opt = self.get_optimizer(self.random_state, parameters=self.parameters)
            self.rngs = [self.opt.rs]

    def apply_space_reduction(
        self, jobs: int, trials_state: TrialsState, epochs: Epochs
    ):
        # fetch all trials
        trials = self.load_trials(
            self.trials_file, self.trials_instance, trials_state, use_backup=False
        )
        if not len(trials):
            return False
        min_trials = self.n_rand
        is_shared_or_single = self.shared or not self.multi
        if is_shared_or_single:
            # update optimizers with new dimensions
            # in shared/single mode only need to compute dimensions once
            trials_flt = self.filter_trials_by_opt(None, trials, min_trials)
            if trials_flt is None:
                return False
            new_pars = self.reduce_spaces(self.parameters.copy(), trials_flt)
        else:
            new_pars = []
        reduced_optimizers = []
        reduced_losses = []
        if self.mode != "single":
            # make sure all workers are idle
            backend.wait_for_lock(epochs.lock, "space_reduction", logger)
            if len(epochs.space_reduction) != 0 or backend.optimizers.qsize() < jobs:
                qs = backend.optimizers.qsize()
                ds = len(epochs.pinned_optimizers)
                raise OperationalException(
                    f" workers unsync, optimizers queue: {qs}, mapping: {ds}"
                )
            for oid, opt in epochs.pinned_optimizers.items():
                if self.multi:
                    opt_trials = self.filter_trials_by_opt(opt.rs, trials, min_trials)
                    if opt_trials is None:
                        continue
                    reduced_losses.extend(opt_trials["loss"].values.tolist())
                opt_pars = (
                    # in shared or single mode
                    new_pars
                    # in multi mode when there are filtered trials
                    or self.reduce_spaces(self.parameters.copy(), opt_trials,)
                )
                # don't update space if this optimizer didn't have filtered trials
                if opt_pars:
                    reduced_optimizers.append(opt.rs)
                    epochs.pinned_optimizers[oid] = opt.create_optimizer(opt_pars)
                epochs.space_reduction[oid] = opt_pars is True
            backend.release_lock(backend.epochs)
        else:
            if new_pars:
                self.opt = self.opt.create_optimizer(new_pars)
                reduced_optimizers.append(self.opt.rs)

        # needed for clearing trials outside of new space from storage
        # if is_shared_or_single:
        #     loss_vals = trials_flt["loss"].unique().tolist()
        # elif is_multi:
        #     loss_vals = reduced_losses
        return True

    def filter_trials_by_opt(
        self, rs: Union[int, None], trials, min_trials
    ) -> Union[None, DataFrame]:
        # filter all the trials at once
        trials = HyperoptData.filter_trials(trials, self.space_reduction_config)
        if rs is not None:
            trials = trials.loc[trials["random_state"].values == rs]
        if len(trials) < min_trials:
            logger.debug(
                "Can't reduce space since filtered trials are less "
                "than starting random points"
            )
            return None
        logger.debug("Applying search space reduction over %s trials..", len(trials))
        return trials

    @staticmethod
    def reduce_spaces(parameters: List, trials: DataFrame) -> List:
        # iterate over each parameter to find new min max
        rs_trials = json_normalize(trials["params_dict"])
        for n, par in enumerate(parameters):
            p = parameters[n]
            HyperoptData.min_max_parameter(p, rs_trials)

        return parameters

    @staticmethod
    def col_min_max(p, df):
        p.low = df[p.name].values.min()
        p.high = df[p.name].values.max()

    @staticmethod
    def min_max_parameter(p: Parameter, rs_trials: DataFrame):
        # if it's a range, set low and high bounds
        if p.kind == 1:
            HyperoptData.col_min_max(p, rs_trials)
        # category elements not present in filtered trials are removed
        elif p.kind == 0:
            new_cats = rs_trials[p.name].unique().tolist()
            # reduce priors list to filtered categories since
            # categories probabilities are user defined
            p.sub = new_cats
            # if prior is a list of probabilities, renormalize to 1.
            if "prior" in p.meta:
                prior = p.meta["prior"]
                if isinstance(prior, Iterable):
                    arr = np.asarray(prior)
                    prior = arr / arr.sum()
                    p.meta["prior"] = prior
        else:
            if isinstance(p.sub, np.ndarray):
                HyperoptData.col_min_max(p, rs_trials)
            elif isinstance(p.sub, Parameter):
                HyperoptData.min_max_parameter(p.sub, rs_trials)
            elif isinstance(p.sub, Iterable):
                for el in p.sub:
                    if isinstance(el, Parameter):
                        HyperoptData.min_max_parameter(el, rs_trials)
                    elif isinstance(el, np.ndarray):
                        raise NotImplementedError("mixed types reduction not supported")
                        # HyperoptData.col_min_max(el, rs_trials)
