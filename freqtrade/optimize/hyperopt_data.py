import logging
import os
import warnings
import json
import random
from numpy import iinfo, int32
from pathlib import Path
from typing import Dict, List, Any
from abc import abstractmethod
from time import sleep
from os import makedirs


from multiprocessing.managers import Namespace
from pandas import DataFrame, HDFStore, concat, isna, read_hdf
from numpy import arange, float64, isfinite, nanmean
from os import path
import io


from freqtrade.constants import HYPEROPT_LIST_STEP_VALUES
from freqtrade.exceptions import OperationalException, DependencyException
from freqtrade.misc import plural, round_dict

# Import IHyperOpt and IHyperOptLoss to allow unpickling classes from these modules
import freqtrade.optimize.hyperopt_backend as backend

# from freqtrade.optimize.hyperopt_backend import Trial
from freqtrade.optimize.hyperopt_interface import IHyperOpt  # noqa: F401
from freqtrade.optimize.hyperopt_loss_interface import IHyperOptLoss  # noqa: F401

from freqtrade.optimize.hyperopt_constants import VOID_LOSS

# Suppress scikit-learn FutureWarnings from skopt
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from skopt import Optimizer
# Additional regressors already pluggable into the optimizer
# from sklearn.linear_model import ARDRegression, BayesianRidge
# possibly interesting regressors that need predict method override
# from sklearn.ensemble import HistGradientBoostingRegressor
# from xgboost import XGBoostRegressor


logger = logging.getLogger(__name__)


class HyperoptData:
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
    epochs_limit: callable
    total_epochs: int
    trials: DataFrame

    # identifier for HyperOpt class, set of parameters, loss function
    trials_instance: str
    # store the indentifier string here
    trials_instances_file: Path
    Xi_cols: List = []
    # when to save trials
    trials_timeout: float
    trials_maxout: int
    # where to save trials
    trials_file: Path
    trials_dir: Path

    opt: Optimizer

    metrics = ("profit", "avg_profit", "duration", "trade_count", "loss")

    def __init__(self, config):
        self.config = config

        self.total_epochs = self.config["epochs"] if "epochs" in self.config else 0

        self.hyperopt_dir = "hyperopt_data"

        self.trials_dir = self.config["user_data_dir"] / self.hyperopt_dir / "trials"

        if not os.path.exists(self.trials_dir):
            os.makedirs(self.trials_dir)

        self.trials_instances_file = (
            self.config["user_data_dir"] / self.hyperopt_dir / "trials_instances.json"
        )
        self.data_pickle_file = (
            self.config["user_data_dir"] / self.hyperopt_dir / "hyperopt_tickerdata.pkl"
        )

    def clear_hyperopt(self) -> None:
        """
        Remove hyperopt pickle files to restart hyperopt.
        """
        if not self.trials_dir.is_dir():
            makedirs(self.trials_dir)
        for f in [self.trials_file]:
            p = Path(f)
            if p.is_file():
                logger.info(f"Removing `{p}`.")
                p.unlink()

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
        sep = "."
        for col in (
            "total_profit",
            "loss",
            f"results_metrics{sep}total_profit",
            f"results_metrics{sep}avg_profit",
            f"results_metrics{sep}duration",
        ):
            trials[col] = trials[col].astype(float64, copy=False).fillna(0)
        return trials

    @staticmethod
    def alias_cols(trials: DataFrame, prefix: str, sep=".") -> (DataFrame, List[str]):
        cols_names = trials.filter(regex=f"^{prefix}{sep}").columns  # with dot
        stripped_names = []
        for name in cols_names:
            stripped = name.replace(f"{prefix}{sep}", "")
            trials[stripped] = trials[name]
            stripped_names.append(stripped)
        return trials, stripped_names

    @staticmethod
    def save_trials(
        trials: DataFrame,
        trials_state: Namespace,
        trials_file: str,
        instance_name: str,
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
        # strings need to be padded when using append mode
        min_itemsize = {"results_explanation": 110}
        if "roi" in trials.columns:  # roi stored as json
            min_itemsize["roi"] = 190

        while not saved:
            try:
                logger.debug(f"\nSaving {num_trials} {plural(num_trials, 'epoch')}.")
                # cast types
                trials = HyperoptData.cast_trials_types(trials)
                # print(trials.loc[trials.isna().any(axis=1)])
                trials = HyperoptData.cast_trials_types(trials)
                # save on storage to hdf, lock is blocking
                locked = trials_state.lock.acquire()
                if locked:
                    trials.to_hdf(
                        trials_file,
                        key=instance_name,
                        mode="a",
                        complib="blosc:zstd",
                        append=append,
                        format="table",
                        # NOTE: make sure to only index the columns really used for indexing
                        # since each parameter is stored in a different col, the col number can
                        # be very high
                        data_columns=["Xi_h", "random_state", "loss"],
                        # this is needed because it is a string that can exceed
                        # the size preset by pd, and in append mode it can't be changed
                        min_itemsize=min_itemsize,
                    )
                if not backup:
                    trials_state.num_saved += num_trials
                    if final:
                        logger.info(
                            f"{trials_state.num_saved} {plural(trials_state.num_saved, 'epoch')} "
                            f"saved to '{trials_file}'."
                        )
                saved = True
            except (IOError, OSError, AttributeError) as e:
                if locked:
                    trials_state.lock.release()
                # If a lock on the hdf file can't be acquired
                if type(e) == AttributeError:
                    # reset table as it has been corrupted
                    append = False
                logger.warn(f"Couldn't access the trials storage, retrying in {interval}..")
                sleep(interval)
                interval += 0.5
            finally:
                if locked:
                    trials_state.lock.release()

    @abstractmethod
    def log_trials(self, asked, trials_list: List = [], n=0):
        """ Calculate epochs and save results to storage """

    @staticmethod
    def _read_trials(
        trials_file: Path,
        trials_instance: str,
        trials_state: Namespace,
        backup: bool,
        where="",
        start=None,
    ) -> List:
        """
        Read hyperopt trials file
        """
        # Only log at the beginning
        if hasattr(backend, "trials") and not backend.trials.exit:
            logger.info("Reading Trials from '%s'", trials_file)
        trials = DataFrame()
        try:
            trials = read_hdf(trials_file, key=trials_instance, where=where, start=start)
            # make a copy of the trials in case this optimization run corrupts it
            # (wrongful termination)
            if backup and len(trials) > 0:
                HyperoptData.save_trials(
                    trials,
                    trials_state,
                    trials_file,
                    instance_name=f"{trials_instance}_bak",
                    backup=True,
                )
            elif len(trials) < 1:
                logger.warn(f"Instance table {trials_instance} appears empty, using backup...")
                trials = read_hdf(trials_file, key=f"{trials_instance}_bak", where=where)
        except (
            KeyError,
            AttributeError,
        ) as e:  # trials instance is not in the database or corrupted
            # if corrupted
            if backup or not start:
                try:
                    logger.warn(f"Instance table {trials_instance} either empty or corrupted, trying backup...")
                    trials = read_hdf(trials_file, key=f"{trials_instance}_bak", where=where)
                except KeyError:
                    logger.warn(f"Backup not found...")
        return trials

    @staticmethod
    def trials_to_csv_file(
        config: dict, trials: DataFrame, total_epochs: int, highlight_best: bool, csv_file: str
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
    def trials_to_dict(trials: DataFrame):
        """ Convert back from json normalize, nesting levels:
        results_metrics, params_dict : 1
        params_details : 2
        and drop NaN
        """
        sep = "."  # separator is "." (dot)
        trials = trials.copy()
        for prefix in ("results_metrics", "params_dict", "params_details"):
            # startswith does not accept regexp
            match = f"{prefix}{sep}"
            cols_with_prefix = trials.filter(regex=match)  # with dot
            cols_with_prefix.columns = cols_with_prefix.columns.str.replace(match, "", 1)
            trials = trials.loc[:, ~trials.columns.str.startswith(match)]
            trials[prefix] = cols_with_prefix.to_dict("records")

            if prefix == "params_details":
                # one more level for spaces
                spaces = {c.split(sep, 1)[0] for c in cols_with_prefix.columns}
                for s in spaces:
                    match = f"{s}{sep}"
                    space = cols_with_prefix.filter(regex=f"^{match}")
                    space.columns = space.columns.str.replace(match, "", 1)
                    cols_with_prefix = cols_with_prefix.loc[
                        :, ~cols_with_prefix.columns.str.startswith(match)
                    ]
                    cols_with_prefix[s] = space.to_dict("records")
                trials[prefix] = cols_with_prefix.to_dict("records")

        return trials.to_dict("records")

    @staticmethod
    def _filter_options(config: Dict[str, Any]):
        """ Parse filtering config options into dict """
        return {
            "enabled": config.get("hyperopt_list_filter", True),
            "dedup": config.get("hyperopt_list_dedup", False),
            "best": config.get("hyperopt_list_best", []),
            "pct_best": config.get("hyperopt_list_pct_best", 0.1),
            "cutoff_best": config.get("hyperopt_list_cutoff_best", 0.99),
            "min_trades": config.get("hyperopt_list_min_trades", None),
            "max_trades": config.get("hyperopt_list_max_trades", None),
            "min_avg_time": config.get("hyperopt_list_min_avg_time", None),
            "max_avg_time": config.get("hyperopt_list_max_avg_time", None),
            "min_avg_profit": config.get("hyperopt_list_min_avg_profit", None),
            "max_avg_profit": config.get("hyperopt_list_max_avg_profit", None),
            "min_total_profit": config.get("hyperopt_list_min_total_profit", None),
            "max_total_profit": config.get("hyperopt_list_max_total_profit", None),
            "step_values": config.get("hyperopt_list_step_values", HYPEROPT_LIST_STEP_VALUES),
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
    def filter_trials(trials: Any, config: Dict[str, Any], return_list=False) -> Any:
        """
        Filter our items from the list of hyperopt trials
        """
        hd = HyperoptData
        filters = hd._filter_options(config)
        # add columns without prefixes for metrics
        trials, _ = hd.alias_cols(trials, "results_metrics")

        if not filters["enabled"] or len(trials) < 1:
            return hd.list_or_df(trials, return_list)
        if filters["min_trades"] is not None:
            trials = trials.loc[trials["trade_count"] > filters["min_trades"]]
        if filters["max_trades"] is not None:
            trials = trials.loc[trials["trade_count"] < filters["max_trades"]]

        with_trades = trials.loc[trials["trade_count"] > 0]
        with_trades_len = len(with_trades)
        if filters["min_avg_time"]:
            with_trades = with_trades.loc[with_trades["duration"] > filters["min_avg_time"]]
        if filters["max_avg_time"]:
            with_trades = with_trades.loc[with_trades["duration"] < filters["max_avg_time"]]
        if filters["min_avg_profit"] is not None:
            with_trades = with_trades.loc[with_trades["avg_profit"] > filters["min_avg_profit"]]
        if filters["max_avg_profit"] is not None:
            with_trades = with_trades.loc[with_trades["avg_profit"] < filters["max_avg_profit"]]
        if filters["min_total_profit"] is not None:
            with_trades = with_trades.loc[with_trades["profit"] > filters["min_total_profit"]]
        if filters["max_total_profit"] is not None:
            with_trades = with_trades.loc[with_trades["profit"] < filters["max_total_profit"]]
        if len(with_trades) != with_trades_len:
            trials = with_trades

        if len(trials) > 0:
            flt_trials = []
            if filters["dedup"]:
                flt_trials.append(hd.dedup_trials(trials))
            if filters["best"]:
                flt_trials.append(hd.norm_best(trials, filters))
            flt_trials.append(hd.sample_trials(trials, filters))
            return hd.list_or_df(
                concat(flt_trials).drop_duplicates(subset="current_epoch"), return_list
            )
        else:
            return hd.list_or_df(trials, return_list)

    @staticmethod
    def norm_best(trials: Any, filters: dict) -> List:
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
            trials[f"norm_{m}"] = (m_col - m_min) / (m_max - m_min)
        # re-invert
        trials["loss"] = trials["loss"].mul(-1)
        trials["duration"] = trials["duration"].mul(-1)

        # Calc cutoff percentage based on normalization
        types_best = filters["best"]
        if filters["cutoff_best"] == "std":
            min_ratio = lambda m: 1 - trials[m].values.std()
        elif filters["cutoff_best"] == "mean":
            min_ratio = lambda m: 1 - trials[m].values.mean()
        else:
            min_ratio = lambda m: filters["cutoff_best"]

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
        n_best = int(len(trials) * pct_best // len(types_best))
        if n_best < 2:
            n_best = 2

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
    def sample_trials(trials: DataFrame, filters: Dict) -> DataFrame:
        """
        Pick one trial, every `step_value` of `step_metric`...
        or pick n == `range` trials for every `step_metric`...
        for every `step_metric`, sorted by `sort_metric` for every `sort_metric`...
        """
        metrics = HyperoptData.metrics
        if filters["step_keys"]:
            step_keys = metrics if filters["step_keys"] == ["all"] else filters["step_keys"]
            sort_keys = metrics if filters["sort_keys"] == ["all"] else filters["sort_keys"]
            step_values = filters["step_values"]
            flt_trials = []
            for step_k in step_keys:
                for sort_k in sort_keys:
                    flt_trials.extend(
                        HyperoptData.step_over_trials(step_k, step_values, sort_k, trials)
                    )
        else:
            flt_trials = [trials]
        return concat(flt_trials).drop_duplicates(subset="current_epoch")

    @staticmethod
    def step_over_trials(step_k: str, step_values: Dict, sort_k: str, trials: DataFrame) -> List:
        """ Apply the sampling of a metric_key:sort_key combination over the trials """
        # for duration and loss we sort by the minimum
        ascending = sort_k in ("duration", "loss")
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
            except ValueError:
                steps = []
        flt_trials = []
        last_epoch = None
        if len(steps) > len(trials):
            min_step = step_v * (len(steps) / len(trials))
            if not defined_range:
                logger.warn(
                    f"Step value of {step_v} for metric {step_k} is too small. "
                    f"Using a minimum of {min_step:.4f}"
                )
            step_v = min_step
        for n, s in enumerate(steps):
            try:
                t = (
                    # the trials between the current step
                    trials.loc[(trials[step_k].values >= s) & (trials[step_k].values <= s + step_v)]
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
        trials_state: Namespace = None,
        where="",
        backup=False,
        clear=False,
    ) -> DataFrame:
        """
        Load data for epochs from the file if we have one
        """
        trials: DataFrame = DataFrame()
        if trials_file.is_file() and trials_file.stat().st_size > 0:
            trials = HyperoptData._read_trials(
                trials_file, trials_instance, trials_state, backup, where
            )
            # clear the table by replacing it with an empty df
            if clear:
                try:
                    store = HDFStore(trials_file)
                    store.remove("/{}".format(trials_instance))
                except KeyError:
                    pass
                finally:
                    store.close()
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
            raise OperationalException(f"No instances were found at {trials_instances_file}")
        else:
            if cv:
                return "{}_cv".format(instances[-1])
            else:
                return instances[-1]

    @staticmethod
    def clear_instance(trials_file: Path, instance_name: str) -> bool:
        success = False
        try:
            store = HDFStore(trials_file)
            store.remove("/{}".format(instance_name))
            success = True
        except KeyError:
            pass
        finally:
            store.close()
            return success

    @staticmethod
    def get_trials_file(config: dict, trials_dir: Path) -> Path:
        hyperopt = config["hyperopt"]
        strategy = config["strategy"]
        if not hyperopt or not strategy:
            raise DependencyException("Strategy or Hyperopt name not specified, both are required.")
        trials_file = trials_dir / f"{hyperopt}_{strategy}.hdf"
        return trials_file

    def setup_optimizers(self):
        """
        Setup the optimizers objects, applies random state from saved trials if present,
        adds a few attributes to determine logic of execution during trials evaluation
        """
        # try to load previous optimizers
        if self.multi:
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
            n_explorers = self.n_explorers
            n_exploiters = self.n_exploiters
            for _ in range(max_opts):  # generate optimizers
                # random state is preserved
                if len(prev_rngs) > 0:
                    rs = prev_rngs.pop(0)
                else:
                    rs = random.randint(0, iinfo(int32).max)
                # in multi mode generate a new optimizer
                # to randomize the base estimator
                opt_copy = self.get_optimizer(rs)
                opt_copy.void_loss = VOID_LOSS
                opt_copy.void = False
                opt_copy.rs = rs
                # assign role
                if n_explorers:
                    # explorers are positive
                    opt_copy.role = n_explorers
                    n_explorers -= 1
                elif n_exploiters:
                    # exploiters are negative
                    opt_copy.role = -n_exploiters
                    n_exploiters -= 1
                rngs.append(rs)
                backend.optimizers.put(opt_copy)
            del opt_copy
        else:
            self.opt = self.get_optimizer()
            self.opt.void_loss = VOID_LOSS
            self.opt.void = False
            self.opt.rs = self.random_state

    @staticmethod
    def opt_rand(opt: Optimizer, rand: int = None, seed: bool = True) -> Optimizer:
        """
        Return a new instance of the optimizer with modified rng, from the previous
        optimizer random state
        """
        if seed:
            if not rand:
                rand = opt.rng.randint(0, VOID_LOSS)
            opt.rng.seed(rand)
        opt, opt.void_loss, opt.void, opt.rs, opt.role = (
            opt.copy(random_state=opt.rng),
            opt.void_loss,
            opt.void,
            opt.rs,
            opt.role,
        )
        return opt

    @staticmethod
    def opt_clear(opt: Optimizer):
        """ Delete models and points from an optimizer instance """
        del opt.models[:], opt.Xi[:], opt.yi[:]
        return opt
