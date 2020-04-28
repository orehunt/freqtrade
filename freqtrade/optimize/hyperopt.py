# pragma pylint: disable=too-many-instance-attributes, pointless-string-statement
"""
This module contains the hyperopt logic
"""

import random
import warnings
import pickle
import logging
import json
import os
import tempfile
from collections import deque
from math import factorial, log
from operator import itemgetter
from typing import Any, Dict, List, Optional, Tuple, Set
from time import time as now

from colorama import init as colorama_init
from joblib import Parallel, cpu_count, delayed, dump, load, wrap_non_picklable_objects, hash
from joblib import parallel_backend
from multiprocessing.managers import Namespace
from filelock import FileLock, Timeout
from pandas import DataFrame, concat, Categorical, json_normalize, HDFStore, read_hdf
from numpy import isfinite
import rapidjson
from colorama import Fore, Style
from joblib import Parallel, cpu_count, delayed, dump, load, wrap_non_picklable_objects
from pandas import DataFrame, json_normalize, isna
import progressbar
import tabulate
from os import path
import io

from freqtrade.data.converter import trim_dataframe
from freqtrade.data.history import get_timerange
from freqtrade.optimize.backtesting import Backtesting
from freqtrade.exceptions import OperationalException

# Import IHyperOpt and IHyperOptLoss to allow unpickling classes from these modules
import freqtrade.optimize.hyperopt_backend as backend

from freqtrade.optimize.hyperopt_multi import HyperoptMulti
from freqtrade.optimize.hyperopt_out import HyperoptOut
from freqtrade.optimize.hyperopt_cv import HyperoptCV
from freqtrade.optimize.hyperopt_constants import (
    VOID_LOSS,
    CYCLE_LIE_STRATS,
    CYCLE_ESTIMATORS,
    CYCLE_ACQ_FUNCS,
    columns,
)

# from freqtrade.optimize.hyperopt_backend import Trial
from freqtrade.optimize.hyperopt_interface import IHyperOpt  # noqa: F401
from freqtrade.optimize.hyperopt_loss_interface import IHyperOptLoss  # noqa: F401
from freqtrade.resolvers.hyperopt_resolver import HyperOptLossResolver, HyperOptResolver

# Suppress scikit-learn FutureWarnings from skopt
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from skopt import Optimizer
    from skopt.space import Dimension
# Additional regressors already pluggable into the optimizer
# from sklearn.linear_model import ARDRegression, BayesianRidge
# possibly interesting regressors that need predict method override
# from sklearn.ensemble import HistGradientBoostingRegressor
# from xgboost import XGBoostRegressor

logger = logging.getLogger(__name__)


class Hyperopt(HyperoptMulti, HyperoptCV):
    """
    Hyperopt class, this class contains all the logic to run a hyperopt simulation

    To run a backtest:
    hyperopt = Hyperopt(config)
    hyperopt.start()
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

        self.backtesting = Backtesting(self.config)

        self.custom_hyperopt = HyperOptResolver.load_hyperopt(self.config)
        self.custom_hyperoptloss = HyperOptLossResolver.load_hyperoptloss(self.config)
        self.calculate_loss = self.custom_hyperoptloss.hyperopt_loss_function

        # runtime
        self.n_jobs = self.config.get("hyperopt_jobs", -1)
        if self.n_jobs < 0:
            self.n_jobs = cpu_count() // 2 or 1
        self.effort = max(0.01, self.config["effort"] if "effort" in self.config else 1)
        # save trials to disk every 10s * jobs
        self.trials_timeout = self.config.get("hyperopt_trials_timeout", 10 * self.n_jobs)
        # or every n jobs
        self.trials_maxout = self.config.get("hyperopt_trials_maxout", self.n_jobs)

        # configure multi mode
        self.setup_multi()

        # Populate functions here (hasattr is slow so should not be run during "regular" operations)
        if hasattr(self.custom_hyperopt, "populate_indicators"):
            self.backtesting.strategy.advise_indicators = (
                self.custom_hyperopt.populate_indicators  # type: ignore
            )
        if hasattr(self.custom_hyperopt, "populate_buy_trend"):
            self.backtesting.strategy.advise_buy = (
                self.custom_hyperopt.populate_buy_trend  # type: ignore
            )
        if hasattr(self.custom_hyperopt, "populate_sell_trend"):
            self.backtesting.strategy.advise_sell = (
                self.custom_hyperopt.populate_sell_trend  # type: ignore
            )

        # Use max_open_trades for hyperopt as well, except --disable-max-market-positions is set
        if self.config.get("use_max_market_positions", True):
            self.max_open_trades = self.config["max_open_trades"]
        else:
            logger.debug("Ignoring max_open_trades (--disable-max-market-positions was used) ...")
            self.max_open_trades = 0
        self.position_stacking = self.config.get("position_stacking", False)

        if self.has_space("sell"):
            # Make sure use_sell_signal is enabled
            if "ask_strategy" not in self.config:
                self.config["ask_strategy"] = {}
            self.config["ask_strategy"]["use_sell_signal"] = True

    @staticmethod
    def get_lock_filename(config: Dict[str, Any]) -> str:
        return str(config["user_data_dir"] / "hyperopt.lock")

    def _get_params_dict(self, raw_params: List[Any]) -> Dict:

        dimensions: List[Dimension] = self.dimensions

        # Ensure the number of dimensions match
        # the number of parameters in the list.
        if len(raw_params) != len(dimensions):
            raise ValueError("Mismatch in number of search-space dimensions.")

        # Return a dict where the keys are the names of the dimensions
        # and the values are taken from the list of parameters.
        return {d.name: v for d, v in zip(dimensions, raw_params)}

    def _get_params_details(self, params: Dict) -> Dict:
        """
        Return the params for each space
        """
        result: Dict = {}

        if self.has_space("buy"):
            result["buy"] = {p.name: params.get(p.name) for p in self.hyperopt_space("buy")}
        if self.has_space("sell"):
            result["sell"] = {p.name: params.get(p.name) for p in self.hyperopt_space("sell")}
        if self.has_space("roi"):
            # convert roi keys for json normalization support
            result["roi"] = {p.name: params.get(p.name) for p in self.hyperopt_space("roi")}
        if self.has_space("stoploss"):
            result["stoploss"] = {
                p.name: params.get(p.name) for p in self.hyperopt_space("stoploss")
            }
        if self.has_space("trailing"):
            result["trailing"] = self.custom_hyperopt.generate_trailing_params(params)

        return result

    @staticmethod
    def is_best_loss(trial, current_best_loss: float) -> bool:
        return isfinite(trial["loss"]) & (trial["loss"] < current_best_loss)

    def has_space(self, space: str) -> bool:
        """
        Tell if the space value is contained in the configuration
        """
        # The 'trailing' space is not included in the 'default' set of spaces
        if space == "trailing":
            return any(s in self.config["spaces"] for s in [space, "all"])
        else:
            return any(s in self.config["spaces"] for s in [space, "all", "default"])

    def hyperopt_space(self, space: Optional[str] = None) -> List[Dimension]:
        """
        Return the dimensions in the hyperoptimization space.
        :param space: Defines hyperspace to return dimensions for.
        If None, then the self.has_space() will be used to return dimensions
        for all hyperspaces used.
        """
        spaces: List[Dimension] = []

        if space == "buy" or (space is None and self.has_space("buy")):
            logger.debug("Hyperopt has 'buy' space")
            spaces += self.custom_hyperopt.indicator_space()

        if space == "sell" or (space is None and self.has_space("sell")):
            logger.debug("Hyperopt has 'sell' space")
            spaces += self.custom_hyperopt.sell_indicator_space()

        if space == "roi" or (space is None and self.has_space("roi")):
            logger.debug("Hyperopt has 'roi' space")
            spaces += self.custom_hyperopt.roi_space()

        if space == "stoploss" or (space is None and self.has_space("stoploss")):
            logger.debug("Hyperopt has 'stoploss' space")
            spaces += self.custom_hyperopt.stoploss_space()

        if space == "trailing" or (space is None and self.has_space("trailing")):
            logger.debug("Hyperopt has 'trailing' space")
            spaces += self.custom_hyperopt.trailing_space()

        return spaces

    def backtest_params(
        self, raw_params: List[Any] = None, iteration=None, params_dict: Dict[str, Any] = None
    ) -> Dict:
        if not params_dict:
            if raw_params:
                params_dict = self._get_params_dict(raw_params)
            else:
                logger.debug("Epoch evaluation didn't receive any parameters")
                return {}
        params_details = self._get_params_details(params_dict)

        if self.has_space("roi"):
            self.backtesting.strategy.amounts["roi"] = self.custom_hyperopt.generate_roi_table(
                params_dict
            )

        if self.has_space("buy"):
            self.backtesting.strategy.advise_buy = self.custom_hyperopt.buy_strategy_generator(
                params_dict
            )

        if self.has_space("sell"):
            self.backtesting.strategy.advise_sell = self.custom_hyperopt.sell_strategy_generator(
                params_dict
            )

        if self.has_space("stoploss"):
            self.backtesting.strategy.amounts["stoploss"] = params_dict["stoploss"]

        if self.has_space("trailing"):
            d = self.custom_hyperopt.generate_trailing_params(params_dict)
            self.backtesting.strategy.amounts["trailing_stop"] = d["trailing_stop"]
            self.backtesting.strategy.amounts["trailing_stop_positive"] = d[
                "trailing_stop_positive"
            ]
            self.backtesting.strategy.amounts["trailing_stop_positive_offset"] = d[
                "trailing_stop_positive_offset"
            ]
            self.backtesting.strategy.amounts["trailing_only_offset_is_reached"] = d[
                "trailing_only_offset_is_reached"
            ]

        processed = load(self.data_pickle_file)

        min_date, max_date = get_timerange(processed)

        backtesting_results = self.backtesting.backtest(
            processed=processed,
            stake_amount=self.config["stake_amount"],
            start_date=min_date,
            end_date=max_date,
            max_open_trades=self.max_open_trades,
            position_stacking=self.position_stacking,
        )
        return self._get_result(
            backtesting_results, min_date, max_date, params_dict, params_details, processed
        )

    def _get_result(
        self, backtesting_results, min_date, max_date, params_dict, params_details, processed
    ):
        results_metrics = self._calculate_results_metrics(backtesting_results)
        results_explanation = HyperoptOut._format_results_explanation_string(
            self.config["stake_currency"], results_metrics
        )

        trade_count = results_metrics["trade_count"]
        total_profit = results_metrics["total_profit"]

        # If this evaluation contains too short amount of trades to be
        # interesting -- consider it as 'bad' (assigned max. loss value)
        # in order to cast this hyperspace point away from optimization
        # path. We do not want to optimize 'hodl' strategies.
        loss: float = VOID_LOSS
        if trade_count >= self.config["hyperopt_min_trades"]:
            loss = self.calculate_loss(
                results=backtesting_results,
                trade_count=trade_count,
                min_date=min_date.datetime,
                max_date=max_date.datetime,
                processed=processed,
            )
        return {
            "loss": loss,
            "params_dict": params_dict,
            "params_details": params_details,
            "results_metrics": results_metrics,
            "results_explanation": results_explanation,
            "total_profit": total_profit,
        }

    def _calculate_results_metrics(self, backtesting_results: DataFrame) -> Dict:
        return {
            "trade_count": len(backtesting_results.index),
            "avg_profit": backtesting_results.profit_percent.mean() * 100.0,
            "total_profit": backtesting_results.profit_abs.sum(),
            "profit": backtesting_results.profit_percent.sum() * 100.0,
            "duration": backtesting_results.trade_duration.mean(),
        }

    @staticmethod
    def lie_strategy():
        """ Choose a strategy randomly among the supported ones, used in multi opt mode
        to increase the diversion of the searches of each optimizer """
        return next(CYCLE_LIE_STRATS)

    @staticmethod
    def estimators():
        return next(CYCLE_ESTIMATORS)

    @staticmethod
    def acq_funcs():
        return next(CYCLE_ACQ_FUNCS)

    def get_optimizer(self, random_state: int = None) -> Optimizer:
        " Construct an optimizer object "
        # https://github.com/scikit-learn/scikit-learn/issues/14265
        # lbfgs uses joblib threading backend so n_jobs has to be reduced
        # to avoid oversubscription
        if self.opt_acq_optimizer == "lbfgs":
            n_jobs = 1
        else:
            n_jobs = self.n_jobs
        return Optimizer(
            self.dimensions,
            base_estimator=self.opt_base_estimator(),
            acq_optimizer=self.opt_acq_optimizer,
            acq_func=self.opt_acq_func(),
            n_initial_points=self.opt_n_initial_points,
            acq_optimizer_kwargs={
                "n_jobs": n_jobs,
                "n_points": self.calc_n_points(len(self.dimensions), self.n_jobs, self.ask_points),
            },
            model_queue_size=self.n_models,
            random_state=random_state or self.random_state,
        )

    def run_backtest_parallel(self, parallel: Parallel, jobs: int):
        """ launch parallel in single opt mode, return the evaluated epochs """
        parallel(
            delayed(wrap_non_picklable_objects(self.parallel_objective))(
                t, asked, backend.epochs, backend.trials
            )
            for t, asked in self.ask_and_tell(jobs)
        )

    def ask_and_tell(self, jobs: int):
        """
        loop to manage optimizer state in single optimizer mode, everytime a job is
        dispatched, we check the optimizer for points, to ask and to tell if any,
        but only fit a new model every n_points, because if we fit at every result previous
        points become invalid.
        """
        vals = []
        fit = False
        to_ask: deque = deque()
        evald: Set[Tuple] = set()
        opt = self.opt

        # this is needed because when we ask None points, the optimizer doesn't return a list
        if self.opt_ask_points:

            def point():
                if to_ask:
                    return tuple(to_ask.popleft())
                else:
                    to_ask.extend(opt.ask(n_points=self.opt_ask_points, strategy=self.lie_strat()))
                    return tuple(to_ask.popleft())

        else:

            def point():
                return tuple(opt.ask(strategy=self.lie_strat()))

        if len(self.trials) > 0:
            if self.random_state != self.trials.iloc[-1]["random_state"]:
                logger.warn("Random state in saved trials doesn't match runtime...")

        locked = False
        t = 0
        for _ in iter(lambda: 0, 1):
            fit = len(to_ask) < 1  # # only fit when out of points
            # tell every once in a while
            if not t % self.trials_maxout:
                try:
                    # only lock if its fitting time
                    locked = backend.trials.lock.acquire(fit)
                    if locked:
                        params_df = read_hdf(
                            self.trials_file,
                            key=self.trials_instance,
                            columns=[*self.Xi_cols, "loss"],
                            start=t,
                        )
                        backend.trials.lock.release()
                        if len(params_df) > 0:
                            opt.tell(
                                params_df.loc[:, self.Xi_cols].values.tolist(),
                                params_df["loss"].values.tolist(),
                                fit=fit,
                            )
                except (KeyError, FileNotFoundError, IOError, OSError):
                    if locked:
                        backend.trials.lock.release()
                    logger.debug("Couldn't read trials from disk")

            a = point()
            # check for termination when getting duplicates
            if a in evald:
                opt.update_next()
                a = point()
                if backend.trials.exit or self._maybe_terminate(
                    t, jobs, backend.trials, backend.epochs
                ):
                    break
            evald.add(a)
            t += 1
            yield t, a

    def parallel_objective_sig_handler(
        self, t: int, params: list, epochs: Namespace, trials_state: Namespace
    ):
        """
        To handle Ctrl-C the worker main function has to be wrapped into a try/catch;
        NOTE: The Manager process also needs to be configured to handle SIGINT (in the backend)
        """
        try:
            return self.parallel_objective(t, params, epochs, trials_state)
        except KeyboardInterrupt:
            trials_state.exit = True
            return self.parallel_objective(t, params, epochs, trials_state)

    def parallel_objective(self, t: int, params, epochs: Namespace, trials_state: Namespace):
        """ Run one single test and eventually save trials """
        HyperoptOut.log_results_immediate(t, epochs)
        # flush trials if terminating
        if trials_state.exit:
            trials_state.tail.extend(backend.trials_list)
            del backend.trials_list[:]
            return
        if not backend.timer:
            backend.timer = now()

        if self.cv:
            v = self.backtest_params(params_dict=params)
        else:
            v = self.backtest_params(raw_params=params)

        # set flag and params for indexing
        if v:
            v["is_initial_point"] = t < self.opt_n_initial_points
            v["random_state"] = self.random_state  # this is 0 in CV
            v["Xi_h"] = hash(HyperoptMulti.opt_params_Xi(v))
            backend.trials_list.append(v)
            trials_state.num_done += 1

        self.maybe_log_trials(trials_state, epochs)

    def log_trials(self, trials_state: Namespace, epochs: Namespace) -> int:
        """
        Log results if it is better than any previous evaluation
        """
        locked = epochs.lock.acquire(False)
        if not locked:
            # on the last run sit in queue for saving
            if trials_state.exit:
                epochs.lock.acquire()
            else:
                return
        ep = epochs

        HyperoptOut.reset_line()

        batch_start = trials_state.num_saved
        current = batch_start + 1
        current_best = ep.current_best_epoch
        has_roi_space = "roi" in self.config["spaces"]
        i = 0
        for i, v in enumerate(backend.trials_list, 1):
            is_best = self.is_best_loss(v, ep.current_best_loss)
            current = batch_start + i
            v["is_best"] = is_best
            v["current_epoch"] = current
            # store roi as json to remember dict k:v mapping
            # without custom hyperopt class
            if has_roi_space:
                v["roi"] = json.dumps({
                    str(k): v
                    for k, v in self.custom_hyperopt.generate_roi_table(
                        v["params_dict"]
                    ).items()
                })
            logger.debug(f"Optimizer epoch evaluated: {v}")
            if is_best:
                current_best = current
                ep.current_best_loss = v["loss"]
        self.update_max_epoch(current_best, current, ep)
        # Save results and optimizers after every batch
        trials = json_normalize(backend.trials_list)

        # make a copy since print results modifies cols
        self.print_results(trials.copy(), trials_state.table_header, epochs)
        trials_state.table_header = 2

        self.save_trials(trials, trials_state, self.trials_file, self.trials_instance)
        # release lock and clear saved trials from global state
        epochs.lock.release()
        del backend.trials_list[:]

        HyperoptOut.clear_line(columns)
        return i

    def setup_trials(self, load_trials=True, backup=None):
        """ The trials instance is the key used to identify the hdf table """
        # If the Hyperopt class has been previously initialized
        if self.config.get("skip_trials_setup", False):
            return
        self.dimensions: List[Any]
        self.dimensions = self.hyperopt_space()
        hyperopt_loss = self.config["hyperopt_loss"]
        hyperopt_params = hash([d.name for d in self.dimensions])
        self.trials_file = self.get_trials_file(self.config, self.trials_dir)
        self.trials_instance = f"{hyperopt_loss}_{hyperopt_params}"
        logger.info(f"Hyperopt state will be saved to " f"key {self.trials_instance:.40}[...]")
        # clean state depending on mode
        try:
            if self.config.get("hyperopt_clear") and not self.cv:
                self.clear_hyperopt()
            store = HDFStore(self.trials_file)
            # optionally remove previous trials of an instance
            if self.config.get("hyperopt_reset"):
                store.remove("/{}".format(self.trials_instance))
            # save a list of all the tables in the store except backups
            keys = store.keys()
            s_keys = set()
            # unique keys and exclude backups
            keys = set([k.lstrip("/").rstrip("_bak") for k in keys])
            keys.add(self.trials_instance)
            logger.debug(f"Saving list of store keys to...{self.trials_instances_file}")
            with open(self.trials_instances_file, "w") as ti:
                json.dump(list(keys), ti)
        except KeyError:
            pass
        finally:
            if 'store' in vars():
                store.close()

        load_instance = self.config.get("hyperopt_trials_instance")
        if load_trials:
            # Optionally load thinned trials list from previous CV run, and clear them after load
            if load_instance == "cv":
                trials_instance = self.trials_instance + "_cv"
            # or load directly from specified instance
            elif load_instance:
                trials_instance = load_instance
            else:
                trials_instance = self.trials_instance
            # Load trials before checking for cross validation, and epochs/points setup
            # and make a copy of the table in case the run is wrongly terminated
            self.trials = self.load_trials(
                self.trials_file,
                trials_instance,
                backend.trials,
                backup=(backup or not self.cv),
                clear=False,
            )
            if self.cv:
                # in cross validation apply filtering
                self.target_trials = self.filter_trials(self.trials, self.config)
                self.dimensions = [
                    k for k in self.target_trials.filter(regex="^params_dict\.").columns
                ]
        if self.cv:
            # CV trials are saved in their own table
            self.trials_instance += "_cv"
            # reset cv trials only if not specified
            if self.cv and not load_instance or load_instance == "cv":
                self.clear_instance(self.trials_file, self.trials_instance)

    def setup_epochs(self) -> bool:
        """ used to resume the best epochs state from previous trials """
        locked = backend.epochs.lock.acquire(True, timeout=60)
        if not locked:
            raise OperationalException("Couldn't acquire lock at startup during epochs setup.")
        self.epochs_limit = lambda: self.total_epochs or backend.epochs.max_epoch
        ep = backend.epochs
        ep.current_best_epoch = 0
        ep.current_best_loss = VOID_LOSS
        # shared collections have to use the manager
        ep.epochs_since_last_best = backend.manager.list([0, 0])
        len_trials = len(self.trials)
        if self.cv:
            self.total_epochs = len(self.target_trials)
            backend.trials.num_saved = 0
        else:
            backend.trials.num_saved = len_trials
        resumed = len_trials > 0
        if resumed and not self.cv:
            best_epochs = self.trials.loc[self.trials["is_best"], :]
            len_best = len(best_epochs)
            if len_best > 0:
                # sorting from lowest to highest, the first value is the current best
                best = self.trials.sort_values(by=["loss"]).iloc[0].to_dict()
                ep.current_best_epoch = best["current_epoch"]
                ep.current_best_loss = best["loss"]
                ep.avg_last_occurrence = max(self.n_jobs, len_trials // len_best)
                ep.epochs_since_last_best = (
                    (best_epochs["current_epoch"] - best_epochs["current_epoch"].shift(1))
                    .dropna()
                    .astype(int)
                    .values.tolist()
                ) or [1, 1]
                avg_best_occurrence = sum(ep.epochs_since_last_best) / len(
                    ep.epochs_since_last_best
                )
            else:
                ep.avg_last_occurrence = min(self.min_epochs, self.opt_points)
                avg_best_occurrence = self.min_trials
            ep.max_epoch = len_trials + avg_best_occurrence
        # if total epochs are not set, max_epoch takes its place
        if self.total_epochs < 1:
            ep.max_epoch = int(self.min_epochs + len(self.trials))
        else:
            ep.max_epoch = self.total_epochs
        # at the start done are equal saved
        backend.trials.num_done = backend.trials.num_saved
        ep.lock.release()
        return resumed

    def _set_random_state(self, random_state: Optional[int]) -> int:
        if self.cv:
            return 0
        else:
            return random_state or random.randint(1, 2 ** 16 - 1)

    @staticmethod
    def _stub_dimension(k):
        d = Dimension()
        d.name = k
        return d

    @staticmethod
    def calc_epochs(
        dimensions: List[Dimension],
        n_jobs: int,
        effort: float,
        start_epochs: int,
        total_epochs: int,
        ask_points: int,
    ):
        """ Compute a reasonable number of initial points and
        a minimum number of epochs to evaluate """
        n_dimensions = len(dimensions)
        n_parameters = 0
        opt_points = n_jobs * ask_points
        # sum all the dimensions discretely, granting minimum values
        for d in dimensions:
            if type(d).__name__ == "Integer":
                n_parameters += max(1, d.high - d.low)
            elif type(d).__name__ == "Real":
                n_parameters += max(10, int(d.high - d.low))
            else:
                n_parameters += len(d.bounds)
        # in case bounds between parameters are too far, fall back to use dimensions
        n_parameters = min(n_dimensions * 100, n_parameters)
        # guess the size of the search space as the count of the
        # unordered combination of the dimensions entries
        try:
            search_space_size = max(
                int(
                    (
                        factorial(n_parameters)
                        / (factorial(n_parameters - n_dimensions) * factorial(n_dimensions))
                    )
                ),
                factorial(n_parameters),
            )
            logger.info(f"Search space size: {search_space_size:e}")
        except OverflowError:
            search_space_size = VOID_LOSS

        log_opt = max(2, int(log(opt_points, 2)))
        # fixed number of epochs
        n_initial_points = opt_points
        if total_epochs > 0:
            min_epochs = total_epochs
        # search space is small
        elif search_space_size < opt_points:
            n_initial_points = max(1, search_space_size // opt_points)
            min_epochs = search_space_size
        else:
            min_epochs = int(max(n_initial_points, opt_points) + 2 * n_initial_points)

        # after calculation, ensure limits
        n_initial_points = max(n_dimensions, int(n_initial_points))
        min_epochs = min(search_space_size, int(min_epochs * effort)) + start_epochs

        return n_initial_points, min_epochs, search_space_size

    def update_max_epoch(self, current_best: int, current: int, ep: Namespace):
        """ calculate max epochs: store the number of non best epochs
            between each best, and get the mean of that value """
        # if there isn't a new best, update the last period
        if ep.current_best_epoch == current_best:
            ep.epochs_since_last_best[-1] = current - ep.current_best_epoch
        else:
            ep.current_best_epoch = current_best
            ep.epochs_since_last_best.append(current - ep.current_best_epoch)
        # this tracks the tip of the average, which is used to compute batch_len
        ep.avg_last_occurrence = sum(ep.epochs_since_last_best) // (
            len(ep.epochs_since_last_best) or 1
        )
        # how many epochs between bests on average
        avg_best_occurrence = sum(ep.epochs_since_last_best[:-1]) // (
            len(ep.epochs_since_last_best[:-1]) or 1
        )
        # the max epoch starts from the current best epoch, and adds the best average
        # has to be at least min_epochs and not bigger than the search space
        ep.max_epoch = int(
            min(
                max(ep.current_best_epoch + avg_best_occurrence, self.min_epochs)
                * max(1, self.effort),
                self.search_space_size,
            )
        )
        logger.debug(f"Max epoch set to: {ep.max_epoch}")

    def setup_points(self):
        """
        Calc starting points, based on parameters, given epochs, mode
        """
        if self.cv:
            self.search_space_size = VOID_LOSS
            self.min_epochs = self.total_epochs
            self.n_initial_points = self.min_epochs
        else:
            self.n_initial_points, self.min_epochs, self.search_space_size = self.calc_epochs(
                self.dimensions,
                self.n_jobs,
                self.effort,
                len(self.trials),
                self.total_epochs,
                self.ask_points,
            )
        logger.debug(f"Min epochs set to: {self.min_epochs}")
        # reduce random points in multi mode by the number of jobs
        # because otherwise each optimizer would ask n_initial_points
        if self.multi:
            self.opt_n_initial_points = self.n_initial_points // self.n_jobs or (
                1 if self.shared else 3
            )
        else:
            self.opt_n_initial_points = self.n_initial_points
        if not self.cv:
            logger.debug(f"Initial points: ~{self.n_initial_points}")
        # each column is a parameter, needed to read points from storage
        # in cv mode we take the params names from the saved epochs columns
        col = "params_dict.{d.name}" if not self.cv else "{d}"
        self.Xi_cols = [col.format(d=d) for d in self.dimensions]
        logger.info(f"Parameters set for optimization: {len(self.Xi_cols)}")

    def _maybe_terminate(
        self, t: int, jobs: int, trials_state: Namespace, epochs: Namespace
    ) -> bool:
        """ signal workers to terminate if conditions are met """
        done = trials_state.num_done - trials_state.num_saved
        if done < t - jobs and not self.void_output_backoff:
            print("\r")
            logger.warn(
                "Some evaluated epochs were void, " "check the loss function and the search space."
            )
            self.void_output_backoff = True
        if not done:
            trials_state.empty_strikes += 1
        if (
            not done
            and self.search_space_size < trials_state.num_done + self.epochs_limit()
            and not self.cv
        ) or trials_state.empty_strikes > self.trials_maxout:
            logger.info("Terminating Hyperopt because trials were empty.")
            trials_state.exit = True
            return True
        # give up if no best since max epochs
        if trials_state.num_done >= self.epochs_limit():
            logger.debug("Max epoch reached, terminating.")
            trials_state.exit = True
            return True
        return False

    def main_loop(self, jobs_scheduler):
        """ main parallel loop """
        with parallel_backend("loky", inner_max_num_threads=2):
            with Parallel(n_jobs=self.n_jobs, verbose=0, backend="loky") as parallel:
                try:
                    # reset parallel state
                    jobs = parallel._effective_n_jobs()
                    logger.info(f"Effective parallel workers: {jobs}")
                    self.run_setup_backend_parallel(parallel, jobs)

                    # run the jobs
                    jobs_scheduler(parallel, jobs)

                # keyboard interrupts should be caught within each worker too
                except KeyboardInterrupt:
                    print("User interrupted..")
                # collect remaining unsaved epochs
                backend.trials.exit = True
                jobs_scheduler(parallel, 2 * jobs)
                # since the list was passed through the manager, make a copy
                if backend.trials.tail:
                    backend.trials_list = [t for t in backend.trials.tail]
                    self.log_trials(backend.trials, backend.epochs)

    def start(self) -> None:
        """ Broom Broom """
        self.random_state = self._set_random_state(self.config.get("hyperopt_random_state", None))
        logger.info(f"Using optimizer random state: {self.random_state}")
        backend.trials.table_header = 0
        data, timerange = self.backtesting.load_bt_data()
        preprocessed = self.backtesting.strategy.ohlcvdata_to_dataframe(data)

        # Trim startup period from analyzed dataframe
        # make a new list of the preprocessed pairs because
        # we delete from the preprocessed dict within the loop
        pairs = [pair for pair in preprocessed.keys()]
        for pair in pairs:
            preprocessed[pair] = trim_dataframe(preprocessed[pair], timerange)
            len_pair_df = len(preprocessed[pair])
            if len_pair_df < 1:
                del preprocessed[pair]
            else:
                self.n_candles += len_pair_df
        if len(preprocessed) < 1:
            raise OperationalException(
                "Not enough data to support the provided startup candle count."
            )
        min_date, max_date = get_timerange(preprocessed)

        logger.info(
            "Hyperopting with data from %s up to %s (%s days)..",
            min_date.isoformat(),
            max_date.isoformat(),
            (max_date - min_date).days,
        )
        dump(preprocessed, self.data_pickle_file)

        # We don't need exchange instance anymore while running hyperopt
        self.backtesting.exchange = None  # type: ignore
        self.backtesting.pairlists = None  # type: ignore

        # Set dimensions, trials instance and paths and load from storage
        self.setup_trials()

        # Set number of initial points, and optimizer related stuff
        self.setup_points()

        # Count the epochs
        self.setup_epochs()

        if self.print_colorized:
            colorama_init(autoreset=True)

        if not self.cv:
            self.setup_optimizers()
            # After setup, trials are only needed by CV so can delete
            self.trials.iloc[0:] = None

        if self.cv:
            jobs_scheduler = self.run_cv_backtest_parallel
        elif self.multi:
            jobs_scheduler = self.run_multi_backtest_parallel
        else:
            jobs_scheduler = self.run_backtest_parallel

        self.main_loop(jobs_scheduler)

        # Print best epoch
        if backend.trials.num_saved:
            best_trial = self.load_trials(
                self.trials_file,
                self.trials_instance,
                backend.trials,
                where=f"loss in {backend.epochs.current_best_loss}",
            )
            best = self.trials_to_dict(best_trial)[0]
            self.print_epoch_details(best, self.epochs_limit(), self.print_json)
        else:
            # This is printed when Ctrl+C is pressed quickly, before first epochs have
            # a chance to be evaluated.
            print("No epochs evaluated yet, no best result.")

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["trials"]
        if self.cv:
            del state["target_trials"]
        elif not self.multi:
            del state["opt"]
        return state
