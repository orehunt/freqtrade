"""
This module contains the hyperopt logic
"""

import atexit
import json
import logging
import random
from collections import deque
from datetime import datetime
from functools import partial
from itertools import cycle
from time import sleep
from time import time as now
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from colorama import init as colorama_init
from joblib import (
    Parallel,
    cpu_count,
    delayed,
    dump,
    hash,
    load,
    parallel_backend,
    wrap_non_picklable_objects,
)
from joblib.externals.loky import get_reusable_executor
from numpy import iinfo, int32, isfinite
from pandas import DataFrame, Timedelta, concat, json_normalize

# Import IHyperOpt and IHyperOptLoss to allow unpickling classes from these modules
import freqtrade.optimize.hyperopt_backend as backend
from freqtrade.constants import DATETIME_PRINT_FORMAT
from freqtrade.data.history import get_timerange
from freqtrade.exceptions import OperationalException
from freqtrade.optimize.backtest_utils import check_data_startup
from freqtrade.optimize.hyperopt_backend import Epochs, TrialsState
from freqtrade.optimize.hyperopt_backtest import HyperoptBacktesting
from freqtrade.optimize.hyperopt_cv import HyperoptCV
from freqtrade.optimize.hyperopt_loss_interface import IHyperOptLoss  # noqa: F401
from freqtrade.optimize.hyperopt_multi import HyperoptMulti
from freqtrade.optimize.hyperopt_out import HyperoptOut
from freqtrade.optimize.optimizer import (
    VOID_LOSS,
    IOptimizer,
    Parameter,
    guess_search_space,
)
from freqtrade.resolvers.hyperopt_resolver import HyperOptLossResolver, HyperOptResolver
from freqtrade.strategy.interface import IStrategy


logger = logging.getLogger(__name__)
hash = partial(hash, hash_name="sha1")


class Hyperopt(HyperoptMulti, HyperoptCV):
    """
    Hyperopt class, this class contains all the logic to run a hyperopt simulation

    To run a backtest:
    hyperopt = Hyperopt(config)
    hyperopt.start()
    """

    # NOTE: The hyperopt class is pickled, attributes can't have default values otherwise
    # they would override the current state when pickled, only use default values for
    # attributes not used by workers

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.logger = logger

        self._setup_workers()

        # configure parallel backend, before backtesting
        # to avoid spawning another exchange instance inside the manager
        self._setup_parallel()

        self._setup_backtesting()

    def _setup_workers(self):
        # runtime
        self.n_jobs = self.config.get("hyperopt_jobs", -1)
        if self.n_jobs < 0:
            self.n_jobs = cpu_count() // 2 or 1

        # save trials to disk every 10s * jobs
        self.trials_timeout = self.config.get(
            "hyperopt_trials_timeout", 10 * self.n_jobs
        )
        # or every n jobs
        self.trials_maxout = self.config.get("hyperopt_trials_maxout", self.n_jobs)
        self.trials_max_empty = self.config.get(
            "hyperopt_trials_max_empty", self.trials_maxout
        )
        self.ask_points = self.config.get("hyperopt_ask_points", 1)
        self.opt_ask_timeout = self.config.get("hyperopt_optimizer", {}).get(
            "ask_timeout"
        )
        self.n_rand = self.config.get("hyperopt_initial_points", self.n_jobs * 3)
        self.use_progressbar = self.config.get("hyperopt_use_progressbar", True)

    def _setup_backtesting(self):
        self.backtesting = HyperoptBacktesting(self.config)

        self.custom_hyperopt = HyperOptResolver.load_hyperopt(self.config)

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
            logger.debug(
                "Ignoring max_open_trades (--disable-max-market-positions was used) ..."
            )
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

    def _get_params_dict(self, raw_params: Union[Tuple[Any, ...], List[Any]]) -> Dict:

        parameters: List[Parameter] = self.parameters

        # Ensure the number of dimensions match
        # the number of parameters in the list.
        if len(raw_params) != len(parameters):
            raw_params_len = len(raw_params)
            params_len = len(parameters)
            raise ValueError(
                "Mismatch in number of search-space dimensions. received:"
                f" {raw_params_len}, expected: {params_len}"
            )

        # Return a dict where the keys are the names of the dimensions
        # and the values are taken from the list of parameters.
        return {d.name: v for d, v in zip(parameters, raw_params)}

    def _get_params_details(self, params: Dict) -> Dict:
        """
        Return the params for each space
        """
        result: Dict = {}

        if self.has_space("buy"):
            result["buy"] = {
                p.name: params.get(p.name) for p in self.hyperopt_space("buy")
            }
        if self.has_space("sell"):
            result["sell"] = {
                p.name: params.get(p.name) for p in self.hyperopt_space("sell")
            }
        if self.has_space("roi"):
            # convert roi keys for json normalization support
            result["roi"] = {
                p.name: params.get(p.name) for p in self.hyperopt_space("roi")
            }
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

    def hyperopt_space(self, space: Optional[str] = None) -> List[Parameter]:
        """
        Return the dimensions in the hyperoptimization space.
        :param space: Defines hyperspace to return dimensions for.
        If None, then the self.has_space() will be used to return dimensions
        for all hyperspaces used.
        """
        spaces: List[Parameter] = []

        if space == "buy" or (space is None and self.has_space("buy")):
            logger.log(5, "Hyperopt has 'buy' space")
            spaces += self.custom_hyperopt.indicator_space()

        if space == "sell" or (space is None and self.has_space("sell")):
            logger.log(5, "Hyperopt has 'sell' space")
            spaces += self.custom_hyperopt.sell_indicator_space()

        if space == "roi" or (space is None and self.has_space("roi")):
            logger.log(5, "Hyperopt has 'roi' space")
            spaces += self.custom_hyperopt.roi_space()

        if space == "stoploss" or (space is None and self.has_space("stoploss")):
            logger.log(5, "Hyperopt has 'stoploss' space")
            spaces += self.custom_hyperopt.stoploss_space()

        if space == "trailing" or (space is None and self.has_space("trailing")):
            logger.log(5, "Hyperopt has 'trailing' space")
            spaces += self.custom_hyperopt.trailing_space()
        return spaces

    def _set_params(self, params_dict: Dict[str, Any]):
        if self.has_space("roi"):
            self.backtesting.strategy.amounts[
                "minimal_roi"
            ] = self.custom_hyperopt.generate_roi_table(params_dict)
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

    def backtest_params(
        self,
        raw_params: Tuple[Tuple, Dict] = None,
        iteration=None,
        params_dict: Dict[str, Any] = None,
        rs: Union[None, int] = None,
    ) -> Dict:
        if not params_dict:
            if raw_params:
                params_dict = self._get_params_dict(raw_params[0])
                params_meta = raw_params[1]
            else:
                raise OperationalException(
                    "Epoch evaluation didn't receive any parameters"
                )
        else:
            params_meta = {}

        self._set_params(params_dict)

        if backend.data:
            processed = backend.data
            min_date, max_date = backend.min_date, backend.max_date
        else:
            processed = load(self.data_pickle_file)
            backend.data = processed
            min_date, max_date = get_timerange(processed)
            backend.min_date, backend.max_date = min_date, max_date

        backtesting_results = self.backtesting.backtest(
            processed=processed,
            stake_amount=self.config["stake_amount"],
            start_date=min_date.datetime,
            end_date=max_date.datetime,
            max_open_trades=self.max_open_trades,
            position_stacking=self.position_stacking,
        )
        return self._get_result(
            backtesting_results,
            min_date,
            max_date,
            params_dict,
            params_meta,
            self._get_params_details(params_dict),
            processed,
            rs,
        )

    @staticmethod
    def _set_hyperoptloss_attrs(
        hyperoptloss: IHyperOptLoss,
        config: dict,
        min_date: datetime,
        max_date: datetime,
    ):
        # Assign timeframe to be used in hyperopt
        hyperoptloss.ticker_interval = str(config["timeframe"])
        hyperoptloss.timeframe = str(config["timeframe"])
        hyperoptloss.weighted_timeranges = config.get(
            "hyperopt_weighted_timeranges", {}
        )
        hyperoptloss.min_date = min_date
        hyperoptloss.max_date = max_date

    def _setup_loss_funcs(self):
        """ Map a (cycled) list of loss functions to the optimizers random states """
        self.adjust_acquisition = self.config.get("hyperopt_adjust_acquisition", True)
        config = self.config.copy()
        if self.multi:
            self.calculate_loss_dict = {}
            loss_func_list = cycle(self.config.get("hyperopt_loss_multi", []))
            logger.debug("Cycling over loss functions: %s", loss_func_list)
            # NOTE: the resolver walks over all the files in the dir on each call
            for rs in self.rngs:
                config["hyperopt_loss"] = next(loss_func_list)
                hyperoptloss = HyperOptLossResolver.load_hyperoptloss(config)
                self._set_hyperoptloss_attrs(
                    hyperoptloss, config, self.min_date, self.max_date
                )
                self.calculate_loss_dict[rs] = hyperoptloss.hyperopt_loss_function
        else:
            self.custom_hyperoptloss = HyperOptLossResolver.load_hyperoptloss(
                self.config
            )
            self._set_hyperoptloss_attrs(
                self.custom_hyperoptloss, config, self.min_date, self.max_date
            )
            self.calculate_loss = self.custom_hyperoptloss.hyperopt_loss_function

    def _get_result(
        self,
        backtesting_results,
        min_date,
        max_date,
        params_dict,
        params_meta,
        params_details,
        processed,
        rs,
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
            loss_func = (
                self.calculate_loss_dict[rs] if rs is not None else self.calculate_loss
            )
            loss = loss_func(
                results=backtesting_results,
                trade_count=trade_count,
                min_date=min_date.datetime,
                max_date=max_date.datetime,
                processed=processed,
            )
            if not np.isfinite(loss):
                logger.warning(
                    "loss result by %s is not finite, using VOID_LOSS value",
                    self.calculate_loss_dict.__name__,
                )
                loss = VOID_LOSS
        return {
            "loss": loss,
            "params_dict": params_dict,
            "params_meta": params_meta,
            "params_details": params_details,
            "results_metrics": results_metrics,
            "results_explanation": results_explanation,
            "total_profit": total_profit,
        }

    def _calculate_results_metrics(self, backtesting_results: DataFrame) -> Dict:
        wins = len(backtesting_results[backtesting_results.profit_percent > 0])
        draws = len(backtesting_results[backtesting_results.profit_percent == 0])
        losses = len(backtesting_results[backtesting_results.profit_percent < 0])
        return {
            "trade_count": len(backtesting_results.index),
            # "wins": wins,
            # "draws": draws,
            # "losses": losses,
            # "winsdrawslosses": f"{wins}/{draws}/{losses}",
            "avg_profit": backtesting_results.profit_percent.mean() * 100.0,
            # "median_profit": backtesting_results.profit_percent.median() * 100.0,
            "total_profit": backtesting_results.profit_abs.sum(),
            "profit": backtesting_results.profit_percent.sum() * 100.0,
            "duration": backtesting_results.trade_duration.mean(),
        }

    def get_optimizer(self, random_state: int = None, parameters=[]) -> IOptimizer:
        " Construct an optimizer object "
        config = self.config.get("hyperopt_optimizer", {})
        config["mode"] = self.mode
        config["n_rand"] = self.n_rand
        config["ask_points"] = self.ask_points
        config["n_jobs"] = self.config.get("hyperopt_jobs", -1)
        config["n_epochs"] = self.config.get("hyperopt_epochs", 10)
        opt_type = config.get("type", "Skopt")
        kwargs = {"seed": random_state}
        if opt_type == "Skopt":
            from freqtrade.optimize.opts.skopt import Skopt

            opt = Skopt(parameters, config=config, **kwargs)
        elif opt_type == "Sherpa":
            from freqtrade.optimize.opts.sherpa import Sherpa

            opt = Sherpa(parameters, config=config, **kwargs)
        elif opt_type == "Emukit":
            from freqtrade.optimize.opts.emukit import EmuKit

            opt = EmuKit(parameters, config=config, **kwargs)
        else:
            raise OperationalException(f"Error loading optimizer type {opt_type}")
        return opt.create_optimizer(parameters, config)

    def run_backtest_parallel(self, parallel: Parallel, jobs: int):
        """ launch parallel in single opt mode, return the evaluated epochs """
        parallel(
            delayed(backend.parallel_sig_handler)(
                self.parallel_objective,
                self.cls_file,
                self.logger,
                t,
                asked,
                epochs=backend.epochs,
                trials_state=backend.trials,
            )
            for t, asked in self.ask_and_tell(jobs)
        )

    def point_func(self, opt: IOptimizer, to_ask: deque) -> Tuple:
        """
        this is needed because when we ask None points, the optimizer doesn't return a list
        """
        if not to_ask:
            wait_start = now()
            to_ask.extend(opt.ask(n=self.ask_points))
            wait_time = now() - wait_start
            backend.epochs.avg_wait_time = (
                (backend.epochs.avg_wait_time or wait_time) + wait_time
            ) / 2
        return tuple(to_ask.popleft()) if to_ask else None

    @staticmethod
    def _unfinished():
        """ return the number of unfinished (in flight) jobs """
        # logger.debug("dispatched: %s", backend.epochs.dispatched)
        # logger.debug("saved: %s", backend.trials.num_saved)
        # logger.debug("done: %s", backend.trials.num_done)
        # logger.debug("done: %s", backend.trials.n_void)
        return backend.epochs.dispatched - (
            backend.trials.num_saved + backend.trials.num_done + backend.trials.n_void
        )

    def _check_points(self, t, to_ask, opt):
        # tell points if trials dispatched are above told and ask queue if empty
        if (not opt.is_blocking) or (len(to_ask) < 1 and not self.points_checked):
            self.points_checked = True
            try:
                # only lock if its fitting time
                locked = backend.acquire_lock(backend.trials, True)
                if locked:
                    self._tell_points(t, opt)
            except (KeyError, FileNotFoundError, IOError, OSError) as e:
                if locked:
                    backend.release_lock(backend.trials)
                logger.debug("Couldn't tell points to the optimizer, %s", e)
        elif self.points_checked:
            self.points_checked = False

    def _tell_points(self, t, opt: IOptimizer):
        """ This is blocking. Assumes there are buffered points, flushes them
        and tells the optimizer. """
        Xi = []
        yi = []
        t_points = len(opt.Xi)
        backoff = 0

        while t > t_points + len(Xi):
            for n_res, res in backend.trials.results.items():
                Xi.append((list(res["params_dict"].values()), res["params_meta"],))
                yi.append(res["loss"])
                del backend.trials.results[n_res]
            logger.debug(
                "fetched %s results to tell the opt, unf: %s, disp: %s, opt: %s, void: %s",
                len(Xi),
                self._unfinished(),
                backend.epochs.dispatched,
                len(opt.Xi),
                backend.trials.n_void,
            )
            if not opt.is_blocking:
                break
            sleep(backoff)
            backoff += 0.3

        backend.release_lock(backend.trials)
        if len(Xi) > 0:
            # params_df = concat(trials, axis=0)
            # read_index = t
            try:
                opt.tell(Xi, yi, fit=True)
                del backend.trials.tail_list[:]
                logger.debug(f"Optimizer now has %s points", len(opt.Xi))
            # If space reduction has just been performed points
            # might be out of space
            except ValueError as e:
                logger.debug("Failed telling optimizer results, %s", e)
            backend.epochs.average = (
                np.nanmean(opt.yi[-self.n_jobs :]) + backend.epochs.average
            ) / 2

    def ask_and_tell(self, jobs: int):
        """
        loop to manage optimizer state in single optimizer mode, everytime a job is
        dispatched, we check the optimizer for points, to ask and to tell if any,
        but only fit a new model every n_points, because if we fit at every result previous
        points become invalid.
        """
        to_ask: deque = deque()
        opt: IOptimizer = self.opt
        point = self.point_func

        t = 0
        sri = self.space_reduction_interval
        opt = self.opt

        # loop indefinitely
        for _ in iter(lambda: 0, 1):
            # update acquisition
            opt = self.opt_adjust_acq(
                opt, jobs, backend.epochs, backend.trials, is_shared=True
            )
            if sri and sri // (t or 1) < 1:
                if self.apply_space_reduction(jobs, backend.trials, backend.epochs):
                    read_index = 0

            self._check_points(t, to_ask, opt)

            try:
                if (backend.trials and backend.trials.exit) or self._maybe_terminate(
                    t, jobs, backend.trials, backend.epochs
                ):
                    break
            except ConnectionError as e:
                logger.debug("connection error %s", e)
                break

            logger.debug("before asking, opt had %s points", len(opt.Xi))
            # NOTE: the optimizer is in charge of ensuring given points are not dups
            p = point(opt, to_ask)

            if self.use_progressbar:
                HyperoptOut._print_progress(t, jobs, self.trials_maxout)
            if not p:
                break
            t += 1
            yield t, p

    @staticmethod
    def parallel_objective(t: int, params, epochs: Epochs, trials_state: TrialsState):
        """ Run one single test and eventually save trials """
        # flush trials if terminating
        cls: Hyperopt = backend.cls
        backend.acquire_lock(epochs, True)
        epochs.dispatched += 1
        backend.release_lock(epochs)
        if not backend.flush_registered:
            atexit.register(cls.flush_remaining_trials, trials_state, False, None)
            backend.flush_registered = True
        if backend.is_exit(trials_state):
            cls.flush_remaining_trials(trials_state, False, None)
            return
        if not backend.timer:
            backend.timer = now()

        if cls.cv:
            # if t not in backend.params_Xi:
            X = cls._from_storage(
                cls.Xi_path, key="X", fields=["params_dict"], indexer=t,
            ).values[0, 0]

            v = cls.backtest_params(params_dict=X)
        else:
            v = cls.backtest_params(raw_params=params)

        # set flag and params for indexing
        if v:
            v["is_initial_point"] = t < cls.n_rand if cls.cv else False
            v["random_state"] = cls.random_state  # this is 0 in CV
            v["Xi_h"] = hash(cls.params_Xi(v))
            backend.trials_list.append(v)
            trials_state.num_done += 1
        else:
            trials_state.n_void += 1
        cls.maybe_log_trials(trials_state, epochs, rs=None)

    def log_trials(
        self, trials_state: TrialsState, epochs: Epochs, rs: Union[None, int]
    ) -> int:
        """
        Log results if it is better than any previous evaluation
        """
        locked = backend.acquire_lock(epochs, False)
        if not locked:
            # in single mode or on the last run, sit in queue for saving
            if (not self.async_sched) or backend.is_exit(trials_state):
                backend.acquire_lock(epochs, True)
            else:
                logger.debug("couldn't acquire log to save trials, skipping %s", rs)
                return 0
        ep = epochs

        batch_start = trials_state.num_saved
        current = batch_start + 1
        has_roi_space = self.has_space("roi")
        i = 0
        # current best loss
        for i, v in enumerate(backend.trials_list, 1):
            is_best = self.is_best_loss(v, ep.current_best_loss[rs])
            current = batch_start + i
            v["is_best"] = is_best
            v["current_epoch"] = current
            # store roi as json to remember dict k:v mapping
            # without custom hyperopt class
            if has_roi_space:
                v["roi"] = json.dumps(
                    {
                        str(k): v
                        for k, v in self.custom_hyperopt.generate_roi_table(
                            v["params_dict"]
                        ).items()
                    }
                )
            logger.debug(
                f"Optimizer epoch evaluated: %s, yi: %s", v["current_epoch"], v["loss"]
            )
            if is_best:
                ep.improvement = abs(
                    ep.current_best_loss[rs] / (v["loss"] or ep.current_best_loss[rs])
                    - 1
                )
                ep.current_best_epoch[rs] = ep.last_best_epoch = current
                ep.current_best_loss[rs] = ep.last_best_loss = v["loss"]
        # in single mode push the results to the main worker
        # to tell the optimizer
        if not self.async_sched:
            for tr in backend.trials_list:
                trials_state.results[tr["current_epoch"]] = tr
        # Save results after every batch
        trials = DataFrame(backend.trials_list)
        # expand results metrics into columns
        trials.drop(columns=["results_metrics", "total_profit"], inplace=True)
        metrics = json_normalize([v["results_metrics"] for v in backend.trials_list],)
        trials = concat([trials, metrics], copy=False, axis=1)

        # make a copy since print results modifies cols
        self.print_results(trials.copy(), trials_state.table_header, epochs)
        # TODO: print the header at the beginning so this is not carried over on each iteration
        trials_state.table_header = 2

        self.save_trials(trials, self.trials_file, self.trials_instance, trials_state)
        # release lock and clear saved trials from global state
        backend.release_lock(epochs)

        del backend.trials_list[:]
        return i

    def cleanup_store_tables(self):
        """ Executes store cleanup options """
        # clean state depending on mode
        try:
            # optionally delete file
            if self.config.get("hyperopt_clear") and not self.cv:
                self.clear_hyperopt()
            keys = self._group.keys()
            # optionally remove previous trials of an instance
            if self.config.get("hyperopt_reset") and not self.cv:
                inst = "/{}".format(self.trials_instance)
                for k in (inst, f"{inst}_bak"):
                    if k in keys:
                        del self._group[k], self._group[k].attrs["columns"]
            # save a list of all the tables in the store except backups
            # unique keys and exclude backups
            keys = [k.lstrip("/").rstrip("_bak") for k in keys]
            keys.append(self.trials_instance)
            logger.debug(
                "Saving last 10 trials instances ids to...%s",
                self.trials_instances_file,
            )
            with open(self.trials_instances_file, "w") as ti:
                json.dump(list(keys[-10:]), ti)
        except KeyError:
            pass

    def _setup_trials(self, load_trials=True, backup=False):
        """ The trials instance is the key used to identify the hdf table """
        # If the Hyperopt class has been previously initialized
        if self.config.get("hyperopt_skip_trials_setup", False):
            return
        self.parameters: List[Any]
        self.parameters = self.hyperopt_space()
        self.trials_instance = "{}-{}/{}/{}".format(
            self.config["hyperopt_loss"],
            len(self.parameters),
            "_".join(sorted(self.config["spaces"])),
            # truncate hash to 4 digits
            str(hash([d.name for d in self.parameters]))[:4],
        )
        logger.info(f"Hyperopt state will be saved to " f"key {self.trials_instance}")

        self.cleanup_store_tables()

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
                backup=backup,
                clear=False,
            )
            logger.info(
                f"Loaded {len(self.trials)} previous trials from {self.trials_file}@{trials_instance}."
            )

            if self.cv:
                tr_len = len(self.trials)
                if tr_len < 1:
                    raise OperationalException("CV requires a starting dataset.")
                # in cross validation apply progressive filtering, tuned by hyperopt_epochs
                intensity = prev_its = 1
                backtrack = False
                prev_tt_len = 0
                max_iterations = 100
                # use a ~0.1 histeresis for desired number of filtered epochs
                min_tt = int(self.total_epochs * 0.9)
                max_tt = int(self.total_epochs * 1.1)
                if tr_len < min_tt or tr_len > max_tt:
                    while max_iterations > 0:
                        max_iterations -= 1
                        logger.debug("filtering trials with intensity: %s", intensity)
                        self.target_trials = self.filter_trials(
                            self.trials, self.config, intensity=intensity
                        )
                        logger.debug("filtered trials to: %s", len(self.target_trials))
                        # backtrack means we applied a previous intensity because we probably
                        # offshoot the calculation
                        tt_len = len(self.target_trials)
                        if backtrack or self.total_epochs < 2:
                            logger.debug(
                                "stopped trials filtering, backtrack: %s", backtrack
                            )
                            break
                        # don't filter if the filtered trials are below setting
                        if tt_len < min_tt and tt_len != prev_tt_len:
                            logger.debug(
                                "decreasing intensity since %s < %s", tt_len, max_tt
                            )
                            prev_its = intensity
                            intensity *= self.total_epochs / tt_len
                            max_iterations -= 1
                        elif tt_len > max_tt:
                            logger.debug(
                                "increasing intensity since %s > %s", tt_len, max_tt
                            )
                            prev_its = intensity
                            intensity *= self.total_epochs / tt_len
                        elif tt_len < min_tt and intensity < prev_its:
                            logger.debug(
                                "backtracking filtering because %s < %s", tt_len, min_tt
                            )
                            backtrack = True
                            intensity = prev_its
                        else:
                            break
                        prev_tt_len = tt_len
                else:
                    self.target_trials = self.trials

                if len(self.target_trials) < 1:
                    logger.warn("Filtering returned 0 trials, using original dataset.")
                    self.target_trials = self.trials
                else:
                    logger.info(
                        "Filtered {} trials down to {}.".format(
                            len(self.trials), len(self.target_trials)
                        )
                    )
            elif len(self.trials) > 0 and not self.async_sched:
                if self.random_state != self.trials.iloc[-1]["random_state"]:
                    logger.warn("Random state in saved trials doesn't match runtime...")
        if self.cv:
            # CV trials are saved in their own table
            self.trials_instance += "_cv"
            # reset cv trials only if not specified
            if not load_instance or load_instance != "cv":
                self.clear_instance(self.trials_file, self.trials_instance)

    @property
    def epochs_limit(self) -> int:
        return self.total_epochs

    def _setup_epochs(self) -> bool:
        """ used to resume the best epochs state from previous trials """
        locked = backend.acquire_lock(backend.epochs, True, timeout=60)
        if not locked:
            raise OperationalException(
                "Couldn't acquire lock at startup during epochs setup."
            )
        ep = backend.epochs
        # signal each worker for space reduction
        ep.space_reduction = backend.manager.dict()
        # map optimizers to workers ids
        ep.pinned_optimizers = backend.manager.dict()

        ep.last_best_epoch = 0
        ep.last_best_loss = float(VOID_LOSS)
        if self.multi:
            for rs in self.rngs:
                ep.current_best_epoch[rs] = 0
                ep.current_best_loss[rs] = VOID_LOSS
        else:
            ep.current_best_epoch[None] = 0
            ep.current_best_loss[None] = VOID_LOSS
        # shared collections have to use the manager
        len_trials = len(self.trials)
        ep.epochs_since_last_best = backend.manager.list([0, 0])

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
                if self.multi:
                    for rs in self.rngs:
                        # sorting from lowest to highest, the first value is the current best
                        opt_best = self.trials[self.trials["random_state"].values == rs]
                        if len(opt_best) > 0:
                            best = opt_best.sort_values(by=["loss"]).iloc[0].to_dict()
                            ep.current_best_epoch[rs] = best["current_epoch"]
                            ep.current_best_loss[rs] = best["loss"]
                    # group by random state, get the lowest loss for each state, then pick the one with
                    # the highest current_epoch
                    last_best_trial = (
                        self.trials.groupby("random_state")
                        .apply(lambda x: x.sort_values(by=["loss"]).iloc[0])
                        .sort_values(by="current_epoch")
                        .iloc[-1]
                    ).to_dict()
                    ep.last_best_epoch = last_best_trial["current_epoch"]
                    ep.last_best_loss = last_best_trial["loss"]
                else:
                    best = self.trials.sort_values(by=["loss"]).iloc[0].to_dict()
                    ep.current_best_epoch[None] = ep.last_best_epoch = best[
                        "current_epoch"
                    ]
                    ep.current_best_loss[None] = ep.last_best_loss = best["loss"]
        ep.max_epoch = self.epochs_limit
        backend.release_lock(ep)
        return resumed

    def _setup_space_reduction(self):
        """ Choose which configuration to use when applying space reduction """
        config = self.config
        if config.get("hyperopt_spc_red_config", False):
            self.space_reduction_interval = config.get(
                "hyperopt_spc_red_interval", iinfo(int32).max
            )
            if self.space_reduction_interval == iinfo(int32).max:
                return
        else:
            if self.shared:
                from freqtrade.optimize.hyperopt_constants import SHARED_SPACE_CONFIG

                config = SHARED_SPACE_CONFIG
            elif self.multi:
                from freqtrade.optimize.hyperopt_constants import MULTI_SPACE_CONFIG

                config = MULTI_SPACE_CONFIG
            else:
                from freqtrade.optimize.hyperopt_constants import SINGLE_SPACE_CONFIG

                config = SINGLE_SPACE_CONFIG
        self.space_reduction_interval = config["hyperopt_spc_red_interval"]
        self.space_reduction_config = config

    def _set_random_state(self, random_state: Optional[int]) -> int:
        if self.cv:
            return 0
        else:
            rs = random_state or random.randint(1, 2 ** 16 - 1)
            np.random.seed(rs)
            return rs

    def _setup_points(self):
        """
        Calc starting points, based on parameters, given epochs, mode
        """
        self.search_space_size = (
            guess_search_space(self.parameters,) if not self.cv else 0
        )
        if not self.cv:
            logger.debug("Initial points: ~%s", self.n_rand)
        # each column is a parameter, needed to read points from storage
        # in cv mode we take the params names from the saved epochs columns
        col = "{d.name}" if not self.cv else "{d}"
        self.Xi_names = tuple(col.format(d=d) for d in self.parameters)
        logger.info(f"Parameters set for optimization: {len(self.Xi_names)}")
        if not self.cv:
            logger.info(f"Search space size: {self.search_space_size:e}")

    def _maybe_terminate(
        self, t: int, jobs: int, trials_state: TrialsState, epochs: Epochs
    ) -> bool:
        """ signal workers to terminate if conditions are met """
        done = trials_state.num_done
        total = trials_state.num_saved + done
        if not self.void_output_backoff and done < t - jobs:
            logger.warn(
                "Some evaluated epochs were void, "
                "check the loss function and the search space."
            )
            self.void_output_backoff = True
        # don't consider an empty strike until the iterator
        # has crossed the number of defined maxout
        if not done and t > (total + self.trials_maxout + jobs * self.ask_points):
            trials_state.empty_strikes += 1
        if backend.epochs.avg_wait_time >= self.opt_ask_timeout:
            logger.debug(
                f"Average wait time for optimizer reached, %s", self.opt_ask_timeout
            )
            trials_state.exit = True
        cvg_ratio = (epochs.convergence / total) if total > 0 else 0
        if cvg_ratio > self.max_convergence_ratio:
            logger.warn(
                f"Max convergence ratio reached ({cvg_ratio:.2f}), terminating."
            )
            trials_state.exit = True
        elif (
            not done
            and self.search_space_size < np.log(total + self.epochs_limit)
            and not self.cv
        ) or trials_state.empty_strikes > self.trials_max_empty:
            logger.error("Terminating Hyperopt because trials were empty.")
            trials_state.exit = True
        # give up if no best since max epochs
        elif total >= self.epochs_limit:
            logger.warn(
                "Max epoch reached %s > %s, terminating.", total, self.epochs_limit
            )
            trials_state.exit = True
        return trials_state.exit

    def main_loop(self, jobs_scheduler):
        """ main parallel loop """
        # dump the object state which will be loaded by every worker
        # instead of pickling functions around
        logger.debug("dumping pickled hyperopt object to path: %s", self.cls_file)
        dump(wrap_non_picklable_objects(self), self.cls_file)
        logger.debug("starting workers pool")
        # in single mode dispatching is synchronized such that
        # it won't wait for a bigger batch to be filled and stall
        pre_dispatch = (
            self.n_jobs
            if (not self.async_sched and not self.cv) and self.opt.is_blocking
            else "2*n_jobs"
        )
        with parallel_backend("loky", inner_max_num_threads=2):
            with Parallel(
                n_jobs=self.n_jobs,
                verbose=0,
                backend="loky",
                pre_dispatch=pre_dispatch,
            ) as parallel:
                try:
                    # reset parallel state
                    jobs = parallel._effective_n_jobs()
                    logger.info(f"Effective parallel workers: {jobs}")
                    self.run_setup_backend_parallel(parallel, jobs)

                    # run the jobs
                    jobs_scheduler(parallel, jobs)

                # exceptions should be caught within each worker too
                except KeyboardInterrupt as e:
                    logger.error(f"Main loop stopped. {e}")
                # collect remaining unsaved epochs
                backend.trials.exit = True
                logger.debug(
                    "ending soon, num_done: %s num_saved: %s",
                    backend.trials.num_done,
                    backend.trials.num_saved,
                )
                if backend.trials.num_done > backend.trials.num_saved:
                    logger.debug(
                        "flushing remaining %s trials to storage",
                        backend.trials.num_done,
                    )
                    get_reusable_executor().shutdown(wait=True)
                    logger.debug(
                        "tail dict is %s and tail list is %s",
                        backend.trials.tail_dict,
                        backend.trials.tail_list,
                    )
                    # since the list was passed through the manager, make a copy
                    if len(backend.trials.tail_dict):
                        for rs in backend.trials.tail_dict.keys():
                            if len(backend.trials.tail_dict[rs]):
                                backend.trials_list = [
                                    t for t in backend.trials.tail_dict[rs]
                                ]
                                backend.just_saved = self.log_trials(
                                    backend.trials, backend.epochs, rs=rs
                                )
                                backend.trials.num_done -= backend.just_saved
                    elif len(backend.trials.tail_list):
                        backend.trials_list = [t for t in backend.trials.tail_list]
                        backend.just_saved = self.log_trials(
                            backend.trials, backend.epochs, rs=None
                        )
                        backend.trials.num_done -= backend.just_saved
                    logger.debug("saved %s trials at the end", backend.just_saved)

                if self.use_progressbar:
                    HyperoptOut._print_progress(
                        backend.just_saved, jobs, self.trials_maxout, finish=True
                    )

    def _setup_data(self):
        """ Load ohlcv data, and check that:
        - startup period is removed
        - there is a global minimum amount of data to test
        """

        data, timerange = self.backtesting.load_bt_data()
        preprocessed = self.backtesting.strategy.ohlcvdata_to_dataframe(data)

        preprocessed, self.n_candles = check_data_startup(
            preprocessed, self.backtesting.required_startup, timerange
        )

        # use the dict provided by backtesting to calc the pairslist length
        # since it applies pairlist filters
        n_pairs = len(data)
        min_candles_ratio = self.config.get("hyperopt_min_candles_ratio", 0.5)
        max_candles = n_pairs * (
            abs(Timedelta(timerange.stopts - timerange.startts))
            / Timedelta(self.config["timeframe"])
        )
        candles_ratio = self.n_candles / max_candles
        if candles_ratio < min_candles_ratio:
            raise OperationalException(
                "Not enough candles for the provided combination of pairs, timeframe "
                f"and timerange, min candle ratio {candles_ratio:.02} < {min_candles_ratio:.02}"
            )
        min_date, max_date = get_timerange(preprocessed)
        self.min_date, self.max_date = min_date, max_date

        logger.info(
            f"Hyperopting with data from {min_date.strftime(DATETIME_PRINT_FORMAT)} "
            f"up to {max_date.strftime(DATETIME_PRINT_FORMAT)} "
            f"({(max_date - min_date).days} days).."
        )

        dump(preprocessed, self.data_pickle_file)

    def start(self) -> None:
        """ Broom Broom """
        self.random_state = self._set_random_state(
            self.config.get("hyperopt_random_state", None)
        )
        logger.info(f"Using optimizer random state: {self.random_state}")
        backend.trials.table_header = 0

        self._setup_data()

        # We don't need exchange instance anymore while running hyperopt
        self.backtesting.exchange = None  # type: ignore
        self.backtesting.pairlists = None  # type: ignore
        self.backtesting.strategy.dp = None  # type: ignore
        IStrategy.dp = None  # type: ignore

        # Set dimensions, trials instance and paths and load from storage
        self._setup_trials()

        # Set number of initial points, and optimizer related stuff
        self._setup_points()

        # Choose space reduction configuration
        self._setup_space_reduction()

        if self.print_colorized:
            colorama_init(autoreset=True)

        if not self.cv:
            self._setup_optimizers()
        self._setup_loss_funcs()

        # Count the epochs
        self._setup_epochs()

        if not self.cv:
            # After setup, trials are only needed by CV so can delete
            self.trials.drop(self.trials.index, inplace=True)

        if self.cv:
            jobs_scheduler = self.run_cv_backtest_parallel
        elif self.async_sched:
            jobs_scheduler = self.run_multi_backtest_parallel
        else:
            jobs_scheduler = self.run_backtest_parallel

        if self.use_progressbar:
            HyperoptOut._init_progressbar(
                self.print_colorized, self.epochs_limit or None, self.cv
            )
        self.main_loop(jobs_scheduler)

        # At the end, print best epoch
        print_best = self.config.get("print_best_at_end")
        if backend.trials.num_saved and print_best:
            trials = self.load_trials(
                self.trials_file, self.trials_instance, backend.trials,
            )
            best_trial = (
                trials.sort_values(by="loss").iloc[:1].to_dict(orient="records")
            )[:1]

            if len(best_trial):
                logger.debug("best trial: %s", best_trial[0])
                self.print_epoch_details(
                    best_trial[0], self.epochs_limit, self.print_json
                )
        elif print_best:
            # This is printed when Ctrl+C is pressed quickly, before first epochs have
            # a chance to be evaluated.
            self.logger.info("No epochs evaluated yet, no best result.")

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["trials"]
        if self.cv:
            del state["target_trials"]
        elif not self.async_sched:
            del state["opt"]
        return state
