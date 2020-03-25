# pragma pylint: disable=too-many-instance-attributes, pointless-string-statement
"""
This module contains the hyperopt logic
"""

import random
import sys
import warnings
from collections import deque
from math import factorial, log
from operator import itemgetter
from typing import Any, Dict, List, Optional, Tuple, Set

from colorama import init as colorama_init
from joblib import Parallel, cpu_count, delayed, dump, load, wrap_non_picklable_objects
from joblib import parallel_backend
from pandas import DataFrame

from freqtrade.data.converter import trim_dataframe
from freqtrade.data.history import get_timerange
from freqtrade.optimize.backtesting import Backtesting

# Import IHyperOpt and IHyperOptLoss to allow unpickling classes from these modules
import freqtrade.optimize.hyperopt_backend as backend

from freqtrade.optimize.hyperopt_data import HyperoptData
from freqtrade.optimize.hyperopt_multi import HyperoptMulti
from freqtrade.optimize.hyperopt_out import HyperoptOut
from freqtrade.optimize.hyperopt_cv import HyperoptCV
from freqtrade.optimize.hyperopt_constants import *

# from freqtrade.optimize.hyperopt_backend import Trial
from freqtrade.optimize.hyperopt_interface import IHyperOpt  # noqa: F401
from freqtrade.optimize.hyperopt_loss_interface import IHyperOptLoss  # noqa: F401
from freqtrade.resolvers.hyperopt_resolver import HyperOptLossResolver, HyperOptResolver
from freqtrade.optimize.hyperopt_backend import filter_trials

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


class Hyperopt(HyperoptData, HyperoptOut, HyperoptMulti, HyperoptCV):
    """
    Hyperopt class, this class contains all the logic to run a hyperopt simulation

    To run a backtest:
    hyperopt = Hyperopt(config)
    hyperopt.start()
    """

    def __init__(self, config: Dict[str, Any]) -> None:

        self.config = config

        self.backtesting = Backtesting(self.config)

        self.custom_hyperopt = HyperOptResolver.load_hyperopt(self.config)

        self.custom_hyperoptloss = HyperOptLossResolver.load_hyperoptloss(self.config)
        self.calculate_loss = self.custom_hyperoptloss.hyperopt_loss_function

        self.trials_file = (
            self.config["user_data_dir"] / "hyperopt_results" / "hyperopt_results.pickle"
        )
        self.data_pickle_file = (
            self.config["user_data_dir"] / "hyperopt_results" / "hyperopt_tickerdata.pkl"
        )
        self.opts_file = (
            self.config["user_data_dir"] / "hyperopt_results" / "hyperopt_optimizers.pickle"
        )
        self.cv_trials_file = (
            self.config["user_data_dir"] / "hyperopt_results" / "hyperopt_cv_results.pickle"
        )

        self.n_jobs = self.config.get("hyperopt_jobs", -1)
        if self.n_jobs < 0:
            self.n_jobs = cpu_count() // 2 or 1
        self.effort = max(0.01, self.config["effort"] if "effort" in self.config else 1)
        self.total_epochs = self.config["epochs"] if "epochs" in self.config else 0
        self.max_epoch = 0
        self.max_epoch_reached = False
        self.min_epochs = 0
        self.epochs_limit = lambda: self.total_epochs or self.max_epoch

        # a guessed number extracted by the space dimensions
        self.search_space_size = 0
        # total number of candles being backtested
        self.n_candles = 0
        # cross validation
        if "hyperopt_cv" in self.config:
            self.cv = self.config["hyperopt_cv"]
        else:
            self.cv = False

        self.current_best_loss = VOID_LOSS
        self.current_best_epoch = 0
        self.epochs_since_last_best: List = []
        self.avg_best_occurrence = 0

        if not self.config.get("hyperopt_continue") and not self.cv:
            self.clean_hyperopt()
        else:
            logger.info("Continuing on previous hyperopt results.")

        self.num_trials_saved = 0

        # evaluations
        self.trials: List = []

        # configure multi mode
        self.setup_multi()

        # Populate functions here (hasattr is slow so should not be run during "regular" operations)
        if hasattr(self.custom_hyperopt, "populate_indicators"):
            self.backtesting.strategy.advise_indicators = (
                self.custom_hyperopt.populate_indicators
            )  # type: ignore
        if hasattr(self.custom_hyperopt, "populate_buy_trend"):
            self.backtesting.strategy.advise_buy = (
                self.custom_hyperopt.populate_buy_trend
            )  # type: ignore
        if hasattr(self.custom_hyperopt, "populate_sell_trend"):
            self.backtesting.strategy.advise_sell = (
                self.custom_hyperopt.populate_sell_trend
            )  # type: ignore

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

        self.print_all = self.config.get("print_all", False)
        self.hyperopt_table_header = 0
        self.print_colorized = self.config.get("print_colorized", False)
        self.print_json = self.config.get("print_json", False)

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
            result["roi"] = self.custom_hyperopt.generate_roi_table(params)
        if self.has_space("stoploss"):
            result["stoploss"] = {
                p.name: params.get(p.name) for p in self.hyperopt_space("stoploss")
            }
        if self.has_space("trailing"):
            result["trailing"] = self.custom_hyperopt.generate_trailing_params(params)

        return result

    @staticmethod
    def is_best_loss(results, current_best_loss: float) -> bool:
        return results["loss"] < current_best_loss

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

    def backtest_params(self, raw_params: List[Any], iteration=None) -> Dict:
        """
        Used Optimize function. Called once per epoch to optimize whatever is configured.
        Keep this function as optimized as possible!
        """
        params_dict = self._get_params_dict(raw_params) if not self.cv else raw_params
        params_details = self._get_params_details(params_dict) if not self.cv else {}
        if self.cv:
            params_values = raw_params
        else:
            params_values = params_dict

        if self.has_space("roi"):
            self.backtesting.strategy.minimal_roi = self.custom_hyperopt.generate_roi_table(
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
            self.backtesting.strategy.stoploss = params_dict["stoploss"]

        if self.has_space("trailing"):
            d = self.custom_hyperopt.generate_trailing_params(params_dict)
            self.backtesting.strategy.trailing_stop = d["trailing_stop"]
            self.backtesting.strategy.trailing_stop_positive = d["trailing_stop_positive"]
            self.backtesting.strategy.trailing_stop_positive_offset = d[
                "trailing_stop_positive_offset"
            ]
            self.backtesting.strategy.trailing_only_offset_is_reached = d[
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
        return self._get_trial(
            backtesting_results, min_date, max_date, params_values, params_details, processed
        )

    def _get_trial(
        self, backtesting_results, min_date, max_date, params_dict, params_details, processed
    ):
        results_metrics = self._calculate_results_metrics(backtesting_results)
        results_explanation = self._format_results_explanation_string(results_metrics)

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

    def lie_strategy(self):
        """ Choose a strategy randomly among the supported ones, used in multi opt mode
        to increase the diversion of the searches of each optimizer """
        return LIE_STRATS[random.randrange(0, LIE_STRATS_N)]

    def estimators(self):
        return ESTIMATORS[random.randrange(0, ESTIMATORS_N)]

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
            n_initial_points=self.opt_n_initial_points,
            acq_optimizer_kwargs={"n_jobs": n_jobs},
            model_queue_size=self.n_models,
            random_state=random_state or self.random_state,
        )

    def run_backtest_parallel(self, parallel: Parallel, tries: int, first_try: int, jobs: int):
        """ launch parallel in single opt mode, return the evaluated epochs """
        parallel(
            delayed(wrap_non_picklable_objects(self.parallel_objective))(
                asked, backend.results_list, i
            )
            for asked, i in zip(
                self.opt_ask_and_tell(jobs, tries), range(first_try, first_try + tries)
            )
        )

    def opt_ask_and_tell(self, jobs: int, tries: int):
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
        if self.ask_points:

            def point():
                if to_ask:
                    return tuple(to_ask.popleft())
                else:
                    to_ask.extend(opt.ask(n_points=self.ask_points, strategy=self.lie_strat()))
                    return tuple(to_ask.popleft())

        else:

            def point():
                return tuple(opt.ask(strategy=self.lie_strat()))

        for r in range(tries):
            fit = len(to_ask) < 1
            if len(backend.results_list) > 0:
                vals.extend(backend.results_list)
                del backend.results_list[:]
            if vals:
                # filter losses
                void_filtered = HyperoptMulti.filter_void_losses(vals, opt)
                if void_filtered:  # again if all are filtered
                    opt.tell(
                        [HyperoptMulti.opt_params_Xi(v) for v in void_filtered],
                        [v["loss"] for v in void_filtered],
                        fit=fit,
                    )  # only fit when out of points
                    self.batch_results.extend(void_filtered)
                del vals[:], void_filtered[:]

            a = point()
            # this usually happens at the start when trying to fit before the initial points
            if a in evald:
                logger.debug("this point was evaluated before...")
                opt.update_next()
                a = point()
                if a in evald:
                    break
            evald.add(a)
            yield a

    def parallel_objective(self, asked, results_list: List = [], n=0):
        """ objective run in single opt mode, run the backtest, store the results into a queue """
        self.log_results_immediate(n)
        v = self.backtest_params(asked)

        v["is_initial_point"] = n < self.opt_n_initial_points
        v["random_state"] = self.random_state
        results_list.append(v)

    def log_results_immediate(self, n) -> None:
        """ Signals that a new job has been scheduled"""
        print(".", end="")
        sys.stdout.flush()

    def log_results(self, batch_results, batch_start, total_epochs: int) -> int:
        """
        Log results if it is better than any previous evaluation
        """
        HyperoptOut.reset_line()

        current = batch_start + 1
        prev_best = self.current_best_epoch
        current_best = prev_best
        i = 0
        for i, v in enumerate(batch_results, 1):
            is_best = self.is_best_loss(v, self.current_best_loss)
            current = batch_start + i
            v["is_best"] = is_best
            v["current_epoch"] = current
            logger.debug(f"Optimizer epoch evaluated: {v}")
            if is_best:
                current_best = current
                self.current_best_loss = v["loss"]
            self.print_results(v)
            self.trials.append(v)
        self.update_max_epoch(prev_best, current_best, current)
        # Save results and optimizers after every batch
        self.save_trials()
        # track new points if in multi mode
        if self.multi and not self.cv:
            self.track_points(trials=self.trials[batch_start:])
            # clear points used by optimizers intra batch
            backend.results_shared.update(self.opt_empty_tuple())
        # give up if no best since max epochs
        if current + 1 > self.epochs_limit():
            self.max_epoch_reached = True

        HyperoptOut.clear_line(columns)
        return i

    def setup_epochs(self) -> bool:
        """ used to resume the best epochs state from previous trials """
        len_trials = len(self.trials)
        if self.cv:
            self.total_epochs = len(self.target_trials)
            self.start_epoch = 0
        else:
            self.start_epoch = len(self.trials)
        if len_trials > 0:
            best_epochs = list(filter(lambda k: k["is_best"], self.trials))
            len_best = len(best_epochs)
            if len_best > 0:
                # sorting from lowest to highest, the first value is the current best
                best = sorted(best_epochs, key=lambda k: k["loss"])[0]
                self.current_best_epoch = best["current_epoch"]
                self.current_best_loss = best["loss"]
                self.avg_best_occurrence = max(self.n_jobs, len_trials // len_best)
                return True
        return False

    def _set_random_state(self, random_state: Optional[int]) -> int:
        return random_state or random.randint(1, 2 ** 16 - 1)

    @staticmethod
    def _stub_dimension(k):
        d = Dimension()
        d.name = k
        return d

    @staticmethod
    def calc_epochs(
        dimensions: List[Dimension], n_jobs: int, effort: float, total_epochs: int, n_points: int
    ):
        """ Compute a reasonable number of initial points and
        a minimum number of epochs to evaluate """
        n_dimensions = len(dimensions)
        n_parameters = 0
        opt_points = n_jobs * n_points
        # sum all the dimensions discretely, granting minimum values
        for d in dimensions:
            if type(d).__name__ == "Integer":
                n_parameters += max(1, d.high - d.low)
            elif type(d).__name__ == "Real":
                n_parameters += max(10, int(d.high - d.low))
            else:
                n_parameters += len(d.bounds)
        # guess the size of the search space as the count of the
        # unordered combination of the dimensions entries
        try:
            search_space_size = int(
                (
                    factorial(n_parameters)
                    / (factorial(n_parameters - n_dimensions) * factorial(n_dimensions))
                )
            )
        except OverflowError:
            search_space_size = VOID_LOSS
        # logger.info(f'Search space size: {search_space_size}')
        log_opt = int(log(opt_points, 2)) if opt_points > 4 else 2
        if search_space_size < opt_points:
            # don't waste if the space is small
            n_initial_points = opt_points // 3
            min_epochs = opt_points
        elif total_epochs > 0:
            # coefficients from total epochs
            log_epp = int(log(total_epochs, 2)) * log_opt
            n_initial_points = min(log_epp, total_epochs // 3)
            min_epochs = total_epochs
        else:
            # extract coefficients from the search space
            log_sss = int(log(search_space_size, 10)) * log_opt
            # never waste
            n_initial_points = min(log_sss, search_space_size // 3)
            # it shall run for this much, I say
            min_epochs = int(max(n_initial_points, opt_points) + 2 * n_initial_points)
        return int(n_initial_points * effort) or 1, int(min_epochs * effort), search_space_size

    def update_max_epoch(self, prev_best: int, current_best: int, current: int):
        """ calculate max epochs: store the number of non best epochs
            between each best, and get the mean of that value """
        if prev_best == current_best:
            last_period = self.epochs_since_last_best[-1]
            self.epochs_since_last_best[-1] = last_period + (current - self.current_best_epoch)
        else:
            self.epochs_since_last_best.append(current_best - prev_best)
        self.avg_best_occurrence = sum(self.epochs_since_last_best) // len(
            self.epochs_since_last_best
        )
        self.max_epoch = int(
            (self.current_best_epoch + self.avg_best_occurrence + self.min_epochs)
            * max(1, self.effort)
        )
        if self.max_epoch > self.search_space_size:
            self.max_epoch = self.search_space_size
        logger.debug(f"\nMax epoch set to: {self.epochs_limit()}")

    def track_points(self, trials: List = None):
        """
        keep tracking of the evaluated points per optimizer random state
        """
        # if no trials are given, use saved trials
        if not trials:
            if len(self.trials) > 0:
                if self.config.get("hyperopt_continue_filtered", False):
                    trials = filter_trials(self.trials, self.config)
                else:
                    trials = self.trials
            else:
                return
        for v in trials:
            rs = v["random_state"]
            try:
                self.Xi[rs].append(HyperoptMulti.opt_params_Xi(v))
                self.yi[rs].append(v["loss"])
            except IndexError:  # Hyperopt was started with different random_state or number of jobs
                pass

    def setup_points(self):
        if self.cv:
            self.search_space_size = VOID_LOSS
            self.min_epochs = self.total_epochs
            self.n_initial_points = self.min_epochs
        else:
            self.n_initial_points, self.min_epochs, self.search_space_size = self.calc_epochs(
                self.dimensions, self.n_jobs, self.effort, self.total_epochs, self.n_points
            )
        logger.info(f"Min epochs set to: {self.min_epochs}")
        # reduce random points by n_points in multi mode because asks are per job
        if self.multi:
            self.opt_n_initial_points = self.n_initial_points // self.n_points
        else:
            self.opt_n_initial_points = self.n_initial_points
        logger.info(f"Initial points: {self.n_initial_points}")
        # if total epochs are not set, max_epoch takes its place
        if self.total_epochs < 1:
            self.max_epoch = int(self.min_epochs + len(self.trials))
        # initialize average best occurrence
        self.avg_best_occurrence = max(self.min_epochs // 3, self.n_jobs * 3)

    def return_results(self):
        """
        results are passed by queue in multi mode, or stored by ask_and_tell in single mode
        """
        batch_results = []
        if self.multi:
            while not backend.results_batch.empty():
                worker_results = backend.results_batch.get()
                batch_results.extend(worker_results)
        else:
            batch_results.extend(self.batch_results)
            del self.batch_results[:]
        return batch_results

    def _batch_len(self, epochs_so_far: int):
        """ calc the length of the next batch """
        jobs = self.n_jobs
        min_occurrence = jobs * 3
        opt_points = self.opt_points
        epochs_limit = self.epochs_limit

        occurrence = max(min_occurrence, int(self.avg_best_occurrence * max(1.1, self.effort)))
        # pad the batch length to the number of jobs to avoid desaturation
        batch_len = occurrence + jobs - occurrence % jobs
        # don't go over the limit
        if epochs_so_far + batch_len * opt_points >= epochs_limit():
            q, r = divmod(epochs_limit() - epochs_so_far, opt_points)
            batch_len = q + r
        print(f"{epochs_so_far+1}-{epochs_so_far+batch_len}" f"/{epochs_limit()}: ", end="")
        return batch_len

    def _batch_check(self, batch_len: int, saved: int, epochs_so_far: int) -> bool:
        # stop if no epochs have been evaluated
        if saved < batch_len:
            logger.warning(
                "Some evaluated epochs were void, " "check the loss function and the search space."
            )
        if batch_len < 1 or (
            not saved and self.search_space_size < batch_len + epochs_limit() and not self.cv
        ):
            logger.info("Terminating Hyperopt because results were empty.")
            return False
        if self.max_epoch_reached:
            logger.info("Max epoch reached, terminating.")
            return False
        return True

    def main_loop(self, jobs_scheduler):
        """ main parallel loop """
        try:
            with parallel_backend("loky", inner_max_num_threads=2):
                with Parallel(n_jobs=self.n_jobs, verbose=0, backend="loky") as parallel:
                    jobs = parallel._effective_n_jobs()
                    logger.info(f"Effective number of parallel workers used: {jobs}")
                    # update epochs count
                    prev_batch = -1
                    epochs_so_far = self.start_epoch
                    epochs_limit = self.epochs_limit
                    while epochs_so_far > prev_batch or epochs_so_far < self.min_epochs:
                        batch_results = []
                        prev_batch = epochs_so_far
                        batch_len = self._batch_len(epochs_so_far)

                        # run the jobs
                        jobs_scheduler(parallel, batch_len, epochs_so_far, jobs)
                        batch_results = self.return_results()

                        # save the results
                        saved = self.log_results(batch_results, epochs_so_far, epochs_limit())

                        if not self._batch_check(batch_len, saved, epochs_so_far):
                            break
                        epochs_so_far += saved

        except KeyboardInterrupt:
            print("User interrupted..")

    def start(self) -> None:
        """ Broom Broom """
        self.random_state = self._set_random_state(self.config.get("hyperopt_random_state", None))
        logger.info(f"Using optimizer random state: {self.random_state}")
        self.hyperopt_table_header = -1
        data, timerange = self.backtesting.load_bt_data()

        preprocessed = self.backtesting.strategy.ohlcvdata_to_dataframe(data)

        # Trim startup period from analyzed dataframe
        for pair, df in preprocessed.items():
            preprocessed[pair] = trim_dataframe(df, timerange)
            self.n_candles += len(preprocessed[pair])
        min_date, max_date = get_timerange(data)

        logger.info(
            "Hyperopting with data from %s up to %s (%s days)..",
            min_date.isoformat(),
            max_date.isoformat(),
            (max_date - min_date).days,
        )
        dump(preprocessed, self.data_pickle_file)

        # We don't need exchange instance anymore while running hyperopt
        self.backtesting.exchange = None  # type: ignore

        self.trials = self.load_previous_results(self.trials_file)
        self.dimensions: List[Any]
        if self.cv:
            self.target_trials = filter_trials(self.trials, self.config)
            self.trials = []
            self.dimensions = [k for k in self.target_trials[-1]["params_dict"]]
            self.total_epochs = len(self.target_trials)
        else:
            self.dimensions = self.hyperopt_space()

        self.setup_epochs()

        self.setup_points()

        if self.print_colorized:
            colorama_init(autoreset=True)

        if not self.cv:
            self.setup_optimizers()

        logger.info(f"Found {cpu_count()} CPU cores. Let's make them scream!")
        logger.info(f"Number of parallel jobs set as: {self.n_jobs}")
        if self.multi and not self.cv:
            jobs_scheduler = self.run_multi_backtest_parallel
        elif self.cv:
            jobs_scheduler = self.run_cv_backtest_parallel
        else:
            jobs_scheduler = self.run_backtest_parallel

        self.main_loop(jobs_scheduler)

        self.save_trials(final=True)

        if self.trials:
            sorted_trials = sorted(self.trials, key=itemgetter("loss"))
            results = sorted_trials[0]
            self.print_epoch_details(results, self.epochs_limit(), self.print_json)
        else:
            # This is printed when Ctrl+C is pressed quickly, before first epochs have
            # a chance to be evaluated.
            print("No epochs evaluated yet, no best result.")

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["trials"]
        if self.cv:
            del state["target_trials"]
        return state
