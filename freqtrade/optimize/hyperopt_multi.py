import warnings
import pickle
import logging
import os
import tempfile
from typing import Any, Dict, List, Tuple
from abc import abstractmethod
from time import time as now
from functools import partial

from joblib import Parallel, delayed, wrap_non_picklable_objects, hash
from multiprocessing.managers import Namespace, SyncManager
import signal
from filelock import FileLock
from queue import Queue
from pandas import read_sql, read_parquet, read_hdf, HDFStore
from sqlalchemy.orm import Session
from sqlalchemy.engine import Engine
from sqlalchemy import create_engine

# Import IHyperOpt and IHyperOptLoss to allow unpickling classes from these modules
import freqtrade.optimize.hyperopt_backend as backend
from freqtrade.optimize.hyperopt_out import HyperoptOut
from freqtrade.optimize.hyperopt_constants import VOID_LOSS
from freqtrade.optimize.hyperopt_interface import IHyperOpt  # noqa: F401
from freqtrade.optimize.hyperopt_loss_interface import IHyperOptLoss  # noqa: F401
from freqtrade.exceptions import OperationalException

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


class HyperoptMulti(HyperoptOut):
    """ Run the optimization with multiple optimizers """

    # tracks the number of batches that were completely void
    empty_batches = 0
    # stop warning against missing results after a while
    void_output_backoff = 0
    Xi_cols = []

    @abstractmethod
    def backtest_params(
        self, raw_params: List[Any] = None, iteration=None, params_dict: Dict[str, Any] = None
    ):
        """
        Used Optimize function. Called once per epoch to optimize whatever is configured.
        Keep this function as optimized as possible!
        """

    def setup_multi(self):
        # optimizer
        self.opt = None
        # start the manager
        self.setup_backend()
        # set mode of operations
        self.mode = self.config.get("mode", "single")
        mode = self.mode
        self.cv = mode == "cv"
        self.multi = mode in ("multi", "shared")
        self.shared = mode == "shared"
        # in multi opt one model is enough
        self.n_models = 1
        if self.multi:
            if self.shared:
                self.opt_base_estimator = lambda: "GBRT"
            else:
                self.opt_base_estimator = self.estimators
            self.opt_acq_optimizer = "sampling"
        else:
            # this is where ask_and_tell stores the results after points are
            # used for fit and predict, to avoid additional pickling
            self.batch_results = []
            # self.opt_base_estimator = lambda: BayesianRidge(n_iter=100, normalize=True)
            self.opt_base_estimator = lambda: "GP"
            self.opt_acq_optimizer = "sampling"
            # The GaussianProcessRegressor is heavy, which makes it not a good default
            # however longer backtests might make it a better tradeoff
            # self.opt_base_estimator = lambda: 'GP'
            # self.opt_acq_optimizer = 'lbfgs'
        # in single opt assume runs are expensive so default to 1 point per ask
        self.n_points = self.config.get("n_points", 1)
        # if 0 n_points are given, don't use any base estimator (akin to random search)
        if self.n_points < 1:
            self.n_points = 1
            self.opt_base_estimator = lambda: "DUMMY"
            self.opt_acq_optimizer = "sampling"
        if self.n_points < 2:
            # ask_points is what is used in the ask call
            # because when n_points is None, it doesn't
            # waste time generating new points
            self.ask_points = None
        else:
            self.ask_points = self.n_points
        # var used in epochs and batches calculation
        self.opt_points = self.n_jobs * (self.n_points or 1)
        # lie strategy
        lie_strat = self.config.get("lie_strat", "default")
        if lie_strat == "default":
            self.lie_strat = lambda: "cl_min"
        elif lie_strat == "random":
            self.lie_strat = self.lie_strategy
        else:
            self.lie_strat = lambda: lie_strat

    @staticmethod
    def setup_backend():
        """ Setup namespaces shared by workers """
        # start the manager with a noop signal handler to allow graceful termination
        backend.manager = SyncManager()
        backend.manager.start(partial(backend.manager_init, backend=backend))
        backend.epochs = backend.manager.Namespace()
        # when changing epochs counting and saving trials
        backend.epochs.lock = backend.manager.Lock()
        backend.trials = backend.manager.Namespace()
        # prevent reading to the store while saving
        backend.trials.lock = backend.manager.Lock()
        # when a worker finishes a dispatch, updates the done count
        backend.trials.num_done = 0
        # when a worker triggers a save, update the save count
        backend.trials.num_saved = 0
        # at the end collect the remaining trials to save here
        backend.trials.tail = backend.manager.list([])
        # at the end one last batch is dispatched to save the remaining trials
        backend.trials.exit = False
        return backend

    def run_multi_backtest_parallel(
        self, parallel: Parallel, tries: int, first_try: int, jobs: int
    ):
        """ Launch parallel in multi opt mode,
        scheduling the specified number of trials, passing the needed objects handled by the manager """
        parallel(
            delayed(wrap_non_picklable_objects(self.parallel_opt_objective_sig_handler))(
                t, jobs, backend.optimizers, backend.epochs, backend.trials
            )
            for t in range(first_try, first_try + tries)
        )

    @staticmethod
    def opt_get_past_points(
        asked: dict, Xi_cols: list, trials_file: str, trials_instance: str, trials_state: Namespace
    ) -> Tuple[dict, int]:
        """ fetch shared results between optimizers """
        # make a list of hashes since on storage points are queried by it
        asked_h = [hash(a) for a in asked]
        past_points = {}
        locked = False
        try:
            # past points are not critical, so if we can't acquire a lock
            # continue execution
            locked = trials_state.lock.acquire(False)
            if locked:
                past_points = read_hdf(
                    trials_file,
                    key=trials_instance,
                    where=[f"Xi_h in {asked_h}"],
                    columns=["loss", "Xi_h"],
                )
                trials_state.lock.release()
            else:
                return asked
        except (KeyError, FileNotFoundError, IOError, OSError):
            if locked:
                trials_state.lock.release()
        if len(past_points) > 0:
            past_points = past_points.set_index("Xi_h").to_dict()
            for n, a in enumerate(asked):
                asked[a] = past_points[asked_h[n]]
        return asked

    @staticmethod
    def opt_state(optimizers: Queue = None, s_opt: Optimizer = None) -> Optimizer:
        """ Return an optimizer in multi opt mode """
        # get an optimizer instance
        global opt
        if s_opt:
            opt = s_opt
            return
        if "opt" not in globals():
            # at worker startup fetch an optimizer from the queue
            if optimizers.qsize() > 0:
                opt = optimizers.get(timeout=1)
                # make a copy of the empty opt to resume from when the global
                # is gced
                backend.opt = opt
                # reset index of read trials since resuming from 0
                backend.trials_index = 0
                # store it back again to restore after global state is gced
                optimizers.put(opt)
            else:
                raise OperationalException(
                    "Global state was reclaimed and no optimizer instance was "
                    "available for recovery"
                )
        return opt

    @staticmethod
    def opt_params_Xi(v: dict):
        return list(v["params_dict"].values())

    def opt_startup_points(self, opt: Optimizer, trials_state: Namespace, is_shared: bool):
        """
        Multi mode: when a job is dispatched, the optimizer might still be present in the global state,
        if it is not (or the run just started) then load from disk.
        Shared mode: check for newly added points at every dispatch
        """
        # fit a model with the known points, either from a previous run, or read
        # from database only at the start when the global points references of each worker are empty
        params_df = []
        rs = opt.rs
        if is_shared or not opt.Xi:
            locked = False
            # fetch all points not already told in shared mode
            # ignoring random state
            if is_shared and backend.Xi_h:
                where = [f"Xi_h != {backend.Xi_h}"]
            elif is_shared:
                where = ""
            else:
                where = [f"random_state in {rs}"]
            try:
                # Only wait if the optimizer has no points at all (startup)
                locked = trials_state.lock.acquire(len(opt.Xi) < 1)
                if locked:
                    params_df = read_hdf(
                        self.trials_file,
                        key=self.trials_instance,
                        where=where,
                        columns=[*self.Xi_cols, "loss", "Xi_h"],
                        start=backend.trials_index,
                    )
                    trials_state.lock.release()
            except (
                KeyError,
                FileNotFoundError,
                IOError,
                OSError,
            ):  # only happens when df is empty and empty df is not saved on disk by pytables or is being written
                if locked:
                    trials_state.lock.release()
                logger.debug("Couldn't read trials from disk")
        if len(params_df) > 0:
            backend.trials_index += len(params_df)
            Xi = params_df.loc[:, self.Xi_cols].values.tolist()
            yi = params_df["loss"].values.tolist()
            # if there are previous points, add them before telling;
            # since points from disk are only saved every once in a while, it is probable
            # that they lag behind the ones stored in the backend, so it makes sense to
            # append (and not prepend) the more recent points
            if backend.Xi:
                Xi.extend(backend.Xi)
                yi.extend(backend.yi)
            opt.tell(Xi, yi)
            # add only the hashes of the query, because the hashes of the points
            # stored in the backend have been already added after the local evaluation
            backend.Xi_h.extend(params_df["Xi_h"].to_list())
        else:
            if backend.Xi:  # or just tell prev points
                opt.tell(backend.Xi, backend.yi)

        del backend.Xi[:], backend.yi[:]
        return opt

    def opt_log_trials(
        self,
        opt: Optimizer,
        void_filtered: list,
        jobs: int,
        is_shared: bool,
        trials_state: Namespace,
        epochs: Namespace,
    ):
        """
        Every workers saves trials to disk after `trials_timeout` seconds or `trials_maxout` number of trials;
        Before saving trials are processed (setting initial_point, random_state, etc..);
        Last tested points are saved in the global state (along with the optimizer instance) to be told in the next run;
        """
        # add points of the current dispatch if any
        if len(void_filtered) > 0:
            void = False
        # the last run for save doesn't run other trials
        elif opt.void_loss != VOID_LOSS and not trials_state.exit:
            void = False
        else:
            void = True
        # NOTE: some trials at the beginning won't be published
        # because they are removed by filter_void_losses
        rs = opt.rs
        if not void:
            # this is the counter used by the optimizer internally to track the initial
            # points evaluated so far..
            initial_points = opt._n_initial_points
            # set initial point flag and optimizer random state
            for n, v in enumerate(void_filtered):
                v["is_initial_point"] = initial_points - n > 0
                v["random_state"] = rs
                # hash Xi to allow queries over it since it's a list
                v["Xi_h"] = hash(HyperoptMulti.opt_params_Xi(v))
            backend.trials_list.extend(void_filtered)
            trials_state.num_done += n + 1

        self.maybe_log_trials(trials_state, epochs)

        # save optimizer stat and the last points that will be told on the next run
        backend.Xi, backend.yi = [], []
        for t in void_filtered:
            backend.Xi.append(HyperoptMulti.opt_params_Xi(t))
            backend.yi.append(t["loss"])
            backend.Xi_h.append(t["Xi_h"])
        self.opt_state(s_opt=opt)

    def maybe_log_trials(self, trials_state: Namespace, epochs: Namespace):
        """
        Check if we should save trials to disk, based on time, and number of local trials
        """
        if (
            now() - backend.timer >= self.trials_timeout
            or len(backend.trials_list) >= self.trials_maxout
            or trials_state.exit
        ):
            self.log_trials(trials_state, epochs)
            backend.timer = now()

    def parallel_opt_objective_sig_handler(
        self, t: int, jobs: int, optimizers: Queue, epochs: Namespace, trials_state: Namespace
    ):
        """
        To handle Ctrl-C the worker main function has to be wrapped into a try/catch;
        NOTE: The Manager process also needs to be configured to handle SIGINT (in the backend)
        """
        try:
            return self.parallel_opt_objective(t, jobs, optimizers, epochs, trials_state)
        except KeyboardInterrupt:
            trials_state.exit = True
            return self.parallel_opt_objective(t, jobs, optimizers, epochs, trials_state)

    def parallel_opt_objective(
        self, t: int, jobs: int, optimizers: Queue, epochs: Namespace, trials_state: Namespace
    ):
        """
        An objective run in multi opt mode;
        Shared: optimizers share the results as soon as they are completed;
        Multi: optimizers share results but only following the points asked by the model
        """
        HyperoptOut.log_results_immediate(t, epochs)
        if not backend.timer:
            backend.timer = now()
        opt = HyperoptMulti.opt_state(optimizers)

        is_shared = self.shared
        asked: Dict[Tuple, Any] = {tuple([]): None}
        asked_d: Dict[Tuple, Any] = {}
        # check early if this is the last run
        if trials_state.exit:
            trials_state.tail.extend(backend.trials_list)
            return

        # at startup always fetch previous points from storage,
        # in shared mode periodically check for new points computed by other workers,
        # every once in a while the optimizer global state is gced, so reload points
        opt = self.opt_startup_points(opt, trials_state, is_shared)

        n_told = 0  # told while looping
        tested_Xi = []  # already tested
        tested_yi = []
        untested_Xi = []  # to test
        # if opt.void == -1 the optimizer failed to give a new point (between dispatches), stop
        # if asked == asked_d  the points returned are the same, stop
        # if opt.Xi > sss the optimizer has more points than the estimated search space size, stop
        while opt.void != -1 and asked != asked_d and len(opt.Xi) < self.search_space_size:
            asked_d = asked
            asked = opt.ask(n_points=self.ask_points, strategy=self.lie_strat())
            # The optimizer doesn't return a list when points are asked with None (skopt behaviour)
            if not self.ask_points:
                asked = {tuple(asked): None}
            else:
                asked = {tuple(a): None for a in asked}
            # check if some points have been evaluated by other optimizers
            prev_asked = HyperoptMulti.opt_get_past_points(
                asked, self.Xi_cols, self.trials_file, self.trials_instance, trials_state
            )
            for a in prev_asked:
                # is the loss set?
                if prev_asked[a] is not None:
                    logger.warn("A point was previously asked by another worker..")
                    if a not in tested_Xi:
                        tested_Xi.append(a)
                        tested_yi.append(prev_asked[a])
                else:
                    # going to test it
                    untested_Xi.append(a)
            # not enough points to test?
            if len(untested_Xi) < self.n_points:
                n_tested_Xi = len(tested_Xi)
                # did other workers test some more points that we asked?
                if n_tested_Xi > n_told:
                    # if yes fit a new model with the new points
                    opt.tell(tested_Xi[n_told:], tested_yi[n_told:])
                    told = n_tested_Xi
                else:  # or get new points from a different random state
                    opt = HyperoptMulti.opt_rand(opt)
            else:
                break
        # return early if there is nothing to test
        if len(untested_Xi) < 1:
            opt.void = -1
            backend.opt = opt
            return
        # run the backtest for each point to do (untested_Xi)
        trials = [self.backtest_params(X) for X in untested_Xi]
        # filter losses
        void_filtered = HyperoptMulti.filter_void_losses(trials, opt)

        self.opt_log_trials(opt, void_filtered, jobs, is_shared, trials_state, epochs)

    @staticmethod
    def filter_void_losses(trials: List, opt: Optimizer) -> List:
        """ Remove out of bound losses from the results """
        if opt.void_loss == VOID_LOSS and len(opt.yi) < 1:
            # only exclude results at the beginning when void loss is yet to be set
            void_filtered = list(filter(lambda t: t["loss"] != VOID_LOSS, trials))
        else:
            if opt.void_loss == VOID_LOSS:  # set void loss once
                opt.void_loss = max(opt.yi)
            void_filtered = []
            # default bad losses to set void_loss
            for n, t in enumerate(trials):
                if t["loss"] == VOID_LOSS:
                    trials[n]["loss"] = opt.void_loss
            void_filtered = trials
        return void_filtered
