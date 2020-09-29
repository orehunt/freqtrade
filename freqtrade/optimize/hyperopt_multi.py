import gc
import logging
import os
import warnings
from decimal import Decimal
from functools import partial

# use math finite check for small loops
from math import isfinite as is_finite
from multiprocessing.managers import SyncManager
from queue import Queue
from time import time as now
from typing import Dict, List, Optional, Tuple, Union

# Import IHyperOpt and IHyperOptLoss to allow unpickling classes from these modules
import freqtrade.optimize.hyperopt_backend as backend
from freqtrade.exceptions import OperationalException
from freqtrade.optimize.hyperopt_backend import Epochs, TrialsState
from freqtrade.optimize.hyperopt_constants import VOID_LOSS
from freqtrade.optimize.hyperopt_out import HyperoptOut
from joblib import Parallel, delayed, hash
from numpy import (
    array,
    asarray,
    inf,
    isfinite,
    isin,
    nan,
    nanmax,
    nanmean,
    nanmin,
    nanstd,
    nanvar,
)
from numpy.core.numeric import flatnonzero
from pandas import DataFrame
from psutil import virtual_memory

# from pyinstrument import Profiler

# profiler = Profiler()


# Suppress scikit-learn FutureWarnings from skopt
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from skopt import Optimizer
# Additional regressors already pluggable into the optimizer
# from sklearn.linear_model import ARDRegression, BayesianRidge
# possibly interesting regressors that need predict method override
# from sklearn.ensemble import HistGradientBoostingRegressor
# from xgboost import XGBoostRegressor


class HyperoptMulti(HyperoptOut):
    """ Run the optimization with multiple optimizers """

    # stop warning against missing results after a while
    void_output_backoff = False
    Xi_names: Tuple = ()
    use_progressbar = True

    def epochs_iterator(self, jobs: int):
        """ Dispatches jobs to parallel indefinitely """
        # at the end dispatch wrap up tasks
        if backend.trials.exit:
            for t in range(jobs):
                if self.use_progressbar:
                    HyperoptOut._print_progress(t, jobs, self.trials_maxout)
                yield t

        # termination window
        w = self.trials_maxout * jobs
        t = len(self.trials)
        yield t
        sri = self.space_reduction_interval
        # give a next trial count indefinitely
        for _ in iter(lambda: 0, 1):
            t += 1
            if t % w < 1:
                if backend.trials.exit or self._maybe_terminate(
                    t, jobs, backend.trials, backend.epochs
                ):
                    break
            if sri and t % sri < 1:
                self.apply_space_reduction(jobs, backend.trials, backend.epochs)
            if self.use_progressbar:
                HyperoptOut._print_progress(t, jobs, self.trials_maxout)
            self.logger.debug("yiedling %s", t)
            yield t

    def setup_multi(self):
        # optimizer
        self.opt = None
        # start the manager
        self.setup_backend()
        # set mode of operations
        self.ask_points = self.config.get("ask_points", 1)
        # if 0 n_points are given, don't use any base estimator (akin to random search)
        # and force single mode as there is no model
        if self.ask_points < 1:
            self.ask_points = (self.ask_points or -1) * -1
            self.opt_base_estimator = lambda: "DUMMY"
            self.opt_acq_func = lambda: "gp_hedge"
            self.mode = "single"
            self.opt_acq_optimizer = "sampling"
        else:
            self.mode = self.config.get("mode", "single")
            if self.mode == "single":
                self.opt_base_estimator = lambda: "GP"
                self.opt_acq_func = lambda: "gp_hedge"
                self.opt_acq_optimizer = "lbfgs"
                # The GaussianProcessRegressor is heavy, which makes it not a good default
                # however longer backtests might make it a better tradeoff
        mode = self.mode
        self.cv = mode == "cv"
        self.multi = mode in ("multi", "shared")
        self.shared = mode == "shared"
        # keep models queue to 1
        self.n_models = 1
        if self.multi:
            self.opt_acq_optimizer = "auto"
            if self.shared:
                self.opt_base_estimator = lambda: "GBRT"
                self.opt_acq_func = self.acq_funcs
            else:
                self.opt_base_estimator = self.estimators
                self.opt_acq_func = lambda: "gp_hedge"
        if self.ask_points < 2:
            # opt_ask_points is what is used in the ask call
            # because when ask_points is None, it doesn't
            # waste time generating new points
            self.opt_ask_points = None
            # when asking only one point, 1 model has to be kept
        else:
            self.opt_ask_points = self.ask_points
        # var used in epochs and batches calculation
        self.opt_points = self.n_jobs * (self.ask_points or 1)
        # lie strategy
        def_lie_strat = "random" if self.shared else "cl_min"
        lie_strat = self.config.get("lie_strat", def_lie_strat)
        if lie_strat == "random":
            self.lie_strat = self.lie_strategy
        else:
            self.lie_strat = lambda: lie_strat
        # split workers to diversify acquisition
        self.n_explorers = self.n_jobs // 2
        self.n_exploiters = self.n_jobs // 2 + self.n_jobs % 2
        self.max_convergence_ratio = self.config.get(
            "hyperopt_max_convergence_ratio", 0.05
        )

    @staticmethod
    def setup_backend():
        """ Setup namespaces shared by workers """
        # start the manager with a noop signal handler to allow graceful termination
        backend.manager = SyncManager()
        backend.manager.start(partial(backend.manager_init, backend=backend))
        backend.epochs = backend.manager.Namespace()
        # when changing epochs counting and saving trials
        backend.epochs.lock = backend.manager.Lock()
        # tracks the balance between explorers (+) and exploiters (-)
        backend.epochs.explo = 0
        # tracks number of duplicate points received by asking
        backend.epochs.convergence = 0
        # in multi mode each optimizer has its own best loss/epoch
        backend.epochs.current_best_loss = backend.manager.dict()
        backend.epochs.current_best_epoch = backend.manager.dict()
        backend.trials = backend.manager.Namespace()
        # prevent reading to the store while saving
        backend.trials.lock = backend.manager.Lock()
        # when a worker finishes a dispatch, updates the done count
        backend.trials.num_done = 0
        # when a worker triggers a save, update the save count
        backend.trials.num_saved = 0
        # terminate early if all tests are being filtered out
        backend.trials.empty_strikes = 0
        # in shared mode the void_loss value is the same across workers
        backend.trials.void_loss = VOID_LOSS
        # at the end collect the remaining trials to save here
        backend.trials.tail_dict = backend.manager.dict()
        backend.trials.tail_list = backend.manager.list()
        # stores the hashes of points currently getting tested
        backend.trials.testing = backend.manager.dict()
        # at the end one last batch is dispatched to save the remaining trials
        backend.trials.exit = False

        return backend

    @staticmethod
    def setup_worker_backend():
        """
        Reset the global variables reference on the backend which is
        used by each worker process; this is not needed in a normal hyperopt run,
        however to run different configurations, within the same loky session
        (same pool of worker), the state has to be cleared at the start.
        """
        global backend
        backend.cls = None
        backend.data = {}
        backend.trials_index = 0
        backend.just_saved = 0
        backend.trials_list = []
        backend.timer = 0
        backend.opt = None
        backend.exploit = 0
        backend.Xi = []
        backend.yi = []
        backend.Xi_h = {}

    def run_setup_backend_parallel(self, parallel: Parallel, jobs: int):
        """ Clear the global state of each worker """
        # run it twice the worker count to work around parallel scheduling multiple
        # tasks to the same worker
        parallel(
            (
                delayed(backend.parallel_sig_handler)(
                    backend, self.setup_worker_backend, None, None
                )
                for _ in range(2 * jobs)
            )
        )

    def run_multi_backtest_parallel(self, parallel: Parallel, jobs: int):
        """ Launch parallel in multi opt mode,
        scheduling the specified number of trials,
        passing the needed objects handled by the manager """
        parallel(
            delayed(backend.parallel_sig_handler)(
                backend,
                self.parallel_opt_objective,
                self.cls_file,
                self.logger,
                t,
                jobs,
                optimizers=backend.optimizers,
                epochs=backend.epochs,
                trials_state=backend.trials,
            )
            for t in self.epochs_iterator(jobs)
        )

    @staticmethod
    def opt_get_past_points(asked: dict) -> dict:
        """ fetch shared results between optimizers """
        # make a list of hashes since on storage points are queried by it
        if backend.Xi_h:
            for h in asked:
                if h in backend.Xi_h:
                    asked[h][1] = backend.Xi_h[h]
        return asked

    @staticmethod
    def opt_state(
        optimizers: Optional[Queue] = None,
        epochs: Epochs = None,
        s_opt: Optimizer = None,
    ) -> Optimizer:
        """ Return an optimizer in multi opt mode """
        # get an optimizer instance
        if s_opt:
            backend.opt = s_opt
            return s_opt
        # check if a space reduction has been performed
        # and delete current parameters if so
        opt_available = hasattr(backend, "opt") and backend.opt
        pid = os.getpid()
        space_reduction = (
            epochs and pid in epochs.space_reduction and not backend.just_reduced
        )
        if opt_available and not space_reduction:
            opt = backend.opt
            # if the Xi_h dict got gced re-fetch all the points
            if not backend.Xi_h and backend.trials_index:
                backend.trials_index = 0
        else:
            # at space reduction fetch an optimizer from the pid map
            if pid in epochs.space_reduction:
                opt = epochs.pinned_optimizers[pid]
                HyperoptMulti.reset_opt_state(opt)
                epochs.pinned_optimizers[pid] = opt
            # at worker startup  fetch an optimizer from the queue
            elif optimizers is not None and optimizers.qsize() > 0:
                opt = optimizers.get(timeout=1)
                HyperoptMulti.reset_opt_state(opt)
                # store it back again to restore after global state is gced
                optimizers.put(opt)
                epochs.pinned_optimizers[pid] = opt
            else:
                raise OperationalException(
                    "Global state was reclaimed and no optimizer instance was "
                    "available for recovery"
                )
        # if space reduction was performed decrease jobs counter
        if space_reduction:
            del epochs.space_reduction[pid]
            backend.just_reduced = True
        elif backend.just_reduced:
            backend.just_reduced = False
        return opt

    @staticmethod
    def opt_params_Xi(v: dict):
        return list(v["params_dict"].values())

    @staticmethod
    def dict_values(dict_list: List[dict]):
        return [list(d.values()) for d in dict_list]

    def opt_startup_points(
        self, opt: Optimizer, trials_state: TrialsState, is_shared: bool
    ) -> Optimizer:
        """
        Check for new points saved by other workers (or by previous runs), once every the workers
        saves a batch
        """
        # fit a model with the known points, either from a previous run, or read
        # from database only at the start when the global points references of each worker are empty
        params_df: DataFrame = []
        if backend.just_saved or not len(opt.Xi):
            locked = False
            # fetch all points not already told in shared mode
            # ignoring random state
            try:
                # Only wait if the optimizer has no points at all (startup)
                locked = trials_state.lock.acquire(len(opt.Xi) < 1)
                if locked:
                    params_df = self._from_group(
                        fields=["loss", "Xi_h", "params_dict", "random_state"],
                        indexer=(slice(backend.trials_index, None)),
                    )
                    if len(params_df) and backend.Xi_h:
                        params_df.drop(
                            flatnonzero(
                                isin(
                                    params_df["Xi_h"].values, list(backend.Xi_h.keys())
                                )
                            ),
                            axis=0,
                            inplace=True,
                        )
                    trials_state.lock.release()
            except (KeyError, FileNotFoundError, IOError, OSError,) as e:
                # only happens when df is empty and empty df is not saved
                # on disk by pytables or is being written
                if locked:
                    trials_state.lock.release()
                self.logger.debug("Couldn't read trials from disk %s", e)
                raise e
        if len(params_df) > 0:
            backend.trials_index += len(params_df)
            # only in shared mode, we tell all the points
            # (these lists are very small)
            if is_shared:
                Xi = self.dict_values(params_df["params_dict"].values.tolist())
                yi = params_df["loss"].values.tolist()
            elif not opt.Xi:
                # while in multi opt, still have to add startup points
                prev_opt_params = params_df.loc[
                    params_df["random_state"].values == opt.rs
                ]
                Xi = self.dict_values(
                    prev_opt_params.loc[:, "params_dict"].values.tolist()
                )
                yi = prev_opt_params["loss"].values.tolist()
            else:
                Xi, yi = [], []
            # if there are previous points, add them before telling;
            # since points from disk are only saved every once in a while, it is probable
            # that they lag behind the ones stored in the backend, so it makes sense to
            # append (and not prepend) the more recent points
            if Xi:
                backend.Xi.extend(Xi)
                backend.yi.extend(yi)
            # add only the hashes of the query, because the hashes of the points
            # stored in the backend have been already added after the local evaluation
            backend.Xi_h.update(
                dict(zip(params_df["Xi_h"].values, params_df["loss"].values))
            )
        if backend.Xi:  # or just tell prev points
            try:
                self.logger.debug(
                    "adjourning the optimizer with %s new points...", len(backend.Xi)
                )
                opt.tell(backend.Xi, backend.yi)
            # this can happen if space reduction has been performed and the
            # points of the previous tests are outside the new search space
            except ValueError as e:
                self.logger.info(e)
        else:
            # get a new point by copy if didn't tell new ones
            self.logger.debug(
                "no new points were found to tell, rolling a new optimizer.."
            )
            opt = HyperoptMulti.opt_rand(opt)

        del backend.Xi[:], backend.yi[:]
        return opt

    def opt_log_trials(
        self,
        opt: Optimizer,
        void_filtered: list,
        t: int,
        jobs: int,
        is_shared: bool,
        trials_state: TrialsState,
        epochs: Epochs,
    ):
        """
        Every workers saves trials to disk after `trials_timeout` seconds
        or `trials_maxout` number of trials; Before saving trials are processed
        (setting initial_point, random_state, etc..); Last tested points are saved
        in the global state (along with the optimizer instance) to be told in the next run;
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
            for n, v in enumerate(void_filtered, 1):
                v["is_initial_point"] = initial_points - n > 0
                v["random_state"] = rs
                # hash Xi to allow queries over it since it's a list
                v["Xi_h"] = hash(HyperoptMulti.opt_params_Xi(v))
            backend.trials_list.extend(void_filtered)
            trials_state.num_done += n

        # save optimizer stat and the last points that will be told on the next run
        backend.Xi, backend.yi = list(), list()
        for v in void_filtered:
            backend.Xi.append(HyperoptMulti.opt_params_Xi(v))
            backend.yi.append(v["loss"])
            backend.Xi_h[v["Xi_h"]] = v["loss"]

        self.maybe_log_trials(trials_state, epochs, rs=None if self.shared else opt.rs)
        # maybe_log_trials updated the just_saved attr
        if backend.just_saved:
            for h in backend.tested_h:
                try:
                    del trials_state.testing[h]
                # it's still possible for other workers to have deleted duplicate points
                # since no locking is applied on points testing
                except KeyError:
                    pass
            del backend.tested_h[:]
            # after saving epochs also update optimizer acquisition
            # since there will be new loss scores
            opt = self.opt_adjust_acq(opt, jobs, epochs, trials_state, is_shared)
            # update the void loss to a worse one
            if opt.yi:
                opt.void_loss = nanmax(opt.yi)
        self.opt_state(s_opt=opt)

    def maybe_log_trials(
        self, trials_state: TrialsState, epochs: Epochs, rs: Union[None, int]
    ):
        """
        Check if we should save trials to disk, based on time, and number of local trials
        """
        if backend.trials_list:
            trials_to_save = len(backend.trials_list)
            if (
                now() - backend.timer >= self.trials_timeout
                or trials_to_save >= self.trials_maxout
                or trials_state.exit
                or (backend.just_reduced and trials_to_save)
            ):
                backend.just_saved = self.log_trials(trials_state, epochs, rs=rs)
                if backend.just_saved:
                    # reset timer
                    backend.timer = now()
                    # decrement the count of buffered trials by the number of saved trials
                    trials_state.num_done -= backend.just_saved

        else:
            backend.just_saved = 0

    @staticmethod
    def parallel_opt_objective(
        t: int, jobs: int, optimizers: Queue, epochs: Epochs, trials_state: TrialsState,
    ):
        """
        An objective run in multi opt mode;
        Shared: optimizers share the results as soon as they are completed;
        Multi: optimizers share results but only following the points asked by the model
        """
        cls = backend.cls

        if not backend.timer:
            backend.timer = now()
        opt = HyperoptMulti.opt_state(optimizers, epochs=epochs)
        # enable gc after referencing optimizer
        gc.enable()

        is_shared = cls.shared
        # check early if this is the last run
        if trials_state.exit:
            if len(backend.trials_list):
                if is_shared:
                    trials_state.tail_list.extend(backend.trials_list)
                else:
                    trials_state.tail_dict[opt.rs] = backend.trials_list
                del backend.trials_list[:]
            return

        # at startup always fetch previous points from storage,
        # in shared mode periodically check for new points computed by other workers,
        # every once in a while the optimizer global state is gced, so reload points
        opt = cls.opt_startup_points(opt, trials_state, is_shared)

        untested_Xi = cls.opt_fetch_points(opt, epochs, trials_state)

        # return early if there is nothing to test
        if len(untested_Xi) < 1:
            opt.void = -1
            cls.opt_state(s_opt=opt)
            # Terminate after enough workers didn't receive a point to test
            # only when not in the process of a space reduction
            if len(epochs.space_reduction) < 1:
                trials_state.empty_strikes += 1
            if trials_state.empty_strikes >= cls.trials_max_empty:
                trials_state.exit = True
                # NOTE: this warning is not formatted..
                self.logger.warn("Reached empty strikes.")
            return
        # run the backtest for each point to do (untested_Xi)
        trials = [
            cls.backtest_params(X, rs=opt.rs if not is_shared else None)
            for X in untested_Xi
        ]
        # filter losses
        void_filtered = HyperoptMulti.filter_void_losses(
            trials, opt, trials_state, is_shared
        )

        cls.opt_log_trials(opt, void_filtered, t, jobs, is_shared, trials_state, epochs)
        # disable gc at the end to prevent disposal of global vars

        gc.disable()

    def opt_fetch_points(
        self, opt: Optimizer, epochs: Epochs, trials_state: TrialsState
    ) -> List:
        asked: Dict[str, List] = {}
        asked_d: Dict[str, List] = {}
        n_told = 0  # told while looping
        tested_Xi = []  # already tested
        tested_yi = []
        untested_Xi = []  # to test
        # if opt.void == -1 the optimizer failed to give a new point (between dispatches), stop
        # if opt.Xi > sss the optimizer has more points than the estimated search space size, stop
        while opt.void != -1 and len(opt.Xi) < self.search_space_size:
            self.logger.debug("asking the oracle for points..")
            asked = opt.ask(n_points=self.opt_ask_points, strategy=self.lie_strat())
            # the optimizer doesn't return a list when points are asked with None (skopt behaviour)
            if not self.opt_ask_points:
                asked = {hash(asked): [asked, None]}
            else:
                asked = {hash(a): [a, None] for a in asked}
            # check if some points have been evaluated by other optimizers
            self.logger.debug("checking if points returned by optimizer are cached...")
            prev_asked = HyperoptMulti.opt_get_past_points(asked)
            for h in prev_asked:
                # is the loss set?
                past_Xi = prev_asked[h][0]
                past_yi = prev_asked[h][1]
                if past_yi is not None:
                    self.logger.debug(
                        "A point was previously asked by another worker.."
                    )
                    epochs.convergence += 1
                    if past_Xi not in tested_Xi:
                        tested_Xi.append(past_Xi)
                        tested_yi.append(past_yi)
                else:
                    # going to test it if it is not being tested
                    # by another optimizer
                    if h not in trials_state.testing:
                        trials_state.testing[h] = True
                        backend.tested_h.append(h)
                        untested_Xi.append(past_Xi)
            # not enough points to test?
            if len(untested_Xi) < self.ask_points:
                n_tested_Xi = len(tested_Xi)
                # did other workers test some more points that we asked?
                if n_tested_Xi > n_told:
                    # if yes fit a new model with the new points
                    opt.tell(tested_Xi[n_told:], tested_yi[n_told:])
                    n_told = n_tested_Xi
                elif (
                    asked != asked_d
                ):  # or get new points from a different random state
                    opt = HyperoptMulti.opt_rand(opt)
                    # getting a point by copy is the last try before
                    # terminating because of convergence
                    asked_d = asked
                else:
                    break
            else:
                break
        return untested_Xi

    @staticmethod
    def opt_acq_window(opt: Optimizer, is_shared: bool, jobs: int, epochs: Epochs):
        """ Determine the ranges of yi to compare for acquisition adjustments """
        last_period = (
            epochs.epochs_since_last_best[-1]
            if is_shared
            else backend.epochs_since_last_best[-1]
        )
        n_tail = backend.just_saved * jobs
        if is_shared:
            loss = array(list(backend.Xi_h.values()))
        else:
            loss = array(opt.yi)
            last_period = last_period // jobs
            n_tail = backend.just_saved * (len(backend.Xi_h) // len(opt.Xi) or 2)
        # calculate base values, dependent on loss score
        if not len(loss[-last_period:]):
            # take the full length at the beginning
            loss_last = loss[-n_tail:]
            loss_tail = loss[-backend.just_saved :]
        else:
            # if exploitation phase was triggered before, if so,
            # track from the beginning of the phase
            # NOTE: it is an absolute index
            if backend.exploit:
                loss_last = loss[backend.exploit :]
            else:
                loss_last = loss[-last_period:]
            loss_tail = loss[-n_tail:]
        return loss, n_tail, last_period, loss_tail, loss_last

    @staticmethod
    def opt_adjust_acq(
        opt: Optimizer,
        jobs: int,
        epochs: Epochs,
        trials_state: TrialsState,
        is_shared: bool,
    ) -> Optimizer:
        """ Tune exploration/exploitaion balance of optimizers """

        rs = None if is_shared else opt.rs
        # can only tune if we know the loss scores, and past initial points
        # in multi mode only use optimizer initial points count
        if (is_shared and len(backend.Xi_h) >= opt.n_initial_points_) or (
            not is_shared and opt._n_initial_points < 0
        ):
            # how many epochs since no best
            xi, kappa = None, None
            (
                loss,
                n_tail,
                last_period,
                loss_tail,
                loss_last,
            ) = HyperoptMulti.opt_acq_window(opt, is_shared, jobs, epochs)

            # increase exploitation around
            # any obs that exhibits a better (lower) score
            if nanmean(loss_tail) < nanmean(loss_last):
                # decrement for exploitation if was previous exploring
                if not backend.exploit:
                    epochs.explo -= 1
                    if is_shared:
                        # update the index of the exploitation session
                        backend.exploit = len(loss) - n_tail
                # in non shared mode update the index at every iteration
                # since only the worker can influence the score which means
                # that only the very last epochs are of interest for the acq
                if not is_shared:
                    backend.exploit = len(loss) - n_tail
                # default xi is 0.01, we want some 10e even lower values
                if opt.acq_func in ("PI", "gp_hedge"):
                    _, digits, exponent = Decimal(nanvar(loss_tail)).as_tuple()
                    if len(digits) and digits[0]:
                        xi = (
                            nanstd(loss_tail) * 10 ** -abs(len(digits) + exponent - 1)
                            or 0.01
                        )
                    else:
                        xi = 0.01
                # with EI that supports negatives we can just use negative std
                elif opt.acq_func == "EI":
                    xi = -nanstd(loss_tail)
                # LCB uses kappa instead of xi, and is based on variance
                if opt.acq_func in ("LCB", "gp_hedge"):
                    kappa = nanvar(
                        [epochs.current_best_loss[rs], nanmin(loss_tail)]
                    ) or nanvar(
                        # use mean when we just found a new best as var would be 0
                        [epochs.current_best_loss[rs], nanmean(loss_tail)]
                    )
            else:  # or use average values from the last period
                # increment for exploration if was previous exploiting
                if backend.exploit:
                    epochs.explo += 1
                # reset exploitation index
                backend.exploit = 0
                # adjust to tail position as we are in general more exploitative
                # at the beginning and more explorative towards the end
                tail_position = (
                    epochs.current_best_epoch[rs] + last_period
                ) / epochs.max_epoch
                if opt.acq_func in ("PI", "EI", "gp_hedge"):
                    xi = nanstd(loss_last) * tail_position
                if opt.acq_func in ("LCB", "gp_hedge"):
                    kappa = nanvar(loss_last) * tail_position
            opt.acq_func_kwargs = {"xi": xi, "kappa": kappa}
            # don't need to update_next point since model is fit
            # at the beginning of the next test
        return opt

    @staticmethod
    def reset_opt_state(opt: Optimizer):
        # make a copy of the empty opt to resume from when the global
        # is gced
        backend.opt = opt
        # reset index of read trials since resuming from 0
        backend.trials_index = 0
        backend.Xi_h = {}

    @staticmethod
    def filter_void_losses(
        trials: List, opt: Optimizer, trials_state: TrialsState, is_shared=False
    ) -> List:
        """ Remove out of bound losses from the results """
        if opt.void_loss == VOID_LOSS and (
            (len(backend.Xi_h) < 1 and is_shared) or (len(opt.Xi) < 1 and not is_shared)
        ):
            # only exclude results at the beginning when void loss is yet to be set
            void_filtered = list(
                filter(
                    lambda t: is_finite(t["loss"]) and t["loss"] is not VOID_LOSS,
                    trials,
                )
            )
            # assert isfinite([t["loss"] for t in void_filtered]).all()
        else:
            if opt.void_loss == VOID_LOSS:  # set void loss once
                if is_shared:
                    if trials_state.void_loss == VOID_LOSS:
                        trials_state.void_loss = nanmax(list(backend.Xi_h.values()))
                    opt.void_loss = trials_state.void_loss
                else:
                    opt.void_loss = nanmax(opt.yi)
            void_filtered = []
            # default bad losses to set void_loss
            for n, t in enumerate(trials):
                if t["loss"] == VOID_LOSS or not isfinite(t["loss"]):
                    trials[n]["loss"] = opt.void_loss
            void_filtered = trials
        return void_filtered

    @staticmethod
    def available_bytes() -> float:
        """ return host available memory in bytes """
        return virtual_memory().available / 1000

    @staticmethod
    def calc_n_points(n_dimensions: int, n_jobs: int, ask_points) -> int:
        """ Calculate the number of points the optimizer samples, based on available host memory """
        available_mem = HyperoptMulti.available_bytes()
        # get size of one parameter
        return int(available_mem / (4 * n_dimensions * n_jobs * ask_points))
