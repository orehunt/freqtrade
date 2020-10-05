import gc
import logging
import os
import atexit
from functools import partial

# use math finite check for small loops
from math import isfinite as is_finite
from multiprocessing.managers import SyncManager

from queue import Queue
from time import time as now
from typing import Dict, Iterable, List, Optional, Tuple, Union, Callable

import numpy as np
from joblib import Parallel, delayed, hash
from numpy import (
    array,
    asarray,
    fromiter,
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

# Import IHyperOpt and IHyperOptLoss to allow unpickling classes from these modules
import freqtrade.optimize.hyperopt_backend as backend
from freqtrade.exceptions import OperationalException
from freqtrade.optimize.hyperopt_backend import Epochs, TrialsState
from freqtrade.optimize.hyperopt_out import HyperoptOut
from freqtrade.optimize.optimizer import VOID_LOSS, IOptimizer


logger = logging.getLogger("freqtrade.optimize.hyperopt")
logger.name += f".{os.getpid()}"
# from pyinstrument import Profiler

# profiler = Profiler()


class HyperoptMulti(HyperoptOut):
    """ Run the optimization with multiple optimizers """

    # stop warning against missing results after a while
    void_output_backoff = False
    Xi_names: Tuple = ()
    use_progressbar = True

    def epochs_iterator(self, jobs: int):
        """ Dispatches jobs to parallel indefinitely """
        # iterators need their own try/catch
        try:
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
                logger.debug("yiedling %s", t)
                yield t
        except ConnectionError as e:
            logger.error(f"Iteration ended abruptly {e}")

    def _setup_parallel(self):
        # start the manager
        self.setup_backend()

        # set mode of operations
        mode = self.mode = self.config.get("hyperopt_mode")
        self.cv = mode == "cv"
        self.async_sched = mode in ("multi", "shared")
        self.shared = mode == "shared"
        self.multi = mode == "multi"

        self.max_convergence_ratio = self.config.get(
            "hyperopt_max_convergence_ratio", 0.05
        )

    @staticmethod
    def setup_backend():
        """ Setup namespaces shared by workers """
        # start the manager with a noop signal handler to allow graceful termination
        backend.manager = SyncManager()
        backend.manager.start(partial(backend.manager_init, backend=backend))
        backend.optimizers = backend.manager.Queue()
        backend.epochs = backend.manager.Namespace()
        # when changing epochs counting and saving trials
        backend.epochs.lock = backend.manager.Lock()
        # tracks the balance between explorers (+) and exploiters (-)
        backend.epochs.explo = 0
        # tracks number of duplicate points received by asking
        backend.epochs.convergence = 0
        backend.epochs.average = np.nan
        backend.epochs.improvement = np.nan
        # tracks the (avg) time it took to fetch the last point from the optimizer
        backend.epochs.avg_wait_time = 0
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
        # store the most recent results for each optimizer/worker
        backend.trials.last_results = backend.manager.dict()

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
        backend.min_date = backend.max_date = None
        backend.trials_index = 0
        backend.just_saved = 0
        backend.just_reduced = False
        backend.trials_list = []
        backend.timer = 0
        backend.opt = None
        backend.exploit = 0
        backend.Xi = []
        backend.yi = []
        backend.Xi_h = {}
        backend.params_Xi = []
        backend.tested_h = []

    def run_setup_backend_parallel(self, parallel: Parallel, jobs: int):
        """ Clear the global state of each worker """
        # run it twice the worker count to work around parallel scheduling multiple
        # tasks to the same worker
        parallel(
            (
                delayed(backend.parallel_sig_handler)(
                    self.setup_worker_backend, None, None
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
                self.parallel_opt_objective,
                self.cls_file,
                logger,
                t,
                jobs,
                optimizers=backend.optimizers,
                epochs=backend.epochs,
                trials_state=backend.trials,
            )
            for t in self.epochs_iterator(jobs)
        )

    @staticmethod
    def opt_get_cached_points(asked: dict) -> dict:
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
        s_opt: IOptimizer = None,
    ) -> IOptimizer:
        """ Return an optimizer in multi opt mode """
        # get an optimizer instance
        if s_opt:
            backend.opt = s_opt
            return s_opt
        # check if a space reduction has been performed
        # and delete current parameters if so
        opt_available = hasattr(backend, "opt") and isinstance(backend.opt, IOptimizer)
        pid = os.getpid()
        space_reduction = (
            epochs and pid in epochs.space_reduction and not backend.just_reduced
        )
        if opt_available and not space_reduction:
            opt = backend.opt
            logger.debug("saved opt has %s points", len(opt.Xi))
            # if the Xi_h dict got gced re-fetch all the points
            if not backend.Xi_h and backend.trials_index:
                backend.trials_index = 0
        else:
            # at space reduction fetch an optimizer from the pid map
            if pid in epochs.space_reduction:
                opt = epochs.pinned_optimizers[pid]
                logger.debug(
                    "retrieving optimizer %s from pinned dict with pid %s", opt.rs, pid
                )
                HyperoptMulti.reset_opt_state(opt)
                epochs.pinned_optimizers[pid] = opt
            # at worker startup  fetch an optimizer from the queue
            elif optimizers is not None and optimizers.qsize() > 0:
                opt = optimizers.get(timeout=1)
                logger.debug("retrieving optimizer %s from queue", opt.rs)
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
            logger.debug("unsubscribing %s from space reduction", pid)
            del epochs.space_reduction[pid]
            backend.just_reduced = True
        elif backend.just_reduced:
            backend.just_reduced = False
        return opt

    @staticmethod
    def dict_values(dict_list: Iterable, iterable=True):
        if iterable:
            return (d.values() for d in dict_list)
        else:
            return [list(d.values()) for d in dict_list]

    @staticmethod
    def zip_points(df):
        return zip(
            HyperoptMulti.dict_values(df["params_dict"], iterable=False),
            df["params_meta"]
        ), df["loss"].values

    def opt_startup_points(
        self, opt: IOptimizer, trials_state: TrialsState, is_shared: bool
    ) -> IOptimizer:
        """
        Check for new points saved by other workers (or by previous runs), once every the workers
        saves a batch
        """
        # fit a model with the known points, either from a previous run, or read
        # from database only at the start when the global points references of each worker are empty
        params_df: Union[List, DataFrame] = []
        if backend.just_saved or not len(opt.Xi):
            locked = False
            # fetch all points not already told in shared mode
            # ignoring random state
            try:
                # Only wait if the optimizer has no points at all (startup)
                locked = trials_state.lock.acquire(len(opt.Xi) < 1)
                if locked:
                    params_df = self._from_group(
                        fields=[
                            "loss",
                            "Xi_h",
                            "params_dict",
                            "params_meta",
                            "random_state",
                        ],
                        indexer=(slice(backend.trials_index, None)),
                    )
                    if len(params_df) and backend.Xi_h:
                        params_df.drop(
                            flatnonzero(
                                isin(
                                    params_df["Xi_h"].values, list(backend.Xi_h)
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
                logger.debug("Couldn't read trials from disk %s", e)
                raise e
        if isinstance(params_df, DataFrame) and len(params_df) > 0:
            backend.trials_index += len(params_df)
            # only in shared mode, we tell all the points
            # (these lists are very small)
            if is_shared:
                Xi, yi = self.zip_points(params_df)
            elif not opt.Xi:
                # while in multi opt, still have to add startup points
                prev_opt_params = params_df.loc[
                    params_df["random_state"].values == opt.rs
                ]
                Xi, yi = self.zip_points(prev_opt_params)
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
            backend.Xi_h.update(zip(params_df["Xi_h"].values, params_df["loss"].values))
        if backend.Xi:  # or just tell prev points
            try:
                logger.debug(
                    "adjourning the optimizer with %s new points...", len(backend.Xi)
                )
                opt.tell(backend.Xi, backend.yi)
            # this can (at least) happen if space reduction has been performed and the
            # points of the previous tests are outside the new search space
            except ValueError as e:
                logger.info(e)
        else:
            # get a new point by copy if didn't tell new ones
            logger.debug("no new points were found to tell, rolling a new optimizer..")
            opt = opt.copy(new_seed=True)

        del backend.Xi[:], backend.yi[:]
        return opt

    @staticmethod
    def flush_remaining_trials(trials_state: TrialsState, is_shared: bool, rs: Union[int, None]):
        rt =len(backend.trials_list)
        logger.debug("flushing remaining trials %s", rt)
        if rt:
            # shared/single
            if is_shared or rs is None:
                trials_state.tail_list.extend(backend.trials_list)
            # multi
            else:
                trials_state.tail_dict[rs] = backend.trials_list
            del backend.trials_list[:]

    def opt_log_trials(
        self,
        opt: IOptimizer,
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
            # set initial point flag and optimizer random state
            n_opt_xi = len(opt.Xi)
            for n, v in enumerate(void_filtered, 1):
                v["is_initial_point"] = n_opt_xi < opt.n_rand
                v["random_state"] = rs
                # hash Xi to allow queries over it since it's a list
                v["Xi_h"] = hash(self.params_Xi(v))
            backend.trials_list.extend(void_filtered)
            trials_state.num_done += n

        # the latest points that will be told on the next run
        last_results = []
        backend.Xi, backend.yi = list(), list()
        for v in void_filtered:
            Xi, yi = (self.params_Xi(v), v["params_meta"]), v["loss"]
            last_results.append((Xi, yi))
            backend.Xi_h[v["Xi_h"]] = v["loss"]
        trials_state.last_results[opt.rs] = last_results

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
            if self.adjust_acquisition:
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
        trials_to_save = len(backend.trials_list)
        if trials_to_save:
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
            logger.debug(
                "skpping save of %s trials because no condition was satisfied",
                trials_to_save,
            )
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
        cls: HyperoptMulti = backend.cls

        if not backend.timer:
            backend.timer = now()
        opt = HyperoptMulti.opt_state(optimizers, epochs=epochs)
        # enable gc after referencing optimizer
        gc.enable()

        is_shared = cls.shared
        if not backend.flush_registered:
            atexit.register(cls.flush_remaining_trials, trials_state, is_shared, opt.rs)
            backend.flush_registered = True
        # check early if this is the last run
        if trials_state.exit:
            cls.flush_remaining_trials(trials_state, is_shared, opt.rs)
            return

        last_results = trials_state.last_results.get(opt.rs, [])
        if len(last_results):
            for r in last_results:
                backend.Xi.append(r[0])
                backend.yi.append(r[1])
            # delete from the managed list, the whole list, since copy is not a reference
            del last_results, trials_state.last_results[opt.rs]
            assert opt.rs not in trials_state.last_results

        # at startup always fetch previous points from storage,
        # in shared mode periodically check for new points computed by other workers,
        # every once in a while the optimizer global state is gced, so reload points
        opt = cls.opt_startup_points(opt, trials_state, is_shared)

        untested_Xi = cls.opt_fetch_points(opt, epochs, trials_state)

        # return early if there is nothing to test
        if len(untested_Xi) < 1:
            logger.debug("updating voidness because no untested points were received")
            opt.void = -1
            cls.opt_state(s_opt=opt)
            # Terminate after enough workers didn't receive a point to test
            # only when not in the process of a space reduction
            if len(epochs.space_reduction) < 1:
                trials_state.empty_strikes += 1
            if trials_state.empty_strikes >= cls.trials_max_empty:
                trials_state.exit = True
                # NOTE: this warning is not formatted..
                cls.logger.warn("Reached empty strikes.")
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
        self, opt: IOptimizer, epochs: Epochs, trials_state: TrialsState
    ) -> List:
        asked: Dict[Union[None, str], List] = {}
        asked_d: Dict[Union[None, str], List] = {}
        n_told = 0  # told while looping
        tested_Xi = []  # already tested
        tested_yi = []
        untested_Xi = []  # to test
        # if opt.void == -1 the optimizer failed to give a new point (between dispatches), stop
        # if opt.Xi > sss the optimizer has more points than the estimated search space size, stop
        logger.debug("starting loop for asking points, voidness is %s", opt.void)
        while opt.void != -1 and np.log1p(len(opt.Xi)) < self.search_space_size:
            logger.debug("asking the oracle for points...")
            wait_start = now()
            # key: hash values: (params, loss, meta)
            asked = {hash(p[0]): [p[0], None, p[1]] for p in opt.ask(self.ask_points)}
            # at start default avg to now
            wait_time = now() - wait_start
            epochs.avg_wait_time = ((epochs.avg_wait_time or wait_time) + wait_time) / 2
            logger.debug(
                "checking if the %s points returned by the optimizer are cached...",
                len(asked),
            )
            # points not previously evald will have loss still set to None
            c_asked = HyperoptMulti.opt_get_cached_points(asked)
            for h in c_asked:
                # is the loss set?
                p_Xi = (c_asked[h][0], c_asked[h][2])
                p_yi = c_asked[h][1]
                if p_yi is not None:
                    logger.debug("A point was previously asked by another worker..")
                    epochs.convergence += 1
                    if p_Xi not in tested_Xi:
                        tested_Xi.append(p_Xi)
                        tested_yi.append(p_yi)
                else:
                    # going to test it if it is not being tested
                    # by another optimizer
                    if h not in trials_state.testing:
                        logger.debug("adding new point %s to the untested list", h)
                        trials_state.testing[h] = True
                        backend.tested_h.append(h)
                        untested_Xi.append(p_Xi)
                    else:
                        logger.debug(
                            "the point %s is being tested by another worker", h
                        )
            # not enough points to test?
            logger.debug(
                "remaining untested: %s , to ask: %s", len(untested_Xi), self.ask_points
            )
            # in case the loop failed to retrieve enough points
            # try to update the optimizer from new points from other workers
            if len(untested_Xi) < self.ask_points:
                n_tested_Xi = len(tested_Xi)
                # did other workers test some more points that we asked?
                if n_tested_Xi > n_told:
                    # if yes add new points to the optimizer
                    logger.debug("updating optimizer with new tested points")
                    opt.tell(tested_Xi[n_told:], tested_yi[n_told:])
                    n_told = n_tested_Xi
                elif (
                    asked != asked_d
                ):  # or get new points from a different random state
                    logger.debug(
                        "rolling a new optimizer because no new points were found"
                    )
                    opt = opt.copy(new_seed=True)
                    # getting a point by copy is the last try before
                    # terminating (because of possibly convergence)
                    asked_d = asked
                else:
                    break
            else:
                break
        return untested_Xi

    @staticmethod
    def opt_acq_window(
        opt: IOptimizer,
        is_shared: bool,
        jobs: int,
        epochs: Epochs,
        trials_state: TrialsState,
    ):
        """ Determine the ranges of yi to compare for acquisition adjustments """
        if is_shared:
            last_period = (
                trials_state.num_saved
                + trials_state.num_done
                + len(backend.trials_list)
                - epochs.current_best_epoch[None]
            )
        else:
            last_period = (
                len(opt.Xi)
                + len(backend.trials_list)
                - epochs.current_best_epoch[opt.rs]
            )
        n_tail = backend.just_saved * jobs
        if is_shared:
            loss = fromiter(backend.Xi_h.values(), dtype=float)
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
        opt: IOptimizer,
        jobs: int,
        epochs: Epochs,
        trials_state: TrialsState,
        is_shared: bool,
    ) -> IOptimizer:
        """ Tune exploration/exploitaion balance of optimizers """

        if not opt.can_tune:
            return opt
        rs = None if is_shared else opt.rs
        # can only tune if we know the loss scores, and past initial points
        # in multi mode only use optimizer initial points count
        if (is_shared and len(backend.Xi_h) >= opt.n_rand) or (
            not is_shared and len(opt.Xi) > opt.n_rand
        ):
            (
                loss,
                n_tail,
                last_period,
                loss_tail,
                loss_last,
            ) = HyperoptMulti.opt_acq_window(opt, is_shared, jobs, epochs, trials_state)

            # increase exploitation around
            # any obs that exhibits a better (lower) score
            last_mean = nanmean(loss_last)
            epochs.average = last_mean
            if nanmean(loss_tail) < last_mean:
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
                opt.exploit(loss_tail, epochs.current_best_loss[rs])
            else:  # or use average values from the last period
                # increment for exploration if was previous exploiting
                if backend.exploit:
                    epochs.explo += 1
                # reset exploitation index
                backend.exploit = 0
                # adjust to tail position as we are in general more exploitative
                # at the beginning and more explorative towards the end
                current_best = epochs.current_best_epoch[rs]
                tail_position = (current_best + last_period) / epochs.max_epoch
                opt.explore(
                    loss_tail,
                    current_best,
                    loss_last=loss_last,
                    tail_position=tail_position,
                )
            # don't need to update next point since model should be fit
            # at the beginning of the next test
        return opt

    @staticmethod
    def reset_opt_state(opt: IOptimizer):
        # make a copy of the empty opt to resume from when the global
        # is gced
        backend.opt = opt
        # reset index of read trials since resuming from 0
        backend.trials_index = 0
        backend.Xi_h = {}
        backend.exploit = 0

    @staticmethod
    def filter_void_losses(
        trials: List, opt: IOptimizer, trials_state: TrialsState, is_shared=False
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
