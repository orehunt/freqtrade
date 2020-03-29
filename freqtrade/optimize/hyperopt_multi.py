import warnings
from typing import Any, Dict, List, Tuple
from abc import abstractmethod

from joblib import Parallel, delayed, wrap_non_picklable_objects
from multiprocessing import Manager
from queue import Queue

# Import IHyperOpt and IHyperOptLoss to allow unpickling classes from these modules
import freqtrade.optimize.hyperopt_backend as backend
from freqtrade.optimize.hyperopt_out import HyperoptOut
from freqtrade.optimize.hyperopt_constants import VOID_LOSS
from freqtrade.optimize.hyperopt_interface import IHyperOpt  # noqa: F401
from freqtrade.optimize.hyperopt_loss_interface import IHyperOptLoss  # noqa: F401

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

    # tracks the number of batches that were completely void
    empty_batches = 0

    @abstractmethod
    def backtest_params(
        self, raw_params: List[Any] = None, iteration=None, params_dict: Dict[str, Any] = None
    ):
        """
        Used Optimize function. Called once per epoch to optimize whatever is configured.
        Keep this function as optimized as possible!
        """

    def setup_multi(self):
        # optimizers
        self.opts: List[Optimizer] = []
        self.opt: Optimizer = None
        self.Xi: Dict = {}
        self.yi: Dict = {}

        backend.manager = Manager()
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
            backend.optimizers = backend.manager.Queue()
            backend.results_batch = backend.manager.Queue()
        else:
            backend.results_list = backend.manager.list([])
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

    def run_multi_backtest_parallel(
        self, parallel: Parallel, tries: int, first_try: int, jobs: int
    ):
        """ launch parallel in multi opt mode, return the evaluated epochs"""
        parallel(
            delayed(wrap_non_picklable_objects(self.parallel_opt_objective))(
                i, backend.optimizers, jobs, backend.results_shared, backend.results_batch
            )
            for i in range(first_try, first_try + tries)
        )

    @staticmethod
    def opt_get_past_points(is_shared: bool, asked: dict, results_shared: Dict) -> Tuple[dict, int]:
        """ fetch shared results between optimizers """
        # a result is (y, counter)
        for a in asked:
            if a in results_shared:
                y, counter = results_shared[a]
                asked[a] = y
                counter -= 1
                if counter < 1:
                    del results_shared[a]
        return asked, len(results_shared)

    @staticmethod
    def opt_state(shared: bool, optimizers: Queue) -> Optimizer:
        """ fetch an optimizer in multi opt mode """
        # get an optimizer instance
        opt = optimizers.get()
        if shared:
            # get a random number before putting it back to avoid
            # replication with other workers and keep reproducibility
            rand = opt.rng.randint(0, VOID_LOSS)
            optimizers.put(opt)
            # switch the seed to get a different point
            opt = HyperoptMulti.opt_rand(opt, rand)
        return opt

    @staticmethod
    def opt_params_Xi(v: dict):
        return list(v["params_dict"].values())

    @staticmethod
    def opt_results(
        opt: Optimizer,
        void_filtered: list,
        jobs: int,
        is_shared: bool,
        results_shared: Dict,
        results_batch: Queue,
        optimizers: Queue,
    ):
        """
        update the board used to skip already computed points,
        set the initial point status
        """
        # add points of the current dispatch if any
        if opt.void_loss != VOID_LOSS or len(void_filtered) > 0:
            void = False
        else:
            void = True
        # send back the updated optimizer only in non shared mode
        if not is_shared:
            opt = HyperoptMulti.opt_clear(opt)
            # is not a replica in shared mode
            optimizers.put(opt)
        # NOTE: some results at the beginning won't be published
        # because they are removed by filter_void_losses
        rs = opt.rs
        if not void:
            # the tuple keys are used to avoid computation of done points by any optimizer
            results_shared.update(
                {
                    tuple(HyperoptMulti.opt_params_Xi(v)): (v["loss"], jobs - 1)
                    for v in void_filtered
                }
            )
            # in multi opt mode (non shared) also track results for each optimizer (using rs as ID)
            # this keys should be cleared after each batch
            Xi, yi = results_shared[rs]
            Xi = Xi + tuple((HyperoptMulti.opt_params_Xi(v)) for v in void_filtered)
            yi = yi + tuple(v["loss"] for v in void_filtered)
            results_shared[rs] = (Xi, yi)
            # this is the counter used by the optimizer internally to track the initial
            # points evaluated so far..
            initial_points = opt._n_initial_points
            # set initial point flag and optimizer random state
            for n, v in enumerate(void_filtered):
                v["is_initial_point"] = initial_points - n > 0
                v["random_state"] = rs
            results_batch.put(void_filtered)

    def parallel_opt_objective(
        self, n: int, optimizers: Queue, jobs: int, results_shared: Dict, results_batch: Queue
    ):
        """
        objective run in multi opt mode, optimizers share the results as soon as they are completed
        """
        HyperoptOut.log_results_immediate(n)
        is_shared = self.shared
        opt = self.opt_state(is_shared, optimizers)
        sss = self.search_space_size
        asked: Dict[Tuple, Any] = {tuple([]): None}
        asked_d: Dict[Tuple, Any] = {}

        # fit a model with the known points, (the optimizer has no points here since
        # it was just fetched from the queue)
        rs = opt.rs
        Xi, yi = self.Xi[rs], self.yi[rs]
        # add the points discovered within this batch
        bXi, byi = results_shared[rs]
        Xi.extend(list(bXi))
        yi.extend(list(byi))
        if Xi:
            opt.tell(Xi, yi)
        told = 0  # told
        Xi_d = []  # done
        yi_d = []
        Xi_t = []  # to do
        # if opt.void == -1 the optimizer failed to give a new point (between dispatches), stop
        # if asked == asked_d  the points returned are the same, stop
        # if opt.Xi > sss the optimizer has more points than the estimated search space size, stop
        while opt.void != -1 and asked != asked_d and len(opt.Xi) < sss:
            asked_d = asked
            asked = opt.ask(n_points=self.ask_points, strategy=self.lie_strat())
            if not self.ask_points:
                asked = {tuple(asked): None}
            else:
                asked = {tuple(a): None for a in asked}
            # check if some points have been evaluated by other optimizers
            p_asked, _ = HyperoptMulti.opt_get_past_points(is_shared, asked, results_shared)
            for a in p_asked:
                if p_asked[a] is not None:
                    print("a point was previously asked by another worker")
                    if a not in Xi_d:
                        Xi_d.append(a)
                        yi_d.append(p_asked[a])
                else:
                    Xi_t.append(a)
            # no points to do?
            if len(Xi_t) < self.n_points:
                len_Xi_d = len(Xi_d)
                # did other workers backtest some points?
                if len_Xi_d > told:
                    # if yes fit a new model with the new points
                    opt.tell(Xi_d[told:], yi_d[told:])
                    told = len_Xi_d
                else:  # or get new points from a different random state
                    opt = HyperoptMulti.opt_rand(opt)
            else:
                break
        # return early if there is nothing to backtest
        if len(Xi_t) < 1:
            if is_shared:
                opt = optimizers.get()
            opt.void = -1
            opt = HyperoptMulti.opt_clear(opt)
            optimizers.put(opt)
            return []
        # run the backtest for each point to do (Xi_t)
        results = [self.backtest_params(a) for a in Xi_t]
        # filter losses
        void_filtered = HyperoptMulti.filter_void_losses(results, opt)

        HyperoptMulti.opt_results(
            opt, void_filtered, jobs, is_shared, results_shared, results_batch, optimizers
        )

    @staticmethod
    def filter_void_losses(vals: List, opt: Optimizer) -> List:
        """ remove out of bound losses from the results """
        if opt.void_loss == VOID_LOSS and len(opt.yi) < 1:
            # only exclude results at the beginning when void loss is yet to be set
            void_filtered = list(filter(lambda v: v["loss"] != VOID_LOSS, vals))
        else:
            if opt.void_loss == VOID_LOSS:  # set void loss once
                opt.void_loss = max(opt.yi)
            void_filtered = []
            # default bad losses to set void_loss
            for k, v in enumerate(vals):
                if v["loss"] == VOID_LOSS:
                    vals[k]["loss"] = opt.void_loss
            void_filtered = vals
        return void_filtered
