import warnings
from decimal import Decimal
from itertools import cycle
from typing import Iterable, List, Optional

from numpy import nanmean, nanmin, nanstd, nanvar
from psutil import virtual_memory

from freqtrade.exceptions import OperationalException

from ..optimizer import CAT, RANGE, IOptimizer


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from skopt import Optimizer
    from skopt.space import Categorical, Dimension, Integer, Real

# supported strategies when asking for multiple points to the optimizer
LIE_STRATS = ["cl_min", "cl_mean", "cl_max"]
LIE_STRATS_N = len(LIE_STRATS)
CYCLE_LIE_STRATS = cycle(LIE_STRATS)

# supported estimators
ESTIMATORS = [
    "GBRT",
    "ET",
    "RF",
]  # "GP" uses too much memory with because of matrix mul...

ESTIMATORS_N = len(ESTIMATORS)
CYCLE_ESTIMATORS = cycle(ESTIMATORS)

ACQ_FUNCS = ["LCB", "EI", "PI"]
ACQ_FUNCS_N = len(ACQ_FUNCS)
CYCLE_ACQ_FUNCS = cycle(ACQ_FUNCS)


class SkoptOptimizer(IOptimizer):

    _space: List[Dimension]
    """ if the space is only made of float ranges """
    _all_real: bool
    _opt: Optimizer
    _fit = False

    def ask(self, n=None, *args, **kwargs):
        if n == 0:
            n = None
        if not self._fit:
            # start = now()
            if hasattr(self._opt, "_next_x"):
                delattr(self._opt, "_next_x")
            self._opt.update_next()
            # print("update next took:", now() - start, self.rs)
        # start = now()
        asked = self._opt.ask(n, strategy=self._lie_strat)
        # print("asking took:", now() - start, self.rs)
        if self._fit:
            self._fit = False
        return [asked] if n is None else asked

    def tell(self, Xi, yi, fit=False, *args, **kwargs):
        told = self._opt.tell(Xi, yi, fit)
        if fit and not self._fit:
            self._fit = True
        return told

    def create_optimizer(
        self, parameters: Optional[Iterable] = None, config={}
    ) -> IOptimizer:
        " Construct an optimizer object "
        if parameters is not None:
            self.update_space(parameters)

        self._setup_mode()
        # keep models queue to 1
        self.n_models = 1

        self._setup_lie_strat()

        # https://github.com/scikit-learn/scikit-learn/issues/14265
        # lbfgs uses joblib threading backend so n_jobs has to be reduced
        # to avoid oversubscription
        base_estimator = self.opt_base_estimator
        acq_optimizer = (
            ("lbfgs" if self._all_real else "sampling")
            if base_estimator == "GP"
            else self.opt_acq_optimizer
        )
        n_jobs = 1 if self.opt_acq_optimizer == "lbfgs" else self.n_jobs
        self._opt = Optimizer(
            self._space,
            base_estimator=base_estimator,
            acq_optimizer=acq_optimizer,
            acq_func=self.chosen_acq_func,
            n_initial_points=self.n_rand,
            acq_optimizer_kwargs={
                "n_jobs": n_jobs,
                "n_points": self.calc_n_points(
                    len(self._params), self.n_jobs, self.ask_points
                ),
            },
            acq_func_kwargs={},
            model_queue_size=self.n_models,
            random_state=self.rs,
        )
        return self

    def update_space(self, parameters: Iterable):
        new_space = []
        self._all_real = True
        for par in parameters:
            m = par.meta
            if par.kind == RANGE:
                dist = m.get("dist", "uni")
                if dist == "uni":
                    prior = "uniform"
                elif dist == "log":
                    prior = "log-uniform"
                else:
                    self.handle_missing_tag(("dist", dist))
                base = m.get("log_base", 10)
                enc = m.get("enc", "idem")
                if enc == "idem":
                    trans = "identity"
                elif enc == "norm":
                    trans = "normalize"
                else:
                    self.handle_missing_tag(("enc", enc))
                if "int" in par.meta:
                    if self._all_real:
                        self._all_real = False
                    new_space.append(
                        Integer(
                            par.low,
                            par.high,
                            name=par.name,
                            base=base,
                            prior=prior,
                            transform=trans,
                            **m["kwargs"]
                        )
                    )
                else:
                    new_space.append(
                        Real(
                            par.low,
                            par.high,
                            name=par.name,
                            base=base,
                            prior=prior,
                            transform=trans,
                            **m["kwargs"]
                        ),
                    )
            elif par.kind == CAT:
                if self._all_real:
                    self._all_real = False
                enc = m.get("enc", "bool")
                if enc == "bool":
                    trans = "onehot"
                elif enc == "int":
                    trans = "label"
                elif enc == "idem":
                    trans = "identity"
                elif enc == "str":
                    trans = "string"
                else:
                    self.handle_missing_tag(("enc", enc))
                new_space.append(
                    Categorical(
                        par.sub, name=par.name, prior=m.get("dist"), transform=trans
                    )
                )
            else:
                raise OperationalException(
                    "mixed parameters are not supported by skopt"
                )
        self._space = new_space

    def _setup_mode(self):
        # if 0 n_points are given, don't use any base estimator (akin to random search)
        # and force single mode as there is no model
        if self.algo == "rand":
            self.opt_base_estimator = "DUMMY"
            self.chosen_acq_func = "gp_hedge"
            self.mode = "single"
            self.opt_acq_optimizer = "sampling"
        # The GaussianProcessRegressor is heavy, which makes it not a good default
        # however longer backtests might make it a better tradeoff
        elif not self.algo or self.algo == "auto":
            if self.mode == "single":
                self.opt_base_estimator = "GP"
                self.chosen_acq_func = "gp_hedge"
                self.opt_acq_optimizer = "lbfgs"
            else:
                self.opt_acq_optimizer = "auto"
                if self.mode == "shared":
                    self.opt_base_estimator = "GBRT"
                    self.chosen_acq_func = self.acq_funcs
                else:
                    self.opt_base_estimator = self.estimators
                    self.chosen_acq_func = "gp_hedge"
        else:
            self.opt_base_estimator = self.algo
            self.chosen_acq_func = self._config.get("acq_func", "gp_hedge")
            self.opt_acq_optimizer = self._config.get("acq_opt", "auto")

    def exploit(self, loss_tail: List[float], current_best: float, *args, **kwargs):
        xi = kappa = None
        # default xi is 0.01, we want some 10e even lower values
        if self._opt.acq_func in ("PI", "gp_hedge"):
            _, digits, exponent = Decimal(nanvar(loss_tail)).as_tuple()
            if len(digits) and digits[0]:
                xi = nanstd(loss_tail) * 10 ** -abs(len(digits) + exponent - 1) or 0.01
            else:
                xi = 0.01
        # with EI that supports negatives we can just use negative std
        elif self._opt.acq_func == "EI":
            xi = -nanstd(loss_tail)
        # LCB uses kappa instead of xi, and is based on variance
        if self._opt.acq_func in ("LCB", "gp_hedge"):
            kappa = nanvar([current_best, nanmin(loss_tail)]) or nanvar(
                # use mean when we just found a new best as var would be 0
                [current_best, nanmean(loss_tail)]
            )
        self._opt.acq_func_kwargs.update({"xi": xi, "kappa": kappa})

    def explore(self, loss_tail: List[float], current_best: float, *args, **kwargs):
        xi = kappa = None
        loss_last = kwargs["loss_last"]
        tail_position = kwargs["tail_position"]
        if self._opt.acq_func in ("PI", "EI", "gp_hedge"):
            xi = nanstd(loss_last) * tail_position
        if self._opt.acq_func in ("LCB", "gp_hedge"):
            kappa = nanvar(loss_last) * tail_position
        self._opt.acq_func_kwargs.update({"xi": xi, "kappa": kappa})

    @property
    def can_tune(self) -> bool:
        return True

    def _setup_lie_strat(self):
        # lie strategy
        def_lie_strat = "mean" if self.mode == "shared" else "cl_min"
        lie_strat = self._config.get("lie_strat", def_lie_strat)
        if lie_strat == "random":
            self._lie_strat = self.lie_strategy
        else:
            self._lie_strat = lie_strat

    @property
    def lie_strategy(self):
        """ Choose a strategy randomly among the supported ones, used in multi opt mode
        to increase the diversion of the searches of each optimizer """
        return next(CYCLE_LIE_STRATS)

    @property
    def estimators(self):
        return next(CYCLE_ESTIMATORS)

    @property
    def acq_funcs(self):
        return next(CYCLE_ACQ_FUNCS)

    @staticmethod
    def available_bytes() -> float:
        """ return host available memory in bytes """
        return virtual_memory().available / 1000

    @staticmethod
    def calc_n_points(n_dimensions: int, n_jobs: int, ask_points) -> int:
        """ Calculate the number of points the optimizer samples, based on available host memory """
        available_mem = SkoptOptimizer.available_bytes()
        # get size of one parameter
        return int(available_mem / (4 * n_dimensions * abs(n_jobs) * ask_points))

    @property
    def supported_tags(self):
        return {"enc", "dist", "log_base", "int"}

    @property
    def yi(self):
        return self._opt.yi

    @property
    def Xi(self):
        return self._opt.Xi

    @property
    def models(self):
        return self._opt.models
