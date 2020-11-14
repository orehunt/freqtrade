import logging
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import emukit as ek
import GPy
import numpy as np
from emukit.core.categorical_parameter import CategoricalParameter
from emukit.core.continuous_parameter import ContinuousParameter
from emukit.core.discrete_parameter import DiscreteParameter
from emukit.core.encodings import OneHotEncoding, OrdinalEncoding
from emukit.core.loop.user_function_result import UserFunctionResult
from emukit.experimental_design.experimental_design_loop import ExperimentalDesignLoop
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper
from emukit.examples.gp_bayesian_optimization.single_objective_bayesian_optimization import (
    GPBayesianOptimization,
)

from freqtrade.exceptions import OperationalException
from freqtrade.optimize.optimizer import IOptimizer

from ..optimizer import CAT, RANGE, Loss


class EmuKit(IOptimizer):
    """ EmuKit supports bayes opt/quadrature, active learning, sensitivy analysis... """

    _results: List[UserFunctionResult] = []
    # True when the optimizer has enough initial points
    _init = True
    _opt = None
    _told_idx = 0
    _params_list: ek.core.Parameter
    logging.getLogger("emukit").setLevel(logging.ERROR)
    logging.getLogger("GP").setLevel(logging.ERROR)

    def ask(self, n=1, *args, **kwargs) -> List[Tuple[Tuple, Dict]]:
        """ Return a new combination of parameters to evaluate """
        if self._init:
            # NOTE: this can overshoot the random points if the batch size is bigger
            Xi = [(p, {"enc_p": ep}) for p, ep in zip(*self._sample_space(n))]
        else:
            Xi = []
            if not self._opt:
                self._setup_mode()
            enc_p = self._opt.get_next_points(self._results[self._told_idx :])[0]
            p = self._parse_points(enc_p)
            self._told_idx = len(self._results)
            Xi.append((p, {"enc_p": enc_p}))
            for _ in range(1, n):
                enc_p = self._opt.get_next_points([])[0]
                p = self._parse_points(enc_p)
                Xi.append((p, {"enc_p": enc_p}))
        return Xi

    def tell(
        self,
        Xi: Iterable[Tuple[Sequence, Dict]],
        yi: Sequence[Loss],
        fit=False,
        *args,
        **kwargs,
    ):
        """ Submit evaluated scores, the scheduler should return
        a list of tuples in the form (parameters, meta) """
        for X, y in zip(Xi, yi):
            _, meta = X
            y = next(iter(y.values()))
            self._results.append(
                UserFunctionResult(np.asarray(meta["enc_p"]), np.asarray((y,)))
            )
        if self._init and len(self.Xi) > self.n_rand:
            self._setup_mode()
            self._init = False

    def _setup_mode(self):
        """ Configure optimizer based on mode of operation """
        self.algo = next(self._algos_pool)
        if self.algo == "rand":
            raise OperationalException("random search not supported by EmuKit")
        elif not self.algo or self.algo == "auto":
            # kwargs = {}
            # self.BO(**kwargs)
            pass
        else:
            algo_fn = getattr(self, self.algo)
            kwargs = {}
            algo_fn(**kwargs)

    def create_optimizer(self, parameters: Iterable, config={}):
        """ Create a new optimizer from given configuration """
        if parameters is not None:
            self.update_space(parameters)
        return self

    @property
    def Xi(self):
        return [r.X for r in self._results]

    @Xi.deleter
    def Xi(self, item):
        del self._results[item]

    @property
    def yi(self):
        return [r.Y for r in self._results]

    @yi.deleter
    def yi(self, item):
        del self._results[item]

    @property
    def supported_tags(self):
        return {"int", "enc", "kwargs"}

    def update_space(self, parameters: Iterable):
        """ Modify the inner representation of the dimensions for the Optimizer """
        new_space = []
        for par in parameters:
            m = par.meta
            self.validate_tags(m)
            if par.kind == RANGE:
                kwargs = {"name": par.name}
                if "int" in m:
                    cls = DiscreteParameter
                    # NOTE: emukit discret parameter is an expanded range..?
                    kwargs.update({"domain": np.arange(par.low, par.high + 1)})
                else:
                    cls = ContinuousParameter
                    kwargs.update(
                        {"min_value": par.low, "max_value": par.high,}
                    )
                new_space.append(cls(**kwargs))
            elif par.kind == CAT:
                enc = m.get("enc", "bool")
                if enc == "bool":
                    cls = OneHotEncoding
                elif enc == "ord":
                    cls = OrdinalEncoding
                else:
                    self.handle_missing_tag(("enc", enc))
                    cls = OneHotEncoding
                encoding = cls(self.sub_to_list(par.sub))
                new_space.append(CategoricalParameter(name=par.name, encoding=encoding))
        self._params_list = new_space
        self._space = ek.core.parameter_space.ParameterSpace(
            new_space, constraints=None
        )

    def _sample_space(self, point_count: int = 1) -> Tuple[np.ndarray, List]:
        sample = np.empty((point_count, len(self._params)), dtype="O")
        enc_sample = [[]] * point_count
        for n, par in enumerate(self._space.parameters):
            if isinstance(par, CategoricalParameter):
                cats = np.asarray(par.encoding.categories)
                enc_pars = par.sample_uniform(point_count)
                if isinstance(par.encoding, OneHotEncoding):
                    indices = enc_pars.argmax(axis=1)
                    for pc in range(point_count):
                        enc_sample[pc].extend(enc_pars[pc].tolist())
                else:
                    indices = enc_pars.ravel().astype(int) - 1
                    for pc in range(point_count):
                        enc_sample[pc].append(enc_pars[pc][0])
                sample[:, n] = cats[indices]
            else:
                par = par.sample_uniform(point_count).ravel()
                sample[:, n] = par
                for pc in range(point_count):
                    enc_sample[pc].append(par[pc])
        return sample, enc_sample

    def _parse_points(self, point: Sequence[Any]):
        parsed = np.empty(len(self._params), dtype="O")
        pt_ofs = 0
        for n_par, par in enumerate(self._space.parameters):
            if isinstance(par, CategoricalParameter):
                cats = np.asarray(par.encoding.categories)
                if isinstance(par.encoding, OneHotEncoding):
                    idx = point[pt_ofs : pt_ofs + len(cats)].argmax()
                    pt_ofs += len(cats)
                else:
                    idx = point[pt_ofs] - 1
                    pt_ofs += 1
                parsed[n_par] = cats[idx.astype(int)]
            else:
                parsed[n_par] = point[pt_ofs]
                pt_ofs += 1
        return tuple(parsed)

    def _validate_points(self):
        n_points = len(self.Xi)
        assert n_points == len(self.yi)
        return n_points

    def ExpD(self):
        n_points = self._validate_points()
        if n_points > 0:
            model = GPy.models.GPRegression(
                np.asarray(self.Xi),
                np.asarray(self.yi),
                GPy.kern.RBF(1, lengthscale=0.08, variance=20),
                noise_var=1e-10,
            )
            self._opt = ExperimentalDesignLoop(
                space=self._space, model=GPyModelWrapper(model)
            )

    def BO(self):
        """ Bayesian optimization """
        n_points = self._validate_points()
        if n_points > 0:
            self._opt = GPBayesianOptimization(
                variables_list=self._params_list,
                X=np.asarray(self.Xi),
                Y=np.asarray(self.yi),
                batch_size=self.ask_points,
                model_update_interval=self._config.get("update_interval", self.n_jobs),
                noiseless=self._config.get("noiseless", False),
            )

# class GPBayesianOptimization(GPBayesianOptimization):
#     def _model_chooser(self):
