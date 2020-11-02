import logging
from itertools import cycle
from typing import Dict, Iterable, List, Sequence, Tuple

import ax
import numpy as np
from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.service.ax_client import AxClient

from freqtrade.optimize.optimizer import CAT, RANGE, IOptimizer

from ..optimizer import IOptimizer, Points


class Xi(Points):
    _client: AxClient
    _parameters_names: List[str]
    counter = 0

    def __init__(self, client: AxClient):
        self._client = client
        self._parameters_names = list(client.experiment.parameters)

    def __getitem__(self, item):
        df = self._client.get_trials_data_frame()
        got = df.loc[self._parameters_names].to_records()
        return got if isinstance(got, Iterable) else [got]

    def __delitem__(self, item):
        del self._client.experiment._trials[item]

    def __len__(self):
        return self.counter


class yi(Xi):
    _client: AxClient

    def __init__(self, client: AxClient):
        self._client = client

    def __getitem__(self, item):
        res = self._client.get_trials_data_frame().loc[item, "objective"]
        if isinstance(res, Iterable):
            return res.values.tolist()
        else:
            return res

    def __delitem__(self, item):
        del self._client.experiment._trials[item]


class Ax(IOptimizer):
    _space: ax.SearchSpace
    _strategy: GenerationStrategy
    _random_sampling = cycle(("SOBOL",))
    _client: AxClient
    _params_names: Sequence[str]
    is_blocking = False

    logging.getLogger("ax.modelbridge.transforms.int_to_float").setLevel(logging.ERROR)
    logging.getLogger("ax.service.ax_client").setLevel(logging.ERROR)

    def copy(self, *args, **kwargs) -> IOptimizer:
        opt = super().copy(*args, **kwargs)
        opt._client = self._client
        opt._params_names = self._params_names
        opt._space = self._space
        opt._strategy = self._strategy
        opt._experiment = self._experiment
        return opt

    def update_space(self, parameters: Iterable):
        new_space = {}
        for par in parameters:
            m = par.meta
            self.validate_tags(m)
            if par.kind == RANGE:
                dist = m.get("dist", "uni")
                if dist == "uni":
                    log_scale = False
                elif dist == "log":
                    log_scape = True
                else:
                    self.handle_missing_tag(("dist", dist))
                kwargs = {
                    "name": par.name,
                    "lower": par.low,
                    "upper": par.high,
                    "parameter_type": ax.ParameterType.INT
                    if "int" in m
                    else ax.ParameterType.FLOAT,
                }
                new_space[par.name] = ax.RangeParameter(**kwargs)
            elif par.kind == CAT:
                enc = m.get("enc", "bool")
                if enc == "bool":
                    is_ordered = False
                elif enc == "ord":
                    is_ordered = True
                else:
                    is_ordered = False
                    self.handle_missing_tag(("enc", enc))
                if "dist" in m:
                    self.handle_missing_tag(("dist", m["dist"]))
                test = par.sub[0]
                if isinstance(test, str):
                    pt = ax.ParameterType.STRING
                elif isinstance(test, (float, np.float64, np.float32)):
                    pt = ax.ParameterType.FLOAT
                elif isinstance(test, (float, np.int64, np.int32)):
                    pt = ax.ParameterType.INT
                elif isinstance(test, (bool, np.bool)):
                    pt = ax.ParameterType.BOOL
                else:
                    raise ValueError("Can't set parameter type for %s", par.name)
                new_space[par.name] = ax.ChoiceParameter(
                    name=par.name,
                    parameter_type=pt,
                    values=par.sub,
                    is_ordered=is_ordered,
                )

        constraints = []
        cs_config = self._config.get("constraints", {})
        for cs, kwargs in cs_config.items():
            if cs == "ParameterConstraint":
                pcs = ax.ParameterConstraint(**kwargs)
            elif cs == "OrderConstraint":
                lp = kwargs.get("lower_parameter")
                up = kwargs.get("upper_parameter")
                if not lp or not up:
                    raise ValueError("OrderConstraint needs lower and upper parameters")
                lpar = new_space[lp]
                upar = new_space[up]
                pcs = ax.OrderConstraint(lower_parameter=lpar, upper_parameter=upar)
            elif cs == "SumConstraint":
                pars = kwargs.get("parameters")
                pars_sum = []
                for pname in pars:
                    pars_sum.append(new_space[pname])
                iub = kwargs.get("is_upper_bound", False)
                bound = kwargs.get("bound")
                if not bound:
                    raise ValueError("Specify bound for SumConstraint")
                pcs = ax.SumConstraint(
                    parameters=pars_sum, is_upper_bound=iub, bound=bound
                )
            else:
                raise NameError("Constraint type not understood")
            constraints.append(pcs)

        self._space = ax.SearchSpace(
            parameters=list(new_space.values()), parameter_constraints=constraints
        )

    @property
    def supported_tags(self):
        return {"dist", "int", "enc", "kwargs"}

    def _rand_model(self):
        return getattr(Models, next(self._random_sampling))

    def _setup_mode(self):

        self._experiment = ax.Experiment(
            name="experiment",
            search_space=self._space,
            optimization_config=OptimizationConfig(
                objective=Objective(
                    metric=Metric(name="objective", lower_is_better=True,),
                ),
                outcome_constraints=self._config.get("outcome_constraints"),
            ),
        )

        self.algo = next(self._algos_pool)
        if not self.algo:
            self.algo = "auto"
        kwargs = {"random_seed": self.rs}
        steps = [
            GenerationStep(
                self._rand_model(),
                num_trials=self.n_rand,
                min_trials_observed=self.ask_points,
                model_kwargs=kwargs
            )
        ]
        if self.algo == "rand":
            model = self._rand_model()
        elif self.algo == "MOO":
            model = Models.MOO
        elif self.algo == "auto":
            model = Models.GPMES
        else:
            model = getattr(Models, self.algo)
        steps.append(GenerationStep(model, num_trials=self.n_epochs, model_kwargs=kwargs))
        self._strategy = GenerationStrategy(steps)

    def create_optimizer(self, parameters: Iterable, config) -> IOptimizer:
        if parameters is not None:
            self.update_space(parameters)
            self._params_names = list(p.name for p in parameters)

        self._setup_mode()

        self._client = AxClient(random_seed=self.rs, verbose_logging=False)
        self._client._experiment = self._experiment
        self._client._generation_strategy = self._strategy
        self._Xi = Xi(self._client)
        self._yi = yi(self._client)
        return self

    def ask(self, n, *args, **kwargs) -> List[Tuple[Tuple, Dict]]:
        asked = []
        for n in range(n):
            tp, idx = self._client.get_next_trial()
            tp_tup = tuple(tp[p] for p in self._params_names)
            asked.append((tp_tup, {"idx": idx}))
        return asked

    def tell(
        self,
        Xi: Iterable[Tuple[Sequence, Dict]],
        yi: Sequence[float],
        fit=True,
        *args,
        **kwargs
    ):
        # at the start attach trials
        if self._Xi.counter == 0:
            for n, obs in enumerate(Xi):
                X, _ = obs[0], obs[1]
                params = {k: v for k, v in zip(self._params_names, X)}
                _, idx = self._client.attach_trial(params)
                self._client.complete_trial(trial_index=idx, raw_data=yi[n])
                self._yi.counter += 1
                self._Xi.counter += 1
        else:
            for n, obs in enumerate(Xi):
                X, meta = obs[0], obs[1]
                self._client.complete_trial(trial_index=meta["idx"], raw_data=yi[n])
                self._yi.counter += 1
                self._Xi.counter += 1
