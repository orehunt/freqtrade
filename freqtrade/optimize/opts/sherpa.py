import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import sherpa as she
import sherpa.core as sc
from sherpa.core import Trial

from freqtrade.exceptions import OperationalException

from ..optimizer import CAT, RANGE, IOptimizer, Points


class Xi(Points):
    _study: she.Study
    _params_names: List[str]

    def __init__(self, study: she.Study):
        self._study = study
        self._params_names = [p.name for p in study.parameters]

    def __getitem__(self, item) -> Any:
        return self._study.results[self._params_names].iloc[item].values.tolist()

    def __delitem__(self, item: int):
        idx = self._study.results.index
        self._study.results.drop(idx[item], inplace=True)

    def __len__(self):
        return len(self._study.results)


class yi(Xi):
    def __getitem__(self, item):
        try:
            ret = self._study.results["Objective"].iloc[item]
            return ret.values.tolist() if isinstance(ret, np.ndarray) else ret
        except KeyError:
            return []


class SherpaOptimizer(IOptimizer):

    _space: List[she.Parameter]
    """ if the space is only made of float ranges """
    _all_real: bool
    _algo: she.algorithms.Algorithm
    _study: she.core.Study
    _fit = False
    _Xi: Xi
    _yi: yi
    _lower_is_better = True
    _port = None
    # params hashes, trial object
    _obs: Dict[int, she.Trial] = {}
    # trial id, iter index
    _iterations: Dict[int, int] = {}
    # params hashes
    _evaluated: Set[int] = set()
    # need the list to preserve the order
    _params_names: List[str]
    _meta_keys: Set[str] = set()

    logging.getLogger("GP").setLevel(logging.ERROR)
    logging.getLogger("rbf").setLevel(logging.ERROR)
    logging.getLogger("variance").setLevel(logging.ERROR)
    logging.getLogger("paramz.transformations").setLevel(logging.ERROR)

    def ask(self, n=None, *args, **kwargs) -> List[Tuple[Tuple, Dict]]:
        asked = []
        for n in range(1 if n is None else n):
            trial = self._study.get_suggestion()
            # check if optimizer is DONE
            if isinstance(trial, str):
                return [((), {})]
            tp = trial.parameters
            # the parameters dict has keys other than just parameters
            # with some algos, (like PBT)
            tp_tup = tuple(tp[name] for name in self._params_names)
            # we only set the meta keys once since we assume they are
            # always the same for a particular mode of operation
            if not self._meta_keys:
                spn = set(self._params_names)
                self._meta_keys = set(k for k in tp if k not in spn)
            meta = {k: trial.parameters[k] for k in self._meta_keys}
            meta['trial_id'] = trial.id
            h = hash(tp_tup)
            self._obs[h] = trial
            asked.append((tp_tup, meta))
            if trial.id not in self._iterations:
                self._iterations[trial.id] = 1

        return asked

    def tell(
        self, Xi: Iterable[Tuple[Sequence, Dict]], yi: Sequence[float], fit=False, *args, **kwargs
    ) -> Any:
        for n, obs in enumerate(Xi):
            X, meta = obs[0], obs[1]
            h = hash(tuple(p for p in X))
            if h in self._evaluated:
                continue
            # if the hash is not in stored trials, the parameters
            # were evaluated from another optimizer instance, however
            # we don't know the trial ID so we just make a new one;
            # NOTE: this means that in shared mode it doesn't make
            # sense to use multiple observation per trial, since
            # trials metadata is not shared
            if h not in self._obs:
                # self._study.num_trials += 1
                # trial_id = self._study.num_trials
                # trial = Trial(
                #     id=trial_id,
                #     parameters={p.name: v for p, v in zip(self._params, X)},
                # )
                parameters = {p.name: v for p, v in zip(self._params, X)}
                parameters.update(meta)
                trial = Trial(id=meta['trial_id'], parameters=parameters)
                self._iterations[meta['trial_id']] = itr = 0
            else:
                trial = self._obs[h]
                itr = self._iterations[trial.id]
                del self._obs[h]
            self._study.add_observation(trial, objective=yi[n], iteration=itr)
            if (itr or 1) >= self.ask_points:
                self._study.finalize(trial)
            self._iterations[trial.id] += 1
            self._evaluated.add(h)

        return

    def create_optimizer(
        self, parameters: Optional[Iterable] = None, config={}
    ) -> IOptimizer:
        " Construct an optimizer object "
        if parameters is not None:
            self.update_space(parameters)
            self._params_names = list(p.name for p in parameters)

        self._setup_mode()

        self._study = she.Study(
            parameters=self._space,
            algorithm=self._algo,
            lower_is_better=self._lower_is_better,
            dashboard_port=self._port,
            disable_dashboard=self._config.get("disable_dashboard", True),
        )
        self._Xi = Xi(self._study)
        self._yi = yi(self._study)

        return self

    @property
    def supported_tags(self):
        return {"dist", "int", "enc", "kwargs"}

    def update_space(self, parameters: Iterable):
        new_space = []
        self._all_real = True
        for par in parameters:
            m = par.meta
            self.validate_tags(m)
            if par.kind == RANGE:
                if dist := m.get("dist", "uni"):
                    if dist == "uni":
                        scale = "linear"
                    elif dist == "log":
                        scale = "log"
                    else:
                        self.handle_missing_tag(("dist", dist))
                kwargs = {
                    "name": par.name,
                    "range": [par.low, par.high],
                    "scale": scale,
                }
                if "int" in m:
                    if self._all_real:
                        self._all_real = False
                    cls = sc.Discrete
                else:
                    cls = sc.Continuous
                new_space.append(cls(**kwargs))
            elif par.kind == CAT:
                if self._all_real:
                    self._all_real = False
                enc = m.get("enc", "bool")
                if enc == "bool":
                    cls = sc.Choice
                elif enc == "int":
                    # integer encoding is equivalent to ordered categories
                    cls = sc.Ordinal
                else:
                    self.handle_missing_tag(("enc", enc))
                    cls = sc.Choice
                if "dist" in m:
                    self.handle_missing_tag(("dist", m["dist"]))
                new_space.append(
                    cls(
                        name=par.name,
                        range=par.sub
                        if isinstance(par.sub, list)
                        else par.sub.tolist()
                        if isinstance(par.sub, np.ndarray)
                        else list(par.sub),
                    )
                )
        self._space = new_space

    def _setup_mode(self):
        # and force single mode as there is no model
        if self.algo == "rand":
            self._algo = she.algorithms.RandomSearch(max_num_trials=self.n_epochs)
        else:
            if self.algo in ("PBT", "ASHA") and self.mode == "shared":
                raise OperationalException(
                    "PBT and ASHA not compatible with shared mode"
                )
            kwargs = self._config.get("opt_kwargs", {})
            if not self.algo or self.algo in ("auto", "BO"):
                self.BO(**kwargs)
            elif self.algo == "PBT":
                self.PBT(**kwargs)
            elif self.algo == "ASHA":
                self.ASHA(**kwargs)
            else:
                cls = getattr(she.algorithms, self.algo)
                self._algo = cls(**kwargs)

    def BO(self, **kwargs):
        """ Bayesian Optimization """
        default = {
            "model_type": "GP",
            "num_initial_data_points": self.n_rand or "infer",
            "acquisition_type": "EI",
            "max_concurrent": self.n_jobs,
            "verbosity": False,
        }
        default.update(kwargs)
        self._algo = she.algorithms.GPyOpt(**default)

    def PBT(self, **kwargs):
        """ Population based training """
        default = {
            "population_size": self.n_rand,
            "num_generations": self.n_epochs or 10,
            "perturbation_factors": (0.8, 1.2),
            "parameter_range": {},
        }
        default.update(kwargs)
        self._algo = she.algorithms.PopulationBasedTraining(**default)

    def ASHA(self, **kwargs):
        """ Asynchronous successive halving """
        default = {
            "r": self.ask_points,
            "R": self.n_jobs * self.n_rand,
            "eta": self.n_rand // self.n_jobs,
            "s": 0,
            "max_finished_configs": 1,
        }
        default.update(kwargs)
        self._algo = she.algorithms.SuccessiveHalving(**default)

    def exploit(self, loss_tail: List[float], current_best: float, *args, **kwargs):
        pass

    def explore(self, loss_tail: List[float], current_best: float, *args, **kwargs):
        pass

    @property
    def models(self):
        pass
