import logging
from itertools import cycle
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from pandas import read_json
import sherpa as she
import sherpa.core as sc
from joblib import hash
from sherpa.core import Trial

from freqtrade.exceptions import OperationalException
from user_data.modules.helper import read_json_file

from ..optimizer import CAT, RANGE, IOptimizer, Points


hash = partial(hash, hash_name="sha1")
logger = logging.getLogger(__name__)


class Xi(Points):
    _study: she.Study
    _params_names: List[str]
    _empty_df = pd.DataFrame()

    def __init__(self, study: she.Study):
        self._study = study
        self._params_names = [p.name for p in study.parameters]

    def _status(self):
        if "Status" in self._study.results:
            return self._study.results["Status"].values == "COMPLETED"
        else:
            return None

    @property
    def __obs(self):
        if len(self._study.results):
            return self._study.results[self._status()]
        else:
            return self._empty_df

    def __getitem__(self, item) -> Any:
        return self.__obs[self._params_names].iloc[item].values.tolist()

    def __delitem__(self, item: int):
        idx = self._study.results.index
        self._study.results.drop(idx[item], inplace=True)

    def __len__(self):
        return len(self.__obs)


class yi(Xi):
    @property
    def __obs(self):
        if len(self._study.results):
            return self._study.results[self._status()]
        else:
            return self._empty_df

    def __getitem__(self, item):
        try:
            ret = self.__obs["Objective"].iloc[item]
            return ret.values.tolist() if isinstance(ret, np.ndarray) else ret
        except KeyError:
            return self._empty_df


class Sherpa(IOptimizer):

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

    # model specifics
    _bo_acq_pool: Optional[Iterable] = None
    _bo_models_pool: Optional[Iterable] = None
    __bo_model: str

    logging.getLogger("GP").setLevel(logging.ERROR)
    logging.getLogger("rbf").setLevel(logging.ERROR)
    logging.getLogger("variance").setLevel(logging.ERROR)
    logging.getLogger("paramz.transformations").setLevel(logging.ERROR)

    def ask(self, n=None, *args, **kwargs) -> List[Tuple[Tuple, Dict]]:
        asked = []
        # if self.algo in ("PBT", "ASHA"):
        #     if len(self._iterations) and not len(self.Xi):
        #         raise OperationalException(f"Can't ask for new points before telling old ones")
        for n in range(1 if n is None else n):
            trial = self._study.get_suggestion()
            # check if optimizer is DONE
            if trial is str or trial is None:
                self.void = True
                return asked
            tp = trial.parameters
            # the parameters dict has keys other than just parameters
            # with some algos, (like PBT)
            tp_tup = tuple(tp[name] for name in self._params_names)
            # we only set the meta keys once since we assume they are
            # always the same for a particular mode of operation
            if not self._meta_keys:
                spn = set(self._params_names)
                self._meta_keys = set(k for k in tp if k not in spn)
                # cols = self._study.results.columns.values.tolist()
                # for mk in self._meta_keys:
                #     if mk not in self._study.results.columns:
                #         cols.append(mk)
                # self._study.results.columns = cols
            meta = {k: trial.parameters[k] for k in self._meta_keys}
            meta["trial_id"] = trial.id
            h = hash(tp_tup)
            self._obs[h] = trial
            asked.append((tp_tup, meta))
            if trial.id not in self._iterations:
                self._iterations[trial.id] = 1

        return asked

    def tell(
        self,
        Xi: Iterable[Tuple[Sequence, Dict]],
        yi: Sequence[float],
        fit=False,
        *args,
        **kwargs,
    ) -> Any:
        for n, obs in enumerate(Xi):
            X, meta = obs[0], obs[1]
            h = hash(tuple(p for p in X))
            # IN PBT gotta tell the same points
            if self.algo != "PBT" and h in self._evaluated:
                continue
            # if the hash is not in stored trials, the parameters
            # were evaluated from another optimizer instance
            # add that to an existing trial if trial_id matches,
            # otherwise add a new trial
            if h not in self._obs:
                parameters = {p.name: v for p, v in zip(self._params, X)}
                parameters.update(meta)
                trial = Trial(id=meta["trial_id"], parameters=parameters)
                if meta["trial_id"] in self._iterations:
                    itr = self._iterations[meta["trial_id"]]
                else:
                    self._iterations[meta["trial_id"]] = itr = 1
            else:
                trial = self._obs[h]
                itr = self._iterations[trial.id]
                del self._obs[h]
            self._study.add_observation(trial, objective=yi[n], iteration=itr)
            if (itr or 1) >= self.epoch_to_obs or self.algo == "PBT":
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

    def copy(self, *args, **kwargs):
        opt = super().copy(*args, **kwargs)
        opt._study = self._study
        opt._params_names = self._params_names
        return opt

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
                elif enc == "ord":
                    # integer encoding is equivalent to ordered categories
                    cls = sc.Ordinal
                else:
                    self.handle_missing_tag(("enc", enc))
                    cls = sc.Choice
                if "dist" in m:
                    self.handle_missing_tag(("dist", m["dist"]))
                new_space.append(cls(name=par.name, range=self.sub_to_list(par.sub)))
        self._space = new_space

    def _setup_mode(self):
        # and force single mode as there is no model
        self.algo = next(self._algos_pool)
        if self.algo == "rand":
            self._algo = she.algorithms.RandomSearch(max_num_trials=self.n_epochs)
        else:
            if self.algo in ("PBT", "ASHA") and self.mode == "shared":
                raise OperationalException(
                    "PBT and ASHA not compatible with shared mode"
                )
            # kwargs = self._config.get("opt_kwargs", {})
            kwargs = self.algo_args()
            if self.algo in dir(self):
                oj = getattr(self, self.algo)
                if isinstance(oj, Callable):
                    oj(**kwargs)
                else:
                    raise OperationalException(f"algo {self.algo} not understood")

    @property
    def _bo_acq(self):
        if not self._bo_acq_pool:
            self._bo_acq_pool = cycle(
                ["EI", "EI_MCMC", "MPI", "MPI_MCMC", "LCB", "LCB_MCMC"]
            )
        acq = next(self._bo_acq_pool)
        # only use _MCMC with _MCMC model
        if self.__bo_model.replace("_MCMC", "") != self.__bo_model:
            while acq.replace("_MCMC", "") == acq:
                acq = next(self._bo_acq_pool)
        else:
            while acq.replace("_MCMC", "") != acq:
                acq = next(self._bo_acq_pool)
        return acq

    @property
    def _bo_model(self):
        if not self._bo_models_pool:
            # NOTE: other models have unresolved problems in gpyopt
            self._bo_models_pool = cycle(["GP", "GP_MCMC"])
        self.__bo_model = next(self._bo_models_pool)
        # don't use MCMC model when jobs > 1
        while (
            self.n_jobs > 1 and self.__bo_model.replace("_MCMC", "") != self.__bo_model
        ):
            self.__bo_model = next(self._bo_models_pool)
        return self.__bo_model

    def BO(self, **kwargs):
        """ Bayesian Optimization """
        default = {
            "model_type": self._bo_model,
            "num_initial_data_points": self.n_rand,
            "acquisition_type": self._bo_acq,
            "max_concurrent": self.n_jobs if self.mode == "single" else self.ask_points,
            "verbosity": False,
        }
        default.update(kwargs)
        self._algo = she.algorithms.GPyOpt(**default)
        self.is_blocking = self.ask_points == self.n_jobs

    def PBT(self, **kwargs):
        """ Population based training """
        default = {
            "population_size": self.ask_points,
            "num_generations": self.n_epochs / self.n_jobs,
            "perturbation_factors": (0.8, 1.2),
            "parameter_range": {},
        }
        default.update(kwargs)
        if self.mode == "single" and self.ask_points != self.n_jobs:
            raise OperationalException(
                f"PBT requires equal number of jobs ({self.n_jobs}) "
                f"and ask points ({self.ask_points}) in sync mode."
            )
        self._algo = she.algorithms.PopulationBasedTraining(**default)
        self.is_blocking = True

    def ASHA(self, **kwargs):
        """ Asynchronous successive halving """
        # NOTE: recommended:
        default = {
            # minimum resource each config will run for
            "r": 1,
            # maximum resource
            "R": self.n_epochs // self.n_jobs,
            # ratio to decide how to split each round
            "eta": self.ask_points,
            # minimum early stopping rate
            "s": 0,
            "max_finished_configs": 1,
        }
        default.update(kwargs)
        logger.info(
            "ASHA: r: %s, R: %s, eta: %s", default["r"], default["R"], default["eta"]
        )
        self._algo = she.algorithms.SuccessiveHalving(**default)

    def LOC(self, **kwargs):
        seed_key = "seed_configuration"
        default = {"perturbation_factors": (0.8, 1.2), "repeat_trials": 1}
        seed = kwargs.get(seed_key, {})
        if isinstance(seed, dict) and seed:
            pass
        elif isinstance(seed, str) and seed:
            kwargs[seed_key] = read_json_file(seed)
        else:
            raise OperationalException(
                f"Local search requires '{seed_key}', "
                "(dict, or as path to json file)"
            )
        default.update(kwargs)
        logger.info(
            "LOC: pertb: %s, repeating: %s",
            default["perturbation_factors"],
            default["repeat_trials"],
        )
        self._algo = she.algorithms.LocalSearch(**default)

    def exploit(self, loss_tail: List[float], current_best: float, *args, **kwargs):
        pass

    def explore(self, loss_tail: List[float], current_best: float, *args, **kwargs):
        pass

    @property
    def models(self):
        pass
