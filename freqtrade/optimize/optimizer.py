from __future__ import annotations

import logging
from abc import abstractmethod
from enum import IntEnum
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union
from itertools import cycle

import numpy as np
from sklearn.utils import check_random_state

from freqtrade.exceptions import OperationalException


VOID_LOSS = float(
    np.iinfo(np.int32).max
)  # just a big enough number to be a bad point in the loss optimization

PARAMETER_TYPES = ["cat", "range", "mix"]

logger = logging.getLogger(__name__)


class ParameterKind(IntEnum):
    """
    cat: categories (ordered, unordered)
    range: reals, integers (with upper and lower bounds)
    mix: matrices or nested parameters
    """

    CAT = 0
    RANGE = 1
    MIX = 2


CAT = ParameterKind.CAT
RANGE = ParameterKind.RANGE
MIX = ParameterKind.MIX


class Parameter(SimpleNamespace):
    # def __init__(
    #     self, name: str, low=None, high=None, kind=ParameterKind.CAT, meta={}, sub=[]
    # ):
    #     self.name = name
    #     self.low = low
    #     self.high = high
    #     self.kind = kind
    #     self.meta = meta
    #     self.sub = sub

    name: str

    low: Union[None, float] = None
    """ Lower bound """

    high: Union[None, float] = None
    """ upper bound """

    kind: ParameterKind

    meta: Dict[str, Any] = {"kwargs": {}}
    """
    metadata used to convert params to the optimizer implementation of parameters
    dist: distribution of parameter space (eg. 'log', List[float])
    enc: encoding of categorical values
    log_base: base to apply to log transormation
    """

    sub: Union[None, Iterable, object] = None
    """ for nested and categorical parameters """


def analyze_parameters(parameters: Iterable[Parameter], precision=16):
    """
    Guesstimate of paramters counts
    param: real_base : base count for a decimal poisition
    """
    n_pars = 0
    for par in parameters:
        if par.kind == 1:
            if "int" in par.meta:
                n_pars += max(1, par.high - par.low)
            else:
                left, right = f"{par.low/(par.high or 1.):.{precision}}".split(".")
                base = len(left)
                exp = int(right)
                n_pars += (exp * 10 + (base * precision * 10)) or 10
        elif par.kind in (0, 2):
            if isinstance(par.sub, (list, tuple)):
                n_pars += len(par.sub)
            elif hasattr(par.sub, "size"):
                mul = par.high - par.low if par.high and par.low else 1
                n_pars += getattr(par.sub, "size") * mul
            elif isinstance(par.sub, Parameter):
                n_pars += analyze_parameters((par.sub,))
    return n_pars


def _factorial(n):
    """ Ramanujan log factorial """
    return (
        n * np.log(n)
        - n
        + np.log(n * (1.0 + 4.0 * n * (1.0 + 2.0 * n))) / 6.0
        + np.log(np.pi) / 2.0
    )


def guess_search_space(parameters: List[Parameter]):
    """ Estimate the search space of size """
    count = len(parameters)
    size = analyze_parameters(parameters)
    # guess the size of the search space as the count of the
    # unordered combination of the non repeated dimensions entries
    try:
        search_space_size = int(
            max(
                _factorial(size) / (_factorial(size - count) * _factorial(count)),
                _factorial(size),
            )
        )
    except OverflowError:
        search_space_size = np.inf

    return search_space_size


class Points:
    """ If Xi,yi points can't be directly indexed an list like class should be used
    (that only requires these methods, a full mutable sequence is not required ) """

    @abstractmethod
    def __getitem__(self):
        raise NotImplementedError()

    @abstractmethod
    def __delitem__(self):
        raise NotImplementedError()

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()


class IOptimizer:
    """ Optimizer interface used to swap out optimizers implementing different optimization algorithms """

    """ random state """
    rs: int

    rng: np.random.RandomState

    """ flag to signal early stopping """
    void: Union[int, bool] = False

    """
    replacement for VOID_LOSS, set after trials with actual
    scores have been evaluated
    """
    void_loss: float = VOID_LOSS

    """
    *  *        algo  config  evaluations
    -  single:  1     1       sync
    -  shared:  1     >1      part-sync
    -  multi:   >1    1       async
    """
    mode: str

    """ tells the scheduler to wait for all trials of the previous ask to finish """
    is_blocking: bool = False

    """ at any epoch, how many observations are scheduled around the optimizer prediction """
    epoch_to_obs: int = 1

    """ Number of parallel workers """
    n_jobs: int

    """ Number of initial random points """
    n_rand: int

    """ A suggested epochs limit """
    n_epochs: int

    """ Points per trial, how many observations to run between epochs """
    ask_points: int

    """ Set to the algo of an instantiated optimizer """
    algo: str
    """ Hint to help the optimizer decide what to use """
    _algos_pool: Iterable

    _params: List
    _args: Tuple
    _kwargs: Dict
    _config: Dict
    _Xi: Points
    _yi: Points
    _models: Points

    def __init__(
        self, parameters: Iterable, seed=None, config={}, *args, **kwargs
    ) -> object:
        """ Create a new optimizer """
        self.rs = seed if seed is not None else np.random.randint(0, VOID_LOSS)
        self.rng = check_random_state(self.rs)
        self.n_jobs = config.get("n_jobs", 1)
        self.n_rand = config.get("n_rand", 1)
        self.n_epochs = config.get("n_epochs", 10)
        self.ask_points = config.get("ask_points", 1) or 1

        # parse the algo if it is comma separated;
        algo = config.get("algo", "rand")
        if isinstance(algo, str):
            algo = algo.split(",")
        # many algos are cycled when create_optimizer is called
        self._algos_pool = cycle(algo)

        self.mode = config.get("mode", "single")
        self._config = config
        self._params = list(parameters)
        self._args = args
        self._kwargs = kwargs
        self._setup_missing_tags_handler()

    def copy(self, rand: int = None, new_seed: bool = False) -> IOptimizer:
        """
        Return a new instance of the optimizer with modified rng, from the previous
        optimizer random state
        """
        if new_seed:
            if not rand:
                rand = self.rng.randint(0, VOID_LOSS)

        opt = self.__class__(
            self._params, rand or self.rs, self._config, *self._args, **self._kwargs
        )
        opt.void_loss = self.void_loss
        opt.void = self.void
        opt._Xi = self._Xi
        opt._yi = self._yi
        # change the random state but keep the initial rs as ID
        if rand:
            opt.rs = self.rs
        return opt

    def validate_tags(self, meta: Dict):
        for tg in meta:
            if tg not in self.supported_tags:
                logger.warning(
                    "metatag {} not supported by optimizer {}".format(
                        tg, self.__class__.__name__
                    )
                )

    def _setup_missing_tags_handler(self):
        mtc = self._config.get("meta_tag_conflict", "warn")
        if mtc == "warn":
            setattr(self, "handle_missing_tag, ", self._warn_missing_tag)
        elif mtc == "term":
            setattr(self, "handle_missing_tag, ", self._term_missing_tag)
        elif mtc == "quiet":
            setattr(self, "handle_missing_tag, ", lambda: None)
        else:
            raise OperationalException(
                "tag conflict option {} not understood".format(mtc)
            )

    def _warn_missing_tag(self, tag: Tuple[str, str]):
        logger.warning("option %s for tag %s is not supported!", tag[0], tag[1])

    def _term_missing_tag(self, tag: Tuple[str, str]):
        raise OperationalException(
            "execution couldn't continue because option %s for tag %s is not supported!".format(
                tag[0], tag[1]
            )
        )

    def algo_args(self) -> Dict[Any, Any]:
        """ Return the algo options provided in the optimizer configuration """
        return (
            self._config.get(self.__class__.__name__, {}).get(self.algo, {})
            if isinstance(self._config, dict)
            else {}
        )

    @abstractmethod
    def _setup_mode(self):
        """ Configure optimizer based on mode of operation """
        if self.algo == "rand":
            pass
        elif not self.algo or self.algo == "auto":
            if self.mode == "single":
                pass
            elif self.mode == "shared":
                pass
            else:
                pass
        else:
            pass

    @abstractmethod
    def handle_missing_tag(self, tag: Tuple[str, str]):
        """ Should terminate or warn about the missing configuration option """

    @staticmethod
    def clear(opt):
        """ Delete models and points from an optimizer instance """
        del opt.models[:], opt.Xi[:], opt.yi[:]
        return opt

    @staticmethod
    def sub_to_list(sub: Any):
        """ expands an iterable into a list"""
        return (
            sub
            if isinstance(sub, list)
            else sub.tolist()
            if isinstance(sub, np.ndarray)
            else list(sub)
        )

    @abstractmethod
    def create_optimizer(self, parameters: Iterable, config={}) -> IOptimizer:
        """ Create a new optimizer from given configuration """

    @abstractmethod
    def update_space(self, parameters: Iterable):
        """ Modify the inner representation of the dimensions for the Optimizer """

    @abstractmethod
    def ask(self, n=1, *args, **kwargs) -> List[Tuple[Tuple, Dict]]:
        """ Return a new combination of parameters to evaluate """

    @abstractmethod
    def tell(
        self,
        Xi: Iterable[Tuple[Sequence, Dict]],
        yi: Sequence[float],
        fit=False,
        *args,
        **kwargs,
    ):
        """ Submit evaluated scores, the scheduler should return
        a list of tuples in the form (parameters, meta) """

    @abstractmethod
    def exploit(self, loss_tail: List[float], current_best: float, *args, **kwargs):
        """ Tune search for exploitation """

    @abstractmethod
    def explore(self, loss_tail: List[float], current_best: float, *args, **kwargs):
        """ Tune search for exploration """

    @property
    def can_tune(self) -> bool:
        """ Optimizer should return True if it provides eploit/explore methods """
        return False

    @property
    @abstractmethod
    def supported_tags(self):
        """ Set of tags supported for parameter definition """

    @property
    @abstractmethod
    def models(self) -> Points:
        """ Where surrogate models are stored """
        return self._models

    @property
    @abstractmethod
    def Xi(self) -> Points:
        """ Where evaluated parameters configurations are stored """
        return self._Xi

    @property
    @abstractmethod
    def yi(self) -> Points:
        """ Where the loss relative to the evaluated (ordered) parameters
        configs are stored """
        return self._yi
