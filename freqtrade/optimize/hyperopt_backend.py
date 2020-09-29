import signal
from abc import abstractmethod
from datetime import datetime
from functools import partial
from logging import Logger
from multiprocessing.managers import Namespace, SyncManager
from pathlib import Path
from queue import Queue
from threading import Lock
from typing import Any, Callable, Dict, List, Union

from pandas import DataFrame
from skopt import Optimizer

hyperopt: Any = None
cls: Any = None
data: Dict[str, DataFrame] = {}
min_date: Union[None, datetime] = None
max_date: Union[None, datetime] = None
manager: SyncManager
pbar: dict = {}

# Each worker stores the optimizer in the global state
opt: Optimizer
optimizers: Queue
exploit: int = 0

trials_list = []  # (worker local)
# each worker index position of known past trials (worker local)
trials_index = 0
# recently saved trials
just_saved = 0
# flag to remember if a worker has recently reset its optimizer parameters pool
# is resetted if worker is not in reduction dict
just_reduced = False
# in multi mode, each optimizer *also* stores its list of last_best periods
epochs_since_last_best = [1, 1]

# timer, keep track how hyperopt runtime, use it to decide when to save on storage (worker local)
timer: float = 0

# save only the last points in order to tell them on the next run (worker local)
Xi: List = []
yi: List = []
# keep track of the points in the worker optimizer
Xi_h: Dict = {}
tested_h: List = []
# used by CV
params_Xi = []

# manage trials state
class TrialsState(Namespace):
    exit: bool
    lock: Lock
    num_saved: int
    num_done: int
    testing: dict
    tail_dict: Dict[int, List]
    tail_list: List
    empty_strikes: int
    void_loss: float
    table_header: int


trials = TrialsState()

# trials counting, accessed by lock
class Epochs(Namespace):
    # optimizers mapped to workers (PID or WID)
    pinned_optimizers: Dict[int, Optimizer]
    lock: Lock
    convergence: int
    epochs_since_last_best: List
    explo: int
    current_best_loss: Dict[Union[None, int], float]
    last_best_loss: float
    current_best_epoch: Dict[Union[None, int], int]
    last_best_epoch: int
    max_epoch: int
    avg_last_occurrence: int
    space_reduction: Dict[int, bool]


epochs: Epochs


def parallel_sig_handler(
    backend,
    fn: Callable,
    cls_file: Union[None, Path],
    logger: Union[None, Logger],
    *args,
    **kwargs,
):
    """
    To handle Ctrl-C the worker main function has to be wrapped into a try/catch;
    NOTE: The Manager process also needs to be configured to handle SIGINT (in the backend)
    """
    if not backend.cls and cls_file:
        from joblib import load

        backend.cls = load(cls_file)
        if logger:
            from freqtrade.loggers import setup_logging, setup_logging_pre

            setup_logging_pre()
            setup_logging(backend.cls.config)
            logger.debug("loaded hyperopt class from %s", cls_file)
    try:
        return fn(*args, **kwargs)
    except KeyboardInterrupt:
        logger.debug("received keyboard interrupt, flushing remaining trials..")
        trials_state = kwargs.get("trials_state")
        if trials_state:
            trials_state.exit = True
        return fn(*args, **kwargs)


def manager_sig_handler(signal, frame, trials_state: TrialsState):
    trials_state.exit = True
    return


def manager_init(backend: Any):
    signal.signal(signal.SIGINT, partial(manager_sig_handler, backend=backend))


def wait_for_lock(lock: Lock, message: str, logger: Logger):
    msg = f"Waiting for lock: {message}"
    logger.debug(msg)
    locked = lock.acquire()
    while not locked:
        logger.debug(msg)
        locked = lock.acquire()


class HyperoptBase:
    dimensions: List[Any]
    logger: Logger

    @abstractmethod
    def backtest_params(
        self,
        raw_params: List[Any] = None,
        iteration=None,
        params_dict: Dict[str, Any] = None,
    ):
        """
        Used Optimize function. Called once per epoch to optimize whatever is configured.
        Keep this function as optimized as possible!
        """

    @abstractmethod
    def _maybe_terminate(
        self, t: int, jobs: int, trials_state: TrialsState, epochs: Epochs
    ):
        """ Decide if the iterator should stop execution based on trials or epochs counts """

    @abstractmethod
    def parallel_objective(
        self, t: int, params, epochs: Epochs, trials_state: TrialsState
    ):
        """ objective run in single opt mode, run the backtest and log the results """

    @abstractmethod
    def log_trials(
        self, trials_state: TrialsState, epochs: Epochs, rs: Union[None, int]
    ):
        """ Calculate epochs and save results to storage """
