from typing import Any, Dict, List, Union
from datetime import datetime
from queue import Queue
from multiprocessing.managers import SyncManager, Namespace
from threading import Lock
from pathlib import Path
import signal
from functools import partial
from logging import Logger

from pandas import DataFrame
from skopt import Optimizer

hyperopt: Any = None
cls: Union[None, Path] = None
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
