from typing import Any, Dict, List, Union
from queue import Queue
from multiprocessing.managers import SyncManager, Namespace
from threading import Lock
from pathlib import Path
import signal
from functools import partial

from pandas import DataFrame
from skopt import Optimizer

hyperopt: Any = None
cls: Path = None
data: Dict[str, DataFrame] = {}
manager: SyncManager
pbar: dict = {}

# Each worker stores the optimizer in the global state
opt: Optimizer
optimizers: Queue
exploit: int = 0

trials_list: List = []  # (worker local)
# each worker index position of known past trials (worker local)
trials_index = 0
# recently saved trials
just_saved = 0
# flag to remember if a worker has recently reset its optimizer parameters pool
# is resetted once space_reduction is again 0 (from n_jobs)
just_reduced = False
# in multi mode, each optimizer *also* stores its list of last_best periods
epochs_since_last_best = []

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
    tail: List
    empty_strikes: int
    void_loss: float
    table_header: int


trials = TrialsState()

# trials counting, accessed by lock
class Epochs(Namespace):
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
    space_reduction: int


epochs: Epochs


def manager_sig_handler(signal, frame, trials_state: TrialsState):
    trials_state.exit = True
    return


def manager_init(backend: Any):
    signal.signal(signal.SIGINT, partial(manager_sig_handler, backend=backend))
