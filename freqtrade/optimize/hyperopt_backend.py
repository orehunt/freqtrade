from typing import Any, Dict, List
from queue import Queue
from multiprocessing.managers import SyncManager, Namespace
from threading import Lock
import signal
from functools import partial

from skopt import Optimizer

hyperopt: Any = None
manager: SyncManager
# pbar: tqdm = None
pbar: dict = {}

# Each worker stores the optimizer in the global state
opt: Optimizer
optimizers: Queue
exploit: int = 0

trials_list: List = []  # (worker local)
# each worker index position of known past trials (worker local)
trials_index = 0
just_saved = 0

# trials counting, variables stored in the epochs namespace, accessed by lock
# lock = Lock()
# current_best_loss = VOID_LOSS
# current_best_epoch = 0
# epochs_since_last_best: List = [0, 0]
# avg_last_occurrence: int
# max_epoch: int

# timer, keep track how hyperopt runtime, use it to decide when to save on storage (worker local)
timer: float = 0

# save only the last points in order to tell them on the next run (worker local)
Xi: List = []
yi: List = []
# keep track of the points in the worker optimizer
Xi_h: Dict = {}
tested_h: List = []

# Manage trials state
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

class Epochs(Namespace):
    lock: Lock
    convergence: int
    epochs_since_last_best: List
    explo: int
    current_best_loss: float
    current_best_epoch: int
    max_epoch: int
    avg_last_occurrence: int

epochs: Epochs

def manager_sig_handler(signal, frame, trials_state: TrialsState):
    trials_state.exit = True
    return


def manager_init(backend: Any):
    signal.signal(signal.SIGINT, partial(manager_sig_handler, backend=backend))
