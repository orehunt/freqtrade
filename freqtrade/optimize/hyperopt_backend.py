from typing import Any, Dict, List, Tuple
from queue import Queue
from multiprocessing.managers import SyncManager, Namespace
from multiprocessing import Lock
import signal
from filelock import FileLock
from pandas import DataFrame, concat, read_hdf
from numpy import arange, isfinite
from pathlib import Path
from time import time as now
from functools import partial

from freqtrade.constants import HYPEROPT_LIST_STEP_VALUES
from freqtrade.exceptions import OperationalException

from skopt import Optimizer

hyperopt: Any = None
manager: SyncManager
# Each worker stores the optimizer in the global state
opt: Optimizer
optimizers: Queue
# Manage trials state
# num_done: int = 0
# num_saved: int = 0
# exit: bool = False
# tail = []
trials: Namespace
trials_list: List = [] # (worker local)
# each worker index position of known past trials (worker local)
trials_index = 0

# trials counting, variables stored in the epochs namespace, accessed by lock
# lock = Lock()
# current_best_loss = VOID_LOSS
# current_best_epoch = 0
# epochs_since_last_best: List = [0, 0]
# avg_last_occurrence: int
# max_epoch: int
epochs: Namespace

# timer, keep track how hyperopt runtime, use it to decide when to save on storage (worker local)
timer: float = 0

# save only the last points in order to tell them on the next run (worker local)
Xi: List = []
yi: List = []
# keep track of the points in the worker optimizer
Xi_h: List = []

def manager_sig_handler(signal, frame, backend: None):
    backend.trials.exit = True
    return

def manager_init(backend: Any):
    signal.signal(signal.SIGINT, partial(manager_sig_handler, backend=backend))

def trials_to_df(trials: List, metrics: bool = False) -> Tuple[DataFrame, str]:
    df = DataFrame(trials)
    if len(df) > 0:
        last_col = df.columns[-1]
    else:
        raise OperationalException("Trials were empty.")
    if metrics:
        df_metrics = DataFrame(t["results_metrics"] for t in trials)
        return concat([df, df_metrics], axis=1), last_col
    else:
        return df, last_col


