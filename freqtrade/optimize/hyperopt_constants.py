import os
import sys
import logging
from numpy import iinfo, int32
from itertools import cycle

# supported strategies when asking for multiple points to the optimizer
LIE_STRATS = ["cl_min", "cl_mean", "cl_max"]
LIE_STRATS_N = len(LIE_STRATS)
CYCLE_LIE_STRATS = cycle(LIE_STRATS)

# supported estimators
ESTIMATORS = ["GBRT", "ET", "RF"] # "GP" uses too much memory with because of matrix mul...
ESTIMATORS_N = len(ESTIMATORS)
CYCLE_ESTIMATORS = cycle(ESTIMATORS)

ACQ_FUNCS = ["LCB", "EI", "PI"]
ACQ_FUNCS_N = len(ACQ_FUNCS)
CYCLE_ACQ_FUNCS = cycle(ACQ_FUNCS)

VOID_LOSS = iinfo(int32).max  # just a big enough number to be a bad point in the loss optimization

testing = "pytest" in sys.modules

if testing:
    columns = 80
else:
    columns, _ = os.get_terminal_size()
    columns -= 1

logger = logging.getLogger(__name__)

MULTI_SPACE_CONFIG = {
  "hyperopt_list_best": ["sum", "ratio"],
  "hyperopt_space_reduction_interval": 1000,
  "hyperopt_list_pct_best": 0.33,
  "hyperopt_list_cutoff_best": 0.66,
  "hyperopt_list_profitable": True,
  "hyperopt_list_step_values": {
    "range": 3
  },
  "hyperopt_list_step_metric": ["loss"],
  "hyperopt_list_sort_metric": ["avg_profit", "profit"],
}


SHARED_SPACE_CONFIG = {
  "hyperopt_list_best": ["sum", "ratio"],
  "hyperopt_space_reduction_interval": 200,
  "hyperopt_list_pct_best": "std",
  "hyperopt_list_cutoff_best": "mean",
  "hyperopt_list_profitable": True,
  "hyperopt_list_step_values": {
    "range": "mean"
  },
  "hyperopt_list_step_metric": ["all"],
  "hyperopt_list_sort_metric": ["loss"],
}


SINGLE_SPACE_CONFIG = {
  "hyperopt_list_best": ["sum", "ratio"],
  "hyperopt_space_reduction_interval": 50,
  "hyperopt_list_pct_best": "mean",
  "hyperopt_list_cutoff_best": "mean",
  "hyperopt_list_profitable": True,
  "hyperopt_list_step_values": {
    "range": "std"
  },
  "hyperopt_list_step_metric": ["loss", "duration", "trade_count"],
  "hyperopt_list_sort_metric": ["loss"],
}
