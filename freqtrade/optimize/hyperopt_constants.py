import os
import sys
import logging
from numpy import iinfo, int32
from itertools import cycle


testing = "pytest" in sys.modules

if testing:
    columns = 80
else:
    columns, _ = os.get_terminal_size()
    columns -= 1

logger = logging.getLogger(__name__)

MULTI_SPACE_CONFIG = {
    "hyperopt_list_best": ["sum", "ratio"],
    "hyperopt_spc_red_interval": 10000,
    "hyperopt_list_pct_best": 0.33,
    "hyperopt_list_cutoff_best": 0.66,
    "hyperopt_list_profitable": False,
    "hyperopt_list_step_values": {"range": 3},
    "hyperopt_list_step_metric": ["loss"],
    "hyperopt_list_sort_metric": ["avg_profit", "profit"],
}


SHARED_SPACE_CONFIG = {
    "hyperopt_list_best": ["sum", "ratio"],
    "hyperopt_spc_red_interval": 10000,
    "hyperopt_list_pct_best": "std",
    "hyperopt_list_cutoff_best": "mean",
    "hyperopt_list_profitable": False,
    "hyperopt_list_step_values": {"range": "mean"},
    "hyperopt_list_step_metric": ["all"],
    "hyperopt_list_sort_metric": ["loss"],
}


SINGLE_SPACE_CONFIG = {
    "hyperopt_list_best": ["sum", "ratio"],
    "hyperopt_spc_red_interval": 100,
    "hyperopt_list_pct_best": "mean",
    "hyperopt_list_cutoff_best": "mean",
    "hyperopt_list_profitable": True,
    "hyperopt_list_step_values": {"range": "std"},
    "hyperopt_list_step_metric": ["loss", "duration", "trade_count"],
    "hyperopt_list_sort_metric": ["loss"],
}
