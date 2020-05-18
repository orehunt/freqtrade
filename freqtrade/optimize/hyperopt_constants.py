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
