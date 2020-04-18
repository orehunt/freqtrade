from numpy import iinfo, int32
import os
import sys
import logging

# supported strategies when asking for multiple points to the optimizer
LIE_STRATS = ["cl_min", "cl_mean", "cl_max"]
LIE_STRATS_N = len(LIE_STRATS)

# supported estimators
ESTIMATORS = ["GBRT", "ET", "RF"]
ESTIMATORS_N = len(ESTIMATORS)

VOID_LOSS = iinfo(int32).max  # just a big enough number to be a bad point in the loss optimization

testing = "pytest" in sys.modules

if testing:
    columns = 80
else:
    columns, _ = os.get_terminal_size()
    columns -= 1

logger = logging.getLogger(__name__)
