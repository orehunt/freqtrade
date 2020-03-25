from numpy import iinfo, int32

# supported strategies when asking for multiple points to the optimizer
LIE_STRATS = ["cl_min", "cl_mean", "cl_max"]
LIE_STRATS_N = len(LIE_STRATS)

# supported estimators
ESTIMATORS = ["GBRT", "ET", "RF"]
ESTIMATORS_N = len(ESTIMATORS)

VOID_LOSS = iinfo(int32).max  # just a big enough number to be bad result in loss optimization
