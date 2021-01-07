#!/usr/bin/env python3
from .julia import get_julia
from .types import njit
from ctypes import CFUNCTYPE, c_double, c_int64
import numpy as np

fn_ptr = None
jl = None
main = None

LIBRARY = "HypothesisTests"
JL_ADF_VAR = "last_adf_test"
JL_ARR = "adf_value_array"
ADF_FUNC = f"(start, stop, tp=:trend) -> ({JL_ADF_VAR} = ADFTest({JL_ARR}[start:stop], tp, 0); {JL_ADF_VAR}.stat)"
P_VALUE_FUNC = f"() -> pvalue({JL_ADF_VAR})"


def adf(start, stop):
    raise NotImplementedError("Run setup before calling adf")


def set_arr(arr):
    global jl, main
    if jl is None:
        jl = get_julia()
    if main is None:
        from julia import Main as main

    setattr(main, JL_ARR, arr)


def setup(arr):
    global fn_ptr, jl

    if jl is None:
        jl = get_julia()

    jl.using(LIBRARY)
    adf_fn_ptr = jl.eval(ADF_FUNC).jl_value
    p_fn_ptr = jl.eval(P_VALUE_FUNC).jl_value

    jl_box_int64 = jl.api.jl_box_int64
    jl_unbox_float64 = jl.api.jl_unbox_float64
    jl_call = jl.api.jl_call
    jl_call2 = jl.api.jl_call2

    set_arr(arr)

    global adf

    @njit(cache=False)
    def adf(start, stop):
        bstart = jl_box_int64(start)
        bstop = jl_box_int64(stop)
        ret = jl_call2(adf_fn_ptr, bstart, bstop)
        if not ret:
            return np.nan, np.nan
        stat = jl_unbox_float64(ret)

        ret = jl_call(p_fn_ptr)
        if not ret:
            return np.nan, np.nan
        pv = jl_unbox_float64(ret)

        return stat, pv
