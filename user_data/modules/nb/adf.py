#!/usr/bin/env python3
from .julia import get_julia
from .types import njit
from ctypes import CFUNCTYPE, c_double, c_int64

fn_ptr = None
jl = None
main = None

LIBRARY = "HypothesisTests"
ANON_FUNC = "(start, stop, tp=:none) -> ADFTest(arr[start:stop], tp, 1).stat"


def adf(start, stop):
    raise NotImplementedError("Run setup before calling adf")


def set_arr(arr):
    global jl, main
    if jl is None:
        jl = get_julia()
    if main is None:
        from julia import Main as main

    main.adf_value_array = arr


def setup(arr):
    global fn_ptr, jl

    if jl is None:
        jl = get_julia()

    jl.using(LIBRARY)
    fn_ptr = jl.eval(ANON_FUNC).jl_value

    jl_box_int64 = jl.api.jl_box_int64
    jl_unbox_float64 = jl.api.jl_unbox_float64
    jl_call2 = jl.api.jl_call2

    @njit(cache=False)
    def jl_adf(start, stop):
        bstart = jl_box_int64(start)
        bstop = jl_box_int64(stop)
        ret = jl_call2(fn_ptr, bstart, bstop)

        return jl_unbox_float64(ret) if ret else ret

    set_arr(arr)

    wrapped_adf = CFUNCTYPE(c_double, c_int64, c_int64)(jl_adf)
    global adf

    @njit(cache=False)
    def adf(start, stop):
        return wrapped_adf(start, stop)
