#!/usr/bin/env python3
from .types import njit
import numpy as np


from julia.api import LibJulia
from ctypes import c_char_p, c_double, c_int64, c_void_p

# from ctypes import *


def set_ctypes(api):
    api.jl_symbol.argtypes = [c_char_p]  #
    api.jl_symbol.restype = c_void_p
    api.jl_get_global.argtypes = [c_void_p, c_void_p]
    api.jl_get_global.restype = c_void_p
    api.jl_symbol.restype = c_void_p

    api.jl_box_voidpointer.argtypes = [c_void_p]
    api.jl_box_voidpointer.restype = c_void_p
    api.jl_box_float64.argtypes = [c_double]
    api.jl_box_float64.restype = c_void_p
    api.jl_box_int64.argtypes = [c_int64]
    api.jl_box_int64.restype = c_void_p
    api.jl_call.argtypes = [c_void_p]
    api.jl_call.restype = c_void_p
    api.jl_call1.argtypes = [c_void_p, c_void_p]
    api.jl_call1.restype = c_void_p
    api.jl_call3.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p]
    api.jl_call3.restype = c_void_p

    api.jl_cstr_to_string.argtypes = [c_char_p]
    api.jl_cstr_to_string.restype = c_void_p

    api.jl_string_ptr.argtyeps = [c_void_p]
    api.jl_string_ptr.restype = c_char_p


def get_julia(jl_args={}):
    from julia.api import Julia

    jl = Julia(**jl_args)
    set_ctypes(jl.api)
    return jl


def get_julia_fn_ptr(fn: str, lib: str, jl=None):
    if jl is None:
        jl = get_julia()

    wrap = jl.eval(f"using {lib}; {fn}")
    return wrap.jl_value, jl
