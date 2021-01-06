import numba as nb
from numba import njit
import numpy as np


@njit(cache=True)
def rebalance_above(arr, min_value: float = 0.1):
    """
    Takes a 1d array. Any value below `min_value` is rebalanced equally over all the others entries.
    The execution is done sequentially such that values that at the beginning are below `min_value` can be
    above it at the end (and therefore are not rebalanced)
    """
    order = arr.argsort()
    ord_arr = arr[order]
    bottom = 0
    len_arr = len(arr)
    top = len_arr - 1
    for n in range(-1, -top, -1):
        # skip weights past minimum
        if ord_arr[n] >= min_value:
            continue
        # add values from the bottom until the pair reached minimum
        while bottom < top and ord_arr[n] < min_value:
            bot_weight = ord_arr[bottom]
            if bot_weight == 0:
                bottom += 1
                continue
            elif bot_weight >= min_value:
                re_weights = ord_arr[n + 1 :] + ord_arr[n] / (top - n)
                ord_arr[n + 1 :] += re_weights * ord_arr[n]
                ord_arr[n] = 0
                break
            re_weights = ord_arr[bottom + 1 :] + bot_weight / (top - bottom)
            ord_arr[bottom + 1 :] += re_weights * bot_weight
            ord_arr[bottom] = 0
            bottom += 1
    return ord_arr, order
