import numpy as np
from enum import IntEnum
from numba import jit, jit_module


def tanh(X):
    return np.tanh(X)


def relu(X):
    return np.maximum(0, X)


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def softmax(X):
    expo = np.exp(X)
    expo_sum = np.sum(np.exp(X))
    return expo / expo_sum


class FnType(IntEnum):
    tanh = 0
    relu = 1
    sigmoid = 2
    softmax = 3


fn = (tanh, relu, sigmoid, softmax)


def AND(x, y):
    return x & y


def OR(x, y):
    return x | y


def ABOVE(x, y):
    return x > y


def BELOWEQ(x, y):
    return x <= y


class OpType(IntEnum):
    AND = 0
    OR = 1
    ABOVE = 2
    BELOWEQ = 3


op = (AND, OR, ABOVE, BELOWEQ)

jit_module(nopython=True, error_model="numpy")
