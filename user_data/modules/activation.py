import numpy as np
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


jit_module(nopython=True, error_model="numpy")
