# activations.py
# Author: Alper Çakan

import numpy as np


def relu_fn(x):
    return (x + np.abs(x)) / 2


def relu_prime_fn(x, _):
    return np.asarray(x > 0, x.dtype)


relu = [relu_fn, relu_prime_fn, 'relu']

name2activation = {
    'relu': relu
}