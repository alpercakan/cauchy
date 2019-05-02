# initializers.py
# Author: Alper Çakan

import numpy as np


def kernel_glorot_uniform(fan_in, fan_out):
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, +limit, size=(fan_out, fan_in))


def vec_all_zero(dim):
    return np.zeros((dim,))
