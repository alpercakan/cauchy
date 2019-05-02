# utils.py
# Author: Alper Çakan

from models import Sequential
from layers import Dense, Activation, Softmax, SimpleRecurrent, VanillaRecurrent
import json
import numpy as np


def load_model(name):
    type_map = {
        "<class 'layers.SimpleRecurrent'>": SimpleRecurrent,
        "<class 'layers.VanillaRecurrent'>": VanillaRecurrent,
        "<class 'layers.Dense'>": Dense,
        "<class 'layers.Activation'>": Activation,
        "<class 'layers.Softmax'>": Softmax
    }

    with open('{}_network.json'.format(name), 'r') as infile:
        network = json.load(infile)

    shallow_params_dict = np.load('{}.npz'.format(name))
    params_dict = {}

    for k, v in shallow_params_dict.items():
        sep_ind = k.find('__')
        layer_ind = int(k[k.find('_') + 1: sep_ind])
        param_key = k[sep_ind + 2:]

        if layer_ind not in params_dict:
            params_dict[layer_ind] = {}

        params_dict[layer_ind][param_key] = v

    model = Sequential()

    layer_types = network['layer_types']
    layer_configs = network['layer_configs']

    for i in range(len(layer_types)):
        lt = type_map[layer_types[i]]
        config = layer_configs[i]

        layer = lt(**config)

        if layer.trainable:
            layer.set_params_from_dict(params_dict[i])

        model.add(layer)

    return model
