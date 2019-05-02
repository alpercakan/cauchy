# layers.py
# Author: Alper Çakan

import numpy as np
from initializers import kernel_glorot_uniform, vec_all_zero
from activations import name2activation


class Dense:
    def __init__(self, num_units, input_dim, kernel_initializer=kernel_glorot_uniform, bias_initializer=vec_all_zero):
        self.num_units = num_units
        self.input_dim = input_dim
        self.trainable = True
        self.stateful = False

        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        self.W = self.get_initial_weights()
        self.b = self.get_initial_bias()

        self.y = np.zeros((self.num_units, ))
        self.dy_dW = np.zeros((self.input_dim, ))
        # dy_db is just a vector of all ones, no need to store

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def get_initial_weights(self):
        return self.kernel_initializer(self.input_dim, self.num_units)

    def get_initial_bias(self):
        return self.bias_initializer(self.num_units)

    def forward(self, x):
        self.y = np.dot(self.W, x) + self.b

        self.dy_dW = x

        return self.y

    def backward(self, d):
        self.dW = np.dot(np.array([d]).T, [self.dy_dW])
        self.db = d

        return np.dot(self.W.T, d)

    def get_grads(self):
        return [self.dW, self.db]

    def get_empty_grads(self):
        return [np.zeros_like(self.dW), np.zeros_like(self.db)]

    def update_params(self, delta_params):
        self.W += delta_params[0]
        self.b += delta_params[1]

    def get_params_dict(self):
        return {'W': self.W, 'b': self.b}

    def set_params_from_dict(self, params):
        self.W = params['W']
        self.b = params['b']

    def get_config(self):
        return {'num_units': self.num_units,
                'input_dim': self.input_dim}


class Softmax:
    """
    Can be used only as the last layer.

    Loss function is categorical cross-entropy.
    """
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.trainable = False
        self.stateful = False

        self.p = np.zeros((self.input_dim, ))
        self.loss = 0

        self.dl_dx = np.zeros((self.input_dim, ))

    def forward(self, x, gt=None):
        x = x - np.ones_like(x) * np.max(x)  # Stabilization. It is guaranteed softmax(x) does not change with this.

        self.p = np.exp(x) / np.sum(np.exp(x))

        if gt is not None:
            self.loss = -np.log(np.dot(self.p, gt))

        self.dl_dx = np.copy(self.p)
        self.dl_dx[np.argmax(gt)] -= 1

        return self.p

    def backward(self, d):
        return d * self.dl_dx

    def get_config(self):
        return {'input_dim': self.input_dim}


class Activation:
    def __init__(self, act, input_dim):
        self.input_dim = input_dim
        self.trainable = False
        self.stateful = False

        if type(act) is str:
            act = name2activation[act]

        self.fn = act[0]
        self.fn_prime = act[1]
        self.act_name = act[2]

        self.y = np.zeros((self.input_dim, ))

        self.dy_dx = np.zeros((self.input_dim, ))

    def forward(self, x):
        self.y = self.fn(x)
        self.dy_dx = self.fn_prime(x, self.y)

        return self.y

    def backward(self, d):
        return d * self.dy_dx

    def get_config(self):
        return {'act': self.act_name, 'input_dim': self.input_dim}


# This one is equivalent to Keras's SimpleRNN with parameter return_sequences=False.
# After some experiments, I observed that using the last y is better than using all y sequence.
# So, if you add Dense and then Softmax on top of this, you get the eqv. of using only the last y of the RNN
# defined in the lecture slides.
class SimpleRecurrent:
    """
    Can be used only as the first layer.
    """

    def __init__(self, seq_len, input_dim, state_dim,
                 kernel_initializer=kernel_glorot_uniform, bias_initializer=vec_all_zero, state_initializer=vec_all_zero):
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.trainable = True
        self.stateful = False

        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.state_initializer = state_initializer

        self.h_seq = np.zeros((seq_len, self.state_dim))

        self.Whh = self.get_initial_hh_weights()
        self.Wxh = self.get_initial_xh_weights()
        self.b = self.get_initial_bias()

        self.dh_dWhh_seq = np.zeros((self.seq_len, self.state_dim))
        self.dh_dWxh_seq = np.zeros((self.seq_len, self.input_dim))

        self.reset_ds()

    def reset_ds(self):
        self.dWhh = np.zeros_like(self.Whh)
        self.dWxh = np.zeros_like(self.Wxh)
        self.db = np.zeros_like(self.b)

    def get_initial_hh_weights(self):
        return self.kernel_initializer(self.state_dim, self.state_dim)

    def get_initial_xh_weights(self):
        return self.kernel_initializer(self.input_dim, self.state_dim)

    def get_initial_bias(self):
        return self.bias_initializer(self.state_dim)

    def get_initial_state(self):
        return self.bias_initializer(self.state_dim)

    def forward(self, x_seq):
        h = self.get_initial_state()

        h_seq = self.h_seq = np.zeros((len(x_seq), self.state_dim))

        for t, xt in enumerate(x_seq):
            self.dh_dWhh_seq[t] = h  # previous h

            h_seq[t] = h = np.tanh(np.dot(self.Whh, h) + np.dot(self.Wxh, xt) + self.b)

            self.dh_dWxh_seq[t] = xt

        return h

    def backward(self, d):
        self.reset_ds()

        for t in reversed(range(1, self.seq_len)):
            ht = self.h_seq[t]

            d = d * (np.ones_like(ht) - ht*ht)

            self.db += d
            self.dWhh += np.dot(np.array([d]).T, [self.dh_dWhh_seq[t]])
            self.dWxh += np.dot(np.array([d]).T, [self.dh_dWxh_seq[t]])

            d = np.dot(self.Whh.T, d)

        return None  # This layer is always the first layer of a network

    def get_grads(self):
        return [self.dWhh, self.dWxh, self.db]

    def get_empty_grads(self):
        return [np.zeros_like(self.dWhh), np.zeros_like(self.dWxh), np.zeros_like(self.db)]

    def update_params(self, delta_params):
        self.Whh += delta_params[0]
        self.Wxh += delta_params[1]
        self.b += delta_params[2]

    def get_params_dict(self):
        return {'Whh': self.Whh, 'Wxh': self.Wxh, 'b': self.b}

    def set_params_from_dict(self, params):
        self.Whh = params['Whh']
        self.Wxh = params['Wxh']
        self.b = params['b']

    def get_config(self):
        return {'seq_len': self.seq_len,
                'input_dim': self.input_dim,
                'state_dim': self.state_dim}


# TODO refactor this to make it more flexible. now, you can only use this as the one and only layer of a model
# Also, there is too much code intersection with the SimpleRecurrent
class VanillaRecurrent:
    """
    Can be used only as the first layer.
    """

    def __init__(self, output_dim, seq_len, input_dim, state_dim,
                 stateful=False,
                 kernel_initializer=kernel_glorot_uniform, bias_initializer=vec_all_zero, state_initializer=vec_all_zero):
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.trainable = True
        self.stateful = stateful

        self.loss = 0

        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.state_initializer = state_initializer

        self.h_seq = np.zeros((seq_len, self.state_dim))
        self.y_seq = np.zeros((seq_len, self.output_dim))
        self.h = self.get_initial_state()

        self.Whh = self.get_initial_hh_weights()
        self.Wxh = self.get_initial_xh_weights()
        self.b = self.get_initial_bias()

        self.dh_dWhh_seq = np.zeros((self.seq_len, self.state_dim))
        self.dh_dWxh_seq = np.zeros((self.seq_len, self.input_dim))

        self.dense = Dense(output_dim, state_dim)
        self.softmax = Softmax(output_dim)

        self.reset_ds()

    def reset_ds(self):
        self.dWhh = np.zeros_like(self.Whh)
        self.dWxh = np.zeros_like(self.Wxh)
        self.db = np.zeros_like(self.b)

    def reset_state(self):
        self.h = self.get_initial_state()

    def get_initial_hh_weights(self):
        return self.kernel_initializer(self.state_dim, self.state_dim)

    def get_initial_xh_weights(self):
        return self.kernel_initializer(self.input_dim, self.state_dim)

    def get_initial_bias(self):
        return self.bias_initializer(self.state_dim)

    def get_initial_state(self):
        return self.bias_initializer(self.state_dim)

    def forward(self, x_seq, gt_seq):
        if self.stateful:
            h = self.h
        else:
            h = self.get_initial_state()

        self.loss = 0

        h_seq = self.h_seq = np.zeros((len(x_seq), self.state_dim))
        y_seq = self.y_seq = np.zeros((len(x_seq), self.output_dim))

        for t, xt in enumerate(x_seq):
            self.dh_dWhh_seq[t] = h  # previous h

            h_seq[t] = h = np.tanh(np.dot(self.Whh, h) + np.dot(self.Wxh, xt) + self.b)
            y_seq[t] = self.dense.forward(h)

            if gt_seq is not None:
                self.softmax.forward(y_seq[t], gt_seq[t])
                self.loss += self.softmax.loss

            self.dh_dWxh_seq[t] = xt

        return y_seq

    def backward(self, d):
        self.reset_ds()

        # TODO since this layer will always be the only layer, we know that d=1. so, we'll ignore it for now.

        dh = np.ones_like(self.db)

        for t in reversed(range(1, self.seq_len)):
            ht = self.h_seq[t]

            dy = self.softmax.backward(1) # TODO
            dy = self.dense.backward(dy)

            dh = dh * (np.ones_like(ht) - ht*ht) + dy

            self.db += dh
            self.dWhh += np.dot(np.array([dh]).T, [self.dh_dWhh_seq[t]])
            self.dWxh += np.dot(np.array([dh]).T, [self.dh_dWxh_seq[t]])

            dh = np.dot(self.Whh.T, dh)

        return None  # This layer is always the first layer of a network

    def get_grads(self):
        return [self.dWhh, self.dWxh, self.db, self.dense.dW, self.dense.db]

    def get_empty_grads(self):
        return [np.zeros_like(self.dWhh), np.zeros_like(self.dWxh), np.zeros_like(self.db),
                np.zeros_like(self.dense.dW), np.zeros_like(self.dense.db)]

    def update_params(self, delta_params):
        self.Whh += delta_params[0]
        self.Wxh += delta_params[1]
        self.b += delta_params[2]

        self.dense.update_params(delta_params[3:])

    def get_params_dict(self):
        return {'Whh': self.Whh, 'Wxh': self.Wxh, 'b': self.b, 'Why': self.dense.W, 'by': self.dense.b}

    def set_params_from_dict(self, params):
        self.Whh = params['Whh']
        self.Wxh = params['Wxh']
        self.b = params['b']

        self.dense.W = params['Why']
        self.dense.b = params['by']

    def get_config(self):
        return {'seq_len': self.seq_len,
                'input_dim': self.input_dim,
                'state_dim': self.state_dim,
                'output_dim': self.output_dim,
                'stateful': self.stateful}
