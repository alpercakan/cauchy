# classifier.py
# Author: Alper Çakan

import numpy as np
import json
import math


class Sequential:
    DEFAULT_LR = 0.001
    DEFAULT_GAMMA = 0.9
    EPS = 1e-8

    def __init__(self):
        self.lr = Sequential.DEFAULT_LR
        self.gamma = Sequential.DEFAULT_GAMMA
        self.metric = None
        self.compiled = False

        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def _check_compiled(self):
        if not self.compiled:
            raise Exception("Model not yet compiled")

    def fit(self, x_train, y_train, epochs=1, batch_size=1, after_epoch=None, shuffle=True, log_count=5,
            x_val=None, y_val=None):
        self._check_compiled()

        sample_count = len(x_train)
        batch_count = math.ceil(float(sample_count) / batch_size)

        last_layer = self.layers[-1]
        trainable_layers = [layer for layer in self.layers if layer.trainable]

        history = []

        for epoch in range(epochs):
            history.append({})

            shuffled_indices = list(range(sample_count))

            if shuffle:
                np.random.shuffle(shuffled_indices)

            loss = 0

            # RMSProp. TODO: other optimizers
            normalizers = [[layer.get_empty_grads() for layer in trainable_layers]
                           for _ in range(batch_count)]

            for batch_index in range(batch_count):
                # Print in-batch progress log_count times in a single epoch
                if batch_index % max(int(batch_count / log_count), 1) == 0:
                    print('Started batch {} of {} (epoch {})'.format(batch_index, batch_count, epoch))

                batch_start = batch_index * batch_size
                batch_indices = shuffled_indices[batch_start:(batch_start + batch_size)]

                xs = x_train[batch_indices]
                ys = y_train[batch_indices]

                grads = [layer.get_empty_grads() for layer in trainable_layers]

                for i in range(len(xs)):
                    gt = ys[i]
                    self._forward(xs[i], gt)

                    self._backward()

                    for li, layer in enumerate(trainable_layers):
                        layer_grads = layer.get_grads()

                        for pi in range(len(grads[li])):
                            grads[li][pi] += np.clip(layer_grads[pi] / batch_size, -5, 5)

                    loss += last_layer.loss

                for li, layer in enumerate(trainable_layers):
                    deltas = layer.get_empty_grads()

                    for pi in range(len(grads[li])):
                        # RMSProp. TODO: other optimizers
                        normalizer = normalizers[batch_index][li][pi] = \
                            ((self.gamma * normalizers[batch_index - 1][li][pi]) if batch_index > 0 else 0) +\
                            (1 - self.gamma) * (grads[li][pi] ** 2)
                        deltas[pi] = -(grads[li][pi] * self.lr) / np.sqrt(normalizer + self.EPS)

                    layer.update_params(deltas)

            loss /= sample_count
            print('Loss of epoch {} is {}'.format(epoch, loss))

            eval_res = self.evaluate(x_train, y_train)
            metric_key = 'train_{}'.format(self.metric)
            print('{} = {}'.format(metric_key, eval_res))
            history[-1][metric_key] = eval_res

            if x_val is not None:
                eval_res = self.evaluate(x_val, y_val)
                metric_key = 'val_{}'.format(self.metric)
                print('{} = {}'.format(metric_key, eval_res))
                history[-1][metric_key] = eval_res

            print()

            if after_epoch is not None:
                after_epoch(self, epoch)

            for li, layer in enumerate(trainable_layers):
                if layer.stateful:
                    layer.reset_state()

        return history

    def compile(self, learning_rate=DEFAULT_LR, metric=None):  # TODO implement gamma rate etc
        self.lr = learning_rate
        self.metric = metric

        self.compiled = True

    def _forward(self, x, gt=None):
        output = x

        for layer in self.layers[:-1]:
            output = layer.forward(output)

        output = self.layers[-1].forward(output, gt)

        return output

    def _backward(self):
        d = 1

        for layer in reversed(self.layers):
            d = layer.backward(d)

    def predict(self, xs):
        self._check_compiled()

        xs = np.array(xs)

        return [self._forward(x) for x in xs]

    def evaluate(self, xs, gts):
        self._check_compiled()

        ys = self.predict(xs)

        return self._measure_metric(xs,  ys, gts)

    def _measure_metric(self, xs, ys, gts):
        measure_fn_by_metric = {
            'accuracy': measure_accuracy,
            'seq_accuracy': measure_seq_accuracy,
            # TODO implement other metrics
        }

        return measure_fn_by_metric[self.metric](xs, ys, gts)

    def save(self, name):
        layer_types = [str(type(l)) for l in self.layers]
        layer_configs = [l.get_config() for l in self.layers]

        params_dict = {}

        for i, layer in enumerate(self.layers):
            if not layer.trainable:
                continue

            for param_name, param_val in layer.get_params_dict().items():
                params_dict['layer_{}__{}'.format(i, param_name)] = param_val

        np.savez_compressed(name, **params_dict)

        # TODO implement saving model params such as the learning rate

        with open('{}_network.json'.format(name), 'w') as outfile:
            json.dump({'layer_types': layer_types, 'layer_configs': layer_configs}, outfile)


def measure_accuracy(xs, ys, gts):
    true_count = 0

    for i in range(len(xs)):
        if np.argmax(ys[i]) == np.argmax(gts[i]):
            true_count += 1

    return true_count / len(xs)


def measure_seq_accuracy(xs, ys, gts):
    true_count = 0

    for i in range(len(xs)):
        for t in range(len(xs[i])):
            if np.argmax(ys[i][t]) == np.argmax(gts[i][t]):
                true_count += 1

    return true_count / (len(xs) * len(xs[0]))

