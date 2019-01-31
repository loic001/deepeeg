import logging
import sys
import uuid

from skorch.callbacks import Callback
from skorch.utils import to_numpy
from skorch.exceptions import SkorchException

from itertools import cycle
from skorch.utils import Ansi
from numbers import Number
from tabulate import tabulate


class PrintLog(Callback):
    """Print out useful information from the model's history.
    By default, ``PrintLog`` prints everything from the history except
    for ``'batches'``.
    To determine the best loss, ``PrintLog`` looks for keys that end on
    ``'_best'`` and associates them with the corresponding loss. E.g.,
    ``'train_loss_best'`` will be matched with ``'train_loss'``. The
    ``Scoring`` callback takes care of creating those entries, which is
    why ``PrintLog`` works best in conjunction with that callback.
    *Note*: ``PrintLog`` will not result in good outputs if the number
    of columns varies between epochs, e.g. if the valid loss is only
    present on every other epoch.
    Parameters
    ----------
    keys_ignored : str or list of str (default='batches')
      Key or list of keys that should not be part of the printed
      table. Note that keys ending on '_best' are also ignored.
    sink : callable (default=print)
      The target that the output string is sent to. By default, the
      output is printed to stdout, but the sink could also be a
      logger, etc.
    tablefmt : str (default='simple')
      The format of the table. See the documentation of the ``tabulate``
      package for more detail. Can be 'plain', 'grid', 'pipe', 'html',
      'latex', among others.
    floatfmt : str (default='.4f')
      The number formatting. See the documentation of the ``tabulate``
      package for more details.
    """
    def __init__(
            self,
            keys_ignored='batches',
            keys_ignored_start_with='_',
            sink=print,
            tablefmt='simple',
            floatfmt='.4f',
    ):
        if isinstance(keys_ignored, str):
            keys_ignored = [keys_ignored]
        self.keys_ignored = keys_ignored
        self.keys_ignored_start_with = keys_ignored_start_with
        self.sink = sink
        self.tablefmt = tablefmt
        self.floatfmt = floatfmt

    def initialize(self):
        self.first_iteration_ = True
        return self

    def format_row(self, row, key, color):
        """For a given row from the table, format it (i.e. floating
        points and color if applicable.
        """
        value = row[key]
        if not isinstance(value, Number):
            return value

        # determine if integer value
        is_integer = float(value).is_integer()
        template = '{}' if is_integer else '{:' + self.floatfmt + '}'

        # if numeric, there could be a 'best' key
        key_best = key + '_best'
        if (key_best in row) and row[key_best]:
            template = color + template + Ansi.ENDC.value
        return template.format(value)

    def _sorted_keys(self, keys):
        """Sort keys alphabetically, but put 'epoch' first and 'dur'
        last.
        Ignore keys that are in ``self.ignored_keys`` or that end on
        '_best'.
        """
        sorted_keys = []
        if ('epoch' in keys) and ('epoch' not in self.keys_ignored):
            sorted_keys.append('epoch')

        for key in sorted(keys):
            if not (
                    (key in ('epoch', 'dur')) or
                    (key in self.keys_ignored) or
                    key.startswith(self.keys_ignored_start_with) or
                    key.endswith('_best')
            ):
                sorted_keys.append(key)

        if ('dur' in keys) and ('dur' not in self.keys_ignored):
            sorted_keys.append('dur')
        return sorted_keys

    def _yield_keys_formatted(self, row):
        colors = cycle([color.value for color in Ansi if color != color.ENDC])
        color = next(colors)
        for key in self._sorted_keys(row.keys()):
            formatted = self.format_row(row, key, color=color)
            yield key, formatted
            color = next(colors)

    def table(self, row):
        headers = []
        formatted = []
        for key, formatted_row in self._yield_keys_formatted(row):
            headers.append(key)
            formatted.append(formatted_row)

        return tabulate(
            [formatted],
            headers=headers,
            tablefmt=self.tablefmt,
            floatfmt=self.floatfmt,
        )

    def _sink(self, text, verbose):
        if (self.sink is not print) or verbose:
            self.sink(text)

    # pylint: disable=unused-argument
    def on_epoch_end(self, net, **kwargs):
        data = net.history[-1]
        verbose = net.verbose
        tabulated = self.table(data)

        if self.first_iteration_:
            header, lines = tabulated.split('\n', 2)[:2]
            self._sink(header, verbose)
            self._sink(lines, verbose)
            self.first_iteration_ = False

        self._sink(tabulated.rsplit('\n', 1)[-1], verbose)
        if self.sink is print:
            sys.stdout.flush()

class NetCheckpoint(Callback):
    def __init__(
        self,
        save_function,
        monitor='valid_loss_best'
    ):
        self.save_function = save_function
        self.target = '{type}_{last_epoch[epoch]}'
        self.monitor = monitor

    def on_epoch_end(self, net, **kwargs):
        if self.monitor is None:
            do_checkpoint = True
        elif callable(self.monitor):
            do_checkpoint = self.monitor(net)
        else:
            try:
                do_checkpoint = net.history[-1, self.monitor]
            except KeyError as e:
                raise SkorchException(
                    "Monitor value '{}' cannot be found in history. "
                    "Make sure you have validation data if you use "
                    "validation scores for checkpointing.".format(e.args[0]))
        if do_checkpoint:
            target = self.target
            if isinstance(self.target, str):
                target = self.target.format(
                    net=net,
                    last_epoch=net.history[-1],
                    last_batch=net.history[-1, 'batches', -1],
                    type=self.monitor
                )
            if net.verbose > 0:
                print("Checkpoint! : {}.".format(target))
            self.save_function(target)


class GetModuleOnTrainBegin(Callback):
    def __init__(self, callback_func):
        self.callback_func = callback_func

    def on_train_begin(self, net, **kwargs):
        print('callback_getmodule')
        model = net.module_
        self.callback_func(model)

class EpochSummary(Callback):
    def __init__(self, callback_func): # data={'name': 'data-test', 'X': [], 'y': []} or str
        self.callback_func = callback_func

    def on_epoch_end(self, net, **kwargs):
        data = net.history[-1]
        obj = {
            'epoch': len(net.history),
            'data': data
        }
        self.callback_func(obj)

class LayerDataViz(Callback):
    def __init__(self, layer_name, callback_func, monitor='valid_loss_best', data='valid', name=None): # data={'name': 'data-test', 'X': [], 'y': []} or str
        self.name = name
        if not self.name:
            self.name = str(uuid.uuid4())[:8]
        self.data_string = {
            'train': {'X': 'X', 'y': 'y'},
            'valid': {'X': 'X_valid', 'y': 'y_valid'}
        }
        self.layer_name = layer_name
        assert self.layer_name is not None
        self.monitor = monitor
        self.data = data

        self.callback_func = callback_func
        assert callable(callback_func)

        self.out_temp = None
        self.in_temp = None
        self.epoch_counter = -1

    def _forward_layer_hook(self, _self, _inputs, _output):
        _input, *_ = _inputs
        in_np = to_numpy(_input)
        out_np = to_numpy(_output)
        self.out_temp = out_np
        self.in_temp = in_np

    def _infer_log_layer(self, net, data):
        self.out_temp = None
        self.in_temp = None
        #infer will trigger the registered hook for the targeted layer
        X_data = data['X']
        net.infer(X_data)
        #get the result in self.in_temp and self.out_temp
        res = {'layer_input': self.in_temp, 'layer_output': self.out_temp}
        return res

    def _process_data(self, data, **kwargs):
        to_infer_data = None
        #string accepted 'train' or 'valid'
        if isinstance(data, str):
            _data_key = self.data_string.get(data, None)
            if _data_key:
                _X = kwargs.get(_data_key['X'], None)
                _y = kwargs.get(_data_key['y'], None)
                if _X is not None and _y is not None:
                    _d = {'name': data, 'X': _X, 'y': _y}
                    to_infer_data = _d
        return to_infer_data
# import numpy as np
# d = {'name': 'sfd', 'X': np.array([1,2]), 'y': np.array([1,2])}
#
# set(['name', 'X']).issubset(d)

    def on_epoch_end(
            self,
            net,
            **kwargs):
        if self.monitor is None:
            do_checkpoint = True
        elif callable(self.monitor):
            do_checkpoint = self.monitor(net)
        else:
            try:
                do_checkpoint = net.history[-1, self.monitor]
            except KeyError as e:
                raise SkorchException(
                    "Monitor value '{}' cannot be found in history. "
                    "Make sure you have validation data if you use "
                    "validation scores for checkpointing.".format(e.args[0]))
        if do_checkpoint:
            model = net.module_
            selected_layer = None
            for name, module in model.named_modules():
                if name == self.layer_name:
                    selected_layer = module
                    break

            # do nothing if layer does not exist
            if not selected_layer:
                return

            handle = selected_layer.register_forward_hook(
                self._forward_layer_hook)
            to_infer_data = self._process_data(self.data, **kwargs)
            input_output = self._infer_log_layer(net, to_infer_data)
            handle.remove()

            record_obj = {
                'name': self.name,
                'layer_name': self.layer_name,
                'epoch': len(net.history),
                'data': to_infer_data,
                'input_output': input_output
            }
            self.callback_func(record_obj)

# tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=1500)
# print('fitting tsne')
# low_dim_embs = tsne.fit_transform(self.out_temp[:1000])
# plot_with_labels(low_dim_embs, y_valid[:1000])

# from sklearn.manifold import TSNE

#
#
# class TSNEViewer(Callback):
#
#     def __init__(self, layer_name):
#         self.layer_name = layer_name
#         assert self.layer_name is not None
#         self.out_temp = None
#         self.epoch_counter = -1
#
#     def _forward_layer_hook(self, _self, _inputs, _output):
#         # _input, _ = _inputs
#         # in_np = to_numpy(_input)
#         out_np = to_numpy(_output)
#         self.out_temp = out_np
#
#     def _check_data(self, data):
#         return bool(data
#
#     def on_epoch_end(
#             self,
#             net,
#             X,
#             y,
#             X_valid,
#             y_valid,
#             **kwargs):
#         if self.data is None:
#             return
#         self.epoch_counter = self.epoch_counter + 1
#         if self.epoch_counter % 10 != 0:
#             return
#         model = net.module_
#         selected_layer = None
#         for name, module in model.named_modules():
#             if name == self.layer_name:
#                 selected_layer = module
#                 break
#
#         # do nothing if layer does not exist
#         if not selected_layer:
#             return
#
#         handle = selected_layer.register_forward_hook(self._forward_layer_hook)
#         net.infer(X_valid)
#         handle.remove()
#
#         if self.out_temp is not None:
#             tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=1500)
#             print('fitting tsne')
#             low_dim_embs = tsne.fit_transform(self.out_temp[:1000])
#             plot_with_labels(low_dim_embs, y_valid[:1000])
#
#         # register_forward_hook(self._layer_hook)
#         # y_pred = net.infer(X_valid, tsne_params={})
#         # dataset_valid = net.get_dataset(X_valid, y_valid)
#         # for Xi, yi in net.get_iterator(dataset_valid, training=False):
#         #     y_pred = net.infer(Xi)
#         #     print(y_pred)


import matplotlib.pyplot as plt
from matplotlib import cm


def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9))
        plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.title('Visualize last layer')
    plt.show()
