import logging
import torch

import skorch
from skorch.net import NeuralNet
from skorch.utils import to_var
from skorch.utils import to_numpy
from skorch.utils import duplicate_items
from skorch.callbacks import EpochTimer
from skorch.callbacks import EpochScoring
from skorch.callbacks import ProgressBar

from skorch_ext.callbacks import PrintLog


from skorch.dataset import get_len

# from .utils import hash_string
import numpy as np
import traceback

from skorch.net import train_loss_score, valid_loss_score

# skorch_logger = logging.getLogger('skorch_train')
# hdlr = logging.FileHandler('/tmp/myapp.log')
# skorch_logger.addHandler(hdlr)

#pickle raise an error if logger is not wrapped into an accessible module method
def logger_info(msg):
    skorch_logger = logging.getLogger('skorch_train')
    skorch_logger.info(msg)

class Classifier(NeuralNet):
    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.temptest = None

    def get_loss(self, y_pred, y_true, X=None, training=False):
        """Return the loss for this batch.
        Parameters
        ----------
        y_pred : torch tensor
          Predicted target values
        y_true : torch tensor
          True target values.
        X : input data, compatible with skorch.dataset.Dataset
          By default, you should be able to pass:
            * numpy arrays
            * torch tensors
            * pandas DataFrame or Series
            * a dictionary of the former three
            * a list/tuple of the former three
          If this doesn't work with your data, you have to pass a
          ``Dataset`` that can deal with the data.
        train : bool (default=False)
          Whether train mode should be used or not.
        """
        y_true = to_var(y_true, use_cuda=self.use_cuda)

        return self.criterion_(y_pred, y_true[:, 0])

    def get_default_callbacks(self):
        return [
            ('epoch_timer', EpochTimer()),
            ('train_loss', EpochScoring(
                train_loss_score,
                name='train_loss',
                on_train=True,
            )),
            ('valid_loss', EpochScoring(
                valid_loss_score,
                name='valid_loss',
            )),
            ('valid_acc', EpochScoring(
                'accuracy',
                name='valid_acc',
                lower_is_better=False,
            )),
            ('print_log', PrintLog(sink=logger_info, keys_ignored=['batches', ])),
            ('auc', EpochScoring(scoring='roc_auc', lower_is_better=False))
        ]

    def predict_proba(self, X):
        """Where applicable, return probability estimates for
        samples.
        If the module's forward method returns multiple outputs as a
        tuple, it is assumed that the first output contains the
        relevant information. The other values are ignored.
        Parameters
        ----------
        X : input data, compatible with skorch.dataset.Dataset
          By default, you should be able to pass:
            * numpy arrays
            * torch tensors
            * pandas DataFrame or Series
            * a dictionary of the former three
            * a list/tuple of the former three
          If this doesn't work with your data, you have to pass a
          ``Dataset`` that can deal with the data.
        Returns
        -------
        y_proba : numpy ndarray
        """
        # Only the docstring changed from parent.
        # pylint: disable=useless-super-delegation
        log_proba_pred = super().predict_proba(X)
        proba_pred = np.exp(log_proba_pred)
        return proba_pred

    def predict(self, X):
        """Where applicable, return class labels for samples in X.
        If the module's forward method returns multiple outputs as a
        tuple, it is assumed that the first output contains the
        relevant information. The other values are ignored.
        Parameters
        ----------
        X : input data, compatible with skorch.dataset.Dataset
          By default, you should be able to pass:
            * numpy arrays
            * torch tensors
            * pandas DataFrame or Series
            * a dictionary of the former three
            * a list/tuple of the former three
          If this doesn't work with your data, you have to pass a
          ``Dataset`` that can deal with the data.
        Returns
        -------
        y_pred : numpy ndarray
        """
        y_preds = []
        for yp in self.forward_iter(X, training=False):
            yp = yp[0] if isinstance(yp, tuple) else yp
            y_preds.append(to_numpy(yp.max(-1)[-1]))
        y_pred = np.concatenate(y_preds, 0)
        return y_pred


    def infer(self, x, **fit_params):
        """Perform a single inference step on a batch of data.

        Parameters
        ----------
        x : input data
          A batch of the input data.

        **fit_params : dict
          Additional parameters passed to the ``forward`` method of
          the module and to the train_split call.

        """
        x = to_var(x, use_cuda=self.use_cuda)
        if isinstance(x, dict):
            x_dict = self._merge_x_and_fit_params(x, fit_params)
            return self.module_(**x_dict)
        return self.module_(x, **fit_params)

import hashlib
def hashDict(d):
    return hashlib.sha256(str(d).encode('utf-8')).hexdigest()

def intro(n, name):
    print('{} nan: {}, inf: {}'.format(name, np.isnan(n).mean(), np.isinf(n).mean()))
