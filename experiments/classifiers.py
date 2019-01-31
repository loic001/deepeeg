import logging
import os

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from skorch_ext.netsaver import NetSaver
from skorch_ext.utils import SliceDict

from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace

from mne.decoding import CSP


class Classifier(object):
    def __init__(self, params):
        self.params = params

    def fit(self, X_train_dict, y_train, X_valid_dict, y_valid, **kwargs):
        assert isinstance(X_train_dict, dict) and isinstance(
            X_valid_dict, dict)
        return self._fit(X_train_dict, y_train, X_valid_dict, y_valid, **kwargs)

    def _fit(self, X_train_dict, y_train, X_valid_dict, y_valid, **kwargs):
        raise NotImplementedError("Subclasses should implement this!")

    def _predict(self, X_test_dict, **kwargs):
        raise NotImplementedError("Subclasses should implement this!")

    def predict(self, X_test_dict, proba=False, **kwargs):
        assert isinstance(X_test_dict, dict)
        return self._predict_proba(X_test_dict, **kwargs) if proba else self._predict(X_test_dict, **kwargs)


class SkorchClassifier(Classifier):
    def __init__(self, params, net_saver_dir):
        super(SkorchClassifier, self).__init__(params)
        self.net_saver_dir = net_saver_dir
        self.net_saver = NetSaver(net_saver_dir=self.net_saver_dir)
        self.net_saver.init(skorch_net_params=self.params)

    # def _init(self, checkpoint=None):
    #     if checkpoint:
    #         self.net_saver.init(checkpoint=checkpoint)
    #     else:
    #         self.net_saver.init(skorch_net_params=self.params)
    #     self._net = self.net_saver.net()

    def updated_skorch_net_params(self, skorch_net_params):
        self.net_saver.updated_skorch_net_params(skorch_net_params)

    def _fit(self, X_train_dict, y_train, X_valid_dict, y_valid, **kwargs):
        X_train_dict = {key: value[:, :, :, None]
                        for key, value in X_train_dict.items()}
        X_valid_dict = {key: value[:, :, :, None]
                        for key, value in X_valid_dict.items()}

        X_train, y_train, X_valid, y_valid = SliceDict(
            **X_train_dict), y_train, SliceDict(**X_valid_dict), y_valid

        self.net_saver.net().fit(X_train, y_train, X_valid=X_valid, y_valid=y_valid)
        name = self.net_saver_dir.rstrip(os.path.sep).split('/')[-1]
        train_result = {'net_history': self.net_saver.net().history}
        return train_result

    def _predict(self, X_test_dict, **kwargs):
        force_reload = kwargs.get('force_reload', True)
        if force_reload:
            print('trying to reload')
            load_criterion = kwargs.get('load_criterion', 'valid_loss_best')
            self.net_saver.init(criterion=load_criterion, session='keep_current', show_log=True)

        X_test_dict = {key: value[:, :, :, None]
                       for key, value in X_test_dict.items()}
        X_test = SliceDict(**X_test_dict)
        return self.net_saver.net().predict(X_test)

    def _predict_proba(self, X_test_dict, **kwargs):
        force_reload = kwargs.get('force_reload', True)
        if force_reload:
            print('trying to reload')
            load_criterion = kwargs.get('load_criterion', 'valid_loss_best')
            self.net_saver.init(criterion=load_criterion, session='keep_current', show_log=True)

        X_test_dict = {key: value[:, :, :, None]
                       for key, value in X_test_dict.items()}
        X_test = SliceDict(**X_test_dict)
        return self.net_saver.net().predict_proba(X_test)


class CSPSVCClassifier(Classifier):
    def __init__(self, params, net_saver_dir):
        super(CSPSVCClassifier, self).__init__(params)
        self._init()

    def _init(self):
        self.n_components = self.params.get('n_components', 4)
        self.svc = SVC(C=1, kernel='linear', probability=True)
        self.csp = CSP(n_components=self.n_components, norm_trace=False)

    def _fit(self, X_train_dict, y_train, X_valid_dict, y_valid, **kwargs):
        X_train = X_train_dict['x']
        X_train_transformed = self.csp.fit_transform(X_train, y_train)
        self.svc.fit(X_train_transformed, y_train)
        return {'csp_filters': self.csp.patterns_}

    def _predict(self, X_test_dict, **kwargs):
        X_test = X_test_dict['x']
        X_test_transformed = self.csp.transform(X_test)
        return self.svc.predict(X_test_transformed)

    def _predict_proba(self, X_test_dict, **kwargs):
        X_test = X_test_dict['x']
        X_test_transformed = self.csp.transform(X_test)
        return self.svc.predict_proba(X_test_transformed)

class RiemannClassifier(Classifier):
    def __init__(self, params, net_saver_dir):
        super(RiemannClassifier, self).__init__(params)
        self._init()

    def _init(self):
        self.n_components = self.params.get('n_components', 3)
        self.pipeline = make_pipeline(XdawnCovariances(
            self.n_components), TangentSpace(metric='riemann'), LogisticRegression())

    def _fit(self, X_train_dict, y_train, X_valid_dict, y_valid, **kwargs):
        X_train = X_train_dict['x']
        self.pipeline.fit(X_train, y_train)
        return {}

    def _predict(self, X_test_dict, **kwargs):
        X_test = X_test_dict['x']
        return self.pipeline.predict(X_test)

    def _predict_proba(self, X_test_dict, **kwargs):
        X_test = X_test_dict['x']
        return self.pipeline.predict_proba(X_test)
