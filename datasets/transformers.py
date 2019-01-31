import logging
from collections import Counter
from sklearn.utils import shuffle as sickit_shuffle


class Transformer(object):
    def __init__(self):
        self.logger = logging.getLogger(__class__.__name__)

    def transform(self, X, y, shuffle=True):
        self.logger.info('sampling data, counter: {}'.format(dict(Counter(y))))
        X, y, indices = self._transform(X, y)
        self.logger.info('resampled to, shape: {}'.format(dict(Counter(y))))
        if shuffle:
            X, y = sickit_shuffle(X, y)
        return X, y, indices

    def _transform(self, X, y):
        return X, y, indices


class SamplingTransformer(Transformer):
    def __init__(self, sampler):
        super().__init__()
        self.sampler = sampler

    def _transform(self, X, y):
        print(X.shape)
        first_dim, second_dim, third_dim = X.shape
        X_reshaped = X.reshape(first_dim, second_dim * third_dim)
        sampled_res = self.sampler.fit_sample(X_reshaped, y)
        X_resampled, y_resampled, *selected_index = sampled_res
        selected_index = selected_index[0] if selected_index else None
        X = X_resampled.reshape(X_resampled.shape[0], second_dim, third_dim)
        y = y_resampled
        return X, y, selected_index
