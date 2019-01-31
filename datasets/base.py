import numpy as np
import mne

import sklearn
import imblearn
from collections import Counter

import logging

class EEGDataset(object):
    def get_name(self):
        raise NotImplementedError()
        #should return the name of the dataset

    def subjects(self):
        raise NotImplementedError()
        #should return a list of available subjects in dataset (string name)

    def get_subject_data(self):
        raise NotImplementedError()
        #should return mne raw array for this subject

class P300Dataset(object):
    def __init__(self, X, y, y_stim, y_names, y_stim_names, info, name='no_name'):
        assert len(X) == len(y) == len(y_stim)
        self.name = name
        self.info = info
        self.X = X
        self.y = y
        self.y_stim = y_stim
        self.y_names = y_names
        self.y_stim_names = y_stim_names

    def __repr__(self):
        return '{} containing :\nCounter: {}\nX: {}\ny: {}\ny_names: {}\ny_stim: {}\ny_stim_names: {}\ninfo: {}\n'.format(self.name, dict(Counter(self.y)), self.X.shape, self.y.shape, self.y_names, self.y_stim.shape, self.y_stim_names, self.info)

    def split(self, frac=0.8, split_names=(None, None)):
        split1_name = split_names[0] if split_names[0] else 'split1_from_{}'.format(self.name)
        split2_name = split_names[1] if split_names[1] else 'split2_from_{}'.format(self.name)
        length = self.X.shape[0]
        split_index = int(length * frac)
        split1 = P300Dataset(self.X[:split_index], self.y[:split_index], self.y_stim[:split_index], self.y_names, self.y_stim_names, self.info, split1_name)
        split2 = P300Dataset(self.X[split_index:], self.y[split_index:], self.y_stim[split_index:], self.y_names, self.y_stim_names, self.info, split2_name)
        return split1, split2

    def select(self, indices, subset_name=None):
        subset_name = subset_name if subset_name else 'subset_from_{}'.format(self.name)
        return P300Dataset(self.X[indices], self.y[indices], self.y_stim[indices], self.y_names, self.y_stim_names, self.info, subset_name)

    def split_by_indices(self, indices, split_names=(None, None)):
        split1_name = split_names[0] if split_names[0] else 'split1_from_{}'.format(self.name)
        split2_name = split_names[1] if split_names[1] else 'split2_from_{}'.format(self.name)
        indices = list(set(list(indices)))
        length = self.X.shape[0]
        indices_complement = [index for index in range(length) if index not in indices]
        return self.select(indices, subset_name=split1_name), self.select(indices_complement, subset_name=split2_name)

    def _duplicate_epoch(self, epoch_index, repeat=1):
        stacked_X = np.array([self.X[epoch_index],]*repeat)
        stacked_y = np.array([self.y[epoch_index],]*repeat)
        stacked_y_stim = np.array([self.y_stim[epoch_index],]*repeat)
        self.X = np.vstack((self.X, stacked_X))
        self.y = np.concatenate((self.y, stacked_y), axis=0)
        self.y_stim = np.concatenate((self.y_stim, stacked_y_stim), axis=0)
        return self

    def shuffle(self, random_state=None):
        self.X, self.y, self.y_stim = sklearn.utils.shuffle(self.X, self.y, self.y_stim, random_state=random_state)
        return self

    def clone(self):
        return P300Dataset(self.X, self.y, self.y_stim, self.y_names, self.y_stim_names, self.info, self.name)

    def sample(self, sampler, shuffle=True): #http://contrib.scikit-learn.org/imbalanced-learn/stable/api.html#module-imblearn.under_sampling
        assert isinstance(sampler, imblearn.base.BaseSampler)
        first_dim, second_dim, third_dim = self.X.shape
        X_reshaped = self.X.reshape(first_dim, second_dim*third_dim)
        ly = len(self.y)
        index_hash = {hashDict(item):index for index, item in enumerate(X_reshaped)}
        X_resampled, y_resampled, *selected_indices = sampler.fit_sample(X_reshaped, self.y)
        if X_resampled.shape[0] > X_reshaped.shape[0]:
            index_hash_resampled = {index:hashDict(item) for index, item in enumerate(X_resampled)}

            print(X_reshaped.shape)
            print(X_resampled.shape)
            sampled = self.clone()
            for h, index in index_hash.items():
                c = list(index_hash_resampled.values()).count(h)
                if c > 1:
                    sampled._duplicate_epoch(index, repeat=c-1)
            if shuffle: sampled.shuffle()
            return sampled
        else:
            if not selected_indices: raise ValueError('sampler must return indices to apply sampler')
            split1, split2 = self.split_by_indices(selected_indices[0], split_names=('sampled_from_{}'.format(self.name), 'excluded_from_{}'.format(self.name)))
            if shuffle:
                split1.shuffle()
                split2.shuffle()
            return split1, split2

    # def resample_up(self):
    #     non_target = np.where(self.y == 1)[0]
    #     target = np.where(self.y == 1)[1]
    #     counter = dict(Counter(self.y))
    #     majority_class_nb = max(list(counter.values()))
    #     logging.info('resampling to majority class {}'.format(majority_class_nb))
    #     self.X, self.y, self.y_stim = sklearn.utils.resample(self.X, self.y, self.y_stim, n_samples=majority_class_nb, replace=True)
    #     return self
    #
    # def resample_down(self):
    #     counter = dict(Counter(self.y))
    #     minority_class_nb = min(list(counter.values()))
    #     logging.info('resampling to minority class {}'.format(minority_class_nb))
    #     self.X, self.y, self.y_stim = sklearn.utils.resample(self.X, self.y, self.y_stim, n_samples=minority_class_nb, replace=False)
    #     return self
    #
    # def resample(self, up=True):
    #     target_indices = target = np.where(self.y == 1)[1]
    #     target, non_target = a.split_by_indices(target_indices, split_names=('target', 'non-target'))


    @staticmethod
    def resample(a, up=True):
        counter = dict(Counter(a.y))
        class_nb = max(list(counter.values())) if up else min(list(counter.values()))
        target_indices_majority = np.where(a.y == max(counter, key=counter.get))[0]

        majority, minority = a.split_by_indices(target_indices_majority, split_names=('target', 'non-target'))

        resampled = minority if up else majority
        to_concat = majority if up else minority
        resampled.X, resampled.y, resampled.y_stim = sklearn.utils.resample(resampled.X, resampled.y, resampled.y_stim, n_samples=class_nb, replace=up)

        concatenated = P300Dataset.concat(resampled, to_concat)
        concatenated.name = a.name
        return concatenated
    @staticmethod
    def concat(a, b):
        X = np.concatenate((a.X, b.X), axis=0)
        y = np.concatenate((a.y, b.y), axis=0)
        y_stim = np.concatenate((a.y_stim, b.y_stim), axis=0)
        concatenated = a.clone()
        concatenated.X = X
        concatenated.y = y
        concatenated.y_stim = y_stim
        return concatenated

    @staticmethod
    def have_duplicates(a, b):
        assert isinstance(a, P300Dataset) and isinstance(b, P300Dataset)
        a_hash = [hashDict(row) for row in a.X]
        b_hash = [hashDict(row) for row in b.X]
        a_in_b = [h for h in a_hash if h in b_hash]
        b_in_a = [h for h in b_hash if h in a_hash]
        return bool(a_in_b) or bool(b_in_a)

    @staticmethod
    def from_mne_epoched(mne_epoched):
        assert isinstance(mne_epoched, mne.Epochs)
        X = (mne_epoched.get_data() * 1e6).astype(np.float32)
        y = (mne_epoched.events[:,2]).astype(np.int64)
        y_names = mne_epoched.info['y_names']
        y_stim = mne_epoched.info['y_stim']
        y_stim_names = mne_epoched.info['y_stim_names']
        info = {'nb_iterations': mne_epoched.info['nb_iterations'], 'nb_items': mne_epoched.info['nb_items'], 'sfreq': mne_epoched.info['sfreq']}
        return P300Dataset(X, y, y_stim, y_names, y_stim_names, info, mne_epoched.info['name'])


import hashlib
def hashDict(d):
    return hashlib.sha256(str(d).encode('utf-8')).hexdigest()
