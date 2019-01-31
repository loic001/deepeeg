
import sys
# sys.path.append('/home/lolo/projects/torchexp')
sys.path.append('/home/lolo/projects/crnl/deepeeg')

from skorch_ext.expe import Expe

import torchexp as te
import numpy as np
import torch

import datasets.Rsvp as rsvp

from models.shallow_fbcsp import ShallowFBCSP

import itertools
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter

from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek

from sklearn.model_selection import train_test_split

import shelve


def launch(subject_id):
    epochedDataOneSubject = rsvp.getRsvpDatasetForSubject(subject_id).toMneEpoched(tmin=0.0, tmax=0.5)

    def toNumpyXY(epochedData):
        X = (epochedData.get_data() * 1e6).astype(np.float32)
        y = (epochedData.events[:,2] - 1).astype(np.int64)
        return X,y

    X,y = toNumpyXY(epochedDataOneSubject)

    # first_dim, second_dim, third_dim = X.shape
    # X_reshaped = X.reshape(first_dim, second_dim*third_dim)
    # X_reshaped.shape
    #
    # print(Counter(y))
    # rus = NearMiss(random_state=42)
    # X_resampled, y_resampled = rus.fit_sample(X_reshaped, y)
    # X_resampled.shape
    # X = X_resampled.reshape(X_resampled.shape[0], second_dim, third_dim)
    # print(Counter(y_resampled))
    #
    # y = y_resampled

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=54)


    X_train = X_train[:,:,:,None]
    x_dict_train = {
        'x': X_train
    }

    X_test = X_test[:,:,:,None]
    x_dict_test = {
        'x': X_test
    }

    with shelve.open('datacheck_nonearmiss_'+subject_id) as db:
        try:
            x_dict_train=db['x_dict_train']
        except KeyError:
            db['x_dict_train']=x_dict_train
        try:
            x_dict_test=db['x_dict_test']
        except KeyError:
            db['x_dict_test']=x_dict_test
        try:
            y_test=db['y_test']
        except KeyError:
            db['y_test']=y_test
        try:
            y_train=db['y_train']
        except KeyError:
            db['y_train']=y_train


    data = x_dict_train['x']

    data = data[:, :, :, 0]
    print(data.shape)


    max_ = np.max(data)
    min_ = np.min(data)

    data_r = data.reshape((data.shape[0], data.shape[1]*data.shape[2]))
    m_over_chan = data_r.mean(axis=0)
    m_over_time = m_over_chan.mean()

    a = np.array([True, False, True])



    print('subject: {}, max value: {}, min value: {}, mean: {}, m: {}, nan: {}, inf: {}'.format(subject_id, max_, min_, m_over_time, data.mean(), np.isnan(data).mean(), np.isinf(data).mean()))

if __name__ == '__main__':
    subjects = ['VPgcc', 'VPfat', 'VPgcb', 'VPgce', 'VPgcg', 'VPgch', 'VPiay', 'VPicn', 'VPicr', 'VPpia'] #, 'VPgcc'
    [launch(subject) for subject in subjects]
