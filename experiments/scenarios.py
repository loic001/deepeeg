import os
import sys
import logging
import mne
import itertools
import importlib
import shelve
import copy

import numpy as np

from datasets.base import P300Dataset

from utils.db import get_or_load, get_or_load_dict

class Scenario(object):
    def __init__(self, experiment_logger, subjects_data_dict_call, db_load_func=None, db_save_func=None, params={}):
        self.experiment_logger = experiment_logger
        self.subjects_data_dict_call = subjects_data_dict_call
        self.db_load_func = db_load_func
        self.db_save_func = db_save_func
        self.params = params
        self.logger = logging.getLogger(__class__.__name__)

    def _build_check_params(self):
        self.subjects_data_dict = self.subjects_data_dict_call()
        if not self.subjects_data_dict:
            raise ValueError('subjects_data_dict is None or empty')
        if not isinstance(self.subjects_data_dict, dict):
            raise ValueError('subjects_data_dict should be a dict')
        return True

    def run(self):
        if self._build_check_params():
            return self._run()

    def _run(self):
        pass

class WithinSubjectScenario(Scenario):
    def _run(self):
        subjects_data_dict = {}
        for subject, data in self.subjects_data_dict.items():
            train_dataset, test_dataset = get_or_load_dict(data, self.db_load_func)
            test_dataset.name = 'test_{}'.format(subject)
            # train_test_split_frac = self.params.get('train_test_split_frac', 0.8)
            # train_valid_split_frac = self.params.get('train_valid_split_frac', 0.8)
            # self.logger.info('splitting train to train / test, frac: {}'.format(train_test_split_frac))
            valid_split_frac = self.params.get('valid_split_frac', 0.8)
            # dataset.shuffle(random_state=42)

            train_dataset, valid_dataset = train_dataset.split(valid_split_frac, split_names=('train_{}'.format(subject), 'valid_{}'.format(subject)))
            if not isinstance(train_dataset, P300Dataset): raise ValueError('data must be an P300Dataset instance')
            if not isinstance(test_dataset, P300Dataset): raise ValueError('test_dataset must be an P300Dataset instance')
            if not isinstance(valid_dataset, P300Dataset): raise ValueError('valid_dataset must be an P300Dataset instance')

            if not callable(self.db_save_func): raise ValueError('db_save_func must be a callable')
            subjects_data_dict[subject] = {'train': self.db_save_func(train_dataset), 'valid': self.db_save_func(valid_dataset), 'test': self.db_save_func(test_dataset)}
        return subjects_data_dict


class WithinSubjectScenarioTest(Scenario):
    def _run(self):
        subjects_data_dict = {}
        for subject, data in self.subjects_data_dict.items():
            train_dataset, test_valid_data = get_or_load_dict(data, self.db_load_func)
            train_dataset.name = 'train_{}'.format(subject)
            # train_test_split_frac = self.params.get('train_test_split_frac', 0.8)
            # train_valid_split_frac = self.params.get('train_valid_split_frac', 0.8)
            # self.logger.info('splitting train to train / test, frac: {}'.format(train_test_split_frac))
            valid_split_frac = self.params.get('valid_split_frac', 0.8)
            # dataset.shuffle(random_state=42)

            test_dataset, valid_dataset = test_valid_data.split(valid_split_frac, split_names=('test_{}'.format(subject), 'valid_{}'.format(subject)))

            if not isinstance(train_dataset, P300Dataset): raise ValueError('data must be an P300Dataset instance')
            if not isinstance(test_dataset, P300Dataset): raise ValueError('test_dataset must be an P300Dataset instance')
            if not isinstance(valid_dataset, P300Dataset): raise ValueError('valid_dataset must be an P300Dataset instance')

            if not callable(self.db_save_func): raise ValueError('db_save_func must be a callable')
            subjects_data_dict[subject] = {'train': self.db_save_func(train_dataset), 'valid': self.db_save_func(valid_dataset), 'test': self.db_save_func(test_dataset)}
        return subjects_data_dict

class TransferSubjectScenario(Scenario):
    def _run(self):
        train_test_data_dict = {}
        # self.subjects_data_dict = {key:value for key,value in self.subjects_data_dict.items() if value['X'].shape[1] == 63}
        for subject_key in self.subjects_data_dict.keys():

            test_dataset = [get_or_load(sd, self.db_load_func) for sk, sd in self.subjects_data_dict.items() if sk == subject_key][0]
            datas = [get_or_load(sd, self.db_load_func) for sk, sd in self.subjects_data_dict.items() if sk != subject_key]

            train_dataset = datas[0]
            for dataset in datas[1:-1]:
                train_dataset = P300Dataset.concat(train_dataset, dataset)

            valid_dataset = datas[-1]

            if not isinstance(train_dataset, P300Dataset): raise ValueError('data must be an P300Dataset instance')
            if not isinstance(test_dataset, P300Dataset): raise ValueError('test_dataset must be an P300Dataset instance')
            if not isinstance(valid_dataset, P300Dataset): raise ValueError('valid_dataset must be an P300Dataset instance')

            name = 'transfer_on_{}'.format(subject_key)
            train_test_data_dict[name] = {'train': self.db_save_func(train_dataset), 'valid': self.db_save_func(valid_dataset), 'test': self.db_save_func(test_dataset)}
        return train_test_data_dict
