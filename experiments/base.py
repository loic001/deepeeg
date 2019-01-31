import uuid
import importlib
import logging

import os
import json
import json_tricks
import dill as pickle
import shelve
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
import seaborn as sns

from experiments.loaders import DatasetDefLoader
from experiments.trainers import P300DatasetTrainer
from experiments.testers import P300DatasetTester, P300DatasetTesterSpelling

from utils.db import ObjectSaver, get_or_load
import copy

import atexit
from shutil import copyfile

import glob

from experiments.render import build_spelling_test_figure, build_test_figure, build_train_figure

# from experiments.viz import Viz

class DataLogger(object):
    def __init__(self, name):
        self.id = uuid.uuid4().hex
        self.name = name

        self.datas = {}

    def merge_data(self, key, data):
        tomerge = self.datas.get(key, {})
        self.datas[key] = {**tomerge, **data}

    def json(self):
        return json_tricks.dumps(self.datas, sort_keys=True, indent=4)

    def __repr__(self):
        res = self.datas
        return str(res)

class DiskDataLogger(object):
    def __init__(self, db_filename, overwrite):
        self.id = uuid.uuid4().hex
        self.db_filename = db_filename
        self.logger = logging.getLogger(self.__class__.__name__)
        self.open(overwrite)

        atexit.register(self.close)

    def close(self):
        try:
            self.datas.close()
        except Exception:
            self.logger.error('error closing  %s', self.db_filename, exc_info=True)

    def open(self, overwrite):
        try:
            pathlib.Path(self.db_filename).mkdir(parents=True, exist_ok=True)
        except:
            pass
        _flag = 'n' if overwrite else 'c'
        try:
            self.datas = shelve.open(self.db_filename, flag=_flag, writeback=True)
        except Exception:
            self.logger.error('error opening  %s', self.db_filename, exc_info=True)

    def append_data(self, key, append_key, append_data):
        _d = self.datas.get(key, {})
        toappend = _d.get(append_key, [])
        toappend.append(append_data)
        _d[append_key] = toappend
        self.datas[key] = _d

    def merge_data(self, key, data):
        tomerge = self.datas.get(key, {})
        self.datas[key] = {**tomerge, **data}

    def __repr__(self):
        return str(self.datas)

class ExperimentLogger(DiskDataLogger):
    def __init__(self, db_filename, overwrite=True):
        super().__init__(db_filename, overwrite)

    @staticmethod
    def create(expe_dir, expe_name):
        experiment_logger_db = os.path.join(expe_dir, expe_name) + '/log'
        return ExperimentLogger(experiment_logger_db)

    def train_result(self, name, data):
        self.merge_data(name, {'train': data})

    def test_result(self, name, data):
        self.merge_data(name, {'test': data})

    def test_result_train(self, name, data):
        self.merge_data(name, {'test_train': data})

    def test_spelling_result(self, name, data):
        self.merge_data(name, {'test_spelling': data})

    def record_datasets(self, name, train_dataset, valid_dataset, test_dataset):
        self.merge_data(name, {'train_dataset': train_dataset,
                               'valid_dataset': valid_dataset, 'test_dataset': test_dataset})

    def record_eeg_dataset(self, name, dataset):
        assert isinstance(name, str)
        self.merge_data('eeg_datasets', {name: dataset})

    # def record_data_layer_logger(self, obj):
    #     # print(obj)
    #     # key = 'epoch_{}'.format(str(obj['epoch']))
    #     self.append_data(self.current_name, 'viz', obj)

    def render_results(self, out_pdf_dir=None):
        sns.set(context="paper", style="whitegrid", font_scale=0.6,
                rc={"lines.linewidth": 0.8, "axes.grid": True})
        fontsize = 7
        train_test_len = len(self.datas.items())
        figures = []

        name = self.db_filename.split(os.path.sep)[-1]
        # descriptionFig = plt.figure()
        # descriptionFig.axis('off')
        # descriptionFig.text('Test text')
        # total_map = {}
        # total_map['test_spelling'] = []
        # for index, (train_test, values) in enumerate(self.datas.items()):
        #     test_spelling_result = values.get('test_spelling', None)
        #     if test_spelling_result:
        #         grouped_by_item = test_spelling_result['group_by_item']
        #         group_by_item_df = pd.DataFrame(grouped_by_item)
        #         acc_arr = self._compute_spelling_test_acc(group_by_item_df)
        #         # acc_10 = [a for a in acc_arr if a['after_x_rep'] == 10][0]
        #         total_map['test_spelling'].append(acc_arr)

        for index, (train_test, values) in enumerate(self.datas.items()):
            title_prefix = '{}, {}'.format(name, train_test)

            train_result = values.get('train', None)
            if train_result:
                name_fig_train = '{} {}'.format('Train results', title_prefix)
                fig = build_train_figure(
                    name_fig_train, train_result, fontsize)
                if fig:
                    figures.append(fig)

            test_result = values.get('test', None)
            test_result_train = values.get('test_train', None)

            if test_result:
                name_fig_test = '{} {}'.format('Test', title_prefix)
                fig = build_test_figure(
                    name_fig_test, test_result, test_result_train, fontsize)
                if fig:
                    figures.append(fig)

            test_spelling_result = values.get('test_spelling', None)
            if test_spelling_result:
                name_fig_test_spelling = '{} {}'.format(
                    'Test spelling', title_prefix)
                fig = build_spelling_test_figure(
                    name_fig_test_spelling, test_spelling_result, fontsize)
                if fig:
                    figures.append(fig)

        if out_pdf_dir:
            out_pdf = os.path.join(
                out_pdf_dir, '{}_{}'.format(name, self.id))
            multipage(out_pdf, figures)
        else:
            plt.show()
            # test_result = values.get('test', None)
            # if test:
            #     self._build_test_figure(name='{} {}'.format('Test', title_prefix), test_result)


def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()


class ExpeRunner(object):
    def __init__(self, expe_config_module, render, suffix=0):
        self.expe_config_module = expe_config_module
        self.suffix = suffix
        self.render = render

        self.__init()

    def __init(self):
        self.expe_config = importlib.import_module(
            self.expe_config_module).expe_config
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self):
        name = self.expe_config.get('name', None)
        name = '{}_{}'.format(name, self.suffix)
        if not name:
            raise ValueError('Name must be defined')
        self.logger.info('Expe {} : started'.format(name))
        datasets_def = self.expe_config['datasets']
        Scenario = self.expe_config['scenario']
        scenario_params = self.expe_config.get('scenario_params', {})
        experiments_dir = self.expe_config['dir']
        samplers = self.expe_config.get('samplers', {})
        classifier = self.expe_config['classifier']
        expe_dir = os.path.join(experiments_dir, name)
        temp_dir = os.path.join(expe_dir, 'temp')
        experiment_logger_db = os.path.join(expe_dir, 'log')
        self.experiment_logger = self.expe_config.get('experiment_logger', ExperimentLogger(experiment_logger_db))

        # self.viz = Viz()

        saver_name = os.path.join(temp_dir, '{}_saver'.format(name))
        saver = ObjectSaver(saver_name)
        save_func = lambda obj, _id=None: saver.save(obj, _id)

        def load_func(_id): return saver.load(_id)

        cache_dir = self.expe_config.get('cache_dir', '/tmp')

        def subjects_data_dict_call(): return DatasetDefLoader(
            cache_dir).loads(datasets_def, db_save=True, db_save_func=save_func)

        train_test_data_dict = Scenario(self.experiment_logger, subjects_data_dict_call,
                                        db_load_func=load_func, db_save_func=save_func, params=scenario_params).run()
        TrainTestExecutor(self.experiment_logger, classifier, train_test_data_dict,
                          expe_dir, samplers, db_load_func=load_func).run()

        saver.remove_files()

        for k, v in self.experiment_logger.datas.items():
            print(v.keys())

        self.logger.info('Expe {} : terminated'.format(name))
        self.experiment_logger.close()

        if self.render:
            self.experiment_logger.open(overwrite=False)
            self.logger.info('Expe {} : rendering results...'.format(name))
            self.experiment_logger.render_results(out_pdf_dir=expe_dir)
            self.experiment_logger.close()

class TrainTestExecutor(object):
    def __init__(self, experiment_logger, classifier, train_test_data_dict, expe_dir, samplers, db_load_func=None):
        self.experiment_logger = experiment_logger
        self.classifier = classifier
        # self.skorch_net_params = skorch_net_params
        # self.dynamic_skorch_net_params = dynamic_skorch_net_params
        self.train_test_data_dict = train_test_data_dict
        self.expe_dir = expe_dir
        self.samplers = samplers
        self.db_load_func = db_load_func
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self):
        if not self.train_test_data_dict:
            raise ValueError(
                'train_test_data_dict must be defined and not empty')
        for name, data in self.train_test_data_dict.items():
            if not isinstance(data, dict):
                raise ValueError('data must be a dict')
            else:
                if not set(['train', 'test', 'valid']).issubset(set(data.keys())):
                    raise ValueError(
                        'data must be dict with train, test, valid dataset')

            data = {key: get_or_load(value, self.db_load_func)
                    for key, value in data.items()}

            if not any([item for _, item in data.items()]):
                self.logger.warning('%s skipped, db_load_func returned none', name)
                continue

            train_data = data['train']
            valid_data = data['valid']
            test_data = data['test']

            for _, sampler in enumerate(self.samplers.get('train', [])):
                train_data = sampler.sample(train_data)

            for _, sampler in enumerate(self.samplers.get('valid', [])):
                valid_data = sampler.sample(valid_data)

            for _, sampler in enumerate(self.samplers.get('test', [])):
                test_data = sampler.sample(test_data)

            train_data.shuffle()

            # logging.info('checking datasets')
            # duplicates = [P300Dataset.have_duplicates(t1, t2) for t1, t2 in itertools.combinations([train_data, valid_data, test_data], 2)]
            # if any(duplicates):
            #     logging.warn('duplicates epoch sample between datasets')
            # else:
            #     logging.info('datasets OK')

            self.logger.info(train_data)
            self.logger.info(valid_data)
            self.logger.info(test_data)

            net_saver_dir = os.path.join(self.expe_dir, name)
            injected_params = {'expe_name': name, 'expe_dir': copy.copy(self.expe_dir), 'train': train_data, 'valid': valid_data, 'test': test_data}
            classifier_ = self.classifier['type'](
                self.classifier['params'](injected_params), net_saver_dir)

            name = net_saver_dir.rstrip(os.path.sep).split('/')[-1]
            self.experiment_logger.record_datasets(
                name, train_data, valid_data, test_data)

            train_result = P300DatasetTrainer(
                classifier_, train_data, valid_data).run()
            if train_result:
                self.experiment_logger.train_result(name, train_result)
            test_result = P300DatasetTester(classifier_, test_data).run()
            if test_result:
                self.experiment_logger.test_result(name, test_result)

            test_result_on_train = P300DatasetTester(
                classifier_, train_data).run()
            if test_result_on_train:
                self.experiment_logger.test_result_train(
                    name, test_result_on_train)

            test_spelling_result = P300DatasetTesterSpelling(
                classifier_, test_data).run()
            if test_spelling_result:
                self.experiment_logger.test_spelling_result(
                    name, test_spelling_result)

        return True
