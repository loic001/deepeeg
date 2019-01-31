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

import atexit
from shutil import copyfile

import glob

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
        self.overwrite = overwrite
        self.db_filename = db_filename
        try:
            pathlib.Path(self.db_filename).mkdir(parents=True, exist_ok=True)
        except:
            pass
        self.open()

        if overwrite:
            self.datas.clear()
        atexit.register(self.close)

    def close(self):
        try:
            self.datas.close()
        except:
            pass

    def open(self):
        _flag = 'n' if self.overwrite else 'c'
        try:
            self.datas = shelve.open(self.db_filename, flag=_flag, writeback=True)
        except:
            pass

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
        self.current_name = None

    def next(name):
        self.current_name = name

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

    def record_data_layer_logger(self, obj):
        # print(obj)
        # key = 'epoch_{}'.format(str(obj['epoch']))
        print(self.current_name)
        self.append_data(self.current_name, 'viz', obj)

    def _build_train_figure(self, name, train_result, fontsize):
        fig = plt.figure()
        fig.suptitle(name, fontsize=fontsize)

        train_grid = gridspec.GridSpec(2, 4)
        history = train_result.get('net_history', None)
        if history:
            print(np.arange(1, len(history)+1))
            train_loss = history[:, 'train_loss'], np.arange(1, len(history)+1)
            valid_loss = history[:, 'valid_loss'], np.arange(1, len(history)+1)
            roc_auc = history[:, 'roc_auc'], np.arange(1, len(history)+1)
            valid_acc = history[:, 'valid_acc'], np.arange(1, len(history)+1)

            valid_loss_np = np.array(valid_loss)
            best_epoch = np.argmin(valid_loss_np) + 1

            train_loss_until_best = train_loss[0][:best_epoch], np.arange(1, best_epoch+1)
            valid_loss_until_best = valid_loss[0][:best_epoch], np.arange(1, best_epoch+1)
            valid_acc_until_best = valid_acc[0][:best_epoch], np.arange(1, best_epoch+1)
            roc_auc_until_best = roc_auc[0][:best_epoch], np.arange(1, best_epoch+1)

            train_loss_ax = fig.add_subplot(train_grid[0, 0])
            valid_loss_ax = fig.add_subplot(train_grid[0, 1])
            roc_auc_ax = fig.add_subplot(train_grid[0, 2])
            valid_acc_ax = fig.add_subplot(train_grid[0, 3])

            train_loss_until_best_ax = fig.add_subplot(train_grid[1, 0])
            valid_loss_until_best_ax = fig.add_subplot(train_grid[1, 1])
            roc_auc_until_best_ax = fig.add_subplot(train_grid[1, 2])
            valid_acc_until_best_ax = fig.add_subplot(train_grid[1, 3])

            # plot train results
            simple_plot(train_loss_until_best, 'Train Loss', x_label='epochs',
                        y_label='nll loss', ax=train_loss_ax, fontsize=fontsize)
            simple_plot(valid_loss_until_best, 'Valid Loss', x_label='epochs',
                        y_label='nll loss', ax=valid_loss_ax, fontsize=fontsize)
            simple_plot(roc_auc_until_best, 'Roc Auc', x_label='epochs',
                        y_label='', ax=roc_auc_ax, fontsize=fontsize)
            simple_plot(valid_acc_until_best, 'Valid Acc', x_label='epochs',
                        y_label='valid acc', ax=valid_acc_ax, fontsize=fontsize)

            simple_plot(train_loss, 'Train Loss', x_label='epochs',
                        y_label='nll loss', ax=train_loss_until_best_ax, fontsize=fontsize)
            simple_plot(valid_loss, 'Valid Loss', x_label='epochs',
                        y_label='nll loss', ax=valid_loss_until_best_ax, fontsize=fontsize)
            simple_plot(roc_auc, 'Roc Auc', x_label='epochs', y_label='',
                        ax=roc_auc_until_best_ax, fontsize=fontsize)
            simple_plot(valid_acc, 'Valid Acc', x_label='epochs',
                        y_label='valid acc', ax=valid_acc_until_best_ax, fontsize=fontsize)

            fig.subplots_adjust(wspace=0.5, hspace=0.5)

        return fig

    def _compute_spelling_test_acc(self, group_by_item_df):
        data = group_by_item_df
        nb_iterations = len(data['predicted_item_iteration'][0])
        accuracies = []
        for x_iter in range(nb_iterations):
            acc = np.mean(data['predicted_item_iteration'].apply(
                lambda x: x[x_iter]) == data['true_item'])
            accuracies.append({'after_x_rep': x_iter + 1, 'acc': acc})
        return accuracies

    def _build_spelling_test_figure(self, name, test_spelling_result, fontsize):
        fig = plt.figure()
        fig.suptitle(name, fontsize=fontsize)
        test_spelling_grid = gridspec.GridSpec(2, 2)

        acc_iter_ax = fig.add_subplot(test_spelling_grid[0, 0])
        grouped_by_item = test_spelling_result['group_by_item']
        group_by_item_df = pd.DataFrame(grouped_by_item)
        print(group_by_item_df[['predicted_item_iteration', 'true_item']])

        acc_arr = self._compute_spelling_test_acc(group_by_item_df)
        bar_plot_arr(acc_arr, 'Acc after N repetitions', x_key='after_x_rep',
                     y_key='acc', ax=acc_iter_ax, fontsize=fontsize)

        fig.subplots_adjust(wspace=0.7, hspace=0.8)
        fig.tight_layout(pad=5)
        return fig
        # data['on_first'] = data['predicted_item_iteration']
        # print(data[['predicted_item_iteration', 'true_item']])

        # print(data[['predicted_item_iteration', 'acc_after_0_iter']])

        # grouped_by_true_item = group_by_key(grouped_by_item, 'true_item')
        #
        # for item in grouped_by_item:
        #     print(item['true_item'])
        # for item, value in grouped_by_true_item.items():
        #     print('{} {}'.format(item, len(value)))
        # print(grouped_by_true_item.keys())

    def _build_test_figure(self, name, test_result, test_result_on_train, fontsize):
        fig = plt.figure()
        fig.suptitle(name, fontsize=fontsize)
        test_grid = gridspec.GridSpec(2, 2)
        confusion_matrix_ax = fig.add_subplot(test_grid[0, 0])
        confusion_matrix_norm_ax = fig.add_subplot(test_grid[0, 1])

        confusion_matrix_on_train_ax = fig.add_subplot(test_grid[1, 0])
        confusion_matrix_norm_on_train_ax = fig.add_subplot(test_grid[1, 1])

        confusion_matrix_data = test_result.get('confusion_matrix', None)
        if confusion_matrix_data is not None:
            heatmap = confusion_matrix_heatmap(confusion_matrix_data, '', [
                                               'Non-Target', 'Target'], ax=confusion_matrix_ax, fontsize=fontsize)
            heatmap_norm = confusion_matrix_heatmap(confusion_matrix_data, '', [
                                                    'Non-Target', 'Target'], ax=confusion_matrix_norm_ax, fontsize=fontsize, norm=True)

        confusion_matrix_data_on_train = test_result_on_train.get(
            'confusion_matrix', None)
        if confusion_matrix_data_on_train is not None:
            heatmap = confusion_matrix_heatmap(confusion_matrix_data_on_train, '', [
                                               'Non-Target', 'Target'], ax=confusion_matrix_on_train_ax, fontsize=fontsize)
            heatmap_norm = confusion_matrix_heatmap(confusion_matrix_data_on_train, '', [
                                                    'Non-Target', 'Target'], ax=confusion_matrix_norm_on_train_ax, fontsize=fontsize, norm=True)
        fig.subplots_adjust(wspace=0.7, hspace=0.8)
        fig.tight_layout(pad=5)
        return fig

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

            print(values.keys())
            train_result = values.get('train', None)
            if train_result:
                name_fig_train = '{} {}'.format('Train results', title_prefix)
                fig = self._build_train_figure(
                    name_fig_train, train_result, fontsize)
                if fig:
                    figures.append(fig)

            test_result = values.get('test', None)
            test_result_train = values.get('test_train', None)

            if test_result:
                name_fig_test = '{} {}'.format('Test', title_prefix)
                fig = self._build_test_figure(
                    name_fig_test, test_result, test_result_train, fontsize)
                if fig:
                    figures.append(fig)

            test_spelling_result = values.get('test_spelling', None)
            if test_spelling_result:
                name_fig_test_spelling = '{} {}'.format(
                    'Test spelling', title_prefix)
                fig = self._build_spelling_test_figure(
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


def simple_plot(data, name, ax, x_label="", y_label="", fontsize=10):
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.set_title(name, fontsize=fontsize)
    if isinstance(data, tuple):
        X=data[1]
        y=data[0]
    else:
        X=np.arange(len(data))
        y=data
    ax.plot(X, y)


def bar_plot_arr(arr, name, x_key='x', y_key='y', x_label=None, y_label=None, ax=None, fontsize=10):
    return bar_plot_df(pd.DataFrame(arr), name, x_key, y_key, x_label, y_label, ax, fontsize)


def bar_plot_df(df, name, x_key='x', y_key='y', x_label=None, y_label=None, ax=None, fontsize=10):
    bar_plot = sns.barplot(x=x_key, y=y_key, data=df,
                           ax=ax, color=sns.xkcd_rgb["denim blue"])
    for p in ax.patches:
        ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', rotation=0, xytext=(0, 2), textcoords='offset points', fontsize=4)
    if x_label:
        ax.set_xlabel(x_label, fontsize=fontsize)
    if y_label:
        ax.set_ylabel(y_label, fontsize=fontsize)
    ax.set_title(name, fontsize=fontsize)


def confusion_matrix_heatmap(confusion_matrix, name, class_names, ax, fontsize=10, norm=False):
    if norm:
        confusion_matrix = confusion_matrix.astype(
            'float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    title = 'CM {}'.format(name)
    title = 'Normalized ' + title if norm else title
    ax.set_title(title, fontsize=fontsize)
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names)
    fmt = ".3f" if norm else "d"
    heatmap = sns.heatmap(df_cm, annot=True, fmt=fmt,
                          ax=ax, annot_kws={"size": 5})
    heatmap.yaxis.set_ticklabels(
        heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(
        heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)


class ExpeRunner(object):
    def __init__(self, expe_config_module, render, index=0):
        self.expe_config_module = expe_config_module
        self.index = index
        self.render = render

        self.__init()

    def __init(self):
        self.expe_config = importlib.import_module(
            self.expe_config_module).expe_config
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self):
        name = self.expe_config.get('name', None)
        if not name:
            raise ValueError('Name must be defined')
        self.experiment_logger = self.expe_config.get('experiment_logger', ExperimentLogger(name))
        self.logger.info('Expe {} : started'.format(name))
        datasets_def = self.expe_config['datasets']
        Scenario = self.expe_config['scenario']
        scenario_params = self.expe_config.get('scenario_params', {})
        experiments_dir = self.expe_config['dir']
        samplers = self.expe_config.get('samplers', {})
        classifier = self.expe_config['classifier']
        expe_dir = os.path.join(experiments_dir, name)
        temp_dir = os.path.join(expe_dir, 'temp')

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

        # if self.render:
        #     self.logger.info('Expe {} : rendering results...'.format(name))
        #     self.experiment_logger.render_results(out_pdf_dir=expe_dir)

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

    def run(self):
        if not self.train_test_data_dict:
            raise ValueError(
                'train_test_data_dict must be defined and not empty')
        for name, data in self.train_test_data_dict.items():
            self.experiment_logger.current_name = name
            if not isinstance(data, dict):
                raise ValueError('data must be a dict')
            else:
                if not set(['train', 'test', 'valid']).issubset(set(data.keys())):
                    raise ValueError(
                        'data must be dict with train, test, valid dataset')

            data = {key: get_or_load(value, self.db_load_func)
                    for key, value in data.items()}

            if not any([item for _, item in data.items()]):
                logging.warning(
                    '{} skipped, db_load_func returned none'.format(name))
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

            logging.info(train_data)
            logging.info(valid_data)
            logging.info(test_data)

            net_saver_dir = os.path.join(self.expe_dir, name)
            classifier_ = self.classifier['type'](
                self.classifier['params'](train_data), net_saver_dir)

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
