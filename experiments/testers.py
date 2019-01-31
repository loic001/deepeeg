import os
import logging
import mne
import shelve
import copy
from datasets.base import P300Dataset
from utils.db import hash_string

import numpy as np

class SpellingPredictor(object):
    def __init__(self, nb_items):
        self.nb_items = nb_items

        self._priors = None
        self._init()

    def _init(self):
        self._priors = np.ones(self.nb_items)*1/self.nb_items
        self._posts = np.zeros(self.nb_items)

    def reset(self):
        self._init()

    def update(self, target_proba, flashed_item):
        # print(self.priors)
        # print('{}, {}'.format(target_proba, flashed_item))
        assert target_proba >= 0. and target_proba <= 1.
        for item in range(1, self.nb_items+1):
            i = item - 1
            if flashed_item == item:
                # print('up: {}'.format(flashed_item))
                self._posts[i] = np.log(self._priors[i]) + np.log(target_proba)
            else:
                # print('no: {}'.format(flashed_item))
                self._posts[i] = np.log(self._priors[i]) + np.log(0.5)
        self._posts = self._posts - np.max(self._posts) + 1
        self._posts = np.exp(self._posts)
        self._posts=self._posts/np.sum(self._posts)

        self._priors = self._posts

    def get_best_item(self):
        return np.argmax(self._priors) + 1

    def get_priors(self):
        return self._priors
            #
            # self.priors = self.posts
        # posts = np.array([np.log(self.priors[item_index-1]) + np.log(target_proba) for item_index in range(1, self.nb_items+1) if flashed_item == item_index])
        # posts = np.array([np.log(self.priors[item_index-1]) + np.log(0.5) for item_index in range(1, self.nb_items+1) if flashed_item != item_index])
        #

class SpellingTestResult(object):
    def __init__(self):
        self.group_by_item = []
        self.current_item = {}

        self.key_names = {
            'pef': 'priors_every_flash',
            'pei': 'priors_every_iteration',
            'ti': 'true_item',
            'pi': 'predicted_item_iteration'
        }

    def next_item(self):
        self.group_by_item.append(copy.copy(self.current_item))
        self.current_item = {}

    def _append_or_create(self, key, obj):
        if key not in self.current_item or not isinstance(self.current_item[key], list):
            self.current_item[key] = []
        self.current_item[key].append(obj)

    def append_priors_on_flash(self, priors):
        key = self.key_names['pef']
        self._append_or_create(key, priors)

    def set_true_item(self, item):
        key = self.key_names['ti']
        self.current_item[key] = item

    def append_priors_on_iteration(self, priors):
        key = self.key_names['pei']
        self._append_or_create(key, priors)
        # print(len(self.current_item[key]))

    def append_predicted_item_on_iteration(self, predicted_item):
        key = self.key_names['pi']
        self._append_or_create(key, predicted_item)

class P300DatasetTester():
    def __init__(self, classifier, test_data):
        self.test_data = test_data
        self.classifier = classifier
        self.logger = logging.getLogger(__name__)

    def run(self):
        test_data = self.test_data
        X_test_dict = {'x': test_data.X}
        y_pred = self.classifier.predict(X_test_dict)
        y_test = test_data.y

        from sklearn.metrics import confusion_matrix
        confusion_matrix = confusion_matrix(y_test, y_pred)
        test_result = {'confusion_matrix': confusion_matrix}
        return test_result

class P300DatasetTesterSpelling():
    def __init__(self, classifier, test_data):
        self.test_data = test_data
        self.classifier = classifier
        self.logger = logging.getLogger(__name__)
        self.sts = SpellingTestResult()

    def run(self):
        test_data = self.test_data
        nb_items = test_data.info['nb_items']
        nb_iterations = test_data.info['nb_iterations']
        predictor = SpellingPredictor(nb_items)
        predictor.reset()


        test_spelling_result = {'group_by_item': []}

        to_index = int(len(test_data.y_stim) / (nb_items))
        for index in range(0, to_index): #len(test_data.y_stim)

            flash_index = int(index*30)
            iteration_index = int(index%10)+1

            #every item
            if index%nb_iterations == 0:
                predictor.reset()

            #record item
            if index%nb_iterations == 0 and index != 0:
                self.sts.next_item()

            X_one_iteration = test_data.X[flash_index:flash_index+nb_items]
            y_stim_one_iteration = test_data.y_stim[flash_index:flash_index+nb_items]
            y_one_iteration = test_data.y[flash_index:flash_index+nb_items]

            X_one_iteration_dict = {'x': X_one_iteration}
            y_pred_one_iteration = self.classifier.predict(X_one_iteration_dict, proba=True, force_reload=bool(index == 0))

            for i in range(len(y_pred_one_iteration)):
                target_proba = y_pred_one_iteration[i][1]
                predictor.update(target_proba, y_stim_one_iteration[i])
                #every flash
                self.sts.append_priors_on_flash(predictor.get_priors())

            item_true = np.asscalar(y_stim_one_iteration[np.where(y_one_iteration == 1)])
            item_predicted_from_prior = predictor.get_best_item()

            # letter_true = test_data.y_stim_names.get(item_true, None)
            # predicted_letter = test_data.y_stim_names.get(item_predicted_from_prior, None)

            prior_flash = predictor.get_priors()

            #every iteration
            self.sts.append_priors_on_iteration(predictor.get_priors())
            self.sts.append_predicted_item_on_iteration(predictor.get_best_item())

            self.sts.set_true_item(item_true)

        #dont forget last item
        self.sts.next_item()


        return self.sts.__dict__
        # print(len(self.sts.group_by_item[0]['priors_every_flash']))
            # X_test_dict = {'x': test_data.X[flash_index]}
            # self.classifier.predict_proba()
            # predictor.update()
        # X_test_dict = {'x': test_data.X}
        # y_pred = self.classifier.predict(X_test_dict)
        # y_test = test_data.y
        #
        # from sklearn.metrics import confusion_matrix
        # confusion_matrix = confusion_matrix(y_test, y_pred)
        # test_result = {'confusion_matrix': confusion_matrix}

        # self.experiment_logger.testSpellingResult(self.name, test_result)
#
# from experiments.classifiers import RiemannClassifier
# from datasets.rsvp import RSVP
#
# rsvp = RSVP()
# mne_raw = rsvp.get_subject_data('VPfat', calib=False)
# test_data_epoched = cache_or_create_raw_to_epoched_mne(mne_raw, 0.0, 0.5, exclude_channels=[], cache=True)
# test_data = P300Dataset.from_mne_epoched(test_data_epoched)
#
# classifier = RiemannClassifier({'n_components': 3})
# classifier.fit({'x': test_data.X}, test_data.y, {}, {})

# import pickle
# classifier = pickle.load(open('class.pkl', "rb"))
# test_data = pickle.load(open('data.pkl', "rb"))
# #
# test_spelling = P300DatasetTestSpelling(classifier, test_data)
# test_spelling.run()
