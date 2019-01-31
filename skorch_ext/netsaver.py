import os
import pathlib
import datetime
import shelve
import logging
import glob
import errno

import dill as pickle

from skorch_ext.classifier import Classifier
from skorch_ext.callbacks import NetCheckpoint

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import GridSearchCV
from skorch.callbacks import EpochScoring
from skorch.callbacks import Checkpoint

from utils.tools import sorted_nicely

import matplotlib.pyplot as plt

from utils.input import query_yes_no
from utils.input import list_selection_input

logger = logging.getLogger('NetSaver')


class NetSaver(object):
    """
    Class NetSaver
    Parameters
    ----------
    par: str
        Param

    Attributes
    ----------
    att: int
        Att
    """

    def __init__(self, net_saver_dir, neural_net_type=Classifier):
        logging.basicConfig(level=logging.DEBUG)
        self.net_saver_dir = net_saver_dir
        self.neural_net_type = neural_net_type

        pathlib.Path(self.net_saver_dir).mkdir(parents=True, exist_ok=True)

        self.default_forced_params = {
            "callbacks": [],
            "warm_start": True
        }

        self._net = None

        self.add_checkpoint_criterion('valid_loss_best')

    def updated_skorch_net_params(self, skorch_net_params):
        self._net.set_params(**skorch_net_params)

    def remove_checkpoint_criterion(self, criterion):
        self.default_forced_params['callbacks'] = {
            k: v
            for k, v in self.default_forced_params['callbacks']
            if isinstance(v, NetCheckpoint) and v.criterion != criterion
        }

    def add_checkpoint_criterion(self, criterion):
        assert isinstance(criterion, str)
        self.default_forced_params['callbacks'].append(
            ('checkpoints', NetCheckpoint(monitor=criterion, save_function=self.__save_net)))

    def net(self):
        return self._net if hasattr(self, '_net') else None

    def save(self, name):
        self.__save_net(name)

    def get_checkpoints(self):
        return self.__check_net_files(self.session_dir)

    def init(self, skorch_net_params=None, session=None, show_founded_sessions=False, checkpoint=None, criterion='valid_loss_best', show_log=True):
        # if session or (not checkpoint and not criterion):
        self.__init_session(session, show_founded_sessions)
        self.__init_net(skorch_net_params, checkpoint, criterion, show_log)

    def __init_session(self, session=None, show_founded_sessions=False, show_log=True, keep_current_session=False):
        if (not session and not show_founded_sessions) and (not keep_current_session):
            session = 'session_{}'.format(
                datetime.datetime.now().strftime('%m_%d_%Y_%X').replace(':', '_'))
            self.__create_session(session)
        if show_founded_sessions:
            sessions = self.__check_session_dirs()
            if not sessions: raise ValueError('sessions not found in %s', self.net_saver_dir)
            session = list_selection_input(
                [session for session, _ in sessions.items()])
        self.__load_session(session, show_log)

    def __init_net(self, skorch_net_params=None, checkpoint=None, criterion='valid_loss_best', show_log=True):
        if skorch_net_params:
            self._net = self.__create_neural_net(
                self.neural_net_type, skorch_net_params)
        else:
            if not checkpoint and not criterion:
                checkpoints = self.__check_net_files(self.session_dir)
                if len(checkpoints) > 0:
                    checkpoint = list_selection_input(
                        checkpoints).split(os.sep)[-1]
                else:
                    if show_log:
                        logger.warning(
                            'no checkpoints founded in {}'.format(self.session_dir))
                    return False

            if criterion and not checkpoint:
                checkpoint = self.__get_checkpoint_by_criterion(criterion)
            checkpoint_filename = os.path.join(self.session_dir, checkpoint)
            return self.__load_net_from_file(checkpoint_filename, show_log)

    def __get_checkpoint_by_criterion(self, criterion):
        checkpoints = self.get_checkpoints()
        checkpoints_filtered = [checkpoint for checkpoint in checkpoints if checkpoint.split(os.sep)[-1].find(criterion) != -1]
        return checkpoints_filtered[-1] if len(checkpoints_filtered) > 0 else None


    def __create_session(self, session):
        session_dir = os.path.join(self.net_saver_dir, session)
        pathlib.Path(session_dir).mkdir(parents=True, exist_ok=True)
        return session

    def __load_session(self, session, show_log=True):
        # sessions = self.__check_session_dirs()
        # if session not in sessions.keys():
        #     if show_log:
        #         logger.warning('{} not exist in {} dir'.format(
        #             session, self.net_saver_dir))
        #     return False
        # else:
        if session and session != 'keep_current':
            self.session_dir = os.path.join(self.net_saver_dir, session)
            self.session_name = session
            if show_log:
                logger.info('session {} loaded'.format(session))

    def __save_net(self, name):
        filename = 'net_{}.pkl'.format(name)
        final_filename = os.path.join(self.session_dir, filename)
        if not os.path.exists(os.path.dirname(final_filename)):
            try:
                os.makedirs(os.path.dirname(final_filename))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        with open(final_filename, 'wb') as f:
            pickle.dump(self._net, f)
            logging.info('net saved to {}'.format(final_filename))

    def __load_net_from_file(self, filename, show_log=True):
        try:
            with open(filename, 'rb') as f:
                self._net = pickle.load(f)
                if show_log: logging.info('net loaded from {}'.format(filename))
                return True
        except:
            return False

    def __check_net_files(self, d):
        files = []
        glob_p = os.path.join(d, 'net*.pkl')
        for filename in glob.iglob(glob_p, recursive=False):
            files.append(filename)
        return sorted_nicely(files)

    def __check_session_dirs(self):
        dirs_glob = os.path.join(self.net_saver_dir, "session_*/")
        sessions_map = {}
        for session_dir in glob.glob(dirs_glob):
            checkpoints_net = self.__check_net_files(session_dir)
            session_name = session_dir.split(os.sep)[-2]
            sessions_map[session_name] = []
            for checkpoint in checkpoints_net:
                sessions_map[session_name].append(checkpoint)
        return sessions_map

    def _merge_dict_append_list(self, dict1, dict2):
        merged_dict = dict1
        for k, v in dict2.items():
            if isinstance(v, list):
                if k not in merged_dict or not isinstance(merged_dict[k], list):
                    merged_dict[k] = []
                merged_dict[k] = merged_dict[k] + v
            else:
                merged_dict[k] = v
        return merged_dict

    def __create_neural_net(self, neural_net_type, skorch_net_params):
        # skorch_net_params.update(self.default_forced_params)
        skorch_net_params = self._merge_dict_append_list(skorch_net_params, self.default_forced_params)
        net = neural_net_type(**skorch_net_params)
        return net
