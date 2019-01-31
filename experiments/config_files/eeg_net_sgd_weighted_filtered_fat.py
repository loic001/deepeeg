import os

import torch

from datasets.transformers import SamplingTransformer
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import RandomOverSampler

from experiments.scenarios import WithinSubjectScenario, TransferSubjectScenario

from skorch_ext.utils import train_split_validation_data
from sklearn.pipeline import make_pipeline


from experiments.classifiers import SkorchClassifier


from sklearn.linear_model import LogisticRegression

from experiments.samplers import SampleUp, SampleDown

from datasets.rsvp import RSVP

from models.eeg_net_braindecode import EEGNetBraindecodeAdapter

from skorch_ext.callbacks import LayerDataLogger

from experiments.base import ExperimentLogger

#create ExperimentLogger instance to access it from here
expe_dir = '/dycog/Jeremie/Loic/results'
expe_name = os.path.basename(__file__)[:-3]
experiment_logger = ExperimentLogger.create(expe_dir, expe_name)

expe_config = {
    'name': expe_name,
    'experiment_logger': experiment_logger,
    'dir': expe_dir,
    'datasets': [
        {
            'eeg_dataset': RSVP(),
            'include_subjects': ['VPfat'],
            'exclude_subjects': [],
            'apply_filter': {
                'l_freq': 0.1,
                'h_freq': 15.
            },
            #epochs params
            'exclude_channels': ['P8', 'O2'],
            'tmin': 0.0,
            'tmax': 0.8,

            #others params
            'cache': True
        }
    ],
    'scenario': WithinSubjectScenario,
    'scenario_params': {
        'valid_split_frac': 0.8
    },
    'samplers': {
        'train': [],
        'valid': []
    },
    'classifier': {
        'type': SkorchClassifier,
        'params': lambda train_data: {
            'module': EEGNetBraindecodeAdapter,
            'optimizer': torch.optim.SGD,
            # 'callbacks': [('layer_data_logger_1', LayerDataLogger(layer_name='sub_module.flatten', callback_func=experiment_logger.record_data_layer_logger))],
            'optimizer__momentum': 0.9,
            'optimizer__weight_decay': 0.0005,
            'criterion': torch.nn.NLLLoss,
            'criterion__weight': torch.FloatTensor([1, 29]),
            'iterator_train__shuffle': True,
            'max_epochs': 300,
            'batch_size': 128,
            'lr': 0.0001,
            'train_split': train_split_validation_data,
            'module__chan': train_data.X.shape[1],
            'module__input_time_length': train_data.X.shape[2]
        }
    }
}
