import torch

from skorch_ext.utils import train_split_validation_data

from experiments.samplers import SampleUp, SampleDown
from experiments.scenarios import WithinSubjectScenario, TransferSubjectScenario
from experiments.classifiers import RiemannClassifier

from datasets.rsvp import RSVP
from datasets.transformers import SamplingTransformer

expe_config = {
    'name': 'riemann_filtered',
    'dir': '/dycog/Jeremie/Loic/results',
    'datasets': [
        {
            'eeg_dataset': RSVP(),
            'tmin': 0.0,
            'tmax': 0.5,
            'apply_filter': {
                'l_freq': 0.1,
                'h_freq': 15.
            },
            'exclude_channels': ['P8', 'O2'],
            'include_subjects': ['VPfat'],
            'exclude_subjects': [],
            'cache': True
        }
    ],
    'scenario': WithinSubjectScenario,
    'scenario_params': {
        'valid_split_frac': 0.8,
        'shuffle_train': True
    },
    'samplers': {
        'train': [],
        'valid': []
    },
    'classifier': {
        'type': RiemannClassifier,
        'params': lambda train_data: {
            'n_components': 5
        }
    }
}
