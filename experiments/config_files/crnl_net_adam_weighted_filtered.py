import torch
# torch.manual_seed(100)

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

from models.crnl_net import NetAdapter

from skorch_ext.callbacks import LayerDataViz, EpochSummary, GetModuleOnTrainBegin

from functools import partial
from experiments.viz import Viz

viz = Viz()

expe_config = {
    'name': 'crnl_net_adam_weighted_filtered',
    'dir': '/dycog/Jeremie/Loic/results',
    'datasets': [
        {
            'eeg_dataset': RSVP(),
            'include_subjects': ['VPfat', 'VPgcc'],
            'exclude_subjects': [],
            'apply_filter': {
                'l_freq': 0.1,
                'h_freq': 20.
            },
            #epochs params
            'exclude_channels': ['P8', 'O2'],
            'tmin': 0.0,
            'tmax': 0.5,

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
        'params': lambda p:
            {
                'module': NetAdapter,
                'optimizer': torch.optim.Adam,
                # 'optimizer__momentum': 0.9,
                'callbacks': [
                    ('graph_viz', GetModuleOnTrainBegin(callback_func=partial(viz.add_graph, p))),
                    ('epoch_summary', EpochSummary(callback_func=partial(viz.epoch_summary, p))),
                    ('layer_data_viz_flatten', LayerDataViz(layer_name='sub_module.flatten', callback_func=partial(viz.layer_data_viz, p)))
                ],
                'optimizer__weight_decay': 0.0005,
                'criterion': torch.nn.NLLLoss,
                'criterion__weight': torch.FloatTensor([1, 29]),
                'iterator_train__shuffle': True,
                'max_epochs': 100,
                'batch_size': 128,
                'lr': 0.001,
                'train_split': train_split_validation_data,
                'module__chan': p['train'].X.shape[1],
                'module__input_time_length': p['train'].X.shape[2]
            }
    }
}
