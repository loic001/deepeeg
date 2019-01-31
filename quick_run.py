import importlib
import torch
import numpy as np
from experiments.classifiers import SkorchClassifier

from skorch_ext.callbacks import LayerDataViz
from skorch_ext.utils import train_split_validation_data

from models.crnl_net import NetAdapter
from functools import partial

from skorch.utils import to_var, to_numpy

# from tensorboardX import SummaryWriter
# writer = SummaryWriter()
#
#
# def layer_data_log(data):
#     input_output = data.get('input_output', None)
#     if input_output:
#         layer_output = to_var(input_output['layer_output'], use_cuda=False)
#         label = data['data']['y']
#         label_list = to_numpy(label).tolist()[:1000]
#         writer.add_embedding(layer_output.data[:1000], metadata=label_list, global_step=data['epoch'])
#         # writer.add_embedding(layer_output, metadata=label)
#     # features = data.view(100, 784)
#     # writer.add_embedding(features, metadata=label, label_img=images.unsqueeze(1))

from experiments.viz import Viz

viz = Viz()

skorch_net_params = {
    'module': NetAdapter,
    'optimizer': torch.optim.Adam,
    'callbacks': [('layer_data_logger_1', LayerDataViz(layer_name='sub_module.flatten', callback_func=partial(viz.layer_data_viz, {'name': 'toto'})))],
    # 'optimizer__momentum': 0.9,
    'optimizer__weight_decay': 0.0005,
    'criterion': torch.nn.NLLLoss,
    'criterion__weight': torch.FloatTensor([1, 29]),
    'iterator_train__shuffle': False,
    'max_epochs': 50,
    'batch_size': 128,
    'lr': 0.001,
    'train_split': train_split_validation_data,
    'module__chan': 61,
    'module__input_time_length': 101
}

classifier = SkorchClassifier(skorch_net_params, '/tmp/quick')


from datasets.rsvp import RSVP
from experiments.loaders import DatasetDefLoader

data_def = {
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
    'tmax': 0.5,

    #others params
    'cache': False
}


loaded = DatasetDefLoader(cache_dir=None).loads([data_def])
calib_dataset = loaded[list(loaded.keys())[0]]['calib_dataset']
dataset = loaded[list(loaded.keys())[0]]['dataset']

import shelve
with shelve.open('/tmp/quick/db.pkl') as db:
    db['train'] = calib_dataset
    db['valid'] = dataset

import shelve
with shelve.open('/tmp/quick/db.pkl') as db:
    train = db['train']
    valid = db['valid']

# from experiments.samplers import SampleDown
# sampler = SampleDown()
#

# valid = sampler.sample(valid)


X_train_dict = {'x': train.X}
X_valid_dict = {'x': valid.X}
import dill as pickle


# torch.manual_seed(1000)

from tensorboardX import SummaryWriter
from skorch.utils import to_var, to_numpy
from torch.autograd import Variable


train_result = classifier.fit(X_train_dict, train.y, X_valid_dict, valid.y)

model = classifier.net_saver.net().module_

writer = SummaryWriter(log_dir='tensorboard_quickrun')

x_var = Variable(torch.from_numpy(np.ones((1, 61, 101, 1))), requires_grad=True)
writer.add_graph(model, x_var.float(), verbose=True)

writer.close()

# pickle.dump(classifier, open('/home/lolo/projects/crnl/deepeeg/results/quick/ep_50_cla.pkl', 'wb'))
