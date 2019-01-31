import os
import uuid
from tensorboardX import SummaryWriter

from skorch.utils import to_var, to_numpy

tensorboard_writers = {}


class Viz(object):
    def __init__(self, logdir='.', tensorboard=True, tensorboard_dir_name='tensorboard', tensorboard_writers={}):
        self.tensorboard_writers = tensorboard_writers
        self.tensorboard = tensorboard
        self.tensorboard_dir_name = tensorboard_dir_name

    def get_tensorboard_writer_by_logdir(self, log_dir):
        writer = self.tensorboard_writers.get(
            log_dir, SummaryWriter(log_dir=log_dir))
        self.tensorboard_writers[log_dir] = writer
        return writer

    def add_graph(self, expe_params, module):
        print('add_graph')
        shape_dummy = expe_params.get('train', None)
        print(shape_dummy)
        # if self.tensorboard:

    def epoch_summary(self, expe_params, epoch_data):
        # print(epoch_data)
        pass

    def layer_data_viz(self, expe_params, viz_params):
        expe_name = expe_params.get('expe_name', 'nan')
        viz_tag = '{}_{}_'.format(viz_params.get('name', 'nan'), viz_params.get('layer_name', 'nan'), str(uuid.uuid4())[:3])
        input_output = viz_params.get('input_output', None)
        label = viz_params['data'].get('y', None)
        if (input_output is None) or (label is None):
            # do nothing if no data
            return
        layer_output = to_var(input_output['layer_output'], use_cuda=False)
        label_list = to_numpy(label).tolist()

        if self.tensorboard:
            tensorboard_dir = os.path.join(expe_params.get(
                'expe_dir', '.'), self.tensorboard_dir_name)
            tensorboard_expe_dir = os.path.join(tensorboard_dir, expe_name)
            writer = self.get_tensorboard_writer_by_logdir(tensorboard_expe_dir)
            writer.add_embedding(layer_output.data, metadata=label_list,
                                 global_step=viz_params['epoch'], tag=viz_tag)
