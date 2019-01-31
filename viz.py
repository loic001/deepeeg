import shelve

import glob
import os
import numpy as np
import mne


from experiments.base import ExperimentLogger
from skorch_ext.netsaver import NetSaver

from torch_ext.pytorch_smoothgrad.gradients import VanillaGrad, SmoothGrad

from datasets.rsvp import RSVP
raw_mne = RSVP().get_subject_data('VPgcc').pick_types(eeg=True, exclude=['P8', 'O2'])

def ma(values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

if __name__ == '__main__':
    import dill as pickle
    import copy


    d = '/dycog/Jeremie/Loic/results/eeg_net_sgd_weighted_filtered_fat/'

    g = os.path.join(d, '*.dat')
    for f in glob.glob(g):
        experiment_logger = ExperimentLogger(f[:-4], overwrite=False)
        test_train_list = list(experiment_logger.datas.keys())
        for test_train in test_train_list:
            net_saver_dir = os.path.join(d, test_train)
            net_saver = NetSaver(net_saver_dir=net_saver_dir)
            net_saver.init(show_founded_sessions=True)
            print(test_train)
            net = net_saver.net().module_

            test_dataset = experiment_logger.datas[test_train]['test_dataset']
            target_index = list(np.where(test_dataset.y == 1)[0])
            target_dataset, non_target_dataset = test_dataset.split_by_indices(target_index, split_names=('target', 'non-target'))
            # sample = target_dataset.select([0])
            evokeds_target_saliency = np.ones(target_dataset.X.shape)
            for index in range(len(target_dataset.X)):
                print(index)
                sample = target_dataset.select([index])
                X_sample = np.expand_dims(sample.X, -1)
                y_sample = sample.y

                vanilla_grad = SmoothGrad(pretrained_model=net, cuda=False, magnitude=False)
                vanilla_saliency = vanilla_grad(X_sample, index=1)
                epoch_saliency = vanilla_saliency.squeeze(-1)
                evokeds_target_saliency[index] = epoch_saliency


            evokeds_non_target_saliency = np.ones((400, non_target_dataset.X.shape[1], non_target_dataset.X.shape[2]))
            for index in range(400):
                print(index)
                sample = non_target_dataset.select([index])
                X_sample = np.expand_dims(sample.X, -1)
                y_sample = sample.y

                vanilla_grad = SmoothGrad(pretrained_model=net, cuda=False, magnitude=False)
                vanilla_saliency = vanilla_grad(X_sample, index=0)
                epoch_saliency = vanilla_saliency.squeeze(-1)
                evokeds_non_target_saliency[index] = epoch_saliency
                # evokeds_saliency[index] = epoch_saliency * X_sample[0].squeeze(-1)
            evokeds_target_saliency_mean_over_batch = evokeds_target_saliency.mean(axis=0)
            evokeds_non_target_saliency_mean_over_batch = evokeds_non_target_saliency.mean(axis=0)

            evoked_target_saliency = mne.EvokedArray(evokeds_target_saliency_mean_over_batch, raw_mne.info)
            evoked_non_saliency = mne.EvokedArray(evokeds_non_target_saliency_mean_over_batch, raw_mne.info)
            evoked_target_saliency.plot_image()
            evoked_target_saliency.plot_joint()
            evoked_non_saliency.plot_image()
            # mean_over_chan_target = np.abs(evokeds_target_saliency_mean_over_batch.mean(axis=0))
            # mean_over_chan_non_target = np.abs(evokeds_non_target_saliency_mean_over_batch.mean(axis=0))
            # evoked_target_saliency = ma(mean_over_chan_target, 35)
            # evoked_non_target_saliency = ma(mean_over_chan_non_target, 35)
            #
            # # evoked_target_saliency = ma(evoked_target_saliency.mean(axis=1), 20)
            # import matplotlib.pyplot as plt
            # plt.plot(mean_over_chan_target)
            # plt.plot(mean_over_chan_non_target)
            # plt.show()

            # evoked_dict = {'Target saliency': evoked_target_saliency, 'Non-Target saliency': evoked_non_saliency}

            # mne.viz.plot_compare_evokeds(evoked_dict)
