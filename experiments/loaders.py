import logging
import os
import shelve

import mne

from datasets.base import P300Dataset
from datasets.base import EEGDataset

from utils.db import hash_string

class DatasetDefLoader(object):
    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(__class__.__name__)

    def _raw_to_epoched_mne(self, raw_mne, tmin, tmax, exclude_channels):
            self.logger.info('Building epochs : name: {}, tmin: {}, tmax: {}, exclude_channels: {}'.format(raw_mne.info['name'], str(tmin), str(tmax), str(exclude_channels)))
            return mne.Epochs(raw_mne, raw_mne.info['y'], dict(target=1, non_target=0), tmin=tmin, tmax=tmax, preload=True).pick_types(eeg=True, exclude=exclude_channels)

    def _cache_or_create_raw_to_epoched_mne(self, raw_mne, tmin, tmax, exclude_channels=None, cache=False):
        epoched = None
        if cache:
            assert self.cache_dir is not None
            db_name = os.path.join(self.cache_dir, 'datasets_cache')
            db_key = '{}_{}_{}_{}_{}_{}'.format(raw_mne.info['name'], str(tmin), str(tmax), str(exclude_channels), str(raw_mne.info['lowpass']), str(raw_mne.info['highpass']))
            db_key = hash_string(db_key)
            with shelve.open(db_name) as db:
                try:
                    epoched=db[db_key]
                    self.logger.info('Epochs successfully recovered : name: {}, tmin: {}, tmax: {}, exclude_channels: {}'.format(raw_mne.info['name'], str(tmin), str(tmax), str(exclude_channels)))
                except KeyError:
                    self.logger.info('Requested epochs do not exist in cache')
                    epoched = self._raw_to_epoched_mne(raw_mne, tmin, tmax, exclude_channels)
                    db[db_key]=epoched
        else:
            epoched = self._raw_to_epoched_mne(raw_mne, tmin, tmax, exclude_channels)
        return epoched


    def _filter(self, raw_mne, l_freq, h_freq):
        self.logger.info('filtering to l_freq: {}, h_freq: {}'.format(l_freq, h_freq))
        raw_mne.filter(l_freq, h_freq, fir_design='firwin')

    def load(self, dataset_def, db_save=False, db_save_func=None):
        eeg_dataset = dataset_def['eeg_dataset']
        assert isinstance(eeg_dataset, EEGDataset)
        cache = dataset_def.get('cache', False)

        transformers = dataset_def.get('transformers', [])
        subjects = eeg_dataset.subjects()

        include_subjects = dataset_def['include_subjects']
        exclude_subjects = dataset_def['exclude_subjects']
        if '*' not in include_subjects:
            subjects = [subject for subject in subjects if subject in include_subjects]

        subjects = [subject for subject in subjects if subject not in exclude_subjects]

        tmin = dataset_def.get('tmin', 0.0)
        tmax = dataset_def.get('tmax', 0.5)
        exclude_channels = dataset_def.get('exclude_channels', [])
        apply_filter = dataset_def.get('apply_filter', {'l_freq': None, 'h_freq': None})
        l_freq = apply_filter['l_freq']
        h_freq = apply_filter['h_freq']

        subjects_data_dict = {}
        for subject in subjects:
            subject_data_raw_calib = eeg_dataset.get_subject_data(subject, calib=True)
            assert isinstance(subject_data_raw_calib, mne.io.RawArray)
            if l_freq or h_freq:
                self._filter(subject_data_raw_calib, l_freq, h_freq)

            # mne.viz.plot_raw_psd(subject_data_raw_calib, average=False)
            mne_epoched_calib = self._cache_or_create_raw_to_epoched_mne(subject_data_raw_calib, tmin, tmax, exclude_channels, cache)


            # mne.viz.plot_topomap(data[idx], evoked.info, axes=axes[idx], show=False)
            # print('before plot')
            # target_evoked = mne_epoched_calib['target'].average()
            # non_target_evoked = mne_epoched_calib['non_target'].average()
            # edi = {'Target': target_evoked, 'Non-Target': non_target_evoked}
            # target_evoked.plot_joint(times='peaks')
            # target_evoked.plot_topomap()
            # non_target_evoked.plot_topomap()
            # mne.viz.plot_evoked_topo([target_evoked, non_target_evoked], title='T / NT', background_color='w')
            # non_target_evoked.plot_joint(times='peaks')
            # mne.viz.plot_compare_evokeds(edi)

            # av.plot(spatial_colors=True, gfp=True)
            # av.plot_image()
            # av.plot_joint(times='auto')
            # mne_epoched_calib.plot_image(combine='mean')


            subject_data_raw = eeg_dataset.get_subject_data(subject, calib=False)
            assert isinstance(subject_data_raw, mne.io.RawArray)
            if l_freq or h_freq:
                self._filter(subject_data_raw, l_freq, h_freq)

            # mne.viz.plot_raw_psd(subject_data_raw, average=False)

            mne_epoched= self._cache_or_create_raw_to_epoched_mne(subject_data_raw, tmin, tmax, exclude_channels, cache)

            dataset_p300_calib = P300Dataset.from_mne_epoched(mne_epoched_calib)
            dataset_p300 = P300Dataset.from_mne_epoched(mne_epoched)
            #to numpy data
            # X = (epoched.get_data() * 1e6).astype(np.float32)
            # y = (epoched.events[:,2] - 1).astype(np.int64)

            # if len(transformers):
            #     logging.info('applying dataset transformers...')
            #     for transformer in transformers:
            #         X, y = transformer.transform(X, y)
            dataset_p300_calib_obj = db_save_func(dataset_p300_calib) if db_save and callable(db_save_func) else dataset_p300_calib
            dataset_p300_obj = db_save_func(dataset_p300) if db_save and callable(db_save_func) else dataset_p300
            datasets = {'calib_dataset': dataset_p300_calib_obj, 'dataset': dataset_p300_obj}
            key = '{}_{}'.format(eeg_dataset.get_name(), subject)
            subjects_data_dict[key] = datasets
        return subjects_data_dict

    def loads(self, datasets_def, db_save=False, db_save_func=None):
        subjects_data_dict = {}
        for dataset_def in datasets_def:
            subjects_data_dict.update(self.load(dataset_def, db_save, db_save_func))
        return subjects_data_dict
