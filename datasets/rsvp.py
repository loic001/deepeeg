import numpy as np
import mne
import scipy.io as sio
from scipy.io import loadmat
import os
import shelve
import pathlib

from datasets.base import EEGDataset

data_dir = '/dycog/Jeremie/Loic/data/RSVP_speller_010-2015/'

class RSVP(EEGDataset):
    # EEGDataset implementation
    def get_name(self):
        return 'RSVP_speller_010-2015'

    def subjects(self):
        # VPgce
        return ['VPgcc', 'VPfat', 'VPgcb', 'VPgcg', 'VPgcd', 'VPgcf', 'VPgch', 'VPiay', 'VPicn', 'VPicr', 'VPpia']

    def get_subject_data(self, subject, calib=False):
        data_folder = data_dir
        filename = os.path.join(data_folder, 'RSVP_{}.mat'.format(subject))
        return self._build_mne_raw_array_from_filename(filename, calib)

    # private functions
    def _build_mne_raw_array_from_filename(self, filename, calib):
        data_mat = sio.loadmat(filename)

        data = data_mat['data'][0, 0] if calib else data_mat['data'][0, 1]
        bbci_mrk = data_mat['bbci_mrk'][0,
                                        0] if calib else data_mat['bbci_mrk'][0, 1]

        cnt = self._extract_data(data)
        events = self._extract_events(data, bbci_mrk)

        name = filename.rstrip(os.path.sep).split(
            '/')[-1] + ('_calib' if calib else '')
        cnt.info['y'] = events
        cnt.info['nb_iterations'] = bbci_mrk[0][0][5][0][0]
        cnt.info['y_names'] = {0: 'Non-Target', 1: 'Target'}
        cnt.info['y_stim'] = data[0][0][6][0]
        cnt.info['y_stim_names'] = {
            index + 1: s[0] for index, s in enumerate(bbci_mrk[0][0][6][0][0][6][0])}  # what the heck
        cnt.info['nb_items'] = len(cnt.info['y_stim_names'])
        cnt.info['name'] = name
        return cnt

    def _extract_data(self, data):
        cnt = data[0][0][0].T  # cnt = channels * time
        # build RawArray mne object
        ch_names = [s[0] for s in data[0][0][1][0]]
        sfreq = data[0][0][2][0][0]
        ch_types = ["eeg" for _ in range(len(ch_names))]
        montage = mne.channels.read_montage("standard_1020")
        info = mne.create_info(
            ch_names=ch_names, sfreq=sfreq, ch_types=ch_types, montage=montage)
        raw = mne.io.RawArray(cnt, info, verbose='WARNING')
        return raw

    def _extract_events(self, data, bbci_mrk):
        # mne expects events with 3 ints each
        # self.bbci_mrk[0][0][1][0] or self.data[0][0][5][0])
        events = np.column_stack((data[0][0][4][0], np.zeros(
            len(data[0][0][4][0]), dtype='int'), bbci_mrk[0][0][1][0]))
        return events
