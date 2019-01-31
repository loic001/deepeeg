import logging
import mne

def raw_to_epoched_mne(raw_mne, tmin, tmax, exclude_channels):
        logging.info('Building epochs : name: {}, tmin: {}, tmax: {}, exclude_channels: {}'.format(raw_mne.info['name'], str(tmin), str(tmax), str(exclude_channels)))
        return mne.Epochs(raw_mne, raw_mne.info['y'], dict(target=1, non_target=0), tmin=tmin, tmax=tmax, preload=True).pick_types(eeg=True, exclude=exclude_channels)

def cache_or_create_raw_to_epoched_mne(raw_mne, tmin, tmax, exclude_channels, cache=False, cache_dir='/tmp'):
    epoched = None
    if cache:
        assert cache_dir is not None
        db_name = os.path.join(cache_dir, 'datasets_cache')
        db_key = '{}_{}_{}_{}_{}_{}'.format(raw_mne.info['name'], str(tmin), str(tmax), str(exclude_channels), str(raw_mne.info['lowpass']), str(raw_mne.info['highpass']))
        db_key = hash_string(db_key)
        with shelve.open(db_name) as db:
            try:
                epoched=db[db_key]
                logging.info('Epochs successfully recovered : name: {}, tmin: {}, tmax: {}, exclude_channels: {}'.format(raw_mne.info['name'], str(tmin), str(tmax), str(exclude_channels)))
            except KeyError:
                logging.info('Requested epochs do not exist in cache')
                epoched = raw_to_epoched_mne(raw_mne, tmin, tmax, exclude_channels)
                db[db_key]=epoched
    else:
        epoched = raw_to_epoched_mne(raw_mne, tmin, tmax, exclude_channels)
    return epoched
