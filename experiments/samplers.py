import logging
from datasets.base import P300Dataset

class P300DatasetSampler(object):
    def __init__(self, shuffle=True):
        self.shuffle = shuffle

    def sample(self, dataset):
        assert isinstance(dataset, P300Dataset)
        logging.info('sampling dataset\n{}'.format(dataset))
        resampled = self._sample(dataset).shuffle() if self.shuffle else self._sample(dataset)
        logging.info('dataset resampled\n{}'.format(resampled))
        return resampled

    def _sample(self, dataset):
        raise NotImplementedError()

class SampleUp(P300DatasetSampler):
    def _sample(self, dataset):
        return P300Dataset.resample(dataset, up=True)

class SampleDown(P300DatasetSampler):
    def _sample(self, dataset):
        return P300Dataset.resample(dataset, up=False)

class SickitLearnImbalanceSampler(P300DatasetSampler):
    def __init__(self, sampler):
        self.sampler = sampler
    def _sample(self, dataset):
        return dataset.sample(self.sampler)
