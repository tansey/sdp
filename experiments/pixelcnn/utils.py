import os
import numpy as np
import collections

Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test', 'nfeatures', 'nlabels'])

class Dataset(object):

    def __init__(self,
               path,
               test_set=False,
               file_indices=None,
               seed=42):
        self._path = path
        self._test = test_set
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._rng = np.random.RandomState(seed)
        self.p = 0
        self._count_samples()
        if file_indices is not None:
            self._perm = file_indices
            self._nfiles = len(file_indices)
            self._nexamples = self._examples_per_file * self._nfiles

    def _count_samples(self):
        self._nfiles = 0
        self._nexamples = 0
        self._nlabels = None
        self._examples_per_file = None
        prefix = 'test' if self._test else 'train'
        while os.path.exists(os.path.join(self._path, '{}_features_{}.npy'.format(prefix, self._nfiles))):
            if self._nfiles == 0:
                features, labels = self._load(0)
                self._examples_per_file = np.prod(features.shape[:-1])
                self._nfeatures = features.shape[-1]
                self._nlabels = tuple([16] * labels.shape[-1]) # TEMP
            self._nexamples += self._examples_per_file
            self._nfiles += 1
        self._perm = np.arange(self._nfiles)

    def _load(self, file_index):
        prefix = 'test' if self._test else 'train'
        features = np.load(os.path.join(self._path, '{}_features_{}.npy'.format(prefix, self._nfiles)))
        labels = np.load(os.path.join(self._path, '{}_pixels_{}.npy'.format(prefix, self._nfiles)))
        return features.reshape((-1,3)), (labels / 16).reshape((-1,3)) # TEMP

    @property
    def nfeatures(self):
        return self._nfeatures

    @property
    def nlabels(self):
        return self._nlabels

    @property
    def nfiles(self):
        return self._nfiles

    @property
    def nexamples(self):
        return self._nexamples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def reset(self):
        self.p = 0

    def __next__(self, n=None):
        # on first iteration permute all data
        if self.p == 0 and not self._test:
            self._rng.shuffle(self._perm)
            
        # on last iteration reset the counter and raise StopIteration
        if self.p >= self._nfiles:
            self.reset() # reset for next time we get called
            raise StopIteration

        return self._load(self._perm[p])

    def __iter__(self):
        return self

    next = __next__  # Python 2 compatibility (https://stackoverflow.com/questions/29578469/how-to-make-an-object-both-a-python2-and-python3-iterator)



def load_dataset(inputdir='experiments/pixelcnn/data/',
                 validation_pct=0.2, seed=49, **kwargs):
    all_train = Dataset(inputdir)
    rng = np.random.RandomState(seed)
    indices = np.arange(all_train.nfiles)
    rng.shuffle(indices)
    train_start = int(np.round(all_train.nfiles * validation_pct))
    validation_indices, train_indices = indices[:train_start], indices[train_start:]
    train = Dataset(inputdir, file_indices=train_indices)
    validation = Dataset(inputdir, file_indices=validation_indices)
    test = Dataset(inputdir, test_set=True)
    return Datasets(train=train, validation=validation, test=test, nfeatures=train.nfeatures, nlabels=train.nlabels)



