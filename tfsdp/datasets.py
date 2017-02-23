import numpy as np
import collections

Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test', 'bins'])

class DataSet(object):

    def __init__(self,
               features,
               labels,
               batch_size=50,
               seed=42):
        """A basic dataset where each row contains a single feature vector and label
        """
        assert features.shape[0] == labels.shape[0], (
              'features.shape: %s labels.shape: %s' % (features.shape, labels.shape))
        self._num_examples = features.shape[0]
        self._features = features
        self._labels = labels
        self._num_features = features.shape[1]
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._batch_size = batch_size
        self._rng = np.random.RandomState(seed)
        self.p = 0

    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def num_features(self):
        return self._num_features

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size=None):
        """Return the next `batch_size` examples from this data set."""
        if batch_size is None: batch_size = self._batch_size
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._features = self._features[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._features[start:end], self._labels[start:end]

    def savetxt(self, filename, delimiter=','):
        np.savetxt(filename, np.concatenate((self._features, self._raw_labels), axis=1), delimiter=delimiter)

    def reset(self):
        self.p = 0

    def __next__(self, n=None):
        """ n is the number of examples to fetch """
        if n is None: n = self._batch_size

        # on first iteration permute all data
        if self.p == 0:
            inds = self._rng.permutation(self._features.shape[0])
            self._features = self._features[inds]
            self._labels = self._labels[inds]

        # on last iteration reset the counter and raise StopIteration
        if self.p >= self._features.shape[0]:
            self.reset() # reset for next time we get called
            raise StopIteration

        # on intermediate iterations fetch the next batch
        end = min(self.p + n, self._features.shape[0])
        x = self._features[self.p : end]
        y = self._labels[self.p : end]
        self.p += self._batch_size

        return x,y

    def __iter__(self):
        return self

    next = __next__  # Python 2 compatibility (https://stackoverflow.com/questions/29578469/how-to-make-an-object-both-a-python2-and-python3-iterator)

