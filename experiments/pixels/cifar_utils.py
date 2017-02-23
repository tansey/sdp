"""
Utilities for downloading and unpacking the CIFAR-10 dataset, originally published
by Krizhevsky et al. and hosted here: https://www.cs.toronto.edu/~kriz/cifar.html
"""

import os
import sys
import tarfile
from six.moves import urllib
import numpy as np

def maybe_download_and_extract(data_dir, url='http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'):
    if not os.path.exists(os.path.join(data_dir, 'cifar-10-batches-py')):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        filename = url.split('/')[-1]
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                    float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()
            filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
            print()
            statinfo = os.stat(filepath)
            print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
            tarfile.open(filepath, 'r:gz').extractall(data_dir)

def unpickle(file):
    fo = open(file, 'rb')
    if (sys.version_info >= (3, 0)):
        import pickle
        d = pickle.load(fo, encoding='latin1')
    else:
        import cPickle
        d = cPickle.load(fo)
    fo.close()
    return {'x': d['data'].reshape((10000,3,32,32)), 'y': np.array(d['labels']).astype(np.uint8)}

def load(data_dir, subset='train'):
    maybe_download_and_extract(data_dir)
    if subset=='train' or subset=='validate':
        train_data = [unpickle(os.path.join(data_dir,'cifar-10-batches-py','data_batch_' + str(i))) for i in range(1,6)]
        trainx = np.concatenate([d['x'] for d in train_data],axis=0)
        trainy = np.concatenate([d['y'] for d in train_data],axis=0)
        return trainx, trainy
    elif subset=='test':
        test_data = unpickle(os.path.join(data_dir,'cifar-10-batches-py','test_batch'))
        testx = test_data['x']
        testy = test_data['y']
        return testx, testy
    else:
        raise NotImplementedError('subset should be either train or test')

class DataLoader(object):
    """ an object that generates batches of CIFAR-10 data for training """

    def __init__(self, data_dir, subset, sample_size, batch_size, seed=1, validate_pct=0.2, shuffle=False, window=(10,10), dimsize=256):
        """ 
        - data_dir is location where to store files
        - subset is train|test 
        - batch_size is int, of #examples to load at once
        - rng is np.random.RandomState object for reproducibility
        """

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.window = window
        self.dimsize = dimsize
        self.rng = np.random.RandomState(seed)

        # create temporary storage for the data, if not yet created
        if not os.path.exists(data_dir):
            print('creating folder', data_dir)
            os.makedirs(data_dir)

        # load MNIST data to RAM
        self.data, self.labels = load(os.path.join(data_dir,'cifar-10-python'), subset=subset)
        self.data = np.transpose(self.data, (0,2,3,1)) # (N,3,32,32) -> (N,32,32,3)
        indices = self.rng.choice(self.data.shape[0], size=sample_size)
        loc_x = self.rng.randint(window[0], self.data.shape[1], size=sample_size)
        loc_y = self.rng.randint(window[1], self.data.shape[2], size=sample_size)
        # Get the target pixel as label
        self.labels = self.data[indices, loc_x, loc_y, :]
        if dimsize < 256:
            self.labels = np.floor(self.labels / (256/dimsize)).astype(int)
         # Get a local window before the target pixel
        self.data = np.array([self.data[i, x-window[0] : x, y-window[1] : y, :] for i,x,y in zip(indices, loc_x, loc_y)]) / 255.

        validate_start = int(np.round(self.data.shape[0]*(1-validate_pct)))
        if subset == 'validate':
            self.data = self.data[validate_start:]
            self.labels = self.labels[validate_start:]
        else:
            self.data = self.data[:validate_start]
            self.labels = self.labels[:validate_start]
        
        self.p = 0 # pointer to where we are in iteration

    def x_shape(self):
        return self.data.shape

    def y_shape(self):
        return self.labels.shape

    def reset(self):
        self.p = 0

    def __iter__(self):
        return self

    def __next__(self, n=None):
        """ n is the number of examples to fetch """
        if n is None: n = self.batch_size

        # on first iteration lazily permute all data
        if self.p == 0 and self.shuffle:
            inds = self.rng.permutation(self.data.shape[0])
            self.data = self.data[inds]
            self.labels = self.labels[inds]

        # on last iteration reset the counter and raise StopIteration
        if self.p + n > self.data.shape[0]:
            self.reset() # reset for next time we get called
            raise StopIteration

        # on intermediate iterations fetch the next batch
        x = self.data[self.p : self.p + n]
        y = self.labels[self.p : self.p + n]
        self.p += self.batch_size

        return x,y

    next = __next__  # Python 2 compatibility (https://stackoverflow.com/questions/29578469/how-to-make-an-object-both-a-python2-and-python3-iterator)


if __name__ == '__main__':
    dimsize=64
    loader = DataLoader('experiments/pixels/data/', 'train', 100000, 50, dimsize=dimsize)
    print loader.data.shape, loader.labels.shape
    print loader.data[0], loader.labels[0]
    import matplotlib.pylab as plt
    fig,ax = plt.subplots(3)
    ax[0].hist(loader.labels[:,0], bins=np.arange(dimsize+1))
    ax[1].hist(loader.labels[:,1], bins=np.arange(dimsize+1))
    ax[2].hist(loader.labels[:,2], bins=np.arange(dimsize+1))
    ax[0].set_xlim([0,dimsize])
    ax[1].set_xlim([0,dimsize])
    ax[2].set_xlim([0,dimsize])
    plt.show()

