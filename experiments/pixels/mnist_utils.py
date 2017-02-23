import os
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images

def load(data_dir, subset):
    SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'

    if subset == 'train' or subset == 'validate':
        local_file = base.maybe_download(TRAIN_IMAGES, data_dir, SOURCE_URL + TRAIN_IMAGES)
        with open(local_file, 'rb') as f:
            images = extract_images(f)
    elif subset == 'test':
        local_file = base.maybe_download(TEST_IMAGES, data_dir, SOURCE_URL + TEST_IMAGES)
        with open(local_file, 'rb') as f:
            images = extract_images(f)
    else:
        raise Exception('subset must be train or test')

    return images

class DataLoader(object):
    """ an object that generates batches of CIFAR-10 data for training """

    def __init__(self, data_dir, subset, sample_size, batch_size, seed=1, validate_pct=0.2, shuffle=False, window=(10,10), dimsize=256):
        """ 
        - data_dir is location where to store files
        - subset is train|test|validate
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
        self.data = load(os.path.join(data_dir,'mnist'), subset=subset)
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

    def get_num_labels(self):
        return np.amax(self.labels) + 1

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
    dimsize = 64
    loader = DataLoader('experiments/pixels/data/', 'train', 100000, 50, dimsize=64)
    print loader.data.shape, loader.labels.shape
    print loader.data[0], loader.labels[0]
    import matplotlib.pylab as plt
    plt.hist(loader.labels, bins=np.arange(dimsize+1))
    plt.xlim([0,dimsize])
    plt.show()

