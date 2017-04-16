import os
import numpy as np
import collections
from scipy.stats import norm, expon
from tfsdp.datasets import DataSet
from tfsdp.utils import tv_distance, ks_distance
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets, extract_images, extract_labels

MnistDatasets = collections.namedtuple('Datasets', ['train', 'validation', 'test', 'densities', 'bins'])

class MixtureOfNormals(object):
    def __init__(self,
                 weights,
                 means,
                 stdevs):
        """Construct a mixture-of-normals distribution."""
        assert(len(means) == len(stdevs))
        self._weights = weights
        self._means = means
        self._stdevs = stdevs

    @property
    def means(self):
        return self._means

    @property
    def stdevs(self):
        return self._stdevs

    @property
    def weights(self):
        return self._weights

    def pdf(self, x):
        if type(x) is float:
            return np.array([w * norm.pdf(x, loc=m, scale=s) for w,m,s in zip(self._weights, self._means, self._stdevs)]).sum()
        return np.array([w * norm.pdf(x, loc=m, scale=s) for w,m,s in zip(self._weights, self._means, self._stdevs)]).sum(axis=0)

    def sample(self, size=1):
        c = np.random.choice(np.arange(len(self._weights)), p=self._weights, size=size)
        if size == 1:
            return np.random.normal(loc=self._means[c], scale=self._stdevs[c], size=size)
        return np.array([np.random.normal(loc=self._means[ci], scale=self._stdevs[ci]) for ci in c])

class DiscontinuousDistribution(object):
    def __init__(self,
                 splits,
                 dists):
        self._splits = splits
        self._dists = dists

    def pdf(self, x):
        for s,dist in zip(self._splits, self._dists):
            if x < s:
                return dist.pdf(x)
        return dist.pdf(x)

class EdgeBiasedDistribution(object):
    def __init__(self,
                 left_lam,
                 right_lam,
                 middle_mean,
                 middle_stdev,
                 right_edge = 10.):
        self._left_lam = left_lam
        self._right_lam = right_lam
        self._middle_mean = middle_mean
        self._middle_stdev = middle_stdev
        self._right_edge = right_edge

    def pdf(self, x):
        if x > self._right_edge:
            return expon.pdf(x) / 2. + norm.pdf(x, self._middle_mean, self._middle_stdev) / 2.
        if x < 1e-4:
            return expon.pdf(self._right_edge - x) / 2. + norm.pdf(x, self._middle_mean, self._middle_stdev) / 2.
        return (expon.pdf(x)
                + expon.pdf(self._right_edge - x) / 3. 
                + norm.pdf(x, self._middle_mean, self._middle_stdev) / 3.)

class DiscreteProbabilityDistribution(object):
    def __init__(self, x, weights):
        assert(len(x) == len(weights))
        assert(weights.min() >= 0)
        assert(weights.max() > 0)
        self._x = x
        self._weights = weights / weights.sum()
        self._bins = np.arange(len(x))

    @property
    def x(self):
        return self._x

    @property
    def weights(self):
        return self._weights

    def prob(self, x_i):
        return self._weights[x_i]

    def sample(self, size=1):
        chosen = np.random.choice(self._bins, p=self._weights, size=size)
        return self._x[chosen]

def reshape_image(img):
    return img.reshape((img.shape[0], img.shape[1]*img.shape[2]))

def load_or_create_densities(filename, nbins, dist_type='gmm'):
    if dist_type == '2d':
        return load_or_create_2d(filename, nbins, dist_type)
    if os.path.exists(filename):
        weights = np.loadtxt(filename, delimiter=',')
        class_densities = [DiscreteProbabilityDistribution(np.arange(nbins), w) for w in weights]
    else:
        # Create a random mixture-of-normals density for each MNIST class
        bins = np.linspace(0.1,10,nbins)
        dists = {'gmm': mixture_of_normals,
                 'discontinuous_gmm': discontinuous_mixture_of_normals,
                 'edge_biased': edge_biased}
        class_densities = [discretize_pdf(dists[dist_type](), bins) for _ in xrange(10)]
        save_densities(filename, class_densities)
    return np.arange(nbins), class_densities

def load_or_create_2d(filename, nbins, dist_type):
    pass # TODO

def mixture_of_normals(num_components=3, mean_range=(1,7), stdev_range=(0.3, 2)):
    w = np.random.random(size=num_components)
    w /= w.sum()
    means = np.random.uniform(mean_range[0], mean_range[1], size=num_components)
    stdevs = np.random.uniform(stdev_range[0], stdev_range[1], size=num_components)
    return MixtureOfNormals(w, means, stdevs)

def discontinuous_mixture_of_normals(num_dists=3, num_components=3, mean_range=(1,7), stdev_range=(0.3, 2), stick_length=10.):
    stick = stick_length
    dists = []
    splits = []
    for i in xrange(num_dists):
        stick = np.random.random() * stick
        splits.append((stick_length - stick))
        dists.append(mixture_of_normals())
    return DiscontinuousDistribution(splits, dists)

def edge_biased(lam_range=(0.25,2.), mean_range=(1,7), stdev_range=(0.3, 2)):
    left_lam, right_lam = np.random.uniform(lam_range[0], lam_range[1], size=2)
    middle_mean = np.random.uniform(mean_range[0], mean_range[1])
    middle_stdev = np.random.uniform(stdev_range[0], stdev_range[1])
    return EdgeBiasedDistribution(left_lam, right_lam, middle_mean, middle_stdev)

def discretize_pdf(dist, bins):
    weights = np.array([dist.pdf(x) for x in bins])
    return DiscreteProbabilityDistribution(np.arange(len(bins)), weights)

def save_densities(filename, dists):
    x = np.array([d.weights for d in dists])
    np.savetxt(filename, x, delimiter=',')

def sample_labels(labels, dists):
    return np.array([dists[i].sample() for i in labels], dtype=np.int32)

def ints_to_multinomials(samples, bins):
    labels = np.zeros((len(samples), len(bins)))
    for j,b in enumerate(bins):
        for i in (np.where(samples == b)[0]):
            labels[i, j] = 1
    return labels

def load_or_create_labels(train_dir, class_densities, bins, validation_samples,
                          train_id, train_indices, dist_type, one_hot=False,
                          ignore_nbins_in_filename=False):
    dargs = {'traindir': train_dir, 'train_id': train_id, 'dist_type': dist_type, 'nbins': len(bins)}
    if ignore_nbins_in_filename:
        TRAIN_LABELS_FILE = '{traindir}/train_labels_{dist_type}_{train_id}.csv'.format(**dargs)
        TEST_LABELS_FILE = '{traindir}/test_labels_{dist_type}_{train_id}.csv'.format(**dargs)
    else:
        TRAIN_LABELS_FILE = '{traindir}/train_labels_{dist_type}_{nbins}_{train_id}.csv'.format(**dargs)
        TEST_LABELS_FILE = '{traindir}/test_labels_{dist_type}_{nbins}_{train_id}.csv'.format(**dargs)
    if os.path.exists(TRAIN_LABELS_FILE) and os.path.exists(TEST_LABELS_FILE):
        print 'loading training'
        train_labels = np.loadtxt(TRAIN_LABELS_FILE, delimiter=',', dtype=int)
        print 'loading testing'
        test_labels = np.loadtxt(TEST_LABELS_FILE, delimiter=',', dtype=int)
    else:
        SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
        TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
        TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

        local_file = base.maybe_download(TRAIN_LABELS, train_dir,
                                   SOURCE_URL + TRAIN_LABELS)

        with open(local_file, 'rb') as f:
            train_labels = extract_labels(f)

        local_file = base.maybe_download(TEST_LABELS, train_dir,
                                   SOURCE_URL + TEST_LABELS)

        with open(local_file, 'rb') as f:
            test_labels = extract_labels(f)

        # Shuffle the labels to match the shuffled images
        train_labels = np.array(train_labels[train_indices])

        # Replace the class label with a sample from the class's density
        train_labels = bins[sample_labels(train_labels, class_densities)]

        # Replace the label with its corresponding true density
        # test_labels = np.array([class_densities[lab].weights for lab in test_labels])

        np.savetxt(TRAIN_LABELS_FILE, train_labels, delimiter=',', fmt='%d')
        np.savetxt(TEST_LABELS_FILE, test_labels, delimiter=',', fmt='%d')

    print 'Labels loaded.'
    if one_hot:
        train_labels = ints_to_multinomials(train_labels, bins)
    
    validation_labels = train_labels[:validation_samples]
    train_labels = train_labels[validation_samples:]

    return train_labels, validation_labels, test_labels

def read_data_sets(train_dir,
                   train_samples=60000,
                   validation_samples=0.1,
                   train_id=0,
                   nbins=128,
                   one_hot=False,
                   dist_type='gmm',
                   ignore_nbins_in_filename=False,
                   **kwargs):
    SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    if ignore_nbins_in_filename:
        DENSITY_URL = '{traindir}/class_densities_{dist_type}_{trainid}.csv'.format(dist_type=dist_type,
                                                                                    traindir=train_dir,
                                                                                    trainid=train_id)
    else:
        DENSITY_URL = '{traindir}/class_densities_{dist_type}_{nbins}_{trainid}.csv'.format(nbins=nbins,
                                                                                            dist_type=dist_type,
                                                                                            traindir=train_dir,
                                                                                            trainid=train_id)
    local_file = base.maybe_download(TRAIN_IMAGES, train_dir,
                                   SOURCE_URL + TRAIN_IMAGES)
    with open(local_file, 'rb') as f:
        train_images = extract_images(f) / 255.

    local_file = base.maybe_download(TEST_IMAGES, train_dir,
                                   SOURCE_URL + TEST_IMAGES)
    
    with open(local_file, 'rb') as f:
        test_images = extract_images(f) / 255.

    # Enable pre-shuffling of the training set to support multiple trials
    train_indices_filename = '{0}/train_indices_{1}_{2}.csv'.format(train_dir, dist_type, train_id)
    if os.path.exists(train_indices_filename):
        train_indices = np.loadtxt(train_indices_filename, delimiter=',').astype(np.int32)
    else:
        train_indices = np.arange(60000, dtype=np.int32)
        np.random.shuffle(train_indices)
        np.savetxt(train_indices_filename, train_indices, delimiter=',', fmt='%d')

    # Shuffle the images according to the specified indices
    train_images = np.array(train_images[train_indices])

    num_validation = int(np.round(train_samples*validation_samples))
    num_train = train_samples - num_validation

    validation_images = train_images[:num_validation]
    train_images = train_images[num_validation:]

    # Reshape the images to be vectors
    train_images = reshape_image(train_images)
    validation_images = reshape_image(validation_images)
    test_images = reshape_image(test_images)

    # Create latent discrete densities for each image class
    bins, class_densities = load_or_create_densities(DENSITY_URL, nbins, dist_type)

    # Get the labels to regress on
    train_labels, validation_labels, test_labels = load_or_create_labels(train_dir, class_densities, bins, 
                                                                         num_validation, train_id, train_indices,
                                                                         dist_type, one_hot=one_hot,
                                                                         ignore_nbins_in_filename=ignore_nbins_in_filename)

    train_images = train_images[:num_train]
    train_labels = train_labels[:num_train]

    # Create some helper datasets
    train = DataSet(train_images, train_labels)
    validation = DataSet(validation_images, validation_labels)
    test = DataSet(test_images, test_labels)

    return MnistDatasets(train=train, validation=validation, test=test, densities=class_densities, bins=bins)

def validate(model, mnist, use_bic=False):
    validation_loss = 0
    val_iter_size = int(np.round(mnist.validation.features.shape[0] / 100))
    for i in xrange(100):
        if i == 99:
            features = mnist.validation.features[val_iter_size*i:]
            labels = mnist.validation.labels[val_iter_size*i:]
            size = mnist.validation.features.shape[0] - val_iter_size*i
        else:
            features = mnist.validation.features[val_iter_size*i:val_iter_size*(i+1)]
            labels = mnist.validation.labels[val_iter_size*i:val_iter_size*(i+1)]
            size = val_iter_size
        if use_bic:
            from pygfl.utils import calc_plateaus, hypercube_edges
            edges = hypercube_edges(model.layer._num_classes, use_map=True)
            fits = model.density.eval(feed_dict=model.fit_dict(features))
            dofs = [len(calc_plateaus(np.log(fit), edges)) for fit in fits]
            validation_loss += np.sum([-2 * np.log(fit[label]) + dof * (np.log(len(fit)) - np.log(2*np.pi)) for fit, label, dof in zip(fits, labels, dofs)])
        else:
            validation_loss += model.test_loss.eval(feed_dict=model.test_dict(features, labels)) * size
    return validation_loss / float(mnist.validation.features.shape[0])

def fit_to_test(model, mnist, batchsize=1, save_fits=True):
    num_batches = int(np.round(mnist.validation.features.shape[0] / batchsize))
    tv_score = []
    ks_score = []
    if save_fits:
        fits = []
    for i in xrange(num_batches):
        start = batchsize*i
        if start >= mnist.test.labels.shape[0]:
            break
        if i == (num_batches-1):
            stop = mnist.test.labels.shape[0]
        else:
            stop = batchsize*(i+1)
        test_fit = model.density.eval(feed_dict=model.fit_dict(mnist.test.features[start:stop]))
        tv_score.extend(np.array([tv_distance(mnist.densities[label].weights, fit) for label,fit in zip(mnist.test.labels[start:stop], test_fit)]))
        ks_score.extend(np.array([ks_distance(mnist.densities[label].weights, fit) for label,fit in zip(mnist.test.labels[start:stop], test_fit)]))
        if save_fits:
            fits.extend(test_fit)
    return np.mean(tv_score), np.mean(ks_score), fits if save_fits else None

def save_scores_and_fits(scores_filename, fits_filename, model, mnist, save_fits=True):
    validation_score = validate(model, mnist)
    tv_score, ks_score, test_fit = fit_to_test(model, mnist, save_fits=save_fits)
    # tv_score = np.mean([tv_distance(mnist.class_densities[l], test_fit[]))
    # ks_score = ks_distance(mnist.test.labels, test_fit).mean()
    np.savetxt(scores_filename, [validation_score, tv_score, ks_score], delimiter=',')
    if save_fits:
        np.savetxt(fits_filename, test_fit, delimiter=',')

if __name__ == '__main__':
    read_data_sets('experiments/mnist/data')

