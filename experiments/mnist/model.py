import numpy as np
import tensorflow as tf
from tfsdp.utils import weight_variable, bias_variable
from tfsdp.models import MultinomialLayer, DiscreteParametricMixtureLayer,\
                         MultiscaleLayer, TrendFilteringLayer,\
                         SmoothedMultiscaleLayer, DiscreteLogisticMixtureLayer, \
                         LocallySmoothedMultiscaleLayer

class Model(object):
    def __init__(self, layer, x=None, density=None, labels=None,
                       train_loss=None, test_loss=None,
                       train_dict={}, test_dict={}, fit_dict={}):
        self._layer = layer
        self._x = x
        self._density = density
        self._labels = labels
        self._train_loss = train_loss
        self._test_loss = test_loss
        self._saver = tf.train.Saver()
        self._train_dict = train_dict
        self._test_dict = test_dict
        self._fit_dict = fit_dict

    @property
    def layer(self):
        return self._layer

    @property
    def x(self):
        return self._x

    @property
    def density(self):
        return self._density

    @property
    def labels(self):
        return self._labels

    @property
    def train_loss(self):
        return self._train_loss

    @property
    def test_loss(self):
        return self._test_loss

    @property
    def saver(self):
        return self._saver

    def train_dict(self, x, labels):
        self._train_dict[self._x] = x
        if isinstance(self.layer, SmoothedMultiscaleLayer):
            self._train_dict[self._labels] = labels if len(labels.shape) == 2 else labels[:,np.newaxis]
        elif isinstance(self.layer, LocallySmoothedMultiscaleLayer):
            self.layer.fill_train_dict(self._train_dict, labels)
        else:
            self._train_dict[self._labels] = labels
        return self._train_dict

    def test_dict(self, x, labels):
        self._test_dict[self._x] = x
        if isinstance(self.layer, SmoothedMultiscaleLayer):
            self._test_dict[self._labels] = labels if len(labels.shape) == 2 else labels[:,np.newaxis]
        elif isinstance(self.layer, LocallySmoothedMultiscaleLayer):
            self.layer.fill_test_dict(self._test_dict, labels)
        else:
            self._test_dict[self._labels] = labels
        return self._test_dict

    def fit_dict(self, x):
        self._test_dict[self._x] = x
        return self._test_dict

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def create_model(model, variable_scope='mnist-', nbins=128, **kwargs):
    with tf.variable_scope(variable_scope):
        IMAGE_ROWS = 28
        IMAGE_COLS = 28
        NUM_IMAGE_PIXELS = IMAGE_ROWS*IMAGE_COLS
        
        x = tf.placeholder(tf.float32, shape=[None, NUM_IMAGE_PIXELS])
        
        x_image = tf.reshape(x, [-1,28,28,1])

        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])

        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        W_fc1 = weight_variable([IMAGE_ROWS/4 * IMAGE_COLS/4 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, IMAGE_ROWS/4 * IMAGE_COLS/4 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        if model == 'multinomial':
            dist_model = MultinomialLayer(h_fc1_drop, 1024, nbins, **kwargs)
        elif model == 'gmm':
            dist_model = DiscreteParametricMixtureLayer(h_fc1_drop, 1024, nbins, component_dist='gaussian', **kwargs)
        elif model == 'multiscale':
            dist_model = MultiscaleLayer(h_fc1_drop, 1024, nbins, **kwargs)
        elif model == 'trendfiltering':
            dist_model = TrendFilteringLayer(h_fc1_drop, 1024, nbins, **kwargs)
        elif model == 'trendfiltering-multiscale':
            dist_model = SmoothedMultiscaleLayer(h_fc1_drop, 1024, nbins, **kwargs)
            # dist_model = LocallySmoothedMultiscaleLayer(h_fc1_drop, 1024, nbins, neighbor_radius=nbins, **kwargs)
        elif model == 'lmm':
            dist_model = DiscreteLogisticMixtureLayer(h_fc1_drop, 1024, nbins, **kwargs)
        elif model == 'sdp':
            dist_model = LocallySmoothedMultiscaleLayer(h_fc1_drop, 1024, nbins, **kwargs)
        else:
            raise Exception('Unknown model type: {0}'.format(model))

        return Model(dist_model, x=x, density=dist_model.density, labels=dist_model.labels,
                    train_loss=dist_model.train_loss, test_loss=dist_model.test_loss,
                    train_dict={keep_prob: 0.5}, test_dict={keep_prob: 1.0}, fit_dict={keep_prob: 1.0})






