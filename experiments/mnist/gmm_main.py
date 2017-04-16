#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import argparse

from edward.stats import norm
from keras import backend as K
from keras.regularizers import l2
from keras.layers import Dense
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from scipy import stats
from utils import read_data_sets
from tfsdp.utils import tv_distance, ks_distance

IMAGE_ROWS = 28
IMAGE_COLS = 28

class MixtureDensityNetwork:
  """
  Mixture density network for outputs y on inputs x.

  p((x,y), (z,theta))
  = sum_{k=1}^K pi_k(x; theta) Normal(y; mu_k(x; theta), sigma_k(x; theta))

  where pi, mu, sigma are the output of a neural network taking x
  as input and with parameters theta. There are no latent variables
  z, which are hidden variables we aim to be Bayesian about.
  """
  def __init__(self, K):
    self.K = K

  def neural_network(self, X):
    """pi, mu, sigma = NN(x; theta)"""
    X_image = tf.reshape(X, [-1,IMAGE_ROWS,IMAGE_COLS,1])
    conv1 = Convolution2D(32, 5, 5, border_mode='same', activation=K.relu, W_regularizer=l2(0.01),
                          input_shape=(IMAGE_ROWS, IMAGE_COLS, 1))(X_image)
    pool1 = MaxPooling2D(pool_size=(2,2), border_mode='same')(conv1)
    conv2 = Convolution2D(64, 5, 5, border_mode='same', activation=K.relu, W_regularizer=l2(0.01))(pool1)
    pool2 = MaxPooling2D(pool_size=(2,2), border_mode='same')(conv2)
    pool2_flat = tf.reshape(pool2, [-1, IMAGE_ROWS//4 * IMAGE_COLS//4 * 64])
    hidden1 = Dense(1024, W_regularizer=l2(0.01), activation=K.relu)(pool2_flat)
    hidden2 = Dense(64, W_regularizer=l2(0.01), activation=K.relu)(hidden1)
    self.mus = Dense(self.K)(hidden2)
    self.sigmas = Dense(self.K, activation=K.softplus)(hidden2)
    self.pi = Dense(self.K, activation=K.softmax)(hidden2)

  def log_prob(self, xs, zs):
    """log p((xs,ys), (z,theta)) = sum_{n=1}^N log p((xs[n,:],ys[n]), theta)"""
    # Note there are no parameters we're being Bayesian about. The
    # parameters are baked into how we specify the neural networks.
    X, y = xs['X'], xs['y']
    self.neural_network(X)
    result = self.pi * norm.prob(y, self.mus, self.sigmas)
    result = tf.log(tf.reduce_sum(result, 1))
    return tf.reduce_sum(result)

def fit_to_grid(all_pis, all_mus, all_sigmas, nbins=100):
    x = np.linspace(0,1,nbins)
    fits = np.zeros((all_pis.shape[0], nbins))
    for i, (pis, mus, sigmas) in enumerate(zip(all_pis, all_mus, all_sigmas)):
        for j, (weight_mix, mu_mix, sigma_mix) in enumerate(zip(pis, mus, sigmas)):
            fits[i] += stats.norm.pdf(x, mu_mix, sigma_mix) * weight_mix
    fits /= fits.sum(axis=1)[:,np.newaxis]
    return fits

def save_scores_and_fits(scores_filename, fits_filename, mnist, validation_score, test_fit):
    tv_score = tv_distance(mnist.test.labels, test_fit).mean()
    ks_score = ks_distance(mnist.test.labels, test_fit).mean()
    np.savetxt(fits_filename, test_fit, delimiter=',')
    np.savetxt(scores_filename, [validation_score, tv_score, ks_score], delimiter=',')

def validate(sess, inference, X, y, X_validate, y_validate):
    validation_loss = 0
    val_iter_size = int(np.round(X_validate.shape[0] / 100))
    for i in xrange(100):
        if i == 99:
            features = X_validate[val_iter_size*i:]
            labels = y_validate[val_iter_size*i:]
            size = X_validate.shape[0] - val_iter_size*i
        else:
            features = X_validate[val_iter_size*i:val_iter_size*(i+1)]
            labels = y_validate[val_iter_size*i:val_iter_size*(i+1)]
            size = val_iter_size
        validation_loss += sess.run(inference.loss, feed_dict={X: features, y: labels}) * size
    return validation_loss / float(X_validate.shape[0])

def fit_to_test(sess, model, X, X_test, nbins):
    fits = np.zeros((X_test.shape[0], nbins))
    iter_size = int(np.round(X_test.shape[0] / 100))
    for i in xrange(100):
        start = iter_size * i
        if i == 99:
            stop = fits.shape[0]
        else:
            stop = iter_size * (i+1)
        pred_weights, pred_means, pred_std = sess.run([model.pi, model.mus, model.sigmas], feed_dict={X: X_test[start:stop]})
        fits[start:stop] = fit_to_grid(pred_weights, pred_means, pred_std, nbins=nbins)
    return fits

def main():
    parser = argparse.ArgumentParser(description='Trains a conditional density estimation model on the MNIST dataset.')
    parser.add_argument('--inputdir', default='experiments/mnist/data', help='The directory where the input data files will be stored.')
    parser.add_argument('--dist_type', choices=['gmm', 'discontinuous_gmm', 'edge_biased'], default='gmm', help='The type of underlying distribution that the labels are drawn from.')
    parser.add_argument('--train_id', default='0', help='A trial ID. All models trained with the same trial ID will use the same train/validation datasets.')
    parser.add_argument('--train_samples', type=int, default=60000, help='The number of training examples to use.')
    parser.add_argument('--validation_samples', type=float, default=0.2,
                                        help='The number of samples to hold out for a validation set. This is a percentage of the training samples.')
    parser.add_argument('--batchsize', type=int, default=50, help='The number of training samples per mini-batch.')
    parser.add_argument('--max_steps', type=int, default=100000, help='The maximum number of training steps.')
    parser.add_argument('--num_components', type=int, default=3, help='The number of components for the GMM method.')
    parser.add_argument('--nbins', type=int, default=128, help='The number of bins in the discrete distribution.')

    args = parser.parse_args()
    dargs = vars(args)

    dargs['model'] = 'gmm_{num_components}'.format(**dargs)
    dargs['outfile'] = 'experiments/mnist/results/{model}_{dist_type}_{train_samples}_{train_id}'.format(**dargs)
    dargs['variable_scope'] = 'mnist-{model}-{dist_type}-{train_samples}-{train_id}'.format(**dargs)
    mnist = read_data_sets(args.inputdir, **dargs)
    X_validate, y_validate = mnist.validation.features, np.argmax(mnist.validation.labels, axis=-1)[:,np.newaxis] / args.nbins
    X_test, y_test = mnist.test.features, np.argmax(mnist.test.labels, axis=-1)[:,np.newaxis] / args.nbins

    ed.set_seed(42)
    
    X = tf.placeholder(tf.float32, [None, IMAGE_ROWS*IMAGE_COLS], name='X')
    y = tf.placeholder(tf.float32, [None, 1], name='y')
    data = {'X': X, 'y': y}

    # Create the model
    model = MixtureDensityNetwork(args.num_components)

    inference = ed.MAP([], data, model)
    optimizer = tf.train.AdamOptimizer(0.0001, epsilon=10.0)
    sess = ed.get_session()  # Start TF session
    K.set_session(sess)  # Pass session info to Keras
    inference.initialize(optimizer=optimizer)

    init = tf.global_variables_initializer()
    init.run()

    saver = tf.train.Saver()

    # Other models are run for 100K SGD iterations with minibatch size 50
    n_epoch = args.max_steps
    train_loss = np.zeros(n_epoch)
    test_loss = np.zeros(n_epoch)
    best_loss = None
    for i in range(n_epoch):
        X_train, y_train = mnist.train.next_batch(args.batchsize)
        y_train = np.argmax(y_train, axis=-1)[:,np.newaxis] / args.nbins
        info_dict = inference.update(feed_dict={X: X_train, y: y_train})
        if i % 100 == 0:
            train_loss[i] = info_dict['loss']
            test_loss[i] = validate(sess, inference, X, y, X_validate, y_validate)
            print(i, train_loss[i], test_loss[i])
            if i == 0 or test_loss[i] < best_loss:
                best_loss = test_loss[i]
                saver.save(sess, dargs['outfile'])

    saver.restore(sess, dargs['outfile'])

    print('Finished training. Scoring model...')
    sys.stdout.flush()

    # Get the resulting GMM outputs on the test set
    fits = fit_to_test(sess, model, X, X_test, args.nbins)
    save_scores_and_fits('{outfile}_score.csv'.format(**dargs), '{outfile}_fits.csv'.format(**dargs), mnist, best_loss, fits)

    print('Best model saved to {outfile}'.format(**dargs))
    sys.stdout.flush()

if __name__ == '__main__':
    main()






