#!/usr/bin/env python
#import matplotlib.pylab as plt
import numpy as np
import sys
import os
import argparse
import tensorflow as tf
from tfsdp.utils import tv_distance, ks_distance
from model import create_model, train_dict, test_dict
from scipy.stats import multivariate_normal as mvn

def explicit_score(sess, model, dist, data, tf_X):
    logprobs = 0
    squared_err = 0
    indices = np.array(list(np.ndindex(dist._num_classes)))
    rescaled_indices = indices / (np.array(dist._num_classes, dtype=float)[np.newaxis,:]-1) * 2 - 1
    print rescaled_indices
    n = 0
    for X, y in data:
        for i in xrange(len(X)):
            feed_dict = test_dict(model, dist, X[i:i+1], y[i:i+1])
            feed_dict[tf_X] = X[i:i+1]
            pi, mu, chol = sess.run(dist.mvn_params, feed_dict=feed_dict)
            pi = pi[0]
            mu = mu[0]
            chol = chol[0]
            probs = np.zeros(indices.shape[0])
            for k in xrange(dist._num_components):
                pi_k = pi[k]
                mu_k = mu[k]
                chol_k = chol[k]
                probs += pi_k * mvn.pdf(rescaled_indices, mu_k, chol_k.dot(chol_k.T)).clip(1e-12,np.inf) / mvn.pdf(indices, mu_k, chol_k.dot(chol_k.T)).sum()
            probs /= probs.sum()
            for idx,p in zip(indices,probs):
                if np.array_equal(idx,y[i]):
                    logprobs += np.log(p)
                    break
            prediction = np.array([p * idx for idx,p in zip(indices,probs)]).sum(axis=0)
            squared_err += np.linalg.norm(y[i] - prediction)**2
            print '#{0} Label: {1} Prediction: {2} Logprobs: {3}'.format(n, y[i], prediction, logprobs)
            sys.stdout.flush()
            n += 1
    rmse = np.sqrt(squared_err / float(n))
    print 'Explicit logprobs: {0} RMSE: {1}'.format(logprobs, rmse)
    return logprobs, rmse

def main():
    parser = argparse.ArgumentParser(description='Predicts pixel intensities given a random subset of an image.')

    # Experiment settings
    parser.add_argument('--inputdir', default='experiments/pixels/data', help='The directory where the input data files will be stored.')
    parser.add_argument('--outputdir', default='experiments/pixels/results', help='The directory where the input data files will be stored.')
    parser.add_argument('--variable_scope', default='pixels-', help='The variable scope that the model will be created with.')
    parser.add_argument('--train_id', type=int, default=0, help='A trial ID. All models trained with the same trial ID will use the same train/validation datasets.')
    parser.add_argument('--train_samples', type=int, default=50000, help='The number of training examples to use.')
    parser.add_argument('--test_samples', type=int, default=10000, help='The number of training examples to use.')
    parser.add_argument('--validation_pct', type=float, default=0.2,
                                        help='The number of samples to hold out for a validation set. This is a percentage of the training samples.')
    parser.add_argument('--dimsize', type=int, default=256, help='The number of bins for each subpixel intensity (max 256, must be a power of 2).')
    parser.add_argument('--batchsize', type=int, default=50, help='The number of training samples per mini-batch.')

    # GMM/LMM settings
    parser.add_argument('--num_components', type=int, default=5, help='The number of mixture components for gmm or lmm models.')

    # Get the arguments from the command line
    args = parser.parse_args()
    dargs = vars(args)
    dargs['model'] = 'gmm'
    dargs['dataset'] = 'cifar'
    dargs['outfile'] = os.path.join(dargs['outputdir'], '{model}_{dataset}_{train_samples}_{num_components}_{train_id}'.format(**dargs))
    dargs['variable_scope'] = '{model}-{dataset}-{train_samples}-{num_components}-{train_id}'.format(**dargs)


    # Get the data
    from cifar_utils import DataLoader
    train_data = DataLoader(args.inputdir, 'train', args.train_samples, args.batchsize, seed=args.train_id, dimsize=args.dimsize)
    validate_data = DataLoader(args.inputdir, 'validate', args.train_samples, args.batchsize, seed=args.train_id, dimsize=args.dimsize)
    test_data = DataLoader(args.inputdir, 'test', args.test_samples, args.batchsize, seed=args.train_id, dimsize=args.dimsize)

    dargs['x_shape'] = train_data.x_shape()
    dargs['y_shape'] = train_data.y_shape()
    dargs['lazy_density'] = True # density is too big to enumerate for cifar
    dargs['one_hot'] = False # We use just the intensities not a one-hot
    
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

    # Get the X placeholder and the output distribution model
    tf_X, dist = create_model(**dargs)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    # Reset the model back to the best version
    saver.restore(sess, dargs['outfile'])

    logprobs, rmse = explicit_score(sess, args.model, dist, test_data, tf_X)
    print logprobs, rmse
    np.savetxt(dargs['outfile'] + '_score.csv', [best_loss, logprobs, rmse, args.k, args.lam, args.num_components])



if __name__ == '__main__':
    main()

