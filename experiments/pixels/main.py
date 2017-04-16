#!/usr/bin/env python
#import matplotlib.pylab as plt
import numpy as np
import sys
import os
import argparse
import tensorflow as tf
from tfsdp.utils import tv_distance, ks_distance
from model import create_model, train_dict, test_dict

def score_model(sess, model, dist, data, tf_X):
    loss = 0
    for step, (X, y) in enumerate(data):
        feed_dict = test_dict(model, dist, X, y)
        feed_dict[tf_X] = X
        loss += sess.run(dist.test_loss, feed_dict=feed_dict)
    return loss

def explicit_score(sess, model, dist, data, tf_X):
    logprobs = 0
    squared_err = 0
    indices = np.array(list(np.ndindex(dist._num_classes)))
    n = 0
    for X, y in data:
        for i in xrange(len(X)):
            feed_dict = test_dict(model, dist, X[i:i+1], y[i:i+1])
            feed_dict[tf_X] = X[i:i+1]
            density = sess.run(dist.density, feed_dict=feed_dict)[0]
            logprobs += np.log(density[tuple(y[i])])
            prediction = np.array([density[tuple(idx)] * idx for idx in indices]).sum(axis=0)
            squared_err += np.linalg.norm(y[i] - prediction)**2
            n += 1
    rmse = np.sqrt(squared_err / float(n))
    print 'Explicit logprobs: {0} RMSE: {1}'.format(logprobs, rmse)
    return logprobs, rmse
    
def main():
    parser = argparse.ArgumentParser(description='Predicts pixel intensities given a random subset of an image.')

    # Experiment settings
    parser.add_argument('model', choices=['multinomial', 'gmm', 'lmm', 'sdp'], help='The model type. gmm is mixture density networks. lmm is logistic mixture model. sdp is smoothed k-d trees.')
    parser.add_argument('--inputdir', default='experiments/pixels/data', help='The directory where the input data files will be stored.')
    parser.add_argument('--outputdir', default='experiments/pixels/results', help='The directory where the output data files will be stored.')
    parser.add_argument('--dataset', choices=['mnist', 'cifar'], default='cifar', help='The dataset to use. MNIST uses grayscale pixels; CIFAR uses RGB pixels')
    parser.add_argument('--variable_scope', default='pixels-', help='The variable scope that the model will be created with.')
    parser.add_argument('--train_id', type=int, default=0, help='A trial ID. All models trained with the same trial ID will use the same train/validation datasets.')
    parser.add_argument('--train_samples', type=int, default=50000, help='The number of training examples to use.')
    parser.add_argument('--test_samples', type=int, default=10000, help='The number of training examples to use.')
    parser.add_argument('--validation_pct', type=float, default=0.2,
                                        help='The number of samples to hold out for a validation set. This is a percentage of the training samples.')
    parser.add_argument('--dimsize', type=int, default=256, help='The number of bins for each subpixel intensity (max 256, must be a power of 2).')

    # Optimizer settings
    parser.add_argument('--epsilon', type=float, default=1.0, help='The numerical stability constant for Adam.')
    parser.add_argument('--initial_learning_rate', type=float, default=0.1, help='The initial learning rate for the optimizer.')
    parser.add_argument('--min_learning_rate', type=float, default=1e-4, help='The initial learning rate for the optimizer.')
    parser.add_argument('--learning_decay', type=float, default=0.25, help='The decay rate for the learning rate.')
    parser.add_argument('--nepochs', type=int, default=100, help='The maximum number of training epochs.')
    parser.add_argument('--batchsize', type=int, default=50, help='The number of training samples per mini-batch.')
    parser.add_argument('--unimprove_wait', type=int, default=5, help='The number of epochs without improvement before we decrease the learning rate or stop early.')

    # SDP settings
    parser.add_argument('--lam', type=float, default=0.05, help='The lambda penalty value for the smoothed k-d tree.')
    parser.add_argument('--k', type=int, default=1, help='The order of the trend filtering penalty matrix for the smoothed k-d tree.')
    parser.add_argument('--neighbor_radius', type=int, default=5, help='The number of neighbors in each axis-aligned direction along the grid for the smoothed k-d tree.')

    # GMM/LMM settings
    parser.add_argument('--num_components', type=int, default=5, help='The number of mixture components for gmm or lmm models.')

    # Get the arguments from the command line
    args = parser.parse_args()
    dargs = vars(args)

    # Get the parameters
    if args.model == 'sdp':
        dargs['outfile'] = os.path.join(dargs['outputdir'], '{model}_{dataset}_{train_samples}_{k}_{lam}_{train_id}'.format(**dargs))
        dargs['variable_scope'] = '{model}-{dataset}-{train_samples}-{k}-{lam}-{train_id}'.format(**dargs)
    elif args.model in ('gmm', 'lmm'):
        dargs['outfile'] = os.path.join(dargs['outputdir'], '{model}_{dataset}_{train_samples}_{num_components}_{train_id}'.format(**dargs))
        dargs['variable_scope'] = '{model}-{dataset}-{train_samples}-{num_components}-{train_id}'.format(**dargs)
    elif args.model == 'multinomial':
        dargs['outfile'] = os.path.join(dargs['outputdir'], '{model}_{dataset}_{train_samples}_{train_id}'.format(**dargs))
        dargs['variable_scope'] = '{model}-{dataset}-{train_samples}-{train_id}'.format(**dargs)
    else:
        raise Exception('Unknown model type: {model}'.format(**dargs))

    # Get the data
    if args.dataset == 'mnist':
        from mnist_utils import DataLoader
        train_data = DataLoader(args.inputdir, 'train', args.train_samples, args.batchsize, seed=args.train_id, dimsize=args.dimsize)
        validate_data = DataLoader(args.inputdir, 'validate', args.train_samples, args.batchsize, seed=args.train_id, dimsize=args.dimsize)
        test_data = DataLoader(args.inputdir, 'test', args.test_samples, args.batchsize, seed=args.train_id, dimsize=args.dimsize)
    elif args.dataset == 'cifar':
        from cifar_utils import DataLoader
        train_data = DataLoader(args.inputdir, 'train', args.train_samples, args.batchsize, seed=args.train_id, dimsize=args.dimsize)
        validate_data = DataLoader(args.inputdir, 'validate', args.train_samples, args.batchsize, seed=args.train_id, dimsize=args.dimsize)
        test_data = DataLoader(args.inputdir, 'test', args.test_samples, args.batchsize, seed=args.train_id, dimsize=args.dimsize)
    else:
        raise Exception('Unknown dataset: {dataset}'.format(**dargs))

    dargs['x_shape'] = train_data.x_shape()
    dargs['y_shape'] = train_data.y_shape()
    # dargs['lazy_density'] = True # density is too big to enumerate for cifar
    dargs['one_hot'] = False # We use just the intensities not a one-hot
    
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

    # Get the X placeholder and the output distribution model
    tf_X, dist = create_model(**dargs)
    saver = tf.train.Saver()

    # Get the optimizer
    cur_learning_rate = args.initial_learning_rate
    learning_rate = tf.placeholder(tf.float32, shape=[])
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=args.epsilon)
    train_step = opt.minimize(dist.train_loss)

    sess.run(tf.global_variables_initializer())

    print('Beginning training and going for a maximum of {nepochs} epochs'.format(**dargs))
    sys.stdout.flush()

    best_loss = None
    epochs_since_improvement = 0
    for epoch in xrange(args.nepochs):
        # Do all the minibatch updates for this epoch
        for step, (X, y) in enumerate(train_data):
            feed_dict = train_dict(args.model, dist, X, y)
            feed_dict[tf_X] = X
            feed_dict[learning_rate] = cur_learning_rate
            sess.run(train_step, feed_dict=feed_dict)
            if step % 100 == 0:
                print('\tEpoch {0}, step {1}'.format(epoch, step))
                sys.stdout.flush()

        # Test if the model improved on the validation set
        validation_loss = score_model(sess, args.model, dist, validate_data, tf_X)

        # Check if we are improving
        if best_loss is None or validation_loss < best_loss:
            best_loss = validation_loss
            epochs_since_improvement = 0
            print('Found new best model. Saving to {outfile}'.format(**dargs))
            sys.stdout.flush()
            saver.save(sess, dargs['outfile'])
        else:
            epochs_since_improvement += 1

        print('Epoch #{0} Validation loss: {1} Epochs since improvement: {2} (learning rate: {3})'.format(epoch, validation_loss, epochs_since_improvement, cur_learning_rate))
        if epochs_since_improvement >= args.unimprove_wait:
            cur_learning_rate *= args.learning_decay
            if cur_learning_rate < args.min_learning_rate:
                print('Stopping.')
                sys.stdout.flush()
                break
            else:
                print('Decreasing learning rate. New rate: {0}'.format(cur_learning_rate))
                epochs_since_improvement = 0

    # Reset the model back to the best version
    saver.restore(sess, dargs['outfile'])

    # Save the validation score for this model
    print('Finished training. Scoring model...')
    sys.stdout.flush()
    
    logprobs, rmse = explicit_score(sess, args.model, dist, test_data, tf_X)
    np.savetxt(dargs['outfile'] + '_score.csv', [best_loss, logprobs, rmse, args.k, args.lam, args.num_components])


if __name__ == '__main__':
    main()


