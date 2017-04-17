#!/usr/bin/env python
import numpy as np
import sys
import os
import argparse
import tensorflow as tf
from model import create_model
from utils import load_dataset

def score_model(sess, model, dataset):
    loss = 0
    for step, (X, y) in enumerate(dataset.validation):
        for i in xrange(len(X)):
            feed_dict = model.test_dict(X[i:i+1], y[i:i+1])
            loss += sess.run(model.test_loss, feed_dict=feed_dict)
            break # TEMP
        break # TEMP
    return loss

def explicit_score(sess, model, dataset):
    logprobs = 0
    squared_err = 0
    indices = np.array(list(np.ndindex(model.layer._num_classes)))
    for X, y in dataset.test:
        for i in xrange(len(X)):
            feed_dict = model.test_dict(X[i:i+1], y[i:i+1])
            density = sess.run(model.density, feed_dict=feed_dict)[0]
            logprobs -= np.log(density[tuple(y[i])] * (np.prod(dataset.nlabels) / 255.**3))
            break # TEMP
        break # TEMP
            # prediction = np.array([density[tuple(idx)] * idx for idx in indices]).sum(axis=0)
            # squared_err += np.linalg.norm(dataset.test.labels[i] - prediction)**2
    # rmse = np.sqrt(squared_err / float(len(dataset.test.features)))
    # bits_per_dim = logprobs / (np.log(2.) * 3. * dataset.test.nexamples)
    bits_per_dim = logprobs / (np.log(2.) * 3. * 1.) # TEMP
    print 'Explicit logprobs: {} Bits/dim: {}'.format(logprobs, bits_per_dim)
    return logprobs, bits_per_dim


def main():
    parser = argparse.ArgumentParser(description='Trains an SDP model on preprocessed PixelCNN++ features.')

    # Experiment settings
    parser.add_argument('inputdir', default='experiments/pixelcnn/data', help='The directory where the input data files are be stored.')
    parser.add_argument('--model', choices=['multinomial', 'gmm', 'lmm', 'sdp'], default='sdp', help='The model type. gmm is mixture density networks. lmm is logistic mixture model. sdp is smoothed dyadic partitions.')
    parser.add_argument('--outputdir', default='experiments/pixelcnn/results', help='The directory where the output data files will be stored.')
    parser.add_argument('--validation_pct', type=float, default=0.2,
                                        help='The number of samples to hold out for a validation set. This is a percentage of the training samples.')
    parser.add_argument('--nepochs', type=int, default=1000, help='The maximum number of training epochs.')
    parser.add_argument('--batchsize', type=int, default=50, help='The mini-batch size.')
    
    # SDP settings
    parser.add_argument('--lam', type=float, default=0.05, help='The lambda penalty value for the smoothed k-d tree.')
    parser.add_argument('--k', type=int, default=1, help='The order of the trend filtering penalty matrix for the smoothed k-d tree.')
    parser.add_argument('--neighbor_radius', type=int, default=5, help='The number of neighbors in each axis-aligned direction along the grid for the smoothed k-d tree.')

    # GMM/LMM settings
    parser.add_argument('--num_components', type=int, default=5, help='The number of mixture components for gmm or lmm models.')

    # Get the arguments from the command line
    args = parser.parse_args()
    dargs = vars(args)
    dargs['train_id'] = 0
    dargs['dataset'] = 'cifar10'

    # Get the parameters
    if not os.path.exists(dargs['outputdir']):
        os.makedirs(dargs['outputdir'])
    
    if args.model == 'sdp':
        dargs['outfile'] = os.path.join(dargs['outputdir'], '{model}_{dataset}_{k}_{lam}_{train_id}'.format(**dargs))
        dargs['variable_scope'] = '{model}-{dataset}-{k}-{lam}-{train_id}'.format(**dargs)
    elif args.model in ('gmm', 'lmm'):
        dargs['outfile'] = os.path.join(dargs['outputdir'], '{model}_{dataset}_{num_components}_{train_id}'.format(**dargs))
        dargs['variable_scope'] = '{model}-{dataset}-{num_components}-{train_id}'.format(**dargs)
    elif args.model == 'multinomial':
        dargs['outfile'] = os.path.join(dargs['outputdir'], '{model}_{dataset}_{train_id}'.format(**dargs))
        dargs['variable_scope'] = '{model}-{dataset}-{train_id}'.format(**dargs)
    else:
        raise Exception('Unknown model type: {model}'.format(**dargs))
    
    dataset = load_dataset(**dargs)
    dargs['dataset'] = dataset

    sess = tf.Session()

    model = create_model(**dargs)
    saver = tf.train.Saver()

    learning_rate = 1e-4
    epsilon = 1e-1
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)
    train_step = opt.minimize(model.train_loss)

    sess.run(tf.global_variables_initializer())

    print('Beginning training and going for a maximum of {nepochs} epochs'.format(**dargs))
    sys.stdout.flush()

    best_loss = None
    epochs_since_improvement = 0

    for epoch in xrange(args.nepochs):
        # Do all the minibatch updates for this epoch
        for step, (X, y) in enumerate(dataset.train):
            for i in xrange(int(np.ceil(len(X) / float(args.batchsize)))):
                feed_dict = model.train_dict(X[i:i+args.batchsize], y[i:i+args.batchsize])
                # feed_dict[learning_rate] = cur_learning_rate
                sess.run(train_step, feed_dict=feed_dict)
                break # TEMP
            if step % 1 == 0: # TEMP
                print('\tEpoch {0}, step {1}'.format(epoch, step))
                sys.stdout.flush()
                break # TEMP

        # Test if the model improved on the validation set
        validation_loss = score_model(sess, model, dataset)

        # Check if we are improving
        if best_loss is None or validation_loss < best_loss:
            best_loss = validation_loss
            epochs_since_improvement = 0
            print('Found new best model. Saving to {outfile}'.format(**dargs))
            sys.stdout.flush()
            saver.save(sess, dargs['outfile'])
        else:
            epochs_since_improvement += 1

        print('Epoch #{0} Validation loss: {1} Epochs since improvement: {2}'.format(epoch, validation_loss, epochs_since_improvement))

    # Reset the model back to the best version
    saver.restore(sess, dargs['outfile'])

    # Save the validation score for this model
    print('Finished training. Scoring model...')
    sys.stdout.flush()
    
    logprobs, bits_per_dim = explicit_score(sess, model, dataset)
    np.savetxt(dargs['outfile'] + '_score.csv', [best_loss, logprobs, bits_per_dim, args.k, args.lam, args.num_components])



if __name__ == '__main__':
    main()


