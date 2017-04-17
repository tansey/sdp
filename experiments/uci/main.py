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
        feed_dict = model.test_dict(X, y)
        loss += sess.run(model.test_loss, feed_dict=feed_dict)
    return loss

def explicit_score(sess, model, dataset):
    logprobs = 0
    squared_err = 0
    indices = np.array(list(np.ndindex(model.layer._num_classes)))
    for i in xrange(len(dataset.test.features)):
        feed_dict = model.test_dict(dataset.test.features[i:i+1], dataset.test.labels[i:i+1])
        if model.density:
            density = sess.run(model.density, feed_dict=feed_dict)[0]
        else:
            density = model.layer.dist(dataset.test.features[i:i+1], sess, feed_dict)[0]
        if np.abs(density.sum() - 1.) > 1e-10:
            raise Exception('Distribution does not add up: {}'.format(density.sum()))
        if density.min() < 0 or density.max() > 1:
            raise Exception('Distribution outside acceptable bounds: [{}, {}]'.format(density.min(), density.max()))
        logprobs += np.log(density[tuple(dataset.test.labels[i])])
        prediction = np.array([density[tuple(idx)] * idx for idx in indices]).sum(axis=0)
        squared_err += np.linalg.norm(dataset.test.labels[i] - prediction)**2
    rmse = np.sqrt(squared_err / float(len(dataset.test.features)))
    print 'Explicit logprobs: {0} RMSE: {1}'.format(logprobs, rmse)
    return logprobs, rmse

def main():
    parser = argparse.ArgumentParser(description='Discrete conditional distribution estimation on UCI data.')

    # Experiment settings
    parser.add_argument('model', choices=['multinomial', 'gmm', 'lmm', 'sdp', 'fast-sdp'], help='The model type. gmm is mixture density networks. lmm is logistic mixture model. sdp is smoothed k-d trees.')
    parser.add_argument('dataset', choices=['auto_mpg', 'housing', 'energy_efficiency', 'parkinsons', 'concrete', 'abalone', 'student_performance'], help='The dataset to use. See the corresponding preprocessing files for details.')
    parser.add_argument('--inputdir', default='experiments/uci/data', help='The directory where the input data files will be stored.')
    parser.add_argument('--outputdir', default='experiments/uci/results', help='The directory where the output data files will be stored.')
    parser.add_argument('--variable_scope', default='uci-', help='The variable scope that the model will be created with.')
    parser.add_argument('--train_id', type=int, default=0, help='A trial ID. All models trained with the same trial ID will use the same train/validation datasets.')
    parser.add_argument('--validation_pct', type=float, default=0.2,
                                        help='The number of samples to hold out for a validation set. This is a percentage of the training samples.')
    
    # Optimizer settings
    parser.add_argument('--epsilon', type=float, default=0.1, help='The numerical stability constant for Adam.')
    parser.add_argument('--initial_learning_rate', type=float, default=0.1, help='The initial learning rate for the optimizer.')
    parser.add_argument('--min_learning_rate', type=float, default=1e-4, help='The initial learning rate for the optimizer.')
    parser.add_argument('--learning_decay', type=float, default=0.25, help='The decay rate for the learning rate.')
    parser.add_argument('--nepochs', type=int, default=1000, help='The maximum number of training epochs.')
    parser.add_argument('--batchsize', type=int, default=50, help='The number of training samples per mini-batch.')
    parser.add_argument('--unimprove_wait', type=int, default=10, help='The number of epochs without improvement before we decrease the learning rate or stop early.')

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
    if not os.path.exists(dargs['outputdir']):
        os.makedirs(dargs['outputdir'])
    if args.model in ('sdp', 'fast-sdp'):
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

    sess = tf.Session()

    model = create_model(**dargs)
    saver = tf.train.Saver()

    # Get the optimizer
    cur_learning_rate = args.initial_learning_rate
    learning_rate = tf.placeholder(tf.float32, shape=[])
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=args.epsilon)
    train_step = opt.minimize(model.train_loss)

    sess.run(tf.global_variables_initializer())

    print('Beginning training and going for a maximum of {nepochs} epochs'.format(**dargs))
    sys.stdout.flush()

    best_loss = None
    epochs_since_improvement = 0
    for epoch in xrange(args.nepochs):
        # Do all the minibatch updates for this epoch
        for step, (X, y) in enumerate(dataset.train):
            feed_dict = model.train_dict(X, y)
            feed_dict[learning_rate] = cur_learning_rate
            sess.run(train_step, feed_dict=feed_dict)
            if step % 100 == 0:
                print('\tEpoch {0}, step {1}'.format(epoch, step))
                sys.stdout.flush()

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

        print('Epoch #{0} Validation loss: {1} Epochs since improvement: {2} (learning rate: {3})'.format(epoch, validation_loss, epochs_since_improvement, cur_learning_rate))
        if epochs_since_improvement >= args.unimprove_wait:
            cur_learning_rate *= args.learning_decay
            if cur_learning_rate < args.min_learning_rate:
                print('Stopping.')
                sys.stdout.flush()
                break
            else:
                print('Decreasing learning rate and resetting model to current best. New rate: {0}'.format(cur_learning_rate))
                # Reset the model back to the best version
                saver.restore(sess, dargs['outfile'])
                epochs_since_improvement = 0

    # Reset the model back to the best version
    saver.restore(sess, dargs['outfile'])

        # Save the validation score for this model
    print('Finished training. Scoring model...')
    sys.stdout.flush()
    
    logprobs, rmse = explicit_score(sess, model, dataset)
    np.savetxt(dargs['outfile'] + '_score.csv', [best_loss, logprobs, rmse, args.k, args.lam, args.num_components])



if __name__ == '__main__':
    main()


