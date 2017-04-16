#!/usr/bin/env python
#import matplotlib.pylab as plt
import numpy as np
import sys
import os
import argparse
import tensorflow as tf
from tfsdp.utils import tv_distance, ks_distance
from utils import read_data_sets
from model import create_model

def validate(model, mnist):
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
        feed_dict = model._test_dict
        feed_dict[model.x] = features
        labels = np.argmax(labels, axis=-1)[:,np.newaxis]
        model.layer.fill_test_dict(feed_dict, labels)
        validation_loss += model.test_loss.eval(feed_dict=feed_dict) * size
    return validation_loss / float(mnist.validation.features.shape[0])

def fit_to_test(model, mnist):
    test_fit = np.zeros(mnist.test.labels.shape)
    iter_size = int(np.round(mnist.validation.features.shape[0] / 100))
    for i in xrange(100):
        start = iter_size*i
        if start >= len(test_fit):
            break
        if i == 99:
            stop = test_fit.shape[0]
        else:
            stop = iter_size*(i+1)
        feed_dict = model._test_dict
        feed_dict[model.x] = mnist.test.features[start:stop]
        density = model.layer.dist()
        test_fit[start:stop] = density.eval(feed_dict=feed_dict)
    return test_fit

def save_scores_and_fits(scores_filename, fits_filename, model, mnist):
    validation_score = validate(model, mnist)
    test_fit = fit_to_test(model, mnist)
    tv_score = tv_distance(mnist.test.labels, test_fit).mean()
    ks_score = ks_distance(mnist.test.labels, test_fit).mean()
    np.savetxt(fits_filename, test_fit, delimiter=',')
    np.savetxt(scores_filename, [validation_score, tv_score, ks_score], delimiter=',')

def main():
    parser = argparse.ArgumentParser(description='Trains a conditional density estimation model on the MNIST dataset.')

    parser.add_argument('--dist_type', choices=['gmm', 'discontinuous_gmm', 'edge_biased'], default='gmm', help='The type of underlying distribution that the labels are drawn from.')
    parser.add_argument('--inputdir', default='experiments/mnist/data', help='The directory where the input data files will be stored.')
    parser.add_argument('--variable_scope', default='mnist-', help='The variable scope that the model will be created with.')
    parser.add_argument('--verbose', type=int, default=1, help='Print detailed progress information to the console. 0=none, 1=high-level only, 2=all details.')

    parser.add_argument('--train_id', default='0', help='A trial ID. All models trained with the same trial ID will use the same train/validation datasets.')
    parser.add_argument('--train_samples', type=int, default=60000, help='The number of training examples to use.')
    parser.add_argument('--validation_samples', type=float, default=0.2,
                                        help='The number of samples to hold out for a validation set. This is a percentage of the training samples.')

    parser.add_argument('--early_stopping', action='store_true', help='If specified, will stop if the validation score exceeds the max score if the last validation_window models.')
    parser.add_argument('--validation_window', type=int, default=20, help='The number of iterations to look back when calculating convergence.')
    parser.add_argument('--eval_freq', type=int, default=100, help='The number of iterations between each validation evaluation to check for convergence.')
    parser.add_argument('--test_freq', type=int, default=10000, help='The number of iterations between each test evaluation.')
    
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd', 'momentum', 'adagrad', 'ftrl', 'rmsprop'], help='The optimizer to use.')
    parser.add_argument('--momentum', type=float, default=0.9, help='The amount of momentum weight, only valid if optimizer is momentum.')
    parser.add_argument('--epsilon', type=float, default=1.0, help='The numerical stability constant for the AdamOptimizer, only valid if optimizer is adam.')

    parser.add_argument('--decay_learning_rate', action='store_true', help='If specified, the learning rate will decay as the number of trials goes on.')
    parser.add_argument('--decay_size', type=float, default=10.0, help='The amount to divide the learning rate by whenever a decay event occurs.')
    parser.add_argument('--decay_freq', type=int, default=15000, help='The number of iterations to go between decay events.')

    parser.add_argument('--max_steps', type=int, default=100000, help='The maximum number of training steps.')
    parser.add_argument('--batchsize', type=int, default=50, help='The number of training samples per mini-batch.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='The learning rate for the optimizer.')

    parser.add_argument('--lam', type=float, default=0.005, help='The lambda penalty value. Only applies to trend filtering methods.')
    parser.add_argument('--k', type=int, default=2, help='The order of the trend filtering penalty matrix.')
    parser.add_argument('--neighbor_radius', type=int, default=5, help='The number of neighbors in each axis-aligned direction along the grid.')
    
    parser.set_defaults(early_stopping=False, decay_learning_rate=False)

    # Get the arguments from the command line
    args = parser.parse_args()
    dargs = vars(args)

    dargs['model'] = 'local-trendfiltering-multiscale'
    dargs['outfile'] = 'experiments/mnist/results/{model}_{dist_type}_{train_samples}_{k}_{lam}_{train_id}'.format(**dargs)
    dargs['variable_scope'] = 'mnist-{model}-{dist_type}-{train_samples}-{k}-{lam}-{train_id}'.format(**dargs)
    dargs['one_hot'] = False

    validation_window = args.validation_window
    variable_scope = args.variable_scope
    inputdir = args.inputdir
    eval_freq = args.eval_freq
    test_freq = args.test_freq
    max_steps = args.max_steps
    batchsize = args.batchsize
    early_stopping = args.early_stopping
    decay_learning_rate = args.decay_learning_rate
    decay_size = args.decay_size
    decay_freq = args.decay_freq
    epsilon = args.epsilon
    optimizer = args.optimizer
    momentum = args.momentum

    sess = tf.InteractiveSession()

    mnist = read_data_sets(inputdir, **dargs)
    dargs['nbins'] = mnist.bins.shape

    model = create_model(**dargs)

    if decay_learning_rate:
        learning_rate = tf.placeholder(tf.float32, shape=[])
    else:
        learning_rate = args.learning_rate

    if optimizer == 'sgd':
        opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    elif optimizer == 'adam':
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)
    elif optimizer == 'momentum':
        opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)
    elif optimizer == 'adagrad':
        opt = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    elif optimizer == 'ftrl':
        opt = tf.train.FtrlOptimizer(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

    train_step = opt.minimize(model.train_loss)

    sess.run(tf.global_variables_initializer())

    cur_learning_rate = args.learning_rate
    best_model_loss = None
    validation_threshold = None
    validation_losses = []
    step = 0
    steps_since_last_test = 0

    # Load checkpointed model if one exists
    if os.path.exists('{outfile}_checkpoint.txt'.format(**dargs)):
        print('Loading existing checkpoint data')
        sys.stdout.flush()
        model.saver.restore(sess, dargs['outfile'])
        step, best_model_loss = np.loadtxt('{outfile}_checkpoint.txt'.format(**dargs))

    print 'Beginning training and going for a maximum of {max_steps} steps'.format(**dargs)
    sys.stdout.flush()

    while step < max_steps+1:
        if decay_learning_rate and step > 0 and (step % decay_freq) == 0:
            cur_learning_rate /= decay_size

        X_train, y_train = mnist.train.next_batch(args.batchsize)
        y_train = np.argmax(y_train, axis=-1)[:,np.newaxis]
        feed_dict = model._train_dict
        feed_dict[model.x] = X_train
        model.layer.fill_train_dict(feed_dict, y_train)
        
        # Check for convergence against a moving average of the validation error
        if step % eval_freq == 0:
            train_loss = model.train_loss.eval(feed_dict=feed_dict)
            validation_loss = validate(model, mnist)
            print("step {0}, training loss {1}, validation loss {2}, validation threshold {3} learning_rate: {4}".format(
                        step,
                        train_loss,
                        validation_loss,
                        validation_threshold if validation_threshold is not None else 0,
                        cur_learning_rate))

            sys.stdout.flush()
            if np.isnan(validation_loss) or np.isnan(train_loss):
                print 'Reached a bad state! Aborting learning!'
                sys.stdout.flush()
                break

            # Save the model if it's the best we've seen
            if best_model_loss is None or validation_loss < best_model_loss:
                model.saver.save(sess, dargs['outfile'])
                best_model_loss = validation_loss
                if steps_since_last_test >= test_freq:
                    # Save the temporary performance measurements to file
                    print("Saving test results for current best model")
                    sys.stdout.flush()
                    save_scores_and_fits('{outfile}_score.csv'.format(**dargs), '{outfile}_fits.csv'.format(**dargs), model, mnist)
                    steps_since_last_test = 0

            # Check for approximate convergence
            if early_stopping and len(validation_losses) >= validation_window and validation_loss >= validation_threshold:
                print 'Reached convergence on validation loss. Halting training!\n'
                sys.stdout.flush()
                break

            validation_losses.append(validation_loss)
            validation_threshold = np.max(validation_losses[-validation_window:])

            # Checkpoint
            np.savetxt('{outfile}_checkpoint.txt'.format(**dargs), [step,best_model_loss])

        if decay_learning_rate:
            feed_dict[learning_rate] = cur_learning_rate

        train_step.run(feed_dict=feed_dict)
        step += 1
        steps_since_last_test += 1

    # Reset the model back to the best version
    model.saver.restore(sess, dargs['outfile'])

    # Save the validation score for this model
    print 'Finished training. Scoring model...'
    sys.stdout.flush()
    save_scores_and_fits('{outfile}_score.csv'.format(**dargs), '{outfile}_fits.csv'.format(**dargs), model, mnist)

    # Save the final model
    print 'Best model saved to {outfile} with variable scope of "{variable_scope}"'.format(**dargs)
    sys.stdout.flush()


if __name__ == '__main__':
    main()











