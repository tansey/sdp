#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import sys
import time
import numpy as np
import tensorflow as tf
import seaborn as sns
from pygfl.utils import hypercube_edges, matrix_from_edges, get_delta as sp_get_delta, pretty_str
from tfcde.utils import batch_sparse_tensor_dense_matmul, \
                        get_sparse_penalty_matrix, \
                        get_delta as tf_get_delta, \
                        batch_multiscale_label_lookup, \
                        tv_distance

##############################################################################
'''Code for the toy experiments testing the effect of neighborhood sampling on
the sample efficiency and wall-clock time for the SDP model.'''
##############################################################################

class TrendFiltering:
    def __init__(self, length, k, lam):
        with tf.variable_scope(type(self).__name__):
            self.length = length
            self.k = k
            self.lam = lam
            self.D = tf_get_delta(get_sparse_penalty_matrix((length,)), k)
            self.samples = tf.placeholder(tf.int32, [None])
            self.y = tf.one_hot(self.samples, length)
            self.q = tf.Variable([1.]*length)
            self.yhat = tf.nn.softmax(self.q, name='yhat')
            self.acc = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(tf.clip_by_value(self.yhat, 1e-10, 1.0)) 
                                                          + (1 - self.y) * tf.log(tf.clip_by_value(1 - self.yhat, 1e-10, 1.0)),
                                        reduction_indices=[1]))
            self.reg = tf.reduce_sum(tf.abs(tf.sparse_tensor_dense_matmul(self.D, tf.expand_dims(self.q,-1))))
            self.loss = self.acc + self.lam * self.reg

class PatchSampledTrendFiltering:
    '''Smoothed multinomial with negative sampling'''
    def __init__(self, length, k, lam, num_negative_samples, neighbor_radius):
        with tf.variable_scope(type(self).__name__):
            self.length = length
            self.k = k
            self.num_negative_samples = num_negative_samples
            self.neighbor_radius = neighbor_radius
            self.neighborhood_size = 2 * self.neighbor_radius + 1
            self.lam = lam * length / (self.neighborhood_size**2)
            self.D = tf_get_delta(get_sparse_penalty_matrix((self.neighborhood_size,)), k) # Local patch to smooth
            self.q = tf.Variable([1.]*length)
            self.yhat = tf.nn.softmax(self.q, name='yhat')
            
            self.sample_y = tf.placeholder(tf.float32, [None, self.neighborhood_size+self.num_negative_samples]) # one-hot encodings
            self.neighborhood_indexes = tf.placeholder(tf.int32, [None, self.neighborhood_size]) # Beta indices to smooth
            self.negative_indexes = tf.placeholder(tf.int32, [None, self.num_negative_samples]) # Beta indices to down
            self.neighborhood_q = tf.gather(self.q, self.neighborhood_indexes) # Sampled logits
            self.negative_q = tf.gather(self.q, self.negative_indexes) # Sampled negative logits
            #self.sample_yhat = tf.nn.softmax(tf.reshape(self.sample_q, [-1]), name='sampled_yhat') # Sampled softmax
            self.sample_yhat = tf.nn.softmax(tf.concat(1, [self.neighborhood_q, self.negative_q]), name='sampled_yhat')
            self.acc = tf.reduce_mean(-tf.reduce_sum(self.sample_y * tf.log(tf.clip_by_value(self.sample_yhat, 1e-10, 1.0)) 
                                                          + (1 - self.sample_y) * tf.log(tf.clip_by_value(1 - self.sample_yhat, 1e-10, 1.0)),
                                        reduction_indices=[1]))
            # Smooth a local patch centered on the target variables
            self.reg = tf.reduce_sum(tf.abs(batch_sparse_tensor_dense_matmul(self.D, tf.expand_dims(self.neighborhood_q,-1)))) 
            self.loss = self.acc + self.lam * self.reg

    def fill_train_dict(self, feed_dict, batch_positives):
        npos = len(batch_positives)
        batch = list(batch_positives)# + list(np.random.choice(self.length, size=self.num_negative_samples, replace=False))
        negs = np.array([np.random.choice(self.length, size=self.num_negative_samples, replace=False) for _ in xrange(npos)]).astype(np.int32)
        samp_indexes, targets = zip(*[self.get_neighborhood(v) for v in batch])
        samp_y = np.zeros((npos, self.neighborhood_size+self.num_negative_samples))
        for i in xrange(npos):
            samp_y[i,targets[i]] = 1.
        feed_dict[self.sample_y] = samp_y
        feed_dict[self.neighborhood_indexes] = np.array(samp_indexes)
        feed_dict[self.negative_indexes] = negs
        # print feed_dict[self.neighborhood_indexes].shape
        # print feed_dict[self.negative_indexes].shape
        # print feed_dict[self.sample_y].shape
        # print ''

    def get_neighborhood(self, v):
        if v < self.neighbor_radius:
            start = 0
            end = self.neighborhood_size
        elif self.length - v - 1 < self.neighbor_radius:
            start = self.length - self.neighborhood_size
            end = self.length
        else:
            start = v - self.neighbor_radius
            end = v + self.neighbor_radius + 1
        return np.arange(start, end), v - start

class SampledMultiscale:
    '''Unsmoothed dyadic partitioning'''
    def __init__(self, length):
        with tf.variable_scope(type(self).__name__):
            self.length = length
            self.bins = [np.arange(self.length)]
            self.num_nodes = int(2**np.ceil(np.log2(self.length))) - 1
            self.path_length = int(np.ceil(np.log2(length)))
            self.q_indices = tf.placeholder(tf.int32, [None, self.path_length])
            self.splits = tf.placeholder(tf.float32, [None, self.path_length])
            self.q = tf.Variable([0.]*self.num_nodes)
            self.sampled_q = tf.gather(self.q, self.q_indices)
            self.sampled_probs = tf.inv(1 + tf.exp(-self.sampled_q))
            self.log_left_probs = self.splits * tf.log(tf.clip_by_value(self.sampled_probs, 1e-10, 1.0))
            self.log_right_probs = (1 - self.splits) * tf.log(tf.clip_by_value(1 - self.sampled_probs, 1e-10, 1.0))
            self.loss = tf.reduce_mean(-tf.reduce_sum(self.log_left_probs+self.log_right_probs, reduction_indices=[1]))
            self.sampled_density = tf.reduce_prod(tf.select(self.splits > 0, self.sampled_probs, 1 - self.sampled_probs), reduction_indices=[1])

    def fill_train_dict(self, feed_dict, batch_positives):
        indices, splits = batch_multiscale_label_lookup(batch_positives[:,np.newaxis], self.bins)
        feed_dict[self.q_indices] = indices.T
        feed_dict[self.splits] = splits.T

    def fill_dist_dict(self, feed_dict):
        grid = np.array(list(np.ndindex(self.length)))
        indices, splits = batch_multiscale_label_lookup(grid, self.bins)
        feed_dict[self.q_indices] = indices.T
        feed_dict[self.splits] = splits.T

    def yhat(self):
        dist_dict = {}
        self.fill_dist_dict(dist_dict)
        return self.sampled_density.eval(feed_dict=dist_dict)

class LocallySmoothedMultiscale:
    '''Locally smoothed dyadic partitioning'''
    def __init__(self, length, k, lam, neighbor_radius):
        with tf.variable_scope(type(self).__name__):
            self.length = length
            # Trend filtering setup
            self.k = k
            self.neighbor_radius = neighbor_radius
            self.neighborhood_size = 2 * self.neighbor_radius + 1
            self.lam = lam * length / (self.neighborhood_size**2)
            self.D = tf_get_delta(get_sparse_penalty_matrix((self.neighborhood_size,)), k) # Local patch to smooth
            # Multiscale setup
            self.bins = [np.arange(self.length)]
            self.num_nodes = int(2**np.ceil(np.log2(self.length))) - 1
            self.path_length = int(np.ceil(np.log2(length)))
            # Binomial likelihoods loss function
            self.q_indices = tf.placeholder(tf.int32, [None, self.path_length])
            self.splits = tf.placeholder(tf.float32, [None, self.path_length])
            self.q = tf.Variable([0.]*self.num_nodes)
            self.sampled_q = tf.gather(self.q, self.q_indices)
            self.sampled_probs = tf.inv(1 + tf.exp(-self.sampled_q))
            self.log_left_probs = self.splits * tf.log(tf.clip_by_value(self.sampled_probs, 1e-10, 1.0))
            self.log_right_probs = (1 - self.splits) * tf.log(tf.clip_by_value(1 - self.sampled_probs, 1e-10, 1.0))
            self.log_probs = tf.reduce_mean(-tf.reduce_sum(self.log_left_probs+self.log_right_probs, reduction_indices=[1]))
            # Smooth a local patch centered on the target variables
            self.neighborhood_indexes = tf.placeholder(tf.int32, [None, self.neighborhood_size, self.path_length])
            self.neighborhood_splits = tf.placeholder(tf.float32, [None, self.neighborhood_size, self.path_length])
            self.neighborhood_q = tf.gather(self.q, self.neighborhood_indexes)
            self.neighborhood_probs = tf.inv(1 + tf.exp(-self.neighborhood_q))
            self.neighborhood_log_left = self.neighborhood_splits * tf.log(tf.clip_by_value(self.neighborhood_probs, 1e-10, 1.0))
            self.neighborhood_log_right = (1 - self.neighborhood_splits) * tf.log(tf.clip_by_value(1 - self.neighborhood_probs, 1e-10, 1.0))
            self.neighborhood_log_probs = tf.reduce_sum(self.neighborhood_log_left+self.neighborhood_log_right, reduction_indices=[2])
            self.reg = tf.reduce_sum(tf.abs(batch_sparse_tensor_dense_matmul(self.D, tf.expand_dims(self.neighborhood_log_probs, -1))))
            # Add the loss and regularization penalty together
            self.loss = self.log_probs + self.lam * self.reg
            self.sampled_density = tf.reduce_prod(tf.select(self.splits > 0, self.sampled_probs, 1 - self.sampled_probs), reduction_indices=[1])

    def fill_train_dict(self, feed_dict, batch_positives):
        indices, splits = batch_multiscale_label_lookup(batch_positives[:,np.newaxis], self.bins)
        feed_dict[self.q_indices] = indices.T
        feed_dict[self.splits] = splits.T
        to_lookup = np.concatenate([self.get_neighborhood(x) for x in batch_positives])[:, np.newaxis]
        indices, splits = batch_multiscale_label_lookup(to_lookup, self.bins)
        feed_dict[self.neighborhood_indexes] = indices.T.reshape((len(batch_positives), self.neighborhood_size, self.path_length))
        feed_dict[self.neighborhood_splits] = splits.T.reshape((len(batch_positives), self.neighborhood_size, self.path_length))

    def fill_dist_dict(self, feed_dict):
        grid = np.array(list(np.ndindex(self.length)))
        indices, splits = batch_multiscale_label_lookup(grid, self.bins)
        feed_dict[self.q_indices] = indices.T
        feed_dict[self.splits] = splits.T

    def yhat(self):
        dist_dict = {}
        self.fill_dist_dict(dist_dict)
        return self.sampled_density.eval(feed_dict=dist_dict)

    def get_neighborhood(self, v):
        if v < self.neighbor_radius:
            start = 0
            end = self.neighborhood_size
        elif self.length - v - 1 < self.neighbor_radius:
            start = self.length - self.neighborhood_size
            end = self.length
        else:
            start = v - self.neighbor_radius
            end = v + self.neighbor_radius + 1
        return np.arange(start, end)

if __name__ == '__main__':
    sess = tf.InteractiveSession()

    # Generate the data
    truth = np.zeros(1000)
    truth[0] = 0
    rates = [0.5]*299 + [-2]*150 + [0.9]*300 + [0.5]*100 + [-1]*150
    for i in xrange(len(truth) - 1):
        truth[i+1] = truth[i] + rates[i]
    truth -= truth.mean()
    truth /= truth.std()
    truth = np.exp(truth) / np.exp(truth).sum()

    n = 5000
    data = np.random.choice(np.arange(len(truth)), p=truth, size=n)
    empirical = np.zeros(len(truth))
    for i in data:
        empirical[i] += 1.
    empirical /= empirical.sum()

    # Softmax case -- learing the distribution from observed samples
    positive_samples = tf.placeholder(tf.int32, [None])
    negative_samples = tf.placeholder(tf.int32, [None,5])
    y = tf.one_hot(positive_samples, len(truth))

    # Get the trend filtering matrix
    k = 1
    lam = 2e-2
    num_negative_samples = 50
    neighbor_radius = [1,3,5,10,25]
    batchsize = 10
    nepochs = 10
    # full = TrendFiltering(len(truth), k, lam)
    # sampled = PatchSampledTrendFiltering(len(truth), k, lam, num_negative_samples, neighbor_radius)
    multi = SampledMultiscale(len(truth))
    sdp = [LocallySmoothedMultiscale(len(truth), k, lam, r) for r in neighbor_radius]
    
    # Create the optimizer
    learning_rate = 1e-2
    epsilon = 0.1
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)
    # train_full = opt.minimize(full.loss)
    # train_sampled = opt.minimize(sampled.loss)
    train_multi = opt.minimize(multi.loss)
    train_sdp = [opt.minimize(s.loss) for s in sdp]

    sess.run(tf.initialize_all_variables())

    full_results, sampled_results, multi_results, sdp_results = [], [], [], []
    indices = np.arange(len(data))
    steps = 0
    step_counts = []
    timings = [[] for _ in neighbor_radius]
    multi_time = []
    for epoch in xrange(nepochs):
        np.random.shuffle(indices)
        batchnum = 0
        while batchnum < len(indices):
            if batchnum % 100 == 0:
                print '\tBatch {0}'.format(batchnum)
                sys.stdout.flush()
            batch = data[indices[batchnum:batchnum+batchsize]]

            # Run the full model
            # train_dict = {full.samples: batch}
            # train_full.run(feed_dict=train_dict)

            # Run the sampled model
            # train_dict = {}
            # sampled.fill_train_dict(train_dict, batch)
            # train_sampled.run(feed_dict=train_dict)

            # Run the multiscale model
            start = time.clock()
            train_dict = {}
            multi.fill_train_dict(train_dict, batch)
            train_multi.run(feed_dict=train_dict)
            end = time.clock()
            multi_time.append(end-start)

            # Run the smoothed multiscale model
            for s, t, c in zip(sdp, train_sdp, timings):
                start = time.clock()
                train_dict = {}
                s.fill_train_dict(train_dict, batch)
                t.run(feed_dict=train_dict)
                end = time.clock()
                c.append(end - start)

            batchnum += batchsize
            steps += batchsize

            if steps % 100 == 0:
                step_counts.append(steps)
                yhat_multi = multi.yhat()
                yhat_sdp = [s.yhat() for s in sdp]
                multi_results.append(tv_distance(truth,yhat_multi))
                sdp_results.append([tv_distance(truth,yhat) for yhat in yhat_sdp])
        
        if epoch % 1 == 0:
            print 'Epoch {0}'.format(epoch)
            print '\tEmpirical MLE --> TV error: {0}'.format(tv_distance(truth,empirical))
            print '\tKDTree  model --> TV error: {0}'.format(tv_distance(truth,yhat_multi))
            print '\tSDP     model --> TV error: {0}'.format([tv_distance(truth,yhat) for yhat in yhat_sdp])
            sys.stdout.flush()
        

    sdp_results = np.array(sdp_results).T
    step_counts = np.array(step_counts)
    multi_time = np.mean(multi_time)
    timings = np.array([np.mean(c) for c in timings])
    
    with sns.axes_style('white'):
        plt.rc('font', weight='bold')
        plt.rc('grid', lw=2)
        plt.rc('lines', lw=3)
        plt.axhline(tv_distance(truth, empirical), color='red', ls='--', label='Empirical MLE')
        plt.plot(step_counts, multi_results, color='gray', label='No smoothing')
        plt.axhline(np.min(multi_results), color='gray', ls='--')
        markers = ['s', 'd', 'o', '^', '8']
        # styles = ['--', '--', '--', '--', '-', '-', '-', '-']
        for r, s, marker in zip(neighbor_radius, sdp_results, markers):
            plt.plot(step_counts, s, color='orange', label='SDP (radius={0})'.format(r), marker=marker, markevery=len(step_counts) / 10, ms=12)
        plt.xlabel('Steps', weight='bold', fontsize=24)
        plt.ylabel('TV Error', weight='bold', fontsize=24)
        plt.legend(loc='upper right', ncol=2)
        plt.savefig('plots/sampled_error.pdf', bbox_inches='tight')
        plt.clf()

        plt.plot(np.log(neighbor_radius), timings / multi_time, color='blue')
        plt.xlabel('Log(Neighborhood Radius)', weight='bold', fontsize=24)
        plt.ylabel('Relative Time', weight='bold', fontsize=24)
        plt.savefig('plots/sampled_timings.pdf', bbox_inches='tight')
        plt.clf()

        # Plot the learned distributions
        # x = np.arange(len(truth))
        # plt.plot(x, truth, color='black', label='Truth')
        # plt.plot(x, full.yhat.eval(), color='lightblue', label='Full TF')
        # plt.plot(x, sampled.yhat.eval(), color='orange', label='Sampled TF')
        # plt.plot(x, multi.yhat(), color='green', label='KD Tree')
        # plt.plot(x, sdp.yhat(), color='purple', label='SDP')
        # plt.bar(x, empirical, color='lightgray', alpha=0.2)
        # plt.legend(loc='upper left')
        # plt.savefig('plots/sampled_distributions.pdf', bbox_inches='tight')
    




