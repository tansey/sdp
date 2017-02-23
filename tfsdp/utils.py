import sys
import csv
import collections
import itertools
import numpy as np
import tensorflow as tf
from scipy.sparse import coo_matrix
from datasets import Datasets, DataSet
from tensorflow.contrib.distributions import MultivariateNormalCholesky as chol_mvn

######## Convenience functions for creating tensorflow models #########

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def l2_regularization(learned_variables=None, lam=10.0):
    '''Adds scaled L2 regularization to all trainable variables'''
    if learned_variables is None:
        learned_variables = tf.trainable_variables()
    return lam / float(len(learned_variables)) * tf.add_n([tf.reduce_mean(tf.pow(regularized_var, 2)) for regularized_var in learned_variables])

#######################################################################

################### Multiscale ########################################

def discretize_labels(raw_labels, bins):
    '''Bins the real-values and creates a one-hot encoding from them'''
    if type(bins) is int:
        bins = (bins, )
    if type(bins) is tuple and type(bins[0]) is int:
        bins = tuple(np.linspace(raw_labels[:,i].min()-1e-12, raw_labels[:,i].max()+1e-12, b+1) for i, b in enumerate(bins))
    labels = np.zeros(raw_labels.shape, dtype=np.int32)
    for i, b in enumerate(bins):
        labels[:,i] = (np.digitize(raw_labels[:,i].clip(b[0],b[-1]), b, right=True) - 1).reshape(labels[:,i].shape)
    return labels, bins

def polya_tree_descend(data, bins, splits, bounds, dim, level, indices, right_inclusive, polya_levels=None):
    '''The recursive helper function to create the multiscale decomposition bins'''
    if polya_levels is not None and np.array_equal(level, polya_levels):
        # If we've exhausted all levels, just return
        return

    if level.max() == -1:
        return

    # If we've reached the bottom of this dimension's allocated levels
    if polya_levels is not None and level[dim] >= polya_levels[dim]:
        polya_tree_descend(data, bins, splits, bounds, (dim+1)%data.shape[1], level, indices, right_inclusive, polya_levels=polya_levels)
        return

    # Add a new split halfway between the left and right bounds of the current dimension
    left, right = bounds[dim]

    # We assume integers, so all boundaries must be at least 1 apart
    if (right - left) < 1:
        level = np.copy(level)
        level[dim] = -1
        polya_tree_descend(data, bins, splits, bounds, (dim+1)%data.shape[1], level, indices, right_inclusive, polya_levels=polya_levels)
        return

    if bounds in splits:
        print 'Why are we getting duplicates??? {0}'.format(bounds)
        print dim, level, indices
        print splits
        print ''
        print ''
        return

    splits.add(bounds)

    # Just take the midpoint as the split
    mid = (left + right) / 2.0

    # Find the candidate subset of data points bounded by this region
    subset = data[indices]

    # Figure out which way each datapoint splits in the multiscale tree
    left_indices = indices[np.where(np.logical_and(subset[:,dim] >= left, subset[:,dim] < mid))[0]]
    
    # Handle the boundary case where the right bin is the last bin
    if right_inclusive[dim]:
        right_indices = indices[np.where(np.logical_and(subset[:,dim] >= mid, subset[:,dim] <= right))[0]]
    else:
        right_indices = indices[np.where(np.logical_and(subset[:,dim] >= mid, subset[:,dim] < right))[0]]

    # Create a vector of -1s for missing, 0s for left splits, and 1s for right splits
    labels = np.zeros(data.shape[0], dtype=np.int32) - 1
    labels[left_indices] = 0
    labels[right_indices] = 1

    # Add the resulting bounds to the list of output bins
    bins.append((bounds, labels))
    
    # Create the new boundary sets for the next level in the tree
    left_bound = [b for b in bounds]
    left_bound[dim] = (left, mid)
    left_bound = tuple(left_bound)
    left_right_inclusive = np.copy(right_inclusive)
    left_right_inclusive[dim] = 0

    right_bound = [b for b in bounds]
    right_bound[dim] = (mid, right)
    right_bound = tuple(right_bound)

    # Increment the level and dimensions
    level = np.copy(level)
    level[dim] += 1
    dim = (dim+1)%data.shape[1]

    # Recurse left
    polya_tree_descend(data, bins, splits, left_bound, dim, level, left_indices, left_right_inclusive, polya_levels=polya_levels)

    # Recurse right
    polya_tree_descend(data, bins, splits, right_bound, dim, level, right_indices, right_inclusive, polya_levels=polya_levels)

def batch_multiscale_label_lookup(data, bins, polya_levels=None):
    results = [[] for _ in data]
    bounds = tuple((b.min(), b.max()) for b in bins)
    level = np.zeros(len(bins))
    indices = np.arange(len(data))
    right_inclusive = np.ones(len(bins))
    batch_multiscale_label_lookup_helper(data, results, 0, bounds, 0, level, indices, right_inclusive, polya_levels=polya_levels)
    return np.array(results).T

def batch_multiscale_label_lookup_helper(data, results, nodeidx, bounds, dim, level, indices, right_inclusive, polya_levels=None):
    '''The recursive helper function to create the multiscale label lookups'''
    if polya_levels is not None and np.array_equal(level, polya_levels):
        # If we've exhausted all levels, just return
        return nodeidx

    if level.max() == -1:
        return nodeidx

    # If we've reached the bottom of this dimension's allocated levels
    if polya_levels is not None and level[dim] >= polya_levels[dim]:
        return batch_multiscale_label_lookup_helper(data, results, nodeidx, bounds, (dim+1)%data.shape[1], level, indices, right_inclusive, polya_levels=polya_levels)

    # Add a new split halfway between the left and right bounds of the current dimension
    left, right = bounds[dim]

    # We assume integers, so all boundaries must be at least 1 apart
    if (right - left) < 1:
        level = np.copy(level)
        level[dim] = -1
        # print 'End of dim. Level: {0} bounds: {1} dim: {2} newdim: {3} data shape: {4}'.format(level, bounds, dim, (dim+1)%data.shape[1], data.shape)
        return batch_multiscale_label_lookup_helper(data, results, nodeidx, bounds, (dim+1)%data.shape[1], level, indices, right_inclusive, polya_levels=polya_levels)

    # If there are no examples that hit this node, calculate the remaining nodes in
    # the subtree and return that offset + this node
    if len(indices) == 0:
        polya_cutoffs = [np.inf]*len(level) if polya_levels is None else polya_levels - level
        offset = 1
        for i,(b_left, b_right) in enumerate(bounds):
            leaves = np.ceil(np.log2(max(b_right - b_left, 1)))
            if polya_levels is not None:
                leaves = int(min(polya_levels[i] - level[i], leaves))
            offset *= 2**leaves
        # offset = np.prod([max(1,int(min(cutoff, np.ceil(np.log2(max(b_right - b_left,1)))))) for cutoff, (b_right, b_left) in zip(polya_cutoffs,bounds)])
        return nodeidx + offset - 1 # Binary trees have (2^d - 1) nodes

    # Just take the midpoint as the split
    mid = (left + right) / 2.0

    # Find the candidate subset of data points bounded by this region
    subset = data[indices]

    # Figure out which way each datapoint splits in the multiscale tree
    left_indices = indices[np.where(np.logical_and(subset[:,dim] >= left, subset[:,dim] < mid))[0]]
    
    # Handle the boundary case where the right bin is the last bin
    if right_inclusive[dim]:
        right_indices = indices[np.where(np.logical_and(subset[:,dim] >= mid, subset[:,dim] <= right))[0]]
    else:
        right_indices = indices[np.where(np.logical_and(subset[:,dim] >= mid, subset[:,dim] < right))[0]]

    # Add the node ID to the list of examples that traverse this node
    # using 0s for left splits, and 1s for right splits
    for idx in left_indices: results[idx].append([nodeidx,0])
    for idx in right_indices: results[idx].append([nodeidx,1])
    
    # Create the new boundary sets for the next level in the tree
    left_bound = [b for b in bounds]
    left_bound[dim] = (left, mid)
    left_bound = tuple(left_bound)
    left_right_inclusive = np.copy(right_inclusive)
    left_right_inclusive[dim] = 0

    right_bound = [b for b in bounds]
    right_bound[dim] = (mid, right)
    right_bound = tuple(right_bound)

    # Increment the level and dimensions
    level = np.copy(level)
    level[dim] += 1
    dim = (dim+1)%data.shape[1]

    # Recurse left
    right_node_idx = batch_multiscale_label_lookup_helper(data, results, nodeidx+1, left_bound, dim, level, left_indices, left_right_inclusive, polya_levels=polya_levels)

    # Recurse right
    next_node_idx = batch_multiscale_label_lookup_helper(data, results, right_node_idx, right_bound, dim, level, right_indices, right_inclusive, polya_levels=polya_levels)

    return next_node_idx

################ Trend Filtering ######################################

def t_offset(t, dim, offset):
    x = np.array(t)
    x[dim] += offset
    return tuple(x)

def get_sparse_penalty_matrix(num_classes):
    '''Creates a sparse graph-fused lasso penalty matrix (zero'th order trendfiltering)
    under the assumption that the class bins are arranged along an evenly spaced
    p-dimensional grid.'''
    bins = [np.arange(c) for c in num_classes]
    idx_map = {t: idx for idx, t in enumerate(itertools.product(*bins))}
    indices = []
    values = []
    rows = 0
    for idx1,t1 in enumerate(itertools.product(*bins)):
        for dim in xrange(len(t1)):
            if t1[dim] < (num_classes[dim]-1):
                t2 = t_offset(t1, dim, 1)
                idx2 = idx_map[t2]
                indices.append([rows, idx1])
                values.append(1)
                indices.append([rows, idx2])
                values.append(-1)
                rows += 1
    # tensorflow version
    #D_shape = [rows, np.prod(num_classes)]
    #return tf.sparse_reorder(tf.SparseTensor(indices=indices, values=values, shape=D_shape))
    # Use scipy's sparse libraries until tensorflow's sparse matrix multiplication is implemented fully
    D_shape = (rows, np.prod(num_classes))
    row_indices = [x for x,y in indices]
    col_indices = [y for x,y in indices]
    return coo_matrix((values, (row_indices, col_indices)), shape=D_shape)

def scipy_sparse_coo_to_tensorflow_sparse(x):
    values = x.data
    indices = list(zip(x.row, x.col))
    shape = x.shape
    return tf.sparse_reorder(tf.SparseTensor(indices=indices, values=values, shape=shape))

def get_delta(D, k, sparse=True):
    '''Calculate the k-th order trend filtering matrix given the oriented edge
    incidence matrix and the value of k.'''
    if k < 0:
        raise Exception('k must be at least 0th order.')
    result = D
    for i in xrange(k):
        result = D.T.dot(result) if i % 2 == 0 else D.dot(result)
    if sparse:
        return tf.cast(scipy_sparse_coo_to_tensorflow_sparse(result.tocoo()), tf.float32)
    return tf.constant(np.array(result.todense()), tf.float32)

def batch_sparse_tensor_dense_matmul(sp_a, b):
    '''Multiply sp_a by every row of b.'''
    return tf.map_fn(lambda b_i: tf.sparse_tensor_dense_matmul(sp_a, b_i), b)

def trend_filtering_penalty_fn(x, penalty='lasso'):
    if penalty == 'lasso' or penalty == 'gamlasso':
        fv = tf.abs(x)
    elif penalty == 'doublepareto':
        fv = tf.log(1 + tf.abs(x))
    return fv

def trend_filtering_penalty(z, dims, k, penalty='lasso', sparse=True):
    '''Applies the trend filtering matrix to a batch of vectors'''
    # Use a fast explicit calculation to keep everything on the GPU
    if (not hasattr(dims, "__len__") or len(dims) == 1) and k < 3:
        print 'Using fast 1D trend filtering'
        if k == 0:
            v = trend_filtering_penalty_fn(tf.sub(z[:,:-1], z[:,1:]), penalty)
            fv = tf.reduce_mean(tf.reduce_sum(v, reduction_indices=[1]))
        elif k == 1:
            v = 2*z
            v_first = trend_filtering_penalty_fn(tf.sub(v[:,0], tf.add(z[:,1],z[:,0])), penalty)
            v_mid = trend_filtering_penalty_fn(tf.sub(v[:,1:-1], tf.add(z[:,:-2], z[:,2:])), penalty)
            v_last = trend_filtering_penalty_fn(tf.sub(v[:,-1], tf.add(z[:,-1],z[:,-2])), penalty)
            fv = tf.reduce_mean(tf.add_n([v_first,
                                         tf.reduce_sum(v_mid, reduction_indices=[1]),
                                         v_last]))
        elif k == 2:
            v = 3*tf.sub(z[:,:-1], z[:,1:])
            v_first = trend_filtering_penalty_fn(tf.add(tf.sub(v[:,0], z[:,0]), z[:,2]), penalty)
            v_mid = trend_filtering_penalty_fn(tf.add(v[:,1:-1], tf.sub(z[:,3:], z[:,:-3])), penalty)
            v_last = trend_filtering_penalty_fn(tf.sub(tf.add(v[:,-1], z[:,-1]), z[:,-3]), penalty)
            fv = tf.reduce_mean(tf.add_n([v_first,
                                         tf.reduce_sum(v_mid, reduction_indices=[1]),
                                         v_last]))
    elif not sparse:
        print 'Using dense n-dimensional trend filtering'
        D = get_delta(get_sparse_penalty_matrix(dims), k, sparse=False)
        v = trend_filtering_penalty_fn(tf.map_fn(lambda z_i: tf.matmul(D,z_i), tf.expand_dims(z,-1)), penalty)
        fv = tf.reduce_mean(tf.reduce_sum(v, reduction_indices=[1,2]))
    else:
        print 'Using slow-but-sparse n-dimensional trend filtering'
        # NOTE: If you need this, you are slowing down your network substantially by
        # moving everything to the CPU to do the sparse tensor multiplication. This
        # is just a byproduct of tensorflow's garbage sparse tensor support
        # Get the sparse trendfiltering penalty matrix
        D = get_delta(get_sparse_penalty_matrix(dims), k)
        
        # Calculate the trendfiltering penalty for each example and average
        v = trend_filtering_penalty_fn(batch_sparse_tensor_dense_matmul(D, tf.expand_dims(z,-1)), penalty)
        fv = tf.reduce_mean(tf.reduce_sum(v, reduction_indices=[1,2]))
    return fv

def batch_trend_filtering_penalty(z, dims, k, penalty='lasso', sparse=True):
    return tf.map_fn(lambda z_i: trend_filtering_penalty(z_i, dims, k, penalty=penalty, sparse=sparse), z)

#######################################################################

################ Tools to convert between label types #################
def one_hot_to_indices(batch_positives, num_classes):
    batch_positives = batch_positives.reshape([len(batch_positives)] + list(num_classes))
    batch_positives = np.where(batch_positives > 0)[1]
    if len(batch_positives.shape) == 1:
        batch_positives = batch_positives[:,np.newaxis]
    return batch_positives

def ints_to_multiscale(data, bins, max_levels=None):
    bounds = tuple((b.min(), b.max()) for b in bins)
    nodes = []
    level = np.zeros(len(bins))
    indices = np.arange(data.shape[0])
    right_inclusive = np.ones(len(bins))
    polya_tree_descend(data, nodes, set(), bounds, 0, level, indices, right_inclusive, polya_levels=max_levels)
    labels = np.array([lab for _, lab in nodes]).T
    return labels

def ints_to_multinomials(samples, bins, flatten=True):
    shape = tuple([len(samples)] + [len(b) for b in bins])
    labels = np.zeros(shape)
    for i,sample in enumerate(samples):
        idx = tuple([i] + list(sample))
        labels[idx] = 1
    if flatten:
        labels = labels.reshape((len(samples), np.prod(labels.shape[1:])))
    return labels

def ints_to_percentiles(samples, bins):
    results = np.zeros(samples.shape)
    for i,b in enumerate(bins):
        results[:, i] = ((samples[:,i] - b[0]) / float(b[-1] - b[0]))
    return results

#######################################################################

################ Simple string printing tools #########################
def pretty_str(p, decimal_places=2, ignore=None, label_columns=False):
    '''Pretty-print a matrix or vector.'''
    if len(p.shape) == 1:
        return vector_str(p, decimal_places, ignore)
    if len(p.shape) == 2:
        return matrix_str(p, decimal_places, ignore, label_columns)
    raise Exception('Invalid array with shape {0}'.format(p.shape))

def matrix_str(p, decimal_places=2, ignore=None, label_columns=False):
    '''Pretty-print the matrix.'''
    return '[{0}]'.format("\n  ".join([(str(i) if label_columns else '') + vector_str(a, decimal_places, ignore) for i, a in enumerate(p)]))

def vector_str(p, decimal_places=2, ignore=None):
    '''Pretty-print the vector values.'''
    style = '{0:.' + str(decimal_places) + 'f}'
    return '[{0}]'.format(", ".join([' ' if ((hasattr(ignore, "__len__") and a in ignore) or a == ignore) else style.format(a) for a in p]))

#######################################################################

################ Performance scores for densities #####################

def ks_distance(a, b, axis=1):
    '''Get the Kolmogorov-Smirnov (KS) distance between two densities a and b.'''
    if len(a.shape) == 1:
        return np.max(np.abs(a.cumsum() - b.cumsum()))
    return np.max(np.abs(a.cumsum(axis=1) - b.cumsum(axis=1)), axis=axis)

def tv_distance(a, b, axis=1):
    '''Get the Total Variation (TV) distance between two densities a and b.'''
    if len(a.shape) == 1:
        return np.sum(np.abs(a - b))
    return np.sum(np.abs(a - b), axis=axis)

def log_prob(labels, densities):
    '''Get the log-probability of each label given the density.'''
    return np.array([np.log(max(1e-10, density[tuple(x)])) for x, density in zip(labels, densities)])

def rmse(labels, densities):
    '''Get the root mean squared error of the expected value of the density.'''
    x = [np.arange(densities.shape[i]) for i in xrange(1, len(densities.shape))]
    y = np.zeros(labels.shape)
    for i, (label, density) in enumerate(zip(labels, densities)):
        for coords in itertools.product(*x):
            y[i] += np.array(coords) * max(1e-10, density[coords])
    score = 0
    for label, yi in zip(labels, y):
        score += np.linalg.norm(label - yi)
    return score / float(densities.shape[0])

#######################################################################

####### Gaussian Mixture Model (Mixture Density Networks) Utils #######

def univariate_gaussian_likelihood(x, mu, sigma):
    result = tf.sub(x, mu)
    result = tf.mul(result,tf.inv(sigma))
    result = -tf.square(result)/2.
    return tf.mul(tf.exp(result),tf.inv(sigma))/np.sqrt(2*np.pi)

def univariate_gmm_likelihood(x, w, mu, sigma):
    xw_shape = tf.pack([tf.shape(w)[-1], tf.shape(x)[0]])
    x_tiles = tf.transpose(tf.reshape(tf.tile(tf.squeeze(x,[-1]), tf.pack([tf.shape(w)[-1]])), xw_shape))
    result = univariate_gaussian_likelihood(tf.cast(x_tiles, tf.float32), mu, sigma)
    result = tf.mul(result, w)
    result = tf.reduce_sum(result, 1)
    return tf.expand_dims(result, -1)

def sample_univariate_gmm(w,mu,sigma, size=1):
    components = np.random.choice(np.arange(len(w)), p=w, replace=True, size=size)
    return np.random.normal(mu[components], sigma[components])

def unpack_cholesky(q, ndims, num_components=1):
    # Build the lower-triangular Cholesky from the flat buffer
    # (assumes q shape is [batchsize, cholsize*num_components])
    cholsize = (ndims*ndims - ndims) / 2 + ndims
    q = tf.reshape(q, [-1, num_components, cholsize])
    chol_diag = tf.nn.softplus(tf.clip_by_value(q[:,:,:ndims], 1e-10, 1e10))
    chol_offdiag = q[:,:,ndims:]
    chol_rows = []
    chol_start = 0
    chol_end = 1
    for i in xrange(ndims):
        pieces = []
        if i > 0:
            pieces.append(chol_offdiag[:,:,chol_start:chol_end])
            chol_start = chol_end
            chol_end += i+1
        pieces.append(chol_diag[:,:,i:i+1])
        if i < (ndims-1):
            pieces.append(tf.zeros([tf.shape(chol_diag)[0], num_components, ndims-i-1]))
        chol_rows.append(tf.concat(2, pieces))
    return tf.reshape(tf.concat(2, chol_rows), [tf.shape(chol_diag)[0], num_components, ndims, ndims])

def unpack_mvn_params(q, ndims, num_components=3):
    '''Returns the parameters of the MVN output.
    Assumes q is [batchsize,numparams]'''
    pi = tf.nn.softmax(q[:,:num_components])
    mu = tf.reshape(q[:,num_components:num_components*(1+ndims)], [-1, num_components, ndims])
    chol_q = q[:,num_components*(1+ndims):]
    chol = unpack_cholesky(chol_q, ndims, num_components)
    return pi, mu, chol

def mvn_mix_log_probs(samples, q, ndims, num_components=3):
    '''Calculate the log probabilities of a MVN mixture model.
    Assumes q is [batchsize,numparams]'''
    pi = tf.nn.softmax(q[:,:num_components])
    mu = tf.reshape(q[:,num_components:num_components*(1+ndims)], [-1, num_components, ndims])
    chol_q = q[:,num_components*(1+ndims):]
    chol = unpack_cholesky(chol_q, ndims, num_components)
    log_probs = []
    for c in xrange(num_components):
        packed_params = tf.concat(1, [mu[:,c,:],tf.reshape(chol[:,c,:,:], [-1,ndims*ndims]), samples])
        log_p = tf.map_fn(lambda x: chol_mvn(x[:ndims], tf.reshape(x[ndims:ndims*(1+ndims)],[ndims,ndims])).log_prob(x[ndims*(1+ndims):]), packed_params)
        log_probs.append(log_p)
    log_probs = tf.transpose(tf.reshape(tf.concat(0, log_probs), [num_components, -1]))
    log_probs = tf.log(pi)+log_probs
    return log_sum_exp(log_probs)

#######################################################################

################ PixelCNN++ utils #####################################
# Some code below taken from OpenAI PixelCNN++ implementation: https://github.com/openai/pixel-cnn
def int_or_neg1(x):
    try:
        return int(x)
    except:
        return -1
def int_shape(x):
    return list(map(int_or_neg1, x.get_shape()))

def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    axis = len(x.get_shape())-1
    m = tf.reduce_max(x, axis)
    m2 = tf.reduce_max(x, axis, keep_dims=True)
    return m + tf.log(tf.reduce_sum(tf.exp(x-m2), axis))

def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    axis = len(x.get_shape())-1
    m = tf.reduce_max(x, axis, keep_dims=True)
    return x - m - tf.log(tf.reduce_sum(tf.exp(x-m), axis, keep_dims=True))


def discretized_mix_logistic_log_probs_nd(x, l, nr_mix=3, ndims=1, num_classes=256):
    '''n-dimensional version of the the discretized logistic mixture model'''
    ncoeffs = (ndims*ndims - ndims) / 2
    params_per_component = ncoeffs + ndims * 2 + 1
    logit_probs = l[:,:nr_mix]
    tf.concat(0,[tf.shape(x)[:-1],[nr_mix,params_per_component-1]])
    l = tf.reshape(l[:,nr_mix:], [tf.shape(x)[0],nr_mix,params_per_component-1]) # reshape into (B, nr_mix, params_per_component-1)
    means = l[:,:,:ndims]
    log_scales = tf.maximum(l[:,:,ndims:2*ndims], -7.)
    coeffs = tf.nn.tanh(l[:,:,2*ndims:])
    x = tf.reshape(tf.tile(x, [1,nr_mix]), [tf.shape(x)[0],nr_mix,ndims]) # here and below: getting the means and adjusting them based on preceding sub-pixels
    adjusted_means = [means[:,:,0]]
    coeffs_start = 0
    coeffs_end = 1
    for i in xrange(1,ndims):
        adjusted_means.append(means[:,:,i] + tf.reduce_sum(coeffs[:, :, coeffs_start:coeffs_end] * x[:, :, 0:i], [2]))
        coeffs_start = coeffs_end
        coeffs_end += i+1
    means = tf.reshape(tf.concat(1,adjusted_means), [tf.shape(x)[0],nr_mix, ndims])
    centered_x = x - means
    inv_stdv = tf.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1./(num_classes-1.))
    cdf_plus = tf.nn.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1./(num_classes-1.))
    cdf_min = tf.nn.sigmoid(min_in)
    log_cdf_plus = plus_in - tf.nn.softplus(plus_in) # log probability for edge case of 0 (before scaling)
    log_one_minus_cdf_min = -tf.nn.softplus(min_in) # log probability for edge case of [num_classes-1] (before scaling)
    cdf_delta = cdf_plus - cdf_min # probability for all other cases
    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2.*tf.nn.softplus(mid_in) # log probability in the center of the bin, to be used in extreme cases (not actually used in our code)

    # now select the right output: left edge case, right edge case, normal case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation based on the assumption that the log-density is constant in the bin of the observed sub-pixel value
    log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.select(cdf_delta > 1e-5, tf.log(tf.maximum(cdf_delta, 1e-12)), log_pdf_mid - np.log((num_classes-1)/2.))))

    log_probs = tf.reduce_sum(log_probs,2) + log_prob_from_logits(logit_probs)
    return log_sum_exp(log_probs)























