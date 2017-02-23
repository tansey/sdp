import numpy as np
import tensorflow as tf
import itertools
from utils import ints_to_multiscale, \
                  weight_variable, \
                  bias_variable, \
                  get_delta, \
                  get_sparse_penalty_matrix, \
                  batch_sparse_tensor_dense_matmul, \
                  log_sum_exp, \
                  log_prob_from_logits, \
                  univariate_gmm_likelihood, \
                  discretized_mix_logistic_log_probs_nd, \
                  trend_filtering_penalty, \
                  batch_multiscale_label_lookup, \
                  mvn_mix_log_probs, \
                  batch_trend_filtering_penalty, \
                  one_hot_to_indices, \
                  unpack_mvn_params

class DiscreteDistributionLayer(object):
    def __init__(self):
        pass

    @property
    def output(self):
        '''Returns the raw outputs of the density layer.'''
        raise NotImplementedError("Abstract method")

    @property
    def density(self):
        '''Returns the reconstructed discrete density from the output of the layer.
        Note that we are totally bastardizing the word "density" here-- this is a
        discrete distribution, not a density. Density is just a much less overloaded
        term in this project, so we use this.'''
        raise NotImplementedError("Abstract method")
    
    @property
    def train_loss(self):
        '''Returns the appropriate loss function for the raw output layer'''
        raise NotImplementedError("Abstract method")

    @property
    def test_loss(self):
        '''Returns the appropriate loss function for the raw output layer'''
        raise NotImplementedError("Abstract method")

class MultinomialLayer(DiscreteDistributionLayer):
    def __init__(self, input_layer, input_layer_size, num_classes, scope=None, **kwargs):
        if not hasattr(num_classes, "__len__"):
            num_classes = (num_classes, )
        
        self._num_classes = num_classes
        self._num_nodes = np.prod(num_classes)

        with tf.variable_scope(scope or type(self).__name__):
            self._labels = tf.placeholder(tf.float32, shape=[None, self._num_nodes], name='labels')
            W = weight_variable([input_layer_size, self._num_nodes])
            b = bias_variable([self._num_nodes])

            self._q = tf.nn.softmax(tf.matmul(input_layer, W) + b)
            self._loss_function = tf.reduce_mean(-tf.reduce_sum(self._labels * tf.log(tf.clip_by_value(self._q, 1e-10, 1.0)) 
                                                             + (1 - self._labels) * tf.log(tf.clip_by_value(1 - self._q, 1e-10, 1.0)),
                                        reduction_indices=[1]))

            # Reshape to the original dimensions of the density
            density_shape = tf.pack([tf.shape(input_layer)[0]] + list(num_classes))
            self._density = tf.reshape(self._q, density_shape)

    @property
    def labels(self):
        return self._labels

    @property
    def density(self):
        return self._density

    @property
    def output(self):
        return self._q

    @property
    def train_loss(self):
        return self._loss_function

    @property
    def test_loss(self):
        return self._loss_function

class DiscreteParametricMixtureLayer(DiscreteDistributionLayer):
    '''A generic discretized mixture model where each component is a parametric
    distribution like a GMM. Currently just supports GMM and LMM, but easily
    extended. Note that the GMM is not particularly numerically stable-- you
    may need to use a higher epsilon (if training with Adam) or much lower
    learning rate for some problems.'''
    def __init__(self, input_layer, input_layer_size, num_classes, num_components,
                 component_dist='gaussian', scope=None, one_hot=True,
                 lazy_density=False, **kwargs):
        if not hasattr(num_classes, "__len__"):
            num_classes = (num_classes, )

        if component_dist not in ('gaussian', 'logistic'):
            raise Exception('Unavailable parametric form specified: ' + component_dist)

        self._dist_type = component_dist
        self._num_classes = num_classes
        self._ndims = len(num_classes)
        self._num_components = num_components
        self._num_outputs = self._get_output_counts()
        scope = scope or type(self).__name__
        np_num_classes = np.array(self._num_classes, dtype=float)
        with tf.variable_scope(scope):
            '''Fit the LMM onto a finite discrete grid.'''
            W = weight_variable([input_layer_size, self._num_outputs])
            b = bias_variable([self._num_outputs])

            # Convert to the range [-1, 1] for all dimensions
            if one_hot:
                self._labels = tf.placeholder(tf.float32, shape=[None, np.prod(self._num_classes)]) # Get the one-hot labels
                grid_values = tf.constant(np.array(list(np.ndindex(self._num_classes))) / (np_num_classes-1) * 2 - 1, tf.float32)
                grid_indices = tf.to_int32(tf.argmax(self._labels, 1))
                mm_labels = tf.gather(grid_values, grid_indices)
            else:
                self._labels = tf.placeholder(tf.float32, shape=[None, len(self._num_classes)])
                mm_labels = self._labels / (np_num_classes-1) * 2 - 1

            # Get the model output
            self._q = tf.matmul(input_layer, W) + b

            # Get the loss for all the observations
            self._loss = tf.reduce_mean(-self._calc_log_probs(mm_labels, self._q))

            if not lazy_density:
                # Reconstruct the full discrete probability distribution by enumerating everything out
                grid_values = np.array(list(np.ndindex(self._num_classes)), np.float32) / (np.array(self._num_classes, dtype=float) - 1) * 2 - 1
                grid_values = tf.constant(grid_values, tf.float32)
                self._grid_values = grid_values

                q_density_shape = tf.pack([tf.shape(self._q)[0] * tf.shape(self._grid_values)[0], tf.shape(self._q)[1]])
                q_density = tf.reshape(tf.tile(self._q, [1, tf.shape(self._grid_values)[0]]), q_density_shape)
                labels_density = tf.tile(self._grid_values, tf.pack([tf.shape(self._q)[0], 1]))
                logprobs_density_shape = tf.pack([tf.shape(self._q)[0], tf.shape(self._grid_values)[0]])
                logprobs_density = tf.reshape(self._calc_log_probs(labels_density, q_density), logprobs_density_shape)
                bin_probs = tf.exp(logprobs_density) / tf.reduce_sum(tf.exp(logprobs_density), reduction_indices=[1], keep_dims=True)
                density_shape = tf.pack([tf.shape(self._q)[0]] + list(self._num_classes))
                self._density = tf.reshape(bin_probs, density_shape)

            # TEMP: Hack to make calculating the full conditional distribution scalable
            if component_dist == 'gaussian' and len(self._num_classes) > 1:
                self.mvn_params = unpack_mvn_params(self._q, self._ndims, self._num_components)

    def _get_output_counts(self):
        if self._dist_type in ('gaussian', 'logistic'):
            return self._num_components * ((self._ndims * self._ndims - self._ndims) / 2 + 2*self._ndims + 1)
        raise Exception('Unknown dist type')

    def _calc_log_probs(self, labels, q):
        if self._dist_type == 'logistic':
            return discretized_mix_logistic_log_probs_nd(labels, q, nr_mix=self._num_components, ndims=self._ndims)
        if self._dist_type == 'gaussian':
            if len(self._num_classes) == 1:
                pi = tf.nn.softmax(q[:,:self._num_components])
                mus = q[:,self._num_components:2*self._num_components]
                sigmas = tf.nn.softplus(q[:,self._num_components*2:])
                return log_sum_exp(tf.log(pi) + tf.contrib.distributions.Normal(mus, sigmas).log_prob(labels))
            else:
                return mvn_mix_log_probs(labels, q, self._ndims, self._num_components)

    @property
    def labels(self):
        return self._labels

    @property
    def density(self):
        return self._density

    @property
    def output(self):
        return self._q

    @property
    def train_loss(self):
        return self._loss

    @property
    def test_loss(self):
        return self._loss

class MultiscaleLayer(DiscreteDistributionLayer):
    '''An unsmoothed dyadic decomposition layer. Throughout the code, we often
    use the phrase "multiscale" to refer to dyadic decomposition models.'''
    def __init__(self, input_layer, input_layer_size, num_classes, scope=None, **kwargs):
        if not hasattr(num_classes, "__len__"):
            num_classes = (num_classes, )
        
        # All splits are done via half-spaces, so there are always 2^k-1 output
        # nodes. We handle non-power-of-two nodes by keeping track of the buffer
        # sizes vs. the actual multinomial dimensions.
        self._num_classes = num_classes
        self._dim_sizes = [2**(int(np.ceil(np.log2(c)))) for c in num_classes]
        self._num_nodes = np.prod(self._dim_sizes) - 1 # flatten the density into a 1-d grid
        self._split_labels, self._split_masks = self.multinomial_split_masks()

        with tf.variable_scope(scope or type(self).__name__):
            self._labels = tf.placeholder(tf.float32, shape=[None, np.prod(self._num_classes)])
            W = weight_variable([input_layer_size, self._num_nodes])
            b = bias_variable([self._num_nodes])
            split_indices = tf.to_int32(tf.argmax(self._labels, 1))
            splits, z = tf.gather(self._split_labels, split_indices), tf.gather(self._split_masks, split_indices)

            # q is the value of the tree nodes
            # m is the value of the multinomial bins
            self._q = tf.inv(1 + tf.exp(-(tf.matmul(input_layer,W) + b)))
            r = splits * tf.log(tf.clip_by_value(self._q, 1e-10, 1.0))
            s = (1 - splits) * tf.log(tf.clip_by_value(1 - self._q, 1e-10, 1.0))
            self._loss_function = tf.reduce_mean(-tf.reduce_sum(z * (r+s),
                                            reduction_indices=[1]))

            # Convert from multiscale output to multinomial output
            L, R = self.multiscale_splits_masks()
            q_tiles = tf.constant([1, np.prod(self._num_classes)])

            m = tf.map_fn(lambda q_i: self.multiscale_to_multinomial(q_i, L, R, q_tiles), self._q)

            # Reshape to the original dimensions of the density
            density_shape = tf.pack([tf.shape(self._q)[0]] + list(self._num_classes))
            self._density = tf.reshape(m, density_shape)

            self._cross_entropy = tf.reduce_mean(-tf.reduce_sum(self._labels * tf.log(tf.clip_by_value(m, 1e-10, 1.0)) 
                                                             + (1 - self._labels) * tf.log(tf.clip_by_value(1 - m, 1e-10, 1.0)),
                                        reduction_indices=[1]))

    @property
    def labels(self):
        return self._labels

    @property
    def density(self):
        return self._density

    @property
    def output(self):
        return self._q    

    @property
    def train_loss(self):
        return self._loss_function

    @property
    def test_loss(self):
        return self._cross_entropy

    def multinomial_split_masks(self):
        data = np.array(list(np.ndindex(self._num_classes)))
        bins = [np.arange(c) for c in self._dim_sizes]
        multiscale_labels = ints_to_multiscale(data, bins)
        return tf.constant(multiscale_labels, tf.float32), tf.constant((multiscale_labels != -1).astype(int), tf.float32)

    def multiscale_to_multinomial(self, q_i, L, R, q_tiles):
        qt = tf.tile(tf.expand_dims(q_i, -1), q_tiles)
        qt = R * qt + L * (1 - qt)
        qt += tf.to_float(tf.equal(qt, 0))
        return tf.reduce_prod(qt, reduction_indices=[0])

    def multiscale_splits_masks(self):
        data = np.array(list(np.ndindex(self._num_classes)))
        bins = [np.arange(c) for c in self._dim_sizes]
        labels = ints_to_multiscale(data, bins)
        return tf.constant((labels == 0).astype(int).T, tf.float32), tf.constant((labels == 1).astype(int).T, tf.float32)

class TrendFilteringLayer(DiscreteDistributionLayer):
    '''A multinomial model with trend filtering on the logits.'''
    def __init__(self, input_layer, input_layer_size, num_classes, lam=0.05, k=2, penalty='lasso', scope=None, **kwargs):
        if not hasattr(num_classes, "__len__"):
            num_classes = (num_classes, )

        self._num_classes = num_classes
        self._num_nodes = np.prod(num_classes)
        self._lam = lam
        self._k = k
        self._penalty = penalty

        with tf.variable_scope(scope or type(self).__name__):
            self._labels = tf.placeholder(tf.float32, shape=[None, self._num_nodes])
            W = weight_variable([input_layer_size, self._num_nodes])
            b = bias_variable([self._num_nodes])
            lam = tf.constant(self._lam, tf.float32, name='lam')


            z = tf.matmul(input_layer, W) + b
            self._q = tf.nn.softmax(z)
            self._cross_entropy = tf.reduce_mean(-tf.reduce_sum(self._labels * tf.log(tf.clip_by_value(self._q, 1e-10, 1.0)) 
                                                          + (1 - self._labels) * tf.log(tf.clip_by_value(1 - self._q, 1e-10, 1.0)),
                                        reduction_indices=[1]))

            # Get the trend filtering penalty
            fv = trend_filtering_penalty(z, self._num_classes, self._k, penalty=self._penalty)
            reg = tf.mul(lam, fv)
            
            self._loss_function = tf.add(self._cross_entropy, reg)
            # Reshape to the original dimensions of the density
            density_shape = tf.pack([tf.shape(input_layer)[0]] + list(num_classes))
            self._density = tf.reshape(self._q, density_shape)

    @property
    def labels(self):
        return self._labels

    @property
    def density(self):
        return self._density

    @property
    def output(self):
        return self._q

    @property
    def train_loss(self):
        return self._loss_function

    @property
    def test_loss(self):
        return self._cross_entropy

    @property
    def lam(self):
        return self._lam

    @property
    def k(self):
        return self._k

class SmoothedMultiscaleLayer(DiscreteDistributionLayer):
    '''A dyadic decomposition model with trend filtering on the logits. This
    model smooths the entire space and thus does not scale well to larger
    grid sizes.'''
    def __init__(self, input_layer, input_layer_size, num_classes, k=2, lam=0.005,
                 penalty='lasso', gamlasso_hyperparams=(1.,0.1), scope=None,
                 one_hot=True, **kwargs):
        if not hasattr(num_classes, "__len__"):
            num_classes = (num_classes, )
        
        # All splits are done via half-spaces, so there are always 2^k-1 output
        # nodes. We handle non-power-of-two nodes by keeping track of the buffer
        # sizes vs. the actual multinomial dimensions.
        self._k = k
        self._penalty = penalty
        self._gamlasso_gamma, self._gamlasso_alpha = gamlasso_hyperparams
        self._num_classes = num_classes
        self._dim_sizes = [2**(int(np.ceil(np.log2(c)))) for c in num_classes]
        self._num_nodes = np.prod(self._dim_sizes) - 1 # flatten the density into a 1-d grid
        self._split_labels, self._split_masks = self.multinomial_split_masks()

        with tf.variable_scope(scope or type(self).__name__):
            W = weight_variable([input_layer_size, self._num_nodes])
            b = bias_variable([self._num_nodes])
            self._lam = tf.Variable(np.log(lam), tf.float32, name='lam') if self._penalty == 'gamlasso' else tf.constant(lam, tf.float32, name='lam')
            if one_hot:
                self._labels = tf.placeholder(tf.float32, shape=[None, np.prod(self._num_classes)])
                split_indices = tf.to_int32(tf.argmax(self._labels, 1))
            else:
                self._labels = tf.placeholder(tf.int32, shape=[None, len(self._num_classes)])
                split_indices = tf.to_int32(tf.reduce_sum([self._labels[:,i]*int(np.prod(self._num_classes[i+1:])) for i in xrange(len(self._num_classes))], 0))
            splits, masks = tf.gather(self._split_labels, split_indices), tf.gather(self._split_masks, split_indices)

            # q is the value of the tree nodes
            # m is the value of the multinomial bins
            # z is the log-space version of m
            self._q = tf.inv(1 + tf.exp(-(tf.matmul(input_layer,W) + b)))
            r = splits * tf.log(tf.clip_by_value(self._q, 1e-10, 1.0))
            s = (1 - splits) * tf.log(tf.clip_by_value(1 - self._q, 1e-10, 1.0))
            self._multiscale_loss = tf.reduce_mean(-tf.reduce_sum(masks * (r+s),
                                            reduction_indices=[1]))

            # Convert from multiscale output to multinomial output
            L, R = self.multiscale_splits_masks()
            q_tiles = tf.constant([1, np.prod(self._num_classes)])

            m = tf.map_fn(lambda q_i: self.multiscale_to_multinomial(q_i, L, R, q_tiles), self._q)

            z = tf.log(tf.clip_by_value(m, 1e-10, 1.))

            # Get the trend filtering penalty
            fv = trend_filtering_penalty(z, self._num_classes, self._k, penalty=self._penalty)
            reg = tf.mul(self._lam, fv)

            self._loss_function = tf.add(self._multiscale_loss, reg)

            # Reshape to the original dimensions of the density
            density_shape = tf.pack([tf.shape(self._q)[0]] + list(self._num_classes))
            self._density = tf.reshape(m, density_shape)

    @property
    def labels(self):
        return self._labels

    @property
    def density(self):
        return self._density

    @property
    def output(self):
        return self._q    

    @property
    def train_loss(self):
        return self._loss_function

    @property
    def test_loss(self):
        return self._loss_function

    def multinomial_split_masks(self):
        data = np.array(list(np.ndindex(self._num_classes)))
        bins = [np.arange(c) for c in self._dim_sizes]
        multiscale_labels = ints_to_multiscale(data, bins)
        return tf.constant(multiscale_labels, tf.float32), tf.constant((multiscale_labels != -1).astype(int), tf.float32)

    def multiscale_to_multinomial(self, q_i, L, R, q_tiles):
        qt = tf.tile(tf.expand_dims(q_i, -1), q_tiles)
        qt = R * qt + L * (1 - qt)
        qt += tf.to_float(tf.equal(qt, 0))
        return tf.reduce_prod(qt, reduction_indices=[0])

    def multiscale_splits_masks(self):
        data = np.array(list(np.ndindex(self._num_classes)))
        bins = [np.arange(c) for c in self._dim_sizes]
        labels = ints_to_multiscale(data, bins)
        return tf.constant((labels == 0).astype(int).T, tf.float32), tf.constant((labels == 1).astype(int).T, tf.float32)

class DiscreteLogisticMixtureLayer(DiscreteDistributionLayer):
    '''A generalized implementation of the Discrete Logistic Mixture Model from
    the OpenAI PixelCNN++ model: https://github.com/openai/pixel-cnn'''
    def __init__(self, input_layer, input_layer_size, num_classes, num_components, scope=None, one_hot=True, lazy_density=False, **kwargs):
        if not hasattr(num_classes, "__len__"):
            num_classes = (num_classes, )

        self._num_classes = num_classes
        self._ndims = len(num_classes)
        self._num_components = num_components
        self._num_outputs = ((self._ndims * self._ndims - self._ndims) / 2 + 2*self._ndims + 1)*self._num_components
        scope = scope or type(self).__name__
        np_num_classes = np.array(self._num_classes, dtype=float)
        with tf.variable_scope(scope):
            '''Fit the LMM onto a finite discrete grid.'''
            W = weight_variable([input_layer_size, self._num_outputs])
            b = bias_variable([self._num_outputs])

            # Convert to the range [-1, 1] for all dimensions
            if one_hot:
                self._labels = tf.placeholder(tf.float32, shape=[None, np.prod(self._num_classes)]) # Get the one-hot labels
                grid_values = tf.constant(np.array(list(np.ndindex(self._num_classes))) / (np_num_classes-1) * 2 - 1, tf.float32)
                grid_indices = tf.to_int32(tf.argmax(self._labels, 1))
                lmm_labels = tf.gather(grid_values, grid_indices)
            else:
                self._labels = tf.placeholder(tf.float32, shape=[None, len(self._num_classes)])
                lmm_labels = self._labels / (np_num_classes-1) * 2 - 1

            # Get the model output
            self._q = tf.matmul(input_layer, W) + b

            # Get the loss for all the observations
            self._loss = tf.reduce_mean(-discretized_mix_logistic_log_probs_nd(lmm_labels, self._q, nr_mix=self._num_components, ndims=self._ndims, num_classes=np.array(self._num_classes)))

            if not lazy_density:
                # Reconstruct the full discrete probability distribution by enumerating everything out
                grid_values = np.array(list(np.ndindex(self._num_classes)), np.float32) / (np.array(self._num_classes, dtype=float) - 1) * 2 - 1
                grid_values = tf.constant(grid_values, tf.float32)
                self._grid_values = grid_values

                q_density_shape = tf.pack([tf.shape(self._q)[0] * tf.shape(self._grid_values)[0], tf.shape(self._q)[1]])
                q_density = tf.reshape(tf.tile(self._q, [1, tf.shape(self._grid_values)[0]]), q_density_shape)
                labels_density = tf.tile(self._grid_values, tf.pack([tf.shape(self._q)[0], 1]))
                logprobs_density_shape = tf.pack([tf.shape(self._q)[0], tf.shape(self._grid_values)[0]])
                logprobs_density = tf.reshape(discretized_mix_logistic_log_probs_nd(labels_density, q_density, nr_mix=self._num_components, ndims=self._ndims, num_classes=np.array(self._num_classes)), logprobs_density_shape)
                bin_probs = tf.exp(logprobs_density) / tf.reduce_sum(tf.exp(logprobs_density), reduction_indices=[1], keep_dims=True)
                density_shape = tf.pack([tf.shape(self._q)[0]] + list(self._num_classes))
                self._density = tf.reshape(bin_probs, density_shape)

    @property
    def labels(self):
        return self._labels

    @property
    def density(self):
        return self._density

    @property
    def output(self):
        return self._q

    @property
    def train_loss(self):
        return self._loss

    @property
    def test_loss(self):
        return self._loss

class LocallySmoothedMultiscaleLayer(DiscreteDistributionLayer):
    '''A dyadic decomposition model with trend filtering on the logits. This
    model smooths only a local region around the target node, making it much
    more scalable to larger spaces. This model is sometimes referred to as
    trendfiltering-multiscale elsewhere in the code. '''
    def __init__(self, input_layer, input_layer_size, num_classes, 
                    k=2, lam=0.005, penalty_type='lasso', sparse_penalty_matrix=True,
                    neighbor_radius=5, one_hot=True,
                    scope=None, **kwargs):
        if not hasattr(num_classes, "__len__"):
            num_classes = (num_classes, )
        if not hasattr(neighbor_radius, "__len__"):
            neighbor_radius = tuple([neighbor_radius]*len(num_classes))
        
        # All splits are done via half-spaces, so there are always 2^k-1 output
        # nodes. We handle non-power-of-two nodes by keeping track of the buffer
        # sizes vs. the actual multinomial dimensions.
        self._input_layer = input_layer
        self._input_layer_size = input_layer_size
        self._num_classes = num_classes
        self._one_hot = one_hot
        self._dim_sizes = [int(2**(int(np.ceil(np.log2(c))))) for c in num_classes]
        self._num_nodes = np.prod(self._dim_sizes) - 1 # flatten the density into a 1-d grid
        # Local trend filtering setup
        self._k = k
        self._penalty_type = penalty_type
        self._neighbor_radius = np.array([min(r,int(np.floor(c/2-1))) for r in neighbor_radius], dtype=int)
        self._neighborhood_dims = 2 * self._neighbor_radius + 1
        self._neighborhood_size = np.prod(self._neighborhood_dims)
        self._neighborhood_offsets = np.array(list(np.ndindex(tuple(self._neighborhood_dims))))
        # self._lam = lam * np.prod(self._dim_sizes) / (np.prod(2*self._neighbor_radius+1)) # TODO: Hack to correct for the volume discrepancy
        self._lam = lam
        # Multiscale setup
        self._bins = [np.arange(c) for c in self._dim_sizes]
        self._path_length = int(np.log2(self._dim_sizes).sum())

        with tf.variable_scope(scope or type(self).__name__):
            # Bit weird but W is transposed for compatibility with tf.gather
            # See the _compute_sampled_logits function for reference:
            # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/nn_impl.py
            self._W = weight_variable([self._num_nodes, self._input_layer_size])
            self._b = bias_variable([self._num_nodes])
            ##### Binomial likelihoods loss function
            self._q_indices = tf.placeholder(tf.int32, [None, self._path_length], name='loss_indices')
            self._splits = tf.placeholder(tf.float32, [None, self._path_length], name='loss_splits')
            self._log_probs = self.get_log_probs(self._q_indices, self._splits, [self._path_length])
            self._neg_log_likelihood = tf.reduce_mean(-self._log_probs)
            
            # Get the trend filtering penalty
            if self._lam > 0:
                ###### Local Trend Filtering
                self._neighborhood_indexes = tf.placeholder(tf.int32, [None, self._neighborhood_size*self._path_length], name='neighbor_indices')
                self._neighborhood_splits = tf.placeholder(tf.float32, [None, self._neighborhood_size*self._path_length], name='neighbor_splits')
                self._neighborhood_log_probs = self.get_log_probs(self._neighborhood_indexes,
                                                                  self._neighborhood_splits,
                                                                  [self._neighborhood_size, self._path_length])
                
                self._reg = trend_filtering_penalty(self._neighborhood_log_probs, self._neighborhood_dims, self._k, penalty=self._penalty_type, sparse=sparse_penalty_matrix)
                # Add the loss and regularization penalty together
                self._loss_function = self._neg_log_likelihood + self._lam * self._reg
            else:
                # Just use the k-d tree loss
                self._loss_function = self._neg_log_likelihood

            # Eager evaluation of the density grid
            print 'Pre-caching all paths in the k-d tree...'
            grid = np.array(list(np.ndindex(tuple(self._num_classes))))
            indices, splits = batch_multiscale_label_lookup(grid, self._bins)
            self._grid_indices = indices.T.reshape(list(self._num_classes) + [self._path_length]).astype(np.int32)
            self._grid_splits = splits.T.reshape(list(self._num_classes) + [self._path_length])
            print 'Finished caching.'

            self._density = self.get_density_probs()
            
    def get_log_probs(self, indices, splits, dims):
        '''Get the necessary nodes from the tree, calculate the log probs, and reshape appropriately'''
        dim1size = int(np.prod(dims))
        sampled_W = tf.transpose(tf.gather(self._W, indices), [0,2,1]) # [batchsize, inputlayersize, dim1size]
        sampled_b = tf.gather(self._b, indices) # [batchsize, dim1size]
        # input_layer is [batchsize, inputlayersize]
        # sampled_W is [batchsize, inputlayersize, dim1size]
        # sampled_q is [batchsize, dim1size] corresponding to q = X*W + b
        sampled_q = tf.reshape(tf.batch_matmul(tf.expand_dims(self._input_layer,1), sampled_W), 
                                    [-1, dim1size]) + sampled_b
        sampled_probs = tf.inv(1 + tf.exp(-sampled_q))
        log_probs = tf.log(tf.clip_by_value(tf.select(splits > 0, sampled_probs, 1-sampled_probs), 1e-10, 1.0))
        log_probs_dims = tf.reshape(log_probs, [-1] + dims)
        return tf.reduce_sum(log_probs_dims, reduction_indices=[len(dims)])

    def get_density_probs(self):
        q = tf.matmul(self._input_layer, tf.transpose(self._W)) + self._b
        probs = tf.inv(1 + tf.exp(-q)) # [batchsize, num_nodes]
        log_probs = tf.map_fn(lambda x: self._grid_log_probs(x), probs) # [batchsize, gridlen]
        return tf.exp(log_probs) / tf.reduce_sum(tf.exp(log_probs), reduction_indices=range(1,len(self._num_classes)+1), keep_dims=True)

    def _grid_log_probs(self, x):
        prob_nodes = tf.gather(x, self._grid_indices)
        return tf.reduce_sum(tf.log(tf.clip_by_value(tf.select(self._grid_splits > 0, prob_nodes, 1-prob_nodes), 1e-10, 1.0)), [-1])

    def fill_train_dict(self, feed_dict, batch_positives):
        if self._one_hot:
            batch_positives = one_hot_to_indices(batch_positives, self._num_classes)
        if len(batch_positives.shape) == 1:
            batch_positives = batch_positives[:,np.newaxis]
        indices = self._grid_indices[tuple(batch_positives.T)]
        splits = self._grid_splits[tuple(batch_positives.T)]
        feed_dict[self._q_indices] = indices
        feed_dict[self._splits] = splits
        if self._lam > 0:
            to_lookup = np.concatenate([self.get_neighborhood(x) for x in batch_positives])
            indices = self._grid_indices[tuple(to_lookup.T)]
            splits = self._grid_splits[tuple(to_lookup.T)]
            feed_dict[self._neighborhood_indexes] = indices.reshape((len(batch_positives), self._neighborhood_size*self._path_length))
            feed_dict[self._neighborhood_splits] = splits.reshape((len(batch_positives), self._neighborhood_size*self._path_length))

    def fill_test_dict(self, feed_dict, vals):
        if self._one_hot:
            vals = one_hot_to_indices(vals, self._num_classes)
        if len(vals.shape) == 1:
            vals = vals[:,np.newaxis]
        indices, splits = batch_multiscale_label_lookup(vals, self._bins)
        feed_dict[self._q_indices] = indices.T
        feed_dict[self._splits] = splits.T

    def get_neighborhood(self, v):
        '''Get the surrounding points in the discrete space'''
        starts = []
        for dimsize, radius, v_i in zip(self._num_classes, self._neighborhood_dims, v):
            diameter = radius * 2 + 1
            if v_i < radius:
                starts.append(0)
            elif dimsize - v_i - 1 < radius:
                starts.append(dimsize - diameter)
            else:
                starts.append(v_i - radius)
        return self._neighborhood_offsets + starts

    def dist(self):
        return self._density

    @property
    def labels(self):
        return None

    @property
    def density(self):
        return self._density

    @property
    def output(self):
        return None

    @property
    def train_loss(self):
        return self._loss_function

    @property
    def test_loss(self):
        return self._neg_log_likelihood






