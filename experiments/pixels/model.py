import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.regularizers import l2
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from tfsdp.utils import ints_to_multinomials
from tfsdp.models import MultinomialLayer, DiscreteParametricMixtureLayer, \
                         MultiscaleLayer, TrendFilteringLayer, \
                         SmoothedMultiscaleLayer, LocallySmoothedMultiscaleLayer, \
                         DiscreteLogisticMixtureLayer

def train_dict(model, dist, X, y):
    feed_dict = {K.learning_phase(): True}
    if model == 'multinomial':
        feed_dict[dist.labels] = ints_to_multinomials(y, [np.arange(c) for c in dist._num_classes])
    elif model == 'gmm':
        feed_dict[dist.labels] = y
    elif model == 'lmm':
        feed_dict[dist.labels] = y
    elif model == 'sdp':
        dist.fill_train_dict(feed_dict, y)
    else:
        raise Exception('Unknown model type: {0}'.format(model))
    return feed_dict

def test_dict(model, dist, X, y):
    feed_dict = {K.learning_phase(): False}
    if model == 'multinomial':
        feed_dict[dist.labels] = ints_to_multinomials(y, [np.arange(c) for c in dist._num_classes])
    elif model == 'gmm':
        feed_dict[dist.labels] = y
    elif model == 'lmm':
        feed_dict[dist.labels] = y
    elif model == 'sdp':
        dist.fill_test_dict(feed_dict, y)
    else:
        raise Exception('Unknown model type: {0}'.format(model))
    return feed_dict

def create_model(model, x_shape, y_shape, variable_scope='pixels-', dimsize=256, **kwargs):
    with tf.variable_scope(variable_scope):
        X_image = tf.placeholder(tf.float32, [None] + list(x_shape[1:]), name='X')
        conv1 = Convolution2D(32, 3, 3, border_mode='same', activation=K.relu, W_regularizer=l2(0.01),
                                        input_shape=x_shape[1:])(X_image)
        pool1 = MaxPooling2D(pool_size=(2,2), border_mode='same')(conv1)
        drop1 = Dropout(0.5)(pool1)
        conv2 = Convolution2D(64, 5, 5, border_mode='same', activation=K.relu, W_regularizer=l2(0.01))(drop1)
        pool2 = MaxPooling2D(pool_size=(2,2), border_mode='same')(conv2)
        drop2 = Dropout(0.5)(pool2)
        drop2_flat = tf.reshape(drop2, [-1, 3*3*64])
        hidden1 = Dense(1024, W_regularizer=l2(0.01), activation=K.relu)(drop2_flat)
        drop_h1 = Dropout(0.5)(hidden1)
        hidden2 = Dense(128, W_regularizer=l2(0.01), activation=K.relu)(drop_h1)
        drop_h2 = Dropout(0.5)(hidden2)
        hidden3 = Dense(32, W_regularizer=l2(0.01), activation=K.relu)(drop_h2)
        drop_h3 = Dropout(0.5)(hidden3)

        num_classes = tuple([dimsize]*y_shape[1])
        print(num_classes)
        if model == 'multinomial':
            dist_model = MultinomialLayer(drop_h3, 32, num_classes, **kwargs)
        elif model == 'gmm':
            dist_model = DiscreteParametricMixtureLayer(drop_h3, 32, num_classes, **kwargs)
        elif model == 'lmm':
            dist_model = DiscreteLogisticMixtureLayer(drop_h3, 32, num_classes, **kwargs)
        elif model == 'sdp':
            dist_model = LocallySmoothedMultiscaleLayer(drop_h3, 32, num_classes, **kwargs)
        else:
            raise Exception('Unknown model type: {0}'.format(model))

        return X_image, dist_model

