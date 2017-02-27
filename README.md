Smoothed Dyadic Partitions
----------------------------------------------
Deep nonparametric conditional discrete probability estimation via smoothed dyadic partitioning.

- See the `tfsdp` directory (specifically models.py) for the details of all the models we implemented. Note that this file contains a lot of garbage code and legacy naming that needs to be cleaned up. Our SDP model is named LocallySmoothedMultiscaleLayer and is often referred to in some of the experiments as trendfiltering-multiscale. 

- See the `experiments` directory for the code to replicate our experiments.

Note that you should be able to install the package as a local pip package via `pip -e .` in this directory. The best example for how to run the models is in `experiments/uci/main.py`, which contains the most recent code and should not have any API issues.

Installation
============
You can install via Pip: `pip install tf-sdp`

Using SDP
=========
Adding an SDP layer to your code is straightforward:

```python
import tensorflow as tf
from keras import backend as K
from keras.regularizers import l2
from keras.layers import Dense, Dropout, Flatten
from tfsdp.models import LocallySmoothedMulticaleLyaer

# Load everything else you need
# ...
num_classes = (32, 45) # Discrete output space with shape 32 x 45

# Create your awesome deep model with lots of layers and whatnot
# ...
final_hidden_layer = Dense(final_hidden_size, W_regularizer=l2(0.01), activation=K.relu)(...)
final_hidden_drop = Dropout(0.5)(final_hidden_layer)
model = LocallySmoothedMultiscaleLayer(final_hidden_drop, final_hidden_size, num_classes, one_hot=False)

# ...
# You can get the training loss for an optimizer
opt = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=args.epsilon)
train_step = opt.minimize(model.train_loss)

# Training evaluation and learning is straightforward
feed_dict = {} # fill the train dict with other needed params like training flag and input vars
model.fill_train_dict(self._train_dict, labels) # add the relevant dyadic nodes to the dictionary
sess.run(train_step, feed_dict=feed_dict)

# You can also get the testing loss for validation
feed_dict = {} # fill the train dict with other needed params like training flag and input vars
model.fill_test_dict(self._train_dict, labels) # add the relevant dyadic nodes to the dictionary
loss += sess.run(model.test_loss, feed_dict=feed_dict)

# If you want the full conditional distribution over the entire space:
feed_dict = {} # fill the train dict with other needed params like training flag and input vars
density = sess.run(model.density, feed_dict=feed_dict) # density will have shape [batchsize,num_classes]
```

See `experiments/uci/model.py` and `experiments/uci/main.py` for complete examples on how to setup and run the model.

Citation
========
If you use this code in your work, please cite the following:

```
@article{tansey:etal:2017:sdp,
  title={Deep Nonparametric Estimation of Discrete Conditional Distributions via
  Smoothed Dyadic Partitioning},
  author={Tansey, Wesley and Pichotta, Karl and Scott, James G.},
  journal={arXiv preprint arXiv:1702.07398},
  year={2017}
}
```

The paper is [available here](http://arxiv.org/abs/1702.07398).
