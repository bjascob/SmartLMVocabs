# Copyright 2018 Brad Jascob
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Input directory for vocab, training and test data
data_dir = ''   # Needs be declared when model is setup
# output directory for models
model_dir = ''   # Needs be declared when model is setup
# rnn, gru, lstm or lstmp
model = 'lstm'
# size of the word embeddings
embed_size = 512
# size of the (recurrent) hidden layers
hidden_size = 512
# number of neurons in the additional projection layer of the LSTMP model (None = not used)
lstmp_proj_size = 512
# number of recurrent layers
num_layers = 1
# probability of keeping activations in the hidden layer
output_keep_prob = 0.9
# probability of keeping the inputs in the word embeddings
input_keep_prob = 0.9
# number of neurons in the additional bottleneck layer for recurrent models (None = not used)
bottleneck_size = None
# use peepholes in the LSTM/LSTMP model (this parameters is used only in LSTM/LSTMP models)
use_peepholes = False
# activation function of the bottleneck layer (if not None)
activation = 'relu'
# initialization method of embeddings, weights and biases. It can be either xavier or None
# The latter will use the default uniform distribution on the interval given by init_scale
init_method = 'xavier'
# interval for initialization of variables. This is used only if init_method = None
init_scale = 0.05
# clip gradients at this value
grad_clip = 5.0
# history size
history_size = 1

# mini-batch size
batch_size = 128
# word sequence length processed at each forward-backward pass
seq_length = 20

# Optimizer = Adam
optimizer = 'AdamOptimizer'
learning_rate = 1e-3    # for adam (TF default = 1e-3)
epsilon = 1.0e-8        # adam param only (TF default is 1e-8, larger values may be more stable)
decay_rate = 1.0        # No lr decay

# The training and test sizes
max_train_words = int(800e6)    # 762,567,698 words in processed BW corpus
max_test_words  = int(1e5)
epoch_splits = 800              # Split the data across x epochs
num_epochs = 800                # Epoch number to train to
