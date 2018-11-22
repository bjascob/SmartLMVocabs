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
#
# Acknowledgements:
# Portions of this file are based on code from Saarland University, Spoken Language Systems LSV
# which is partially based on the Tensorflow PTB-LM recipe
# See https://github.com/uds-lsv/TF-NNLM-TK
# and https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py
from   __future__ import print_function
from   __future__ import division
import time
import tensorflow as tf
import numpy as np
from   tensorflow.contrib import rnn
from   tensorflow.contrib import legacy_seq2seq


def data_type():
    return tf.float32


act_dict = {
    'tanh':     tf.nn.tanh,
    'sigmoid':  tf.nn.sigmoid,
    'relu':     tf.nn.relu,
    'elu':      tf.nn.elu,
    'relu6':    tf.nn.relu6}


class LMBasic():
    """
    This classe implements the basic RNN-LMs using the built-in Tensorflow cells.
    In particular, this calss can be used to train vanilla-RNN, LSTM (with and
    witout projection) and GRU.
    """

    def __init__(self, config, training=True):
        """
        The constructor of the RNN-LM. We define here the complete graph.
        """
        ###############################################################
        # Setup the model params
        self.training = training
        # bottleneck layer activation function
        self.activation = act_dict.get(config.activation, None)
        self.history_size = 1
        for attr in ['model', 'init_method', 'num_layers', 'input_keep_prob',
                     'output_keep_prob', 'vocab_size', 'use_peepholes', 'embed_size']:
            val = getattr(config, attr)
            setattr(self, attr, val)
        # hidden size (layer): internal to the models (e.g., memory in LSTM).
        self.hidden_state_size = config.hidden_size
        # recurrent layer: layer that feeds back in time into the model.
        self.recurrent_state_size = config.hidden_size
        # last layer: layer right before the output layer (can be bottleneck or recurrent layer).
        self.last_layer_size = config.hidden_size

        # check consistencies in the LSTM parameters
        if config.model == "lstm" or config.model == "lstmp":
            if config.use_peepholes or config.lstmp_proj_size:
                self.model = "lstmp"
            else:
                self.model = "lstm"

        if self.model == "lstmp" and config.bottleneck_size:
            print("[WARNING] you are using a bottleneck layer on the the " +
                  "top of an LSTMP model, which includes an internal " +
                  "bottleneck (projection) layer...!")

        if config.bottleneck_size:
            self.last_layer_size = config.bottleneck_size

        if self.model == "lstmp" and config.lstmp_proj_size:
            self.recurrent_state_size = config.lstmp_proj_size
            self.last_layer_size = config.lstmp_proj_size
            if config.bottleneck_size:
                self.last_layer_size = config.bottleneck_size

        ###############################################################
        # DEFINE THE PLACEHOLDERS
        # placeholder for the training input data and target words
        self.input_data = tf.placeholder(
            tf.int32, [config.batch_size, config.seq_length])
        self.targets = tf.placeholder(
            tf.int32, [config.batch_size, config.seq_length])

        ###############################################################
        # DEFINE TRAINABLE VARIABLES (WEIGHTS AND BIASES)
        # define the initializer of embeddings, weights and biases
        if self.init_method == "xavier":
            initializer = tf.contrib.layers.xavier_initializer(uniform=True)
        else:
            initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

        # word embeddings
        with tf.variable_scope("input_layer"):
            self.embedding = tf.get_variable(
                "embedding", [self.vocab_size, self.embed_size], initializer=initializer)

        # weights and biases of the bottleneck layer (if used)
        if config.bottleneck_size:
            with tf.variable_scope("bottleneck_layer"):
                self.bottleneck_w = tf.get_variable(
                    "bottleneck_w", [self.recurrent_state_size, config.bottleneck_size],
                    initializer=initializer)
                self.bottleneck_b = tf.get_variable(
                    "bottleneck_b", [config.bottleneck_size], initializer=initializer)

        # weights and biases of the hidden-to-output layer
        with tf.variable_scope("output_layer"):
            self.output_w = tf.get_variable("output_w",
                                            [self.last_layer_size, self.vocab_size],
                                            initializer=initializer)
            self.output_b = tf.get_variable("output_b", [self.vocab_size],
                                            initializer=initializer)

        ###############################################################
        # BUILD THE LM NETWORK GRAPH
        # extract the embedding of each char input in the batch
        inputs = tf.nn.embedding_lookup(self.embedding, self.input_data)

        # apply dropout to the input if needed.
        if self.training and self.input_keep_prob < 1:
            inputs = tf.nn.dropout(inputs, self.input_keep_prob)

        # rearrange our input shape to create the training sequence
        # we create a sequence made of the vertical slices the input
        inputs = tf.split(inputs, config.seq_length, 1)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        # build the separate forward and backward lstm cells
        self.cell = self.build_recurrence_cell(config)

        # initialize the hidden (recurrent) state to zero
        self.initial_state = self.cell.zero_state(config.batch_size, tf.float32)

        # build the LM and update the hidden state
        rec_state, self.final_state = self.time_sequence_graph(inputs)

        if config.bottleneck_size:
            last_layer = self.activation(tf.nn.xw_plus_b(
                rec_state, self.bottleneck_w, self.bottleneck_b))
        else:
            last_layer = rec_state

        # self.logits = tf.matmul(output, self.output_w) + self.output_b
        # self.probs = tf.nn.softmax(self.logits)
        logits = tf.nn.xw_plus_b(last_layer, self.output_w, self.output_b)
        # reshape logits to be a 3-D tensor for sequence loss
        self.logits = tf.reshape(logits, [config.batch_size, config.seq_length, self.vocab_size])

        loss = tf.contrib.seq2seq.sequence_loss(
            self.logits,
            self.targets,
            tf.ones([config.batch_size, config.seq_length], dtype=data_type()),
            average_across_timesteps=False,
            average_across_batch=True)

        with tf.name_scope('cost'):
            self.cost = tf.reduce_sum(loss)

        with tf.name_scope('probs'):
            # produces 3D output [epochs, batch_size, seq_len]
            self.probs = tf.nn.softmax(self.logits)

        ###################################################
        # If we are in the training stage, then  calculate the loss, back-propagate
        # the error and update the weights, biases and word embeddings
        if self.training:
            self.lr = tf.Variable(0.0, trainable=False)
            tvars = tf.trainable_variables()
            # clip the gradient by norm
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), config.grad_clip)
            # update variables (weights, biases, embeddings...)
            with tf.name_scope('optimizer'):
                if config.optimizer == 'GradientDescentOptimizer':
                    optimizer = tf.train.GradientDescentOptimizer(self.lr)
                elif config.optimizer == 'AdamOptimizer':
                    epsilon = getattr(config, 'epsilon', 1.0e-8)
                    optimizer = tf.train.AdamOptimizer(self.lr, epsilon=epsilon)
                elif  config.optimizer == 'AdagradOptimizer':
                    optimizer = tf.train.AdagradOptimizer(self.lr, initial_accumulator_value=1.0)
                else:
                    raise ValueError('Invalid optimizer spec: %s' % config.optimizer)
                self.train_op = optimizer.apply_gradients(
                    zip(grads, tvars), global_step=tf.train.get_or_create_global_step())

    def build_recurrence_cell(self, config):
        """
        Build and return the recurrent cell that will be used by our LM.
        This class uses only the built-in Tensorflow
        """
        # if needed, the activation function used by the basic model can change be changed as well
        activation_ = tf.nn.tanh
        if self.model == 'rnn':
            _cell_ = rnn.BasicRNNCell
        elif self.model == 'gru':
            _cell_ = rnn.GRUCell
        elif self.model == "lstmp":
            _cell_ = rnn.LSTMCell
        elif self.model == "lstm":
            _cell_ = rnn.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(self.model))
        cells = []
        # Apply dropout if required
        for _ in range(self.num_layers):
            if  self.model == "lstmp":      # activation function of the project layer
                cell = _cell_(self.hidden_state_size, use_peepholes=self.use_peepholes,
                              num_proj=config.lstmp_proj_size)
            else:
                cell = _cell_(self.hidden_state_size, activation=activation_)

            if self.training and self.output_keep_prob < 1.0:
                cell = rnn.DropoutWrapper(cell, output_keep_prob=self.output_keep_prob)
            cells.append(cell)
        # build and return the TF multi-recurrent cell graph
        return rnn.MultiRNNCell(cells, state_is_tuple=True)

    def time_sequence_graph(self, inputs):
        """
        Apply the recurrence cell to an input sequence (each batch entry is a sequence of words).
        return: stacked cell outputs of the complete sequence in addition to the last hidden state
        (and memory for LSTM/LSTMP) obtained after processing the last word (in each batch entry).
        """
        outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state,
                                                         self.cell, loop_function=None)
        output = tf.reshape(tf.concat(outputs, 1), [-1, self.recurrent_state_size])
        return output, last_state

    def run_model(self, session, data, eval_op=None, verbosity=10000, verbose=False):
        """
        Train or test the current model on some given data.
        This basically trains/applies the model on some data
        loaded by the data processor.
        This will help training on a large corpus by splitting
        them into smaller chunks and processing them one by one.
        """
        data.reset_batch_pointer()

        start_time = time.time()
        costs = 0.0
        iters = 0
        state = session.run(self.initial_state)

        fetches = {
            "cost": self.cost,
            "final_state": self.final_state,
        }
        if eval_op is not None:
            fetches["eval_op"] = eval_op

        print_tresh = verbosity
        for step in range(data.get_num_batches()):
            input, target = data.next_batch()
            feed_dict = {self.initial_state: state, self.input_data: input, self.targets: target}
            vals = session.run(fetches, feed_dict)
            cost = vals["cost"]
            state = vals["final_state"]

            costs += cost
            iters += data.seq_length

            total_proc_words = float((iters - 1) * data.batch_size)

            if verbose and (total_proc_words > print_tresh or step == data.get_num_batches() - 1):
                print("[INFO] Progress: {:6.2f}% | Perplexity: {:7.1f} | "
                      "Total Words: {:7.1f}K | Speed: {:4.1f}K word/second".format(
                          (step + 1) / (data.get_num_batches()) * 100, np.exp(costs / iters),
                          total_proc_words / 1000,
                          total_proc_words / (1000 * (time.time() - start_time))))
                print_tresh += verbosity

        return np.exp(costs / iters)

    def predict(self, session, data):
        data.reset_batch_pointer()
        state = session.run(self.initial_state)
        fetches = {
            "probs": self.probs,
            "final_state": self.final_state}

        probs = []
        for step in range(data.get_num_batches()):
            input, target = data.next_batch()
            feed_dict = {self.initial_state: state, self.input_data: input}
            vals  = session.run(fetches, feed_dict)
            state = vals["final_state"]
            probs.append(vals["probs"])
        probs = np.asarray(probs)           # probs [epoch, batch_size, seq_len]
        return probs

    # This is odd but coming out of predict, the sequences are ordered on the
    # 3rd axis (as expected)but then increment in the 1st axis and finally the 2nd.
    # Here swap the axis and then flatten all but the 4th (softmax).
    # Axis must be swapped first so that the flatten (ie.. reshape) operation is
    # in the correct order.
    @staticmethod
    def flattenProbs3D(probs):
        probs = probs.swapaxes(0, 1)    # probs [batch_size, epoch, seq_len].
        probs = probs.reshape(probs.shape[0] * probs.shape[1] * probs.shape[2], -1)
        return probs

    # Get the total number of trainable parameters
    def getTrainableParams(self):
        return np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])

    # Print the trainable parameters and their sizes / shapes
    def printTrainableParams(self):
        print('Model parameters:')
        total_parameters = 0
        for v in tf.trainable_variables():
            shape = v.get_shape()
            variable_parameters = np.prod(v.get_shape().as_list())
            v_name = '/'.join(v.name.split('/')[-2:])
            print('   {:30s}  shape: {:20s}  Num-parms: {:,}'.format(
                  v_name, str(v.get_shape()), variable_parameters))
            total_parameters += variable_parameters
        print('Total params: {:,}'.format(total_parameters))
        print()

    # Check the directory to see if there's a model or a checkpoint
    @staticmethod
    def get_model_fn(model_dir):
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            return ckpt.model_checkpoint_path
        return None

    # Load a previously saved model
    @staticmethod
    def restore_session(model_fn):
        session = tf.Session()
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(session, model_fn)
        return session
