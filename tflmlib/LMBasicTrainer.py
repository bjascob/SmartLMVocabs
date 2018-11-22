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
# This file is adapted from code created by Saarland University, Spoken Language Systems (LSV)
# which is partially based on the Tensorflow PTB-LM recipe
# See https://github.com/uds-lsv/TF-NNLM-TK
# and https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py
from __future__ import print_function
from __future__ import division
import os
import sys
import tensorflow as tf
from   . InputData  import InputData
from   . LMBasic    import LMBasic
from   . Vocab      import Vocab


def LMBasicTrainer(config):
    ''' Method for creating / training an LMBasic

    Args:
        config (AttributeContrainer): the model/training configuration
    '''
    train_data = InputData(config.batch_size, config.seq_length, history_size=1)
    test_data  = InputData(config.batch_size, config.seq_length, history_size=1)
    max_train_words = getattr(config, 'max_train_words', sys.maxsize)
    max_test_words  = getattr(config, 'max_test_words', sys.maxsize)
    epoch_splits    = getattr(config, 'epoch_splits', 1)
    train_data.loadIndexedCorpus(os.path.join(config.data_dir, 'train'),
                                 max_train_words, epoch_splits)
    test_data.loadIndexedCorpus(os.path.join(config.data_dir, 'test'),
                                max_test_words)

    # Config must have Vocab size is it so it knows how big to make the output vector
    print('Using vocabulary from ', config.data_dir)
    print()
    config.vocab_size = Vocab(config.data_dir).getVocabSize()

    # create the LM graph for training
    with tf.Graph().as_default():
        with tf.name_scope("Train"):
            with tf.variable_scope("Model", reuse=None):
                model_train = LMBasic(config, True)

        # create the LM graph for testing with shared parameters
        with tf.name_scope("Test"):
            with tf.variable_scope("Model", reuse=True):
                model_test = LMBasic(config, False)

        # Print the trainable parameters
        model_train.printTrainableParams()

        # See if there's an existing model in the directory and if so, build on it
        start_epoch = 0
        model_fn = LMBasic.get_model_fn(config.model_dir)
        if not model_fn:
            print('Initializing a new model')
            # save the training configuration for future need
            if not os.path.isdir(config.model_dir):
                os.makedirs(config.model_dir)
            config.saveJSON(os.path.join(config.model_dir, 'config.json'))
            # Create a new session
            session = tf.Session()
            session.run(tf.global_variables_initializer())
        else:
            print('!! Loading exiting model ', model_fn)
            session = LMBasic.restore_session(model_fn)
            chkpt_num = model_fn.split('-')[-1]
            if chkpt_num.isdigit():
                start_epoch = int(chkpt_num)    # actually start_epoch-1
            else:
                # Note - This will only happen if the model is saved without a
                # checkpoint number, which isn't currently done. In this case,
                # to train more, change the value in the config file and
                # manually set the start_epoch here.
                print('Model is already fully trained')
                start_epoch = config.num_epochs
            print()

        # Test the initial perplexity of the un-trained model (should return ~vocab size)
        test_perplexity = model_test.run_model(session, test_data, eval_op=None,
                                               verbosity=10000, verbose=True)
        print("\n[INFO] Starting perplexity of test set: %.1f" % test_perplexity)
        print('========================\n')

        # model saving manager
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        # loop over all epochs
        last_train_perplexity = test_perplexity
        lr = config.learning_rate
        for e in range(start_epoch, config.num_epochs):
            # If we are spanning data across epochs, set that value
            train_data.set_epoch_num(e)
            test_data.set_epoch_num(e)

            # Run the model for train and test
            session.run(tf.assign(model_train.lr, lr))
            print("[INFO] Epoch: %d, Learning rate: %.2e \n" %
                  (e + 1, session.run(model_train.lr)))
            train_perplexity = model_train.run_model(session, train_data,
                                                     eval_op=model_train.train_op,
                                                     verbosity=50000, verbose=True)
            test_perplexity = model_test.run_model(session, test_data)

            print("\n[SUMMARY] Epoch: {} | Train Perplexity: {:.1f} | "
                  "Test Perplexity: {:.1f} \n".format(e + 1, train_perplexity, test_perplexity))
            print('========================')

            # update the LR dynamically if cost (aka train_perplexity) goes up
            if train_perplexity >= last_train_perplexity:
                lr *= getattr(config, 'decay_rate', 1.0)
            last_train_perplexity = train_perplexity

            # check for divergence of the algorithm
            if train_perplexity > 1.0e5:  # approximate starting perplexity
                raise ValueError('Minimization is diverging')

            # save model after each epoch
            model_path = os.path.join(config.model_dir, 'model.ckpt')
            saver.save(session, model_path, global_step=(e + 1))

        # Close the session
        session.close()
