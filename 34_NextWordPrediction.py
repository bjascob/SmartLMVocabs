#!/usr/bin/python3
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
from __future__ import print_function
import os
import readline
import nltk
import tensorflow as tf
import numpy
from   tflmlib import AttribContainer
from   tflmlib import InputData
from   tflmlib import LMBasic
from   tflmlib import SNLPConnection
from   tflmlib import Vocab
from   configs import snlp_server
from   configs import config
try:    # python2/3 compatibility
    input = raw_input
except NameError:
    pass


class Processor(object):
    def __init__(self, model_dir, tokenizer, strip_period):
        self.snlp = SNLPConnection(snlp_server.port)
        self.tokenizer = tokenizer
        self.strip_period = strip_period
        self.config = AttribContainer.fromJSON(os.path.join(model_dir, 'config.json'))
        self.config.batch_size = 5
        self.config.seq_length = 7
        self.indata = InputData(self.config.batch_size, self.config.seq_length,
                                history_size=self.config.history_size)
        self.vocab  = Vocab(self.config.data_dir)
        self.model, self.session = self.model_setup()

    def predict(self, text):
        # Tokenize / index words
        sent = self.snlp.process(text)
        tokens = self.tokenizer.tokenizeSentence(sent)
        # Smart tokenizer automatically puts a '.' at the end of everything, so strip it
        if self.strip_period and tokens[-1] == '.':
            tokens = tokens[:-1]
        indexes = self.vocab.toIndexes(tokens)
        pad_len = self.indata.batch_size * self.config.seq_length - (
            len(indexes) % self.indata.batch_size) + 1
        indexes += [0] * pad_len
        indexes = numpy.array(indexes)
        self.indata.data_to_batches(indexes)    # convert to 3D arrays for input to the model
        self.indata.batches_per_epoch = self.indata.num_batches
        self.indata.epoch_offset = 0
        # Run the model and get back a flattened softmax list
        probs = self.model.predict(self.session, self.indata)
        probs = LMBasic.flattenProbs3D(probs)
        # Find the most likely next words
        maxes        = numpy.argmax(probs, axis=1)
        widx         = len(tokens) - 1  # next predicted word for the last word in the sentence
        next_words   = self.vocab.toWords(list(range(probs.shape[1])))
        next_probs   = [probs[widx, i] for i in range(probs.shape[1])]
        ret_data     = sorted(zip(next_words, next_probs), key=lambda x: x[1], reverse=True)
        return tokens, ret_data

    def model_setup(self):
        # Get the last checkpoint's filename
        model_fn = LMBasic.get_model_fn(self.config.model_dir)
        if not model_fn:
            msg = "Could not open and/or read model from {}"
            raise Exception(msg.format(self.config.model_dir))
        print('Using model ', model_fn)
        print()
        # Setup the model
        with tf.variable_scope("Model", reuse=False):
            model_test = LMBasic(self.config, False)
        # Restore the parameters
        session = LMBasic.restore_session(model_fn)
        return model_test, session


if __name__ == '__main__':
    print('*' * 80)
    print()

    # Pick the vocabulary type
    if 0:   # Simple vocab
        from tflmlib import TokenizerSimple
        # model_dir = os.path.join(config.data_repo, 'L1_512_512-Simple')
        model_dir = os.path.join(config.data_repo, 'L1_2048_512-Simple')
        tokenizer = TokenizerSimple()
        proc = Processor(model_dir, tokenizer, False)
    else:
        from tflmlib import TokenizerSmartA
        # model_dir = os.path.join(config.data_repo, 'L1_512_512-SmartA')
        model_dir = os.path.join(config.data_repo, 'L1_2048_512-SmartA')
        dict_fn   = config.sys_dict
        tokenizer = TokenizerSmartA(dict_fn)
        proc = Processor(model_dir, tokenizer, True)

    print('Loading model/config from ', model_dir)

    topn = 20
    print('Enter a phrase and this will predict the next word')
    print
    while 1:
        # Input the test phrase and correct next word
        text = input('Match phrase > ')
        if not text or text == 'q':
            break
        # Run the model to see what the most likely next words are
        tokens, best_next_words = proc.predict(text)
        print('Best matches for phrase : ', tokens)
        for i, (word, prob) in enumerate(best_next_words):
            print('  %8.2e : %s' % (prob, word))
            if i >= topn - 1: break
        print()
        print()
