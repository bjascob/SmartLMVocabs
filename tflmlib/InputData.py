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
import fnmatch
import numpy


#
#
class InputData(object):
    ''' Class for reading a directory of data and delivering it for training / test

    Read a corpus directory of preprocessed .npy files and transform it into
    arrays of input and target batches.  All data in the directory is read
    until max_words is reached.  Data can be optionally split across epochs.

    Args:
        batch_size (int): number of sequences per batch
        seq_length (int): number of word tokens in a sequence
        history_size (int): size of history = 1
    '''
    def __init__(self, batch_size, seq_length, history_size=1):
        self.batch_size   = batch_size
        self.seq_length   = seq_length
        self.history_size = history_size

    def loadIndexedCorpus(self, data_dir, max_words=int(1e9), epoch_splits=1):
        ''' Load the preprocessed corpus of index data (.npy format)

        Files are read in numerical order.  Reading is stopped when max_words
        is reached.

        Args:
            data_dir (str): name of directory where data is saved
            max_words (int): number or words/tokens to read in
            epoch_splits (int): number of epochs to split the data across
        '''
        fns = [os.path.join(data_dir, fn) for fn in os.listdir(data_dir)
               if fnmatch.fnmatch(fn, '*.npy')]
        data = numpy.zeros(max_words, dtype='int32')
        ptr = 0
        for fn in sorted(fns):
            print('Loading data from ', fn)
            with open(fn, 'r') as f:
                new_data = numpy.load(fn)
                load_size = min(new_data.shape[0], data.shape[0] - ptr)
                data[ptr:ptr + load_size] = new_data[:load_size]
                ptr += load_size
            if ptr >= max_words:
                break
        data = data[:ptr]
        print('Total size of loaded data is {:,} words.'.format(data.shape[0]))
        self.data_to_batches(data)      # create the input/target batches from the data array
        self.reset_batch_pointer()  # make sure that the index points at the first batch
        # Split the data  across epochs if requested.  Truncate data as needed.
        if epoch_splits > 1:
            self.epoch_splits = epoch_splits
            self.batches_per_epoch = int(self.num_batches / epoch_splits)
            print('Data split across {:} epochs.  Batches per epoch is {:,}.  '
                  'Words per epoch is {:,}'.format(
                      self.epoch_splits, self.batches_per_epoch,
                      self.batches_per_epoch * self.batch_size * self.seq_length))
        else:
            self.epoch_splits = 1
            self.batches_per_epoch = self.num_batches
        self.set_epoch_num(0)
        print('')

    def data_to_batches(self, data):
        ''' Create batches from data and store it in self.input/target

        Args:
            data (numpy array): Numpy array of data to be split into batches
        '''
        # Figure number of batches, truncate the data and print an error
        # message when the data array is too small
        self.num_batches = int((data.size - self.history_size) /
                               (self.batch_size * self.seq_length))
        if self.num_batches == 0:
            msg = "ERROR: Cannot create batches ==> data size={}, batch size={}, segment size={}"
            assert False, msg.format(data.size, self.batch_size, self.seq_length)
        data = data[:(self.num_batches * self.batch_size * self.seq_length) + self.history_size]
        # Remove the last words in the input chunk and shift the target words
        input  = data[:-1]
        target = data[self.history_size:]
        # Chunk the data for consumption
        input  = numpy.array(self.chunk(input, (self.num_batches * self.seq_length) +
                             self.history_size - 1, overlap=self.history_size - 1))
        target = numpy.array(self.chunk(target, (self.num_batches * self.seq_length), overlap=0))
        self.input  = self.chunk(input,  self.seq_length + self.history_size - 1,
                                 overlap=self.history_size - 1)
        self.target = self.chunk(target, self.seq_length, overlap=0)
        self.reset_batch_pointer()

    def get_num_batches(self):
        ''' Get he number of batches for an epoch

        Returns:
            int: Number of batches per epoch
        '''
        return self.batches_per_epoch

    def next_batch(self):
        ''' Get X and Y data and increment the internal pointer

        Returns:
            numpy array: X input data
            numpy array: Y target data
        '''
        ptr = self.pointer + self.epoch_offset
        x, y = self.input[ptr], self.target[ptr]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        ''' Reset the batch pointer to the beginning of the data'''
        self.pointer = 0

    def set_epoch_num(self, epoch_num):
        ''' Set the offset into the data based on the epoch number

        This wraps if epoch_num > epoch_splits
        '''
        self.epoch_offset = self.batches_per_epoch * (epoch_num % self.epoch_splits)
        self.pointer  = 0

    @staticmethod
    def chunk(A, seq_len, overlap=0):
        ''' Chunk an data up for use in the net

        This function chunks data up into an array that is indexed appropriately
        for input into the network.  The chunking is done so that a sequences are
        continous across a batch boundaries which makes the order someone unusual.
        For an explanation see https://github.com/uds-lsv/TF-NNLM-TK/issues/4
        '''
        if overlap >= seq_len:
            print("ERROR in function chunk: overlap cannot be >= to sequence length")
            exit(0)
        if A.ndim == 1:
            Alen = A.shape[0]
            return [A[i:i + seq_len] for i in range(0, Alen - seq_len + 1, seq_len - overlap)]
        elif A.ndim == 2:
            Alen = A.shape[1]
            return [A[:, i:i + seq_len] for i in range(0, Alen - seq_len + 1, seq_len - overlap)]
        else:
            print("ERROR in function chunk: this function works only for 1-D and 2-D arrays")
            exit(0)
