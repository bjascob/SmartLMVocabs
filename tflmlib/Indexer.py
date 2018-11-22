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
from   __future__ import print_function
import os
import numpy
from   . ProgressBar    import ProgressBar
from   . DataContainer  import DataContainer
from   . Vocab          import Vocab


class Indexer(object):
    ''' Class to handle indexing a text corpus file

    Given a Vocab, this class will use a supplied Tokenizer to read a corpus
    file and convert it into an integer indexed file which can then be saved
    in .npy format for use in training.

    Args:
        vocab_dir (str): Directory containing vocabulary.pkl to be loaded
    '''
    def __init__(self, vocab_dir):
        self.vocab = Vocab(vocab_dir, log_unknowns=True)

    # Index various types of data, compatible with the tokenizer
    def indexFile(self, fn, tokenizer, output_dir, file_index):
        ''' Index a file and save it to a directory

        Args:
            fn (str): filename of file to load
            tokenizer (Tokenizer): A Tokenizer type class to process the input file
            output_dir (str): Directory to save the data
            file_index (int): Index of the file.  File will be <output_dir>/<file_index>.npy

        Returns:
            int: The number of tokens in the output file
        '''
        sents = tokenizer.read(fn)
        data = []
        token_ctr   = 0
        pb = ProgressBar(len(sents))
        for i, sent in enumerate(sents):
            tokens = tokenizer.tokenizeSentence(sent)
            indexes = self.vocab.toIndexes(tokens)
            data.extend(indexes)   # 1D array
            token_ctr += len(tokens)
            if 0 == i % 100:
                pb.update(i)
        pb.clear()
        self._saveFile(data, output_dir, file_index)
        return token_ctr

    # Save the unknown counter for debug
    def saveUnkCounter(self, fn):
        ''' Save the internal unknown token counter to a file for debugging

        Args:
            fn (str): filename for output
        '''
        self.vocab.saveUnkCounter(fn)

    # Save data in numpy format
    def _saveFile(self, data, output_dir, file_index):
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        fn = '%04d.npy' % file_index
        fn = os.path.join(output_dir, fn)
        print('  Saving the converted corpus to ', fn)
        numpy.save(fn, numpy.array(data, dtype='int32'))
