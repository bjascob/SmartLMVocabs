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
from   collections      import Counter
from . DataContainer    import DataContainer


class Vocab(object):
    ''' Vocabulary for use in training.

    The Vocabulary is the list of word-tokens and their integer index
    equivalents.  Methods here convert word-tokens to indexes and vice-versa.

    Args:
        vocab_dir (str): directory where vocabulary.pkl is located
        log_unknowns (bool): Create an unknown counter for use while indexing
    '''
    def __init__(self, vocab_dir, log_unknowns=False):
        fn = os.path.join(vocab_dir, 'vocabulary.pkl')
        dc = DataContainer.load(fn)
        self.unk_token   = dc.unk_token
        self.eos_token   = dc.eos_token
        self.idx_t_word  = dc.idx_t_word
        self.word_t_idx  = dc.word_t_idx
        if log_unknowns:
            self.unk_counter = Counter()

    def getVocabSize(self):
        ''' Get the number of tokens / max index of the vocabulary'''
        return len(self.idx_t_word)

    def toIndexes(self, words):
        ''' Convert a list of words to vectors

        Don't convert to lower-case because vocab may have upper-case for
        special token, etc..

        Args:
            words (list of strings): words to convert to indexes

        Returns:
            list of ints: the words converted to indexes
        '''
        assert isinstance(words, list)
        indexes   = []
        for word in words:
            try:
                idx  = self.word_t_idx[word]
            except KeyError:
                if hasattr(self, 'unk_counter'):
                    self.unk_counter[word] += 1
                word = self.unk_token
                idx  = self.word_t_idx[word]
            indexes.append(idx)
        return indexes

    def toWords(self, indexes):
        ''' Convert a list of indexes to words

        Args:
            indexes (list of int): indexes to convert to words

        Returns:
            list of str: indexes converted to words
        '''
        assert isinstance(indexes, list)
        words = []
        for idx in indexes:
            words.append(self.idx_t_word[idx])
        return words

    def saveUnkCounter(self, fn):
        ''' Save the unknown counter for debug

        Args:
            fn (str): name of file to save to
        '''
        unknowns = self.unk_counter.most_common()   # list of (word, count)
        unknowns = sorted(unknowns, key=lambda x: x[1], reverse=True)
        with open(fn, 'w') as f:
            for word, count in unknowns:
                f.write('%8d : %s\n' % (count, word))
        print('Unknown counts written to  %s' % fn)
