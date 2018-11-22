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
import sys
import math
from   collections import Counter
from . DataContainer    import DataContainer
from . ProgressBar      import ProgressBar


class VocabBuilder(object):
    ''' Build a vocabulary from the corpus to be loaded by the Vocab class

    Args:
        vocab_count_fn (str): Optional.  Load a previously saved vocabulary.
    '''
    unk_token          = '<unk>'
    eos_token          = '<eos>'
    vocab_pkl_fn       = 'vocabulary.pkl'
    vocab_txt_fn       = 'vocabulary.txt'
    vocab_count_pkl_fn = 'vocab_counts.pkl'
    vocab_count_txt_fn = 'vocab_counts.txt'

    def __init__(self, vocab_count_fn=None):
        if vocab_count_fn is not None:
            dc = DataContainer.load(vocab_count_fn)
            self.counter = dc.counter
        else:
            self.counter   = Counter()
            self.counter[self.unk_token] = 0    # gaurentee this is in vocab

    def addFile(self, fn, tokenizer):
        ''' Add all the tokens from a file to the vocabulary

        Args:
            tokenizer (Tokenizer): Tokenizer class used to read the file and tokenize it

        Returns:
            int: the number of tokens read
            int: the number of sentences read
        '''
        sents = tokenizer.read(fn)
        pb = ProgressBar(len(sents))
        token_ctr = 0
        for i, sent in enumerate(sents):
            tokens = tokenizer.tokenizeSentence(sent)
            for token in tokens:
                self.counter[token] += 1
            token_ctr += len(tokens)
            if 0 == i % 100:
                pb.update(i)
        pb.clear()
        return token_ctr, len(sents)

    def getTopN(self, max_vocab):
        ''' Get up to max_vocab words and their counts in a list of (word, count)

        This will clip the vocabulary to max_vocab so that only the most commonly
        occuring tokens will remain.

        Note that this will clip the vocab list and assign all words counts
        outside of this to <unk>.  This will also gaurentee <unk> is defined,
        even if the count is 0

        Args:
            max_vocab (int): maximum number of tokens in the vocabulary.

        Returns:
            List of str: Sorted list of tokens in the vocabulary
            Counter: the Counter object of the vocabulary
        '''
        new_counter = self.reduceVocabToTopN(self.counter, max_vocab, self.unk_token)
        vocab = new_counter.most_common()           # list of (word, count)
        has_unk = any([item[0] == self.unk_token for item in vocab])
        if not has_unk:
            vocab.insert(0, (self.unk_token, 0))
            vocab = vocab[:-1]
        vocab = sorted(vocab, key=lambda x: x[0])    # vocab is alphabetical
        return vocab, new_counter    # (word, count)

    def save(self, data_dir, max_vocab=sys.maxsize):
        ''' Save the vocabulary to a directory

        Saves multiple files to a directory.

        Args:
            data_dir (str): dirtectory to save the data to.
        '''
        # Create the directory if needed
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Save the vocab to a pkl (this file is used for the Vocab.py class)
        vocab, new_counter = self.getTopN(max_vocab)
        fn = os.path.join(data_dir, self.vocab_pkl_fn)
        dc = DataContainer()
        dc.idx_t_word = [w[0] for w in vocab]
        dc.word_t_idx = {w[0]: i for i, w in enumerate(vocab)}
        dc.unk_token  = self.unk_token
        dc.eos_token  = self.eos_token
        dc.save(fn)
        print('Vocab data written to      ', fn)

        # Save the vocab to a text file in indexed order
        # (ie.. alphabetical) for debug
        fn = os.path.join(data_dir, self.vocab_txt_fn)
        with open(fn, 'w') as f:
            for word in vocab:
                f.write(word[0] + '\n')
        print('Vocab words written to     ', fn)

        # Save the decimated counter to a pkl
        fn = os.path.join(data_dir, self.vocab_count_pkl_fn)
        dc = DataContainer()
        dc.counter = new_counter
        dc.save(fn)
        print('Vocab counter written to   ', fn)

        # Save the decimated counts but sorted by frequency,
        # not alphabetically, for debug
        vocab = sorted(vocab, key=lambda x: x[1], reverse=True)
        fn = os.path.join(data_dir, self.vocab_count_txt_fn)
        with open(fn, 'w') as f:
            for word, count in vocab:
                f.write('%8d : %s\n' % (count, word))
        print('Vocab counts written to    ', fn)

    @staticmethod
    def getUnigramPerplexity(counter):
        ''' Get the unigram perplexity of a vocabulary counter

        This basically calculates perplexity on the same data set
        used to create the counter.

        Args:
            Counter: the Counter with vocabulary data

        Returns:
            int: perplexity of the data based on the unigram
        '''
        word_count = sum(counter.values())
        entropy = 0.0
        for count in counter.values():
            prob = count / float(word_count)
            if prob > 0:
                entropy += count * math.log(prob, 2)
        perplexity = math.pow(2, -1 * entropy / float(word_count))
        return int(perplexity)

    @staticmethod
    def reduceVocabToTopN(counter, topn, unk_token='<unk>'):
        ''' Change the vocab counter to only use the topn

        Args:
            counter (Counter): the vocabulary data
            topn (int): the maximum number of words in the returned vocabulary
            unk_token (str): string to use for "unknown"
        '''
        new_counter = Counter()
        for i, (word, counter) in enumerate(counter.most_common()):
            if i < topn - 1:
                new_counter[word] = counter
            else:
                new_counter[unk_token] += counter
        return new_counter
