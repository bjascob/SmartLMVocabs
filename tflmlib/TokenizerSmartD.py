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
from  . DataContainer import DataContainer
from  . Tokenizer import Tokenizer


# This is the same as SmartA except there is no NER used here at all and
# POS=NNP* is not labeled with POS automatically.  Now only punctuation is
# combined, then dictionary words are used and if all else fails, use the POS.
class TokenizerSmartD(Tokenizer):
    def __init__(self, dict_fn):
        super(TokenizerSmartD, self).__init__()
        self.word_set = self.loadWordSet(dict_fn)

    # Load the dictionary and create a set
    def loadWordSet(self, dict_fn):
        print('Loading dictionary from ', dict_fn)
        word_set = set()
        with open(dict_fn, 'r') as f:
            lines = f.readlines()
            lines = [l.strip().lower() for l in lines]
            word_set.update(lines)
        return word_set

    # sent must be a dict containing 3 lists (words, pos, ner) all of the same length
    # words converted to lower-case but special tokens (ie POS, NER) are not
    def tokenizeSentence(self, sent):
        sent_len = len(sent['words'])
        tokens = []
        i = 0
        while i < sent_len:
            # Use POS token for proper-nouns
            if sent['pos'][i].startswith('NNP'):
                tokens.append(sent['pos'][i])
                i = self.skipMultiTokens(i, sent['pos'])
            # Replace backets and colons with SEP
            elif sent['pos'][i] == '-LRB-' or sent['pos'][i] == '-RRB-' or sent['pos'][i] == ':':
                tokens.append('SEP')
                i = self.skipMultiTokens(i, sent['pos'])
            # STP labels double-quotes to 2 different single ticks (opening or closing)
            elif sent['pos'][i] == "''" or sent['pos'][i] == '``':
                tokens.append('"')
                i = self.skipMultiTokens(i, sent['pos'])
            # STP labels single quotes with a tick in some instances
            elif sent['pos'][i] == "'" or sent['pos'][i] == '`':
                tokens.append("'")
                i = self.skipMultiTokens(i, sent['pos'])
            # Check for end of sentence marker (period, question-mark and exclamation map to this)
            elif sent['pos'][i] == '.':
                tokens.append('.')
                i = self.skipMultiTokens(i, sent['pos'])
            # If not one of the above, check to see if it's in the dictionary
            elif sent['words'][i].lower() in self.word_set:
                tokens.append(sent['words'][i].lower())
                i += 1
            # If all else fails, just use the POS
            else:
                tokens.append(sent['pos'][i])
                i += 1
        # Check to be sure that there's and end of sentence marker at the end (? and ! map to .)
        if tokens[-1] != '.':
            tokens.append('.')
        return tokens

    # Read in all data from a file
    @staticmethod
    def read(fn):
        dc = DataContainer.load(fn)
        return dc.sents

    # return the index of the next token that's not equal to the current
    @staticmethod
    def skipMultiTokens(i, iterable):
        max_i = len(iterable)
        test  = iterable[i]
        while i < max_i - 1 and test == iterable[i + 1]:
            i += 1
        return i + 1
