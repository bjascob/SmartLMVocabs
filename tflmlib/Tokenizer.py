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


class Tokenizer(object):
    ''' Tokenizer base class

    This class is intended to turn raw text into word tokens that will later
    be turned into integer IDs.  This is the generic base class showing what
    methods are required.
    '''
    def __init__(self):
        pass

    def tokenizeSentence(self, sent):
        ''' Tokenize a sentence

        Args:
            sent (str): text string to tokenized

        Returns:
            list of str: list of human-readable tokens
        '''
        assert False, 'All tokenizers must have tokenizeSentence()'

    def read(self, fn):
        ''' Read a file and return a list of sentences.

        Args:
            fn (str): name of file to read

        Returns:
            list of str: list of sentences in the corpus
        '''
        assert False, 'All tokenizers must have read()'
