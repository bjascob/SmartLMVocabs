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
from  . Tokenizer import Tokenizer
from  . DataContainer import DataContainer


class TokenizerSimple(Tokenizer):
    def __init__(self):
        pass

    # sent must be a dict containing 3 lists (words, pos, ner) all of the same length
    # words are converted to lower-case
    @staticmethod
    def tokenizeSentence(sent):
        words = [w.lower() for w in sent['words']]
        return words

    # Read in all data from a file
    @staticmethod
    def read(fn):
        dc = DataContainer.load(fn)
        return dc.sents
