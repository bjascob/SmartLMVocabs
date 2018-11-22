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
from   __future__ import print_function
import os
import codecs
import nltk
from   unidecode    import unidecode
from   configs      import config


if __name__ == '__main__':
    print('*' * 80)

    dict_in_fn  = config.sys_dict
    dict_out_fn = os.path.join(config.data_repo, 'english_dict.txt')

    # Load the dictionary and tokenize the words if needed
    # ie.. split the 's or n't from the ends so it's consistant
    # with the Stanford Parser's tokenization
    print('Loading dictionary from ', dict_in_fn)
    word_set = set()
    with codecs.open(dict_in_fn, "r", "utf-8") as f:
        lines = f.readlines(f)
        for line in lines:
            line = line.strip().lower()
            line = unidecode(line)
            parts = nltk.tokenize.word_tokenize(line)
            word_set.update(parts)

    # Save the dictionary
    with open(dict_out_fn, 'w') as f:
        for word in sorted(word_set):
            f.write('%s\n' % word)
    print('Data written to ', dict_out_fn)
    print()
