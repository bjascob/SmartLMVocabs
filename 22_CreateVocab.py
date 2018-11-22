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
from   fnmatch  import fnmatch
from   tflmlib  import VocabBuilder
from   configs  import config


if __name__ == '__main__':
    print('*' * 80)
    print()

    # Enable/Disable test mode so we only process the first file
    test = False

    # Pick the vocabulary type
    dict_fn = os.path.join(config.data_repo, 'english_dict.txt')
    if 0:   # Simple Vocab
        from tflmlib  import TokenizerSimple
        vocab_dir     = os.path.join(config.data_repo, 'SimpleVocab')
        topn = 64830    # same as Smart vocab A
        tokenizer = TokenizerSimple()
    elif 1:   # Smart Vocabulary A
        from tflmlib  import TokenizerSmartA
        topn = 100000   # include everything
        tokenizer = TokenizerSmartA(dict_fn)
        vocab_dir = os.path.join(config.data_repo, 'SmartVocabA')
    elif 0:   # Smart Vocabulary B
        from tflmlib  import TokenizerSmartB
        topn = 100000   # include everything
        tokenizer = TokenizerSmartB(dict_fn)
        vocab_dir = os.path.join(config.data_repo, 'SmartVocabB')
    elif 0:   # Smart Vocabulary C
        from tflmlib  import TokenizerSmartC
        topn = 100000   # include everything
        tokenizer = TokenizerSmartC(dict_fn)
        vocab_dir = os.path.join(config.data_repo, 'SmartVocabC')
    elif 0:   # Smart Vocabulary D
        from tflmlib  import TokenizerSmartD
        topn = 100000   # include everything
        tokenizer = TokenizerSmartD(dict_fn)
        vocab_dir = os.path.join(config.data_repo, 'SmartVocabD')

    # Setup the directories
    bw_pkl_dir    = os.path.join(config.bw_corpus, 'BWParsed')
    bw_fn_pat     = 'bw_*'

    # Run through all files in the directory to get the vocab
    vb = VocabBuilder()
    print('Gathering the corpus from ', bw_pkl_dir)
    fns = sorted([os.path.join(bw_pkl_dir, fn) for fn in
                 os.listdir(bw_pkl_dir) if fnmatch(fn, bw_fn_pat)])
    if test: fns = fns[:1]
    word_ctr = 0
    sent_ctr = 0
    for i, fn in enumerate(fns):
        print('  %2d/%2d : %s' % (i + 1, len(fns), fn))
        nwords, nsents = vb.addFile(fn, tokenizer)
        word_ctr += nwords
        sent_ctr += nsents
    print('Complete.  Loaded {:,} words from {:,} sentences'.format(word_ctr, sent_ctr))
    print()

    # Save the vocab
    vb.save(vocab_dir, topn)
    print()
