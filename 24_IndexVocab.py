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
from   fnmatch          import fnmatch
from   tflmlib          import Indexer
from   configs          import config


if __name__ == '__main__':
    print('*' * 80)
    print()

    # Enable/Disable test mode so we only process the first 2 files
    test = False

    # Pick the vocabulary type
    dict_fn = os.path.join(config.data_repo, 'english_dict.txt')
    if 0:   # Simple Vocab
        from tflmlib import TokenizerSimple
        vocab_dir = os.path.join(config.data_repo, 'SimpleVocab')
        tokenizer = TokenizerSimple()
    elif 1:   # Smart Vocab A
        from tflmlib import TokenizerSmartA
        vocab_dir = os.path.join(config.data_repo, 'SmartVocabA')
        tokenizer = TokenizerSmartA(dict_fn)
    elif 0:   # Smart Vocab B
        from tflmlib import TokenizerSmartB
        vocab_dir = os.path.join(config.data_repo, 'SmartVocabB')
        tokenizer = TokenizerSmartB(dict_fn)
    elif 0:   # Smart Vocab C
        from tflmlib import TokenizerSmartC
        vocab_dir = os.path.join(config.data_repo, 'SmartVocabC')
        tokenizer = TokenizerSmartC(dict_fn)
    elif 0:   # Smart Vocab D
        from tflmlib import TokenizerSmartD
        vocab_dir = os.path.join(config.data_repo, 'SmartVocabD')
        tokenizer = TokenizerSmartD(dict_fn)

    # Setup the directories
    bw_pkl_dir    = os.path.join(config.bw_corpus, 'BWParsed')
    out_train_dir = os.path.join(vocab_dir, 'train')
    out_test_dir  = os.path.join(vocab_dir, 'test')
    unk_count_fn  = os.path.join(vocab_dir, 'unknown_counts.txt')
    bw_fn_pat     = 'bw_*'

    # Process the data
    indexer = Indexer(vocab_dir)

    print('Gathering the corpus from ', bw_pkl_dir)
    fns = sorted([os.path.join(bw_pkl_dir, fn) for fn in
                 os.listdir(bw_pkl_dir) if fnmatch(fn, bw_fn_pat)])
    if test: fns = fns[:2]
    token_ctr = 0
    for i, fn in enumerate(fns):
        print('  %2d/%2d : %s' % (i + 1, len(fns), fn))
        token_ctr += indexer.indexFile(fn, tokenizer, out_train_dir, i)
    print('Complete.  Indexed {:,} words.'.format(token_ctr))
    print()

    # Save the unknown counts for debug
    indexer.saveUnkCounter(unk_count_fn)
    print()

    # Use the last file for the testing, move that file into a new test directory
    if not os.path.exists(out_test_dir):
        os.mkdir(out_test_dir)
    src_fn = sorted([os.path.join(out_train_dir, fn) for fn in
                    os.listdir(out_train_dir) if fnmatch(fn, '*.npy')])[-1]
    dst_fn = os.path.join(out_test_dir, os.path.basename(src_fn))
    try:
        os.remove(dst_fn)
    except OSError:
        pass
    os.rename(src_fn, dst_fn)
    print('Test data created by moving %s to %s' % (src_fn, dst_fn))
    print()
