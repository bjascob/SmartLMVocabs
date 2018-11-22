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
from   fnmatch      import fnmatch
from   subprocess   import Popen, PIPE
from   tflmlib      import DataContainer
from   tflmlib      import ProgressBar
from   configs          import config


# Billion Word Corpus
bw_pkl_dir    = os.path.join(config.bw_corpus, 'BWParsed_FirstPass')
out_txt_dir   = os.path.join(config.bw_corpus, 'BWTokenized')
out_pkl_dir   = os.path.join(config.bw_corpus, 'BWParsed')
bw_fn_pat     = 'bw_*'


if __name__ == '__main__':
    print('*' * 80)
    print()

    test = False

    # Create directories if needed
    if not os.path.exists(out_txt_dir):
        os.mkdir(out_txt_dir)
    if not os.path.exists(out_pkl_dir):
        os.mkdir(out_pkl_dir)

    # Loop through all the files
    print('Loading the raw corpus')
    fns = sorted([os.path.join(bw_pkl_dir, fn) for fn in
                 os.listdir(bw_pkl_dir) if fnmatch(fn, bw_fn_pat)])
    if test: fns = fns[:1]
    bw_set = set()
    duplicates = 0
    for i, fn in enumerate(fns):
        # Read the data
        print('  %d/%d : %s' % (i + 1, len(fns), fn))
        dcout = DataContainer()
        dcout.sents = []
        txt_sents = []
        dcin = DataContainer.load(fn)
        pb = ProgressBar(len(dcin.sents))
        for i, sent in enumerate(dcin.sents):
            text = ' '.join(sent['words'])
            if text not in bw_set:
                bw_set.add(text)
                dcout.sents.append(sent)
                txt_sents.append(text)
            else:
                duplicates += 1
            if 0 == i % 100: pb.update(i)
        pb.clear()
        # Save the data
        fnbase, _ = os.path.splitext(os.path.basename(fn))
        out_pkl_fn = os.path.join(out_pkl_dir, fnbase + '.pkl')
        out_txt_fn = os.path.join(out_txt_dir, fnbase + '.txt')
        prn_pkl_fn = os.sep.join(out_pkl_fn.split(os.sep)[-3:])
        prn_txt_fn = os.sep.join(out_txt_fn.split(os.sep)[-3:])
        print('  Saving data to %s and %s' % (prn_pkl_fn, prn_txt_fn))
        dcout.save(out_pkl_fn)
        with open(out_txt_fn, 'w') as f:
            for text in txt_sents:
                f.write('%s\n' % text)
    print()
    print('%d duplicates removed from %d files' % (duplicates, len(fns)))
    print()
