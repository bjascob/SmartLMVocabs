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
import io
import math
import random
from   subprocess   import Popen, PIPE
from   tflmlib      import ProgressBar
from   configs      import config

# Billion Word Corpus
# To create the training-monolingual directory ..
#   wget http://statmt.org/wmt11/training-monolingual.tgz
#   tar xvf training-monolingual.tgz --wildcards training-monolingual/news.20??.en.shuffled
#   The 9.9 GB file extracts to 25GB without the wildcards. Other files are non-english.
bw_raw_dir    = config.bw_rawdir
bw_unique_dir = os.path.join(config.bw_corpus, 'BWUniqueSents_FirstPass')


# Get the number of lines in a file
def getFileLines(fname):
    p = Popen(['wc', '-l', fname], stdout=PIPE, stderr=PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])


if __name__ == '__main__':
    print('*' * 80)
    print()

    nshards = 100
    test = False

    # Loop through the corpus
    # Note this takes about 3 minutes, uses 6.5GB of RAM and 3.9GB disk space
    print('#' * 40)
    print('Loading the raw corpus')
    fns = sorted([os.path.join(bw_raw_dir, fn) for fn in os.listdir(bw_raw_dir)])
    bw_set = set()
    sent_ctr = 0
    for i, fn in enumerate(fns):
        print('  %d/%d : %s' % (i + 1, len(fns), fn))
        nlines = getFileLines(fn)
        pb = ProgressBar(nlines)
        with io.open(fn, 'r', encoding='utf8') as f:
            for j, line in enumerate(f):
                line = line.strip()
                bw_set.add(line)
                if 0 == j % 100: pb.update(j)
                if test and j > 100000: break
        pb.clear()
        sent_ctr += nlines
        if test and i >= 0: break
    nunique = len(bw_set)
    print('Corpus has {:,} unique sentences out of {:,} read.'.format(nunique, sent_ctr))
    print()

    # Create the output directory
    if not os.path.exists(bw_unique_dir):
        os.mkdir(bw_unique_dir)

    # Split the sentences into shards and save them
    print('Converting to a list and shuffling')
    bw_set = list(bw_set)
    random.shuffle(bw_set)
    sents_per_shard = int(math.ceil(nunique / float(nshards)))
    for i in range(nshards):
        fn = os.path.join(bw_unique_dir, 'bw_%02d.txt' % i)
        print('Saving the data to ', fn)
        with io.open(fn, 'w', encoding='utf8') as f:
            for sent in bw_set[i * sents_per_shard:(i + 1) * sents_per_shard]:
                f.write('%s\n' % sent)
    print('done')
    print()
