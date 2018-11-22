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
import logging
import time
import unicodedata
from   unidecode        import unidecode    # 3rd party module
from   collections      import Counter
from   fnmatch          import fnmatch
import multiprocessing  as     mp
from   tflmlib          import DataContainer
from   tflmlib          import ProgressBar
from   tflmlib          import SNLPConnection
from   configs          import config
from   configs          import snlp_server

# Billion Word Corpus
bw_unique_dir = os.path.join(config.bw_corpus, 'BWUniqueSents_FirstPass')
bw_txt_dir    = os.path.join(config.bw_corpus, 'BWTokenized_FirstPass')
bw_pkl_dir    = os.path.join(config.bw_corpus, 'BWParsed_FirstPass')
bw_fn_pat     = 'bw_*'


# Gloabl method for multiprocessing
snlp = SNLPConnection(snlp_server.port)


def getParse(text):
    try:
        global snlp
        text = text.strip()
        # Use a decoder to sanitize unicode strings to ascii characters
        # Note that 3rd party unidecode works bettern than unicodedata.normalize
        text = unidecode(text)                          # returns a <str> type
        # the following doesn't eliminante a lot of stuff (ie.. '/u2018')
        # text = unicodedata.normalize('NFKD', text)
        # text = text.encode('ascii', errors='ignore')   # unicode to string - ignore errors
        # text = text.encode('ascii', errors='strict')   # unicode to string - raise error
        data = snlp.process(text)
        # The Stanford POS tagger converts double quotes to `` and ''
        # (depending on if it's an open or close). This is done because it
        # conforms to PTB standard but I find it annoying so let's get rid of it.
        # Note that there are options in the tokenizer to change this but doing
        # so may impact the accuracy of the parse.
        for i in range(len(data['words'])):
            if data['words'][i] == '``' or data['words'][i] == "''":
                data['words'][i] = '"'
            if data['words'][i] == '`':
                data['words'][i] = "'"
        # Error check
        if len(data['words']) != len(data['pos']) or len(data['words']) != len(data['ner']):
            logging.warn('Inconsistant returned sizes: {}, {}, {}'.format(
                data['words'], data['pos'], data['ner']))
            return None
        return data
    except Exception as e:
        logging.error('SNLP error: {:}'.format(str(e)))
        return None


def processFile(infn, out_txt_fn, out_pkl_fn):
    dc = DataContainer()
    dc.sents = []
    txt_sents = []
    lctr = 0
    wctr = 0
    ectr = 0
    st = time.time()
    with io.open(infn, 'r', encoding='utf-8') as f:     # read in as unicode
        lines = f.readlines()
        if test: lines = lines[:10000]
        st2 = time.time()
        pb = ProgressBar(len(lines))
        pool = mp.Pool()
        for data in pool.imap(getParse, lines):
            if not data:
                ectr += 1
                continue
            wctr += len(data['words'])
            dc.sents.append(data)
            txt_sents.append(' '.join(data['words']))
            if 0 == lctr % 100: pb.update(lctr)
            lctr += 1
        pool.close()    # prevents memory leaks
        pool.join()
    pb.clear()

    # print some stats
    dur = int(time.time() - st + 0.5)
    if test: print('   Parsing time is {:.1f} seconds'.format(time.time() - st2))
    print('   Processed {:,} lines with {:,} words in {:,} seconds.'.format(lctr, wctr,  dur))
    if ectr > 0:
        print('   !!! Note there were {:,} SNLP return processing errors.'.format(ectr))
    # Save the pkl
    dc.save(out_pkl_fn)
    print('   Data saved to ', out_pkl_fn)
    # Save the text file
    # Note that SNLP encodes words as ascii strings so sent is ascii
    with open(out_txt_fn, 'w') as f:
        for sent in txt_sents:
            f.write('%s\n' % sent)
    print('   Data saved to ', out_txt_fn)
    print()


def processDirectory(indir, out_txt_dir, out_pkl_dir):
    # Creat the output directories if they don't exist
    if not os.path.exists(out_txt_dir):
        os.mkdir(out_txt_dir)
    if not os.path.exists(out_pkl_dir):
        os.mkdir(out_pkl_dir)
    print('Gathering the corpus from ', indir)
    fns = sorted([os.path.join(indir, fn) for fn in os.listdir(indir) if fnmatch(fn, bw_fn_pat)])
    if test: fns = fns[:1]
    for i, fn in enumerate(fns):
        print('  %2d/%2d : %s' % (i + 1, len(fns), fn))
        base_fn, _ = os.path.splitext(fn)
        base_fn = os.path.basename(base_fn)
        out_txt_fn = os.path.join(out_txt_dir, base_fn) + '.txt'
        out_pkl_fn = os.path.join(out_pkl_dir, base_fn) + '.pkl'
        processFile(fn, out_txt_fn, out_pkl_fn)
    print()


if __name__ == '__main__':
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)
    logfn = os.path.join(config.log_dir, 'parse.log')
    logging.basicConfig(level=logging.WARN, filename=logfn, filemode='w',
                        format='[%(levelname)s %(filename)s ln=%(lineno)s] %(message)s')
    print('*' * 80)
    print()

    test = True
    # On i7-7940x (14 core SkylakeX) in test mode (first 10,000 lines of one 306,078 line file)
    #   Parsing time is 15.8 seconds without "fine-grained" ner
    #   Parsing time is 78.0 seconds with "fine-grained" ner enabled (which adds 23 extra classes)
    # To turn on/off fine-grained see tflmlib/SNLPConnection.  It is off by default.
    #
    # Full parsing takes about 8.5 minutes per file ~= 14 hours for all 100 files
    # RAM required is about 3GB plus 3GB for running the CoreNLP processor
    # Disk space used is 4GB for BWTokenized + 13GB for BWParsed
    processDirectory(bw_unique_dir, bw_txt_dir, bw_pkl_dir)
