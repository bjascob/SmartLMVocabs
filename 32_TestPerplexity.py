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
from __future__ import print_function
from __future__ import division
import os
import tensorflow as tf
from   tflmlib import AttribContainer
from   tflmlib import InputData
from   tflmlib import LMBasic
from   configs import config


# Calculate perplexity for a given (processed) test data
def calculate_perplexity(config, max_test_words):
    # process the test corpus and load it into batches
    test_data = InputData(config.batch_size, config.seq_length, history_size=config.history_size)
    test_data.loadIndexedCorpus(os.path.join(config.data_dir, 'test'), max_test_words)

    # Get the last checkpoint's filename
    model_fn = LMBasic.get_model_fn(config.model_dir)
    if not model_fn:
        raise Exception("Could not open and/or read model from {}".format(config.model_dir))
    print('Using model ', model_fn)
    print()
    # Setup the model
    with tf.variable_scope("Model", reuse=False):
        model_test = LMBasic(config, False)
    # Restore the parameters
    session = LMBasic.restore_session(model_fn)

    # Print the model's trainable params
    model_test.printTrainableParams()

    # run the test
    test_perplexity = model_test.run_model(session, test_data, eval_op=None,
                                           verbosity=10000, verbose=True)
    print("\n[SUMMARY] Perplexity: %.1f" % test_perplexity)
    print('========================\n')
    session.close()


if __name__ == '__main__':
    print('*' * 80)
    print()

    model_dir = os.path.join(config.data_repo, 'L1_2048_512-SmartA')

    print('Loading model/config from ', model_dir)
    config = AttribContainer.fromJSON(os.path.join(model_dir, 'config.json'))

    max_test_words = int(1e9)   # shard 099.npy contains 7,636,320 words.
    calculate_perplexity(config, max_test_words)
