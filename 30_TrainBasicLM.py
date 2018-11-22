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
import importlib
from   tflmlib  import AttribContainer
from   tflmlib  import LMBasicTrainer
from   tflmlib  import LMBasic
from   configs  import config


# Simple helper to load the AttributeContainer config and add
# vocabulary input and model output paths to it
def setupConfig(config_name, data_dir, model_dir):
    model_config = importlib.import_module(config_name)
    model_config.data_dir  = os.path.join(config.data_repo, data_dir)
    model_config.model_dir = os.path.join(config.data_repo, model_dir)
    return model_config


# Loading the full dataset takes about 7GB
if __name__ == '__main__':
    print('*' * 80)
    print()

    # Pick the vocabulary type and setup the directories to use
    # These directories will be saved with the model and used in later scripts
    if 0:
        model_config = setupConfig('configs.L1_512_512', 'SimpleVocab', 'L1_512_512-Simple')
    elif 0:
        model_config = setupConfig('configs.L1_512_512', 'SmartVocabA', 'L1_512_512-SmartA')
    elif 0:
        model_config = setupConfig('configs.L1_2048_512', 'SimpleVocab', 'L1_2048_512-Simple')
    elif 1:
        model_config = setupConfig('configs.L1_2048_512', 'SmartVocabA', 'L1_2048_512-SmartA')
    elif 0:
        model_config = setupConfig('configs.L1_2048_512', 'SmartVocabB', 'L1_2048_512-SmartB')
    elif 0:
        model_config = setupConfig('configs.L1_2048_512', 'SmartVocabC', 'L1_2048_512-SmartC')
    elif 0:
        model_config = setupConfig('configs.L1_2048_512', 'SmartVocabD', 'L1_2048_512-SmartD')

    # Run Training
    print('Training model based on ', model_config.__name__)
    config = AttribContainer(model_config)
    LMBasicTrainer(config)
