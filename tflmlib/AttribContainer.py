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
import json
from   types import ModuleType


class AttribContainer(object):
    ''' Create an empty class to store configuration key, values paris as attributes.

    Args:
        obj (object): Optional object who's non-private attributes will be copied
    '''
    def __init__(self, obj=None):
        if obj is not None:
            for key, value in vars(obj).items():
                if not key.startswith('_') and not isinstance(value, ModuleType):
                    setattr(self, key, value)

    # Read in attributes from a JSON file
    @classmethod
    def fromJSON(cls, filename):
        ''' Load attributes from a JSON file

        Args:
            filename (str) : Name of .json file to load

        Returns:
            object: An instance of this class with loaded attributes
        '''
        obj = cls()
        with open(filename, 'r') as f:
            data = json.load(f)
        for key, value in data.items():
            setattr(obj, key, value)
        return obj

    # Save attributes to a JSON file
    def saveJSON(self, filename):
        ''' Save attributes to a JSON file

        Args:
            filename (str): Full path ame of .json file to save attributes to
        '''
        data = dict()
        for key, value in self.__dict__.items():
            data[key] = value
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, sort_keys=True)
