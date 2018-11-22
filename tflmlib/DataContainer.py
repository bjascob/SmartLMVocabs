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
import gzip
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle


class DataContainer(object):
    ''' Create an empty class to misc data for pickling/unpickling

    This class is designed to to save and load generic python objects.  Data may
    be added to a class instance simple by appending an attributes.  The data
    can be save as a standard pickle or gziped pickle.

    Example:
        >>> dc1 = DataContainer()
        >>> dc1.var1 = [1,2,3]
        >>> dc1.save('data.pkl.gz')
        >>> dc2 = DataContainer.load('data.pkl.gz')
        >>> print(dc2.var1)
        [1,2,3]

    Args:
        obj (object): Optional object who's non-private attributes will be copied
    '''
    def __init__(self, obj=None):
        if obj is not None:
            for key, value in vars(obj).items():
                if not key.startswith('_'):
                    setattr(self, key, value)

    def save(self, filename):
        ''' Save the data to a file.

        By convention pickle files have the suffix .pkl.  Additionally, rf the
        user supplies a filename ending in .gz the file will be gzip'd.

        Args:
            filename (str): Full path of file to save
        '''
        with self._open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):
        ''' Load the data from a file

        If the filename ends in .gz the file will be assumed to be gzip'd.

        Args:
            filename (str): Full path of file to load

        Returns:
            object: instance of the DataContainer with data loaded from the file
        '''
        with cls._open(filename, 'rb') as f:
            dc = cls()
            dc.__dict__ = pickle.load(f)
        return dc

    @staticmethod
    def _open(filename, mode):
        if filename.split('.')[-1] == 'gz':
            return gzip.open(filename, mode)
        return open(filename, mode)
