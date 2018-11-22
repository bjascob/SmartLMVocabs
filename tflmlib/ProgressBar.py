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
import sys


class ProgressBar(object):
    ''' Progress bar for display

    Args:
        end_val (int): The value at 100%
        bar_len (int): Number of ascii characters in length
    '''
    def __init__(self, end_val, bar_len=20):
        self.end_val = end_val
        self.bar_len = bar_len

    def update(self, val):
        ''' Redraw the progress

        Args:
            val (int): value of the item to be displayed
        '''
        percent = float(val) / self.end_val
        if percent > 1.0:
            percent = 1.0
        hashes = '#' * int(round(percent * self.bar_len))
        spaces = ' ' * (self.bar_len - len(hashes))
        sys.stdout.write('\rPercent: [{0}] {1}%'.format(hashes + spaces,
                         int(round(100 * percent))))
        sys.stdout.flush()

    def clear(self):
        ''' Clear the indicator from the screen'''
        spaces = ' ' * (30 + self.bar_len)
        sys.stdout.write('\r{0}'.format(spaces))
        sys.stdout.write('\r')
