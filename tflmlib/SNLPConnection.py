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
import  json
import  requests
import  logging
import  nltk


# Notes*
#   finegrained NER slows down parsing by about 5X but adds 23 news classes
#   default is on.  To disable set ner.applyFineGrained and ner.buildEntityMentions false
#   See https://stanfordnlp.github.io/CoreNLP/ner.html
class SNLPConnection(object):
    ''' Stanford Core NLP Server Connection class

    This class is ued to communicate with a locally running Stanford Parser

    Args:
        port (int): internal port number used for communication
    '''
    def __init__(self, port):
        # Used by "requests" and prints lots of stuff that isn't useful
        logging.getLogger('urllib3').setLevel(logging.ERROR)
        self.server_url = 'http://localhost:%d' % (port)
        self.reqdict = {'annotators': 'tokenize, ssplit, pos, lemma, ner',
                        'ner.applyFineGrained': 'false',
                        'ner.buildEntityMentions': 'false',
                        'outputFormat': 'json'}
        self._checkConnection()

    def process(self, text):
        ''' Call the parser with the given text

        Args:
            text (str): Text string to parse

        Returns:
            dictionary: Parse data containing 'words', 'pos' and 'ner'
        '''
        assert isinstance(text, str)
        snlp_ret  = self._annotate(text)
        if not snlp_ret or not snlp_ret.get('sentences', None):
            logging.warning('Empty return from server. text=%s' % (text))
            return
        data = {}
        # data.text = text
        data['words'] = []
        data['pos']   = []
        data['ner']   = []
        for parsed in snlp_ret['sentences']:
            tokens = parsed['tokens']
            # originally in python2 the return was converted to str
            # data['words'].extend( [x['word'].encode('ascii', 'ignore') for x in tokens] )
            # data['pos'].extend( [str(x['pos']) for x in tokens] )
            # data['ner'].extend( [str(x['ner']) for x in tokens] )
            data['words'].extend([x['word'] for x in tokens])
            data['pos'].extend([x['pos'] for x in tokens])
            data['ner'].extend([x['ner'] for x in tokens])
        # If the below is added, will need to return a list of data since it can't correctly
        # concatenate stp trees like the list of words, pos, etc.. above.
        # data.cparse  = parsed['parse']                  # constituency parse (stp format)
        # data.tree    = nltk.Tree.fromstring(data.cparse)   #significantly increases size
        return data

    # Perform a GET on the server which will get the main interface html page
    def _checkConnection(self):
        try:
            r = requests.get(self.server_url)
        except requests.exceptions.ConnectionError:
            raise Exception('Check whether you have started the CoreNLP server')

    # POST to the server to get the annotated data
    # Note that text needs to be a clean ascii string (not unicode)
    def _annotate(self, text):
        r = requests.post(self.server_url, params={'properties': str(self.reqdict)},
                          data=text.encode(), headers={'Connection': 'close'})
        r.raise_for_status()    # raise an excepection for a bad return
        output = json.loads(r.text, encoding='utf-8', strict=True)
        return output
