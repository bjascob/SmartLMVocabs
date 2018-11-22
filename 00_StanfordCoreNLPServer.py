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
import  os
import  sys
import  time
import  json
import  signal
import  psutil  # 3rd party lib
from    configs import snlp_server
from    configs import config

# Note stanford-corenlp-full-2018-02-27 (and 2017) does not work with openjdk,
# these only work with Oracle java8 or later.
# stanford-corenlp-full-2018-10-05 works with Java 11, but 2018-02-27 doesn't.


# Terminate the java process
def signal_handler(signal, frame):
    global gProc
    if gProc:
        gProc.terminate()
        gone, still_alive = psutil.wait_procs([gProc], timeout=1.0)
        if still_alive:
            gProc.kill()
        global gRun
        gRun = False
        global gLogfile
        gLogfile.close()


# Note running server takes about 3GB of RAM
if __name__ == '__main__':
    # Catch sigint
    signal.signal(signal.SIGINT, signal_handler)

    # Open the logfile
    gLogfile = sys.stdout   # default
    try:
        if not os.path.exists(config.log_dir):
            os.mkdir(config.log_dir)
        gLogfile = open(snlp_server.log_fn, 'w')
    except IOError as e:
        print(e)
        print('Unable to open logfile. Does the directory exist?')
        print('Logging to stdout')

    # Start the server
    # See https://github.com/stanfordnlp/CoreNLP/blob/master/src/edu/stanford/
    #     nlp/pipeline/StanfordCoreNLPServer.java
    working_dir = '.'
    cmd = 'java -mx4g -cp %s/* edu.stanford.nlp.pipeline.' \
          'StanfordCoreNLPServer ' % (snlp_server.core_nlp)
    cmd += '--port %d --preload tokenize,ssplit,pos,lemma,ner,parse ' % (snlp_server.port)
    cmd += '--ner.applyFineGrained 0 '
    cmd += '--ner.buildEntityMentions 0 '
    cmd += '--quiet '    # prevents printing text.  Works for version 3.6.1 onward
    cmd = cmd.split()
    if 0:  # Redirect stderr to the log file
        gProc = psutil.Popen(cmd, cwd=working_dir, stdout=gLogfile, stderr=gLogfile)
    else:  # Redirect stderr to /dev/null
        devnull = open(os.devnull, 'w')
        gProc = psutil.Popen(cmd, cwd=working_dir, stdout=gLogfile, stderr=devnull)
    gLogfile.write('java process started with pid = %d\n\n' % (gProc.pid))
    gLogfile.flush()
    print('Started server. pid=', gProc.pid)

    # Run until signal handler sets this to false
    gRun = True
    while gRun:
        time.sleep(0.1)
