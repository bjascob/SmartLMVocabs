# SmartLMVocabs<br/>
**Improving Language Model Performance through Smart Vocabularies**

This project is designed to explore the use of preprocessing the Billion Word Corpus with Part-Of-Speech labels and Named-Entities in order to create a "smarter" vocabulary.  Using these techniques it's possible to obtain better perplexity scores than using the top-N words in the corpus alone.

For a full explanation of the processing and it's impact on obtainable perplexity scores see [Improving Language Model Performance with Smarter Vocabularies](https://github.com/bjascob/SmartLMVocabs/blob/master/Improving Language Model Performance with Smarter Vocabularies.pdf).

The project includes scripts to
* Extract unique sentences from the Billion Word Corpus
* Create Simple or Smart Vocabularies based on user designated parameters
* Construct, train and test language models using the defined vocabularies


## Installation and Setup
The project is designed to be used in place and there is no pip installation for it.  Simply download the source from GitHub, set configuration options and run the numbered scripts in order.  Note that this code is designed for the researcher so expect to need to review code in the scripts that are being run and modify options as needed to obtain specific results.


## Using the scripts
The main directory's scripts are prefixed with a number to indicate what order to run them in.  Most scripts in the main directory are a relatively short series of calls the associated library functions.  Before running each script be sure to open it up and set any of the configuration parameters to get the results you want.<br/>

** !! Global configuration options used by multiple scripts are set in ./configs/config.py.  Be sure to go into this file and modify the paths to your data prior to running any of the scripts below.**

* **00_StanfordCoreNLPServer.py** : Starts the Stanford Core NLP parser using the configuration file ./configs/snlp_server.py.  The script blocks while the server is running.  If you prefer, the server can be started manually.  This script is for convience only.<br/>

* **10_ExtractUniqueSents.py** : Extracts unique sentences from the Billion Word Corpus prior to parsing.  Near the top of the file are some instructions on how download and setup the raw corpus.<br/>

* **12_STParseBWCorpus.py** : Runs all unique sentences through the Stanford Parser.  Note that step takes a considerable amount of time (possibly a full day) and bennefits greatly from a multi-core processor.  See notes in the file about run-times.<br/>

* **14_RemoveDuplicates.py** : Does a second pass through the data to remove duplicate sentences that occur after parsing/normalization.<br/>

* **20_CreateEnglishDictionary.py** : Required for Smart vocabulary creation.<br/>

* **22_CreateVocab.py** : This script creates the various types of vocabularies.  A large if/elif block in the file exists for the user to choose what vocabulary type to create.<br/>

* **24_IndexVocab.py** : Creates the indexed corpus data used for training the language model.  Again there is a large if/elif block in here that needs to be set by the user to select the vocabulary type to be used.<br/>

* **30_TrainBasicLM.py** : Sets up and runs training of the model.  Again there is a block of statemetns in here allowing the user to choose the vocabulary to train against.  Additionaly the user needs to choose the file with the model configuration parameters.  The model config parameters are stored in the config directory under filenames such as L1_2048_512.py.<br/>

* **32_TestPerplexity** : Runs a perplexity test against the trained model.  Choose the model to test at the bottom of the script.<br/>

* **34_NextWordPrediction.py** : Allows the user to input the first portion of a sentence and calls the model to predict the next word.  Configure which model to use in the main portion of the script.

## Compatibility
* The code is tested to run under python 3 and Linux.
* It was originally setup using python 2 so it's likely only minor changes would need to be made to get it to work under that environment.
* I'm not aware of any limitations to running this under Windows.  If the script to run the Stanford Parser doesn't work, it can simply be run manually.
