# CS5891-NLP Assighnemt 2
This repo contains all the code for CS5891 NLP HW2 and running instructions which will help TA and instructor to execute the code

### Helper modules
* _utils.py_: Contains all the helper functions
* _models.py_: Contains Model for the Pytorch Word2Vec
* _dataloader.py_: Helper script to contain dataloader util functions for both Pytorch and Gensim
* _train_pytorch.py_: Trainning script to train pytorch based model for word2vec
* _gensim_model.py_: Execution entry point for the gensim model


## How to execute the code
You can use basic commands to run the scripts to train word2vec in PyTorch.

If you wish to execute the script the file on level you can use:
```bash
python3 train_pytorch.py [-h] [--path PATH_TO_DATA] [--n WORD_FREQ]
```
Usage:
```bash
usage: parser to run the scripts for training Pytorch word2vec [-h] [--path PATH_TO_DATA] [--n WORD_FREQ]

optional arguments:
  -h, --help           show this help message and exit
  --path PATH_TO_DATA  Path to the data file for loading data
  --n WORD_FREQ        number of word freq to drop while training
```

### Gensim Model (train)
You can use basic commands to run the scripts  to train the models for the gensim.

```bash
python3 gensim_model.py train [-h] [--path PATH] [--path_to_saved_model PATH_TO_SAVED_MODEL]
                                                                         [--N MIN_COUNT]
```
Usage:
```bash
usage: Script to train and predict the word2vec model using gensim train [-h] [--path PATH] [--path_to_saved_model PATH_TO_SAVED_MODEL]
                                                                         [--N MIN_COUNT]

parser to train the model

optional arguments:
  -h, --help            show this help message and exit
  --path PATH           path to data file
  --path_to_saved_model PATH_TO_SAVED_MODEL
                        path to save model file
  --N MIN_COUNT         min_count size for the trainer,default 5
```

### Gensim Model (predict)
You can use basic bash commands to run the scripts to predict some random default words for the gensim model.

If you wish to execute the script the file on level you can use:
```bash
python3 gensim_model.py predict [-h] [--path PATH] [--word WORD] [--top TOP]
```
Usage:
```bash
usage: Script to train and predict the word2vec model using gensim predict [-h] [--path PATH] [--word WORD] [--top TOP]

parser to predict embedding from the trained model

optional arguments:
  -h, --help   show this help message and exit
  --path PATH  path to saved model file
  --word WORD  word who\'s similarity needs to be computed
  --top TOP    topn word similarities to load, default 10
```