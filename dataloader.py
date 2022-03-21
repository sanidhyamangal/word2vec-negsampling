"""
author:Sanidhya Mangal
github:sanidhyamangal
"""

import random  # for random choices
from collections import Counter
from typing import List

import numpy as np
import pandas as pd  # for pd
from gensim.utils import simple_preprocess


def load_body_as_text(path_to_csv:str) -> str:
    """Function to load the dataframe and convert the body into the continous stream of text

    Args:
        path_to_csv (str): path to the data file containing csv

    Returns:
        str: continous string of words which needs to be processed
    """
    # read the lines from the csv file
    lines = pd.read_csv(path_to_csv)['body'].apply(str.strip).values
    # create a text stream based on the lines.
    text = " ".join(i.strip() for i in lines)

    return text


def get_target(words, idx, window_size=5):
    ''' Get a list of words in a window around an index. '''

    # randomly sample the window to get the target words
    R = np.random.randint(1, window_size + 1)
    start = idx - R if (idx - R) > 0 else 0
    stop = idx + R
    target_words = words[start:idx] + words[idx + 1:stop + 1]

    return list(target_words)


def get_batches(words, batch_size, window_size=20):
    ''' Create a generator of word batches as a tuple (inputs, targets) '''

    n_batches = len(words) // batch_size

    # only full batches
    words = words[:n_batches * batch_size]

    for idx in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[idx:idx + batch_size]
        for ii in range(len(batch)):
            batch_x = batch[ii]
            batch_y = get_target(batch, ii, window_size)
            y.extend(batch_y)
            x.extend([batch_x] * len(batch_y))
        yield x, y


def gensim_dataloader(path:str)-> List[List[str]]:
    """function to load the data from csv file and process it for gensim word2vec model

    Args:
        path (str): path to the csv file for data loading

    Returns:
        List[List[str]]: returns the two array of preprocessed lines
    """
    lines = pd.read_csv(path)['body'].apply(str.strip).values

    return [simple_preprocess(i) for i in lines]
