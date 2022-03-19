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

from preprocessing import create_lookup_tables, preprocess


def load_body_as_text(path_to_csv):
    lines =  pd.read_csv(path_to_csv)['body'].apply(str.strip).values
    text = " ".join(i.strip() for i in lines)

    return text

def subsampling_data(path_to_csv, threshold=1e-5):
    text = load_body_as_text(path_to_csv)

    words = preprocess(text)
    
    # generate lookup table
    vocab2idx,idx2vocab = create_lookup_tables(words)
    word_idx = [vocab2idx[i] for i in words]
    word_count = Counter(word_idx)

    # perform the subsampling of the data
    total_count = len(word_idx)

    freqs = {word: count/total_count for word, count in word_count.items()}
    p_drop = {word: 1 - np.sqrt(threshold/freqs[word]) for word in word_count}

    # drop few words based on the random probability to create a better ops
    train_words = [word for word in word_idx if random.random() < (1 - p_drop[word])]

    return {
        "words":words,
        "train_words":train_words,
        "word_idx":word_idx,
        "vocab2idx":vocab2idx,
        "idx2vocab":idx2vocab,
        "word_freqs":np.array(sorted(freqs.values(), reverse=True))
    }

def get_target(words:List[int], idx:int, window_size:int=5):
    R = random.randint(1,window_size+1)
    start = idx-R if idx-R > 0 else 0
    stop = idx + R
    target_words = words[start:idx] + words[idx+1:stop]

    return list(target_words)

def get_batchs(words, batch_size, window_size:int=5):
    n = len(words) // batch_size

    # we need to mask to get only the full batches
    words = words[:n*batch_size]

    for idx in range(0, len(words), batch_size):
        x,y = [], []

        batch = words[idx:idx+batch_size]

        # create a btach         
        for idx in range(len(batch)):
            batch_x = batch[idx]
            batch_y = get_target(batch, idx, window_size)
            y.extend(batch_y)
            x.extend([batch_x]*len(batch_y))
        
        # create a generator for the ops
        yield x, y


def gensim_dataloader(path):
    lines =  pd.read_csv(path)['body'].apply(str.strip).values

    return [simple_preprocess(i) for i in lines]
