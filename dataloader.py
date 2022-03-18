"""
author:Sanidhya Mangal
github:sanidhyamangal
"""

from collections import Counter
import torch
from torch.utils.data import Dataset
import pandas as pd # for pd
from gensim.utils import simple_preprocess
from nltk.tokenize import word_tokenize


class Wor2VecDataLoader(Dataset):

    def __init__(self, path_to_csv, window_size) -> None:
        super().__init__()

        self.window_size = window_size
        self.data = pd.read_csv(path_to_csv)['body'].apply(str.strip).values
        self.text_list = []
        self.create_lookup_tables()
    

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index.tolist()
    

    def create_lookup_tables(self):
        for i in self.data[:100]:
            self.text_list.extend(word_tokenize(i))

        word_counts = Counter(self.text_list)
        sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
        self.word2idx = {word:idx for idx, word in enumerate(sorted_vocab)}
        self.idx2word = {idx:word for idx, word in enumerate(sorted_vocab)}


def gensim_dataloader(path):
    lines =  pd.read_csv(path)['body'].apply(str.strip).values

    return [simple_preprocess(i) for i in lines]