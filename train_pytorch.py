"""
author:Sanidhya Mangal
github:sanidhyamangal
"""

import argparse
import random
from collections import Counter

import numpy as np
import torch
import torch.optim as optim
from torch import nn

import utils
from dataloader import get_batches, load_body_as_text
from models import SkipGramNeg
from utils import NegativeSamplingLoss, cosine_similarity


def train_model(path_to_data: str, word_freq: int = 5):

    text = load_body_as_text(path_to_data)

    # get list of words
    words = utils.preprocess(text, word_freq)

    # generate corpus for the words
    vocab_to_int, int_to_vocab = utils.create_lookup_tables(words)
    int_words = [vocab_to_int[word] for word in words]

    # define the threshold to drop the words.
    threshold = 1e-5
    word_counts = Counter(int_words)
    #print(list(word_counts.items())[0])  # dictionary of int_words, how many times they appear

    total_count = len(int_words)
    freqs = {word: count / total_count for word, count in word_counts.items()}
    p_drop = {
        word: 1 - np.sqrt(threshold / freqs[word])
        for word in word_counts
    }
    # discard some frequent words, according to the subsampling equation
    # create a new list of words for training
    train_words = [
        word for word in int_words if random.random() < (1 - p_drop[word])
    ]

    # define the device type for the network to use cuda if available.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Get our noise distribution
    # Using word frequencies calculated earlier in the notebook
    word_freqs = np.array(sorted(freqs.values(), reverse=True))
    unigram_dist = word_freqs / word_freqs.sum()
    noise_dist = torch.from_numpy(unigram_dist**(0.75) /
                                  np.sum(unigram_dist**(0.75)))

    # init the model with 300 embedding_dim
    embedding_dim = 300
    model = SkipGramNeg(len(vocab_to_int),
                        embedding_dim,
                        noise_dist=noise_dist).to(device)

    # define the criterion and optimizer to train the network
    criterion = NegativeSamplingLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    epochs = 5

    # train for some number of epochs
    for e in range(epochs):
        steps = 0
        # get our input, target batches
        for input_words, target_words in get_batches(train_words, 512):
            steps += 1
            inputs, targets = torch.LongTensor(input_words), torch.LongTensor(
                target_words)
            inputs, targets = inputs.to(device), targets.to(device)

            # input, output, and noise vectors
            input_vectors = model.forward_input(inputs)
            output_vectors = model.forward_output(targets)
            noise_vectors = model.forward_noise(inputs.shape[0], 5)

            # negative sampling loss
            loss = criterion(input_vectors, output_vectors, noise_vectors)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss stats and print similarity of the words after every 1000 iters
            if steps % 100 == 0:
                print(f"Epoch: {e+1}, Steps:{steps}, Loss:{loss.item()}")

            if steps % 1000 == 0:
                valid_examples, valid_similarities = cosine_similarity(
                    model.in_embed, device=device)
                _, closest_idxs = valid_similarities.topk(6)

                valid_examples, closest_idxs = valid_examples.to(
                    'cpu'), closest_idxs.to('cpu')
                for ii, valid_idx in enumerate(valid_examples):
                    closest_words = [
                        int_to_vocab[idx.item()] for idx in closest_idxs[ii]
                    ][1:]
                    print(int_to_vocab[valid_idx.item()] + " | " +
                          ', '.join(closest_words))
                print("...\n")


if __name__ == "__main__":
    # argparser for running the script
    parser = argparse.ArgumentParser(
        "parser to run the scripts for training Pytorch word2vec")

    parser.add_argument("--path",
                        dest="path_to_data",
                        help="Path to the data file for loading data")
    parser.add_argument("--n",
                        dest="word_freq",
                        help="number of word freq to drop while training",
                        type=int)

    args = parser.parse_args()
    train_model(args.path_to_data, args.word_freq)
