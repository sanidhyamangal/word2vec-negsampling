"""
author:Sanidhya Mangal
github:sanidhyamangal
"""
import random

import numpy as np
import torch
import torch.nn as nn # for nn ops


# create a cosine similarity for the model ops
def cosine_similarity(embedding, valid_size=16, valid_window=100, device="cpu"):

    embed_vec = embedding.weight

    # gen the scalar quantity
    magnitudes = embed_vec.pow(2).sum(dim=1).sqrt().unsqueeze(0)

    # pick N words from our ranges (0,window) and (1000,1000+window). lower id implies more frequent 
    valid_examples = np.array(random.sample(range(valid_window), valid_size//2))
    valid_examples = np.append(valid_examples,
                               random.sample(range(1000,1000+valid_window), valid_size//2))
    valid_examples = torch.LongTensor(valid_examples).to(device)
    
    valid_vectors = embedding(valid_examples)
    similarities = torch.mm(valid_vectors, embed_vec.t())/magnitudes
        
    return valid_examples, similarities

class NegativeSamplingLoss(nn.Module):

    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input_vector, output_vector, noise_vector):

        batch_size, embed_size = input_vector.shape

        # reshape the vector shape
        input_vector = input_vector.view(batch_size, embed_size, 1)
        output_vector = output_vector.view(batch_size, 1, embed_size)

        out_loss = torch.bmm(output_vector, input_vector).sigmoid().log()
        out_loss = out_loss.squeeze()
        
        # incorrect log-sigmoid loss
        noise_loss = torch.bmm(noise_vector.neg(), input_vector).sigmoid().log()
        noise_loss = noise_loss.squeeze().sum(1)  # sum the losses over the sample of noise vectors

        # negate and sum correct and noisy log-sigmoid losses
        # return average batch loss
        return -(out_loss + noise_loss).mean()