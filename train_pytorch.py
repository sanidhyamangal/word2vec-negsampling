"""
author:Sanidhya Mangal
github:sanidhyamangal
"""
import numpy as np
import torch  # for torch ops
from torch.optim import Adam

from dataloader import get_batchs, get_target, subsampling_data
from models import SkipGramNegSampling
from utils import NegativeSamplingLoss, cosine_similarity

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = subsampling_data("Dataset/keto_comments_Feb28.csv")

# create negative sampling dist
unigram_dist = dataset["word_freqs"] / dataset["word_freqs"].sum()
noise_dist = torch.from_numpy(unigram_dist**(0.75)/np.sum(unigram_dist**(0.75)))

embedding_size = 300
model = SkipGramNegSampling(len(dataset["vocab2idx"]), embedding_size, noise_dist=noise_dist).to(device)

criterion = NegativeSamplingLoss()
optimizer = Adam(model.parameters(), 0.003)

epochs = 5

for e in range(epochs):
    steps = 0
    for input_words, target_words in get_batchs(dataset["train_words"], batch_size=512, window_size=5):
        steps +=1
        inputs, targets = torch.LongTensor(input_words), torch.LongTensor(target_words)
        inputs, targets = inputs.to(device), targets.to(device)

        # input, output and noise embed
        input_vectors = model.forward_input(inputs)
        output_vectors = model.forward_output(targets)
        noise_vectors = model.forward_noise(inputs.shape[0], 5)

        # neg sampling
        loss = criterion(input_vectors, output_vectors, noise_vectors)

        if steps % 100 == 0:
            print(f"Epoch: {e}, Step: {steps}, Loss: {loss.item()}")
        
        if steps % 1000 == 0:
            valid_examples, valid_similarities = cosine_similarity(model.in_embed, device=device)
            _, closest_idxs = valid_similarities.topk(6)

            valid_examples, closest_idxs = valid_examples.to('cpu'), closest_idxs.to('cpu')
            for ii, valid_idx in enumerate(valid_examples):
                closest_words = [dataset["idx2vocab"][idx.item()] for idx in closest_idxs[ii]][1:]
                print(dataset["idx2vocab"][valid_idx.item()] + " | " + ', '.join(closest_words))
            print("...\n")