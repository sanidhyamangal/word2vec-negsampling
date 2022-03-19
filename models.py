"""
author:Sanidhya Mangal
github:sanidhyamangal
"""

import torch  # for deep learning
import torch.nn as nn  # for nn


class SkipGramNegSampling(nn.Module):

    def __init__(self, vocab_size, embed_size, noise_dist=None) -> None:
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.noise_dist = noise_dist

        self.in_embed = nn.Embedding(vocab_size, embed_size)
        self.out_embed = nn.Embedding(vocab_size, embed_size)

        self.in_embed.weight.data.uniform_(-1, 1)
        self.out_embed.weight.data.uniform_(-1, 1)

    def forward_input(self, input_words):
        input_vectors = self.in_embed(input_words)

        return input_vectors

    def forward_output(self, output_words):
        output_vectors = self.in_embed(output_words)

        return output_vectors

    def forward_noise(self, batch_size, n_samples):

        # use noise dist
        noise_dist = torch.ones(
            self.vocab_size) if self.noise_dist is None else self.noise_dist

        # sample the noise words
        noise_words = torch.multinomial(noise_dist,
                                        batch_size * n_samples,
                                        replacement=True)

        device = "cuda" if self.out_embed.weight.is_cuda else "cpu"

        noise_words.to(device)

        noise_vectors = self.out_embed(noise_words).view(
            batch_size, n_samples, self.embed_size)

        return noise_vectors

