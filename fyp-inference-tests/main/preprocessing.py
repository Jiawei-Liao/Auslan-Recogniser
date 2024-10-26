import torch
import torch.nn as nn


class Preprocessing(nn.Module):
    def __init__(self):
        super(Preprocessing, self).__init__()

    def normalize(self, x):
        nonan = x[~torch.isnan(x)].view(-1, x.shape[-1])
        x = x - nonan.mean(0)[None, None, :]
        x = x / nonan.std(0, unbiased=False)[None, None, :]
        return x

    def fill_nans(self, x):
        x[torch.isnan(x)] = 0
        return x

    def forward(self, x):
        # seq_len, 3* n_landmarks -> seq_len, n_landmarks, 3
        x = x.reshape(x.shape[0], 3, -1).permute(0, 2, 1)

        # Normalize & fill nans
        x = self.normalize(x)
        x = self.fill_nans(x)

        return x
