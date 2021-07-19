import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Embedding, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.active = nn.LeakyReLU(0.05)

    def forward(self, x):
        out = self.fc(x)
        out = self.active(out)
        return out
