import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, in_dim, hidd_dim):
        super(Critic, self).__init__()
        self.fc_1 = nn.Linear(in_dim, hidd_dim)
        self.fc_2 = nn.Linear(hidd_dim, 1)
        self.active = nn.LeakyReLU(0.05)

    def forward(self, x):
        out = self.fc_1(x)
        out = self.active(out)
        out = self.fc_2(out)
        return out
