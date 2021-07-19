import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidd_dims):
        super(Critic, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(obs_dim + act_dim, hidd_dims[0])])
        self.linears.extend([nn.Linear(hidd_dims[i], hidd_dims[i + 1]) for i in range(len(hidd_dims) - 1)])
        self.linears.append(nn.Linear(hidd_dims[-1], 1))
        self.active = nn.LeakyReLU(0.05)

    def forward(self, obs, act):
        x = torch.cat((obs, act), dim=1)
        for i in range(len(self.linears) - 1):
            x = self.active(self.linears[i](x))
        x = self.linears[-1](x)
        return x
