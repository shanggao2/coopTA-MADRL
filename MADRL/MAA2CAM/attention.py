import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math


# Scaled Dot-Product Attention
class Attention(nn.Module):
    # ob_dim: the observational dimensions of agent i
    def __init__(self, ob_dim):
        super(Attention, self).__init__()
        self.query_extractor = nn.Linear(ob_dim, ob_dim, bias=False)
        self.key_extractor = nn.Linear(ob_dim, ob_dim, bias=False)
        self.value_extractor = nn.Linear(ob_dim, ob_dim, bias=False)

    # all_embd_obs: a list, in which each element e_i is the embedding result of the observation of agent i
    # agent_idx: the index of the agent
    def forward(self, all_embd_obs, agent_idx):
        query = self.query_extractor(all_embd_obs[agent_idx])
        d_k = query.size(-1)
        keys = [self.key_extractor(Variable(ob)) for i, ob in enumerate(all_embd_obs) if i != agent_idx]
        values = [self.value_extractor(Variable(ob)) for i, ob in enumerate(all_embd_obs) if i != agent_idx]

        scores = [(query * key).sum(dim=1) / math.sqrt(d_k) for i, key in enumerate(keys)]
        scores = torch.stack(scores)
        weights = F.softmax(scores, dim=0)
        values_t = torch.cat(values, dim=1)

        context_value = []
        for i in range(weights.size(1)):
            weight = weights[:, i].view(-1, 1)
            value = values_t[i].view(len(values), -1)
            cv = (value * weight).sum(dim=0)
            context_value.append(cv)
        context_value = torch.stack(context_value)
        return context_value
