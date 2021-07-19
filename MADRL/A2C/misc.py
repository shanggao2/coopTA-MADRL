import torch
import numpy as np


def get_action(action_probs, n_real_w):
    probs = []
    for i in range(n_real_w):
        probs.append(action_probs[0][i])
    probs.append(action_probs[0][-1])
    probs = torch.FloatTensor(probs)
    m = torch.distributions.categorical.Categorical(probs=probs)
    act_idx = m.sample().item()
    if act_idx == n_real_w:
        act_idx = len(action_probs) - 1
    log_action_prob = torch.log(action_probs[0][act_idx])
    return act_idx, log_action_prob


def compute_target(r_list, d_list, next_v, gamma):
    g = next_v
    td_target = []
    for r, done in zip(r_list[::-1], d_list[::-1]):
        g = r + gamma * g * (1 - done)
        td_target.append(g)
    td_target = torch.stack(td_target[::-1]).view(-1, 1)
    return td_target


def deal_trajectory(s_list, lap_list, e_list, r_list, d_list, next_v, gamma):
    t_batch = compute_target(r_list, d_list, next_v, gamma)
    s_batch = torch.stack(s_list).squeeze()
    lap_batch = torch.stack(lap_list).view(-1, 1)
    e_batch = torch.stack(e_list)
    return t_batch, s_batch, lap_batch, e_batch
