import math
import random
import torch


def get_actions(all_action_probs, masks, n_real_w, n_t, n_w):
    actions = []
    log_action_probs = []
    # for real workers
    for i in range(n_real_w):
        temp_probs = []
        temp_idxs = []
        for idx, m in enumerate(masks[i]):
            if m == 1.0:
                temp_probs.append(all_action_probs[i][0][idx])
                temp_idxs.append(idx)
        temp_probs.append(all_action_probs[i][0][-1])
        temp_idxs.append(n_t)
        t_probs = torch.FloatTensor(temp_probs)
        m = torch.distributions.categorical.Categorical(probs=t_probs)
        # sample_pool = [m.sample(), m.sample(), m.sample()]
        # act_idx = max(sample_pool, key=sample_pool.count)
        act_idx = m.sample()
        action = temp_idxs[act_idx]
        action_prob = temp_probs[act_idx]
        log_action_prob = torch.log(action_prob)
        actions.append(action)
        log_action_probs.append(log_action_prob)
    # for virtual workers
    for i in range(n_real_w, n_w):
        action = n_t
        action_prob = all_action_probs[i][0][-1]
        log_action_prob = torch.log(action_prob)
        actions.append(action)
        log_action_probs.append(log_action_prob)
    # return
    return actions, log_action_probs


def transpose(matrix):
    new_matrix = []
    for i in range(len(matrix[0])):
        matrix_t = []
        for j in range(len(matrix)):
            matrix_t.append(matrix[j][i])
        new_matrix.append(matrix_t)
    return new_matrix


def compute_target(r_list, done_list, next_value, gamma):
    g = next_value
    td_target = []
    for r, done in zip(r_list[::-1], done_list[::-1]):
        g = r + gamma * g * (1 - done)
        td_target.append(g)
    return torch.stack(td_target[::-1]).view(-1, 1)


def deal_trajectory(raw_obs_list, state_list, reward_list, entorpy_list, done_list,
                    ltap_list, next_values, gamma):
    # transpose list
    raw_obs_list_t = transpose(raw_obs_list)
    state_list_t = transpose(state_list)
    entropy_list_t = transpose(entorpy_list)
    ltap_list_t = transpose(ltap_list)
    # get n-step reward for all agents
    target_list = [compute_target(reward_list, done_list, next_value, gamma) for next_value in next_values]
    # get mean entropy for all agents
    mean_entropy_list = [sum(entropy) / len(entropy) for entropy in entropy_list_t]

    # transform inner list to tensor
    state_list_t = [torch.cat(state, dim=0) for state in state_list_t]
    raw_obs_list_t = [torch.cat(raw_obs, dim=0) for raw_obs in raw_obs_list_t]
    ltap_list_t = [torch.stack(ltap).view(-1, 1) for ltap in ltap_list_t]

    return raw_obs_list_t, state_list_t, target_list, mean_entropy_list, ltap_list_t


def zero_model_grad(model):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data.zero_()
