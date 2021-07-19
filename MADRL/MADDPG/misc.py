import math
import random
import torch


def get_explore_prob(step, base_num):
    return math.pow(0.25, step / base_num)


def get_actions_test(all_action_probs, masks, n_real_w, n_t, n_w):
    actions = []
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
        ac_idx = temp_probs.index(max(temp_probs))
        action = temp_idxs[ac_idx]
        actions.append(action)
    # for virtual workers
    for i in range(n_real_w, n_w):
        actions.append(n_t)
    return actions


def get_actions_train(all_action_probs, masks, n_real_w, n_t, n_w, step, exp_it_thr, base_num):
    actions = []
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
        if step < exp_it_thr:
            exp_prob = get_explore_prob(step, base_num)
            if random.random() < exp_prob:
                ac_idx = random.randint(0, len(temp_idxs) - 1)
            else:
                ac_idx = temp_probs.index(max(temp_probs))
        else:
            ac_idx = temp_probs.index(max(temp_probs))
        action = temp_idxs[ac_idx]
        actions.append(action)
    # for virtual workers
    for i in range(n_real_w, n_w):
        actions.append(n_t)
    # to one hot
    actions_one_hot = []
    for i in range(len(actions)):
        act_one_hot = torch.zeros(1, all_action_probs[0].size(1))
        act_one_hot[0][actions[i]] = 1.0
        actions_one_hot.append(act_one_hot)
    return actions, actions_one_hot


def transpose(matrix):
    new_matrix = []
    for i in range(len(matrix[0])):
        matrix_t = []
        for j in range(len(matrix)):
            matrix_t.append(matrix[j][i])
        new_matrix.append(matrix_t)
    return new_matrix


def compute_target(r_list, done_list, next_done, next_value, gamma):
    if not next_done:
        g = next_value[0][0]
    else:
        g = 0.0
    td_target = []
    for r, done in zip(r_list[::-1], done_list[::-1]):
        g = r + gamma * g * (1 - done)
        td_target.append(g)
    return torch.tensor(td_target[::-1]).float()

# def deal_experience(batch, cuda_available):
def deal_experience(batch):
    all_obs = list(batch.obs)
    all_obs = transpose(all_obs)
    all_obs = [torch.cat(obs, dim=0) for obs in all_obs]
    all_actions = list(batch.actions)
    all_actions = transpose(all_actions)
    all_actions = [torch.cat(act, dim=0) for act in all_actions]
    all_next_obs = list(batch.next_obs)
    all_next_obs = transpose(all_next_obs)
    all_next_obs = [torch.cat(next_obs, dim=0) for next_obs in all_next_obs]
    rewards = batch.rewards
    rewards = torch.FloatTensor(rewards).view(-1, 1)
    # if cuda_available:
    #     rewards = torch.cuda.FloatTensor(rewards).view(-1, 1)
    #     all_actions = [act.cuda() for act in all_actions]
    # else:
    #     rewards = torch.FloatTensor(rewards).view(-1, 1)
    return all_obs, all_actions, rewards, all_next_obs


def prob_to_one_hot(probs):
    idxs = torch.argmax(probs, dim=1)
    ohs = torch.zeros(probs.size())
    for i in range(ohs.size(0)):
        ohs[i][idxs[i]] = 1.0
    return ohs
