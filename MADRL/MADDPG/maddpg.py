import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from actor import Actor
from critic import Critic
from misc import deal_experience, prob_to_one_hot
from memory import ReplayMemory, Experience


class MADDPG(object):
    def __init__(self, n_agents, n_actions, ob_dim, a_hidd_dims, c_hidd_dims, a_lr, c_lr, tau):
        obs_dim = ob_dim * n_agents
        act_dim = (n_actions + 1) * n_agents
        self.n_agents = n_agents
        self.tau = tau
        self.on_cuda = False
        self.curr_actors = [Actor(ob_dim, n_actions + 1, a_hidd_dims) for _ in range(n_agents)]
        self.curr_critics = [Critic(obs_dim, act_dim, c_hidd_dims) for _ in range(n_agents)]
        self.target_actors = deepcopy(self.curr_actors)
        self.target_critics = deepcopy(self.curr_critics)
        self.actor_optims = [torch.optim.Adam(params=actor.parameters(), lr=a_lr) for actor in self.curr_actors]
        self.critic_optims = [torch.optim.Adam(params=critic.parameters(), lr=c_lr) for critic in self.curr_critics]

    def soft_update(self):
        for i in range(self.n_agents):
            for t_param, c_param in zip(self.target_actors[i].parameters(), self.curr_actors[i].parameters()):
                t_param.data.copy_((1 - self.tau) * t_param.data + self.tau * c_param.data)
            for t_param, c_param in zip(self.target_critics[i].parameters(), self.curr_critics[i].parameters()):
                t_param.data.copy_((1 - self.tau) * t_param.data + self.tau * c_param.data)

    # def update_actor(self, idx, all_obs, all_actions, cuda_available, max_grad_norm):
    def update_actor(self, idx, all_obs, all_actions, max_grad_norm):
        whole_obs = torch.cat(all_obs, dim=1)
        the_ob = all_obs[idx]
        new_action_prob = self.curr_actors[idx](the_ob)
        new_action_oh = prob_to_one_hot(new_action_prob)
        # if cuda_available:
        #     new_action_oh = new_action_oh.cuda()
        all_actions[idx] = new_action_oh
        whole_actions = torch.cat(all_actions, dim=1)
        loss = - self.curr_critics[idx](whole_obs, whole_actions).mean()
        actor_optim = self.actor_optims[idx]
        actor_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.curr_actors[idx].parameters(), max_grad_norm)
        actor_optim.step()

    # def update_critic(self, idx, all_obs, all_actions, all_next_obs, rewards, gamma, cuda_available, max_grad_norm):
    def update_critic(self, idx, all_obs, all_actions, all_next_obs, rewards, gamma, max_grad_norm):
        whole_next_obs = torch.cat(all_next_obs, dim=1)
        new_actions = [target_actor(ob) for target_actor, ob in zip(self.target_actors, all_obs)]
        # print(whole_next_obs)
        ohs = [prob_to_one_hot(act) for act in new_actions]
        whole_new_actions = torch.cat(ohs, dim=1)
        # if cuda_available:
        #     whole_new_actions = whole_new_actions.cuda()
        target_q = self.target_critics[idx](whole_next_obs, whole_new_actions)
        target = rewards + gamma * target_q
        whole_obs = torch.cat(all_obs, dim=1)
        whole_actions = torch.cat(all_actions, dim=1)
        curr_q = self.curr_critics[idx](whole_obs, whole_actions)
        loss = F.mse_loss(target, curr_q)
        critic_optim = self.critic_optims[idx]
        critic_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.curr_critics[idx].parameters(), max_grad_norm)
        critic_optim.step()

    # def update_model(self, rm, batch_size, cuda_available, gamma, max_grad_norm):
    def update_model(self, rm, batch_size, gamma, max_grad_norm):
        for i in range(self.n_agents):
            experience = rm.sample(batch_size)
            batch = Experience(*zip(*experience))
            # all_obs, all_actions, rewards, all_next_obs = deal_experience(batch, cuda_available)
            # self.update_critic(i, all_obs, all_actions, all_next_obs, rewards, gamma, cuda_available, max_grad_norm)
            # self.update_actor(i, all_obs, all_actions, cuda_available, max_grad_norm)
            all_obs, all_actions, rewards, all_next_obs = deal_experience(batch)
            self.update_critic(i, all_obs, all_actions, all_next_obs, rewards, gamma, max_grad_norm)
            self.update_actor(i, all_obs, all_actions, max_grad_norm)
        self.soft_update()

    def move_to_gpu(self):
        if torch.cuda.is_available() and not self.on_cuda:
            for curr_actor, curr_critic, target_actor, target_critic in \
                    zip(self.curr_actors, self.curr_critics, self.target_actors, self.target_critics):
                curr_actor.cuda()
                curr_critic.cuda()
                target_actor.cuda()
                target_critic.cuda()
            self.on_cuda = True

    def move_to_cpu(self):
        if self.on_cuda:
            for curr_actor, curr_critic, target_actor, target_critic in \
                    zip(self.curr_actors, self.curr_critics, self.target_actors, self.target_critics):
                curr_actor.cpu()
                curr_critic.cpu()
                target_actor.cpu()
                target_critic.cpu()
            self.on_cuda = False

    def save(self, filename):
        if self.on_cuda:
            self.move_to_cpu()
        save_dict = {'curr_actor_params': [curr_actor.state_dict() for curr_actor in self.curr_actors],
                     'curr_critic_params': [curr_critic.state_dict() for curr_critic in self.curr_critics],
                     'target_actor_rams': [target_actor.state_dict() for target_actor in self.target_actors],
                     'target_critic_params': [target_critic.state_dict() for target_critic in self.target_critics]}
        torch.save(save_dict, filename)

    def load(self, filename):
        save_dict = torch.load(filename)
        [curr_actor.load_state_dict(params) for curr_actor, params in
         zip(self.curr_actors, save_dict['curr_actor_params'])]
        [curr_critic.load_state_dict(params) for curr_critic, params in
         zip(self.curr_critics, save_dict['curr_critic_params'])]
        [target_actor.load_state_dict(params) for target_actor, params in
         zip(self.target_actors, save_dict['curr_actor_params'])]
        [target_critic.load_state_dict(params) for target_critic, params in
         zip(self.target_critics, save_dict['curr_critic_params'])]

    def set_eval(self):
        for curr_actor, curr_critic, target_actor, target_critic in \
                zip(self.curr_actors, self.curr_critics, self.target_actors, self.target_critics):
            curr_actor.eval()
            curr_critic.eval()
            target_actor.eval()
            target_critic.eval()
