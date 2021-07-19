import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from actor import Actor
from critic import Critic


class A2C(object):
    def __init__(self, n_actions, s_dim, a_hidd_dims, c_hidd_dims, a_lr, c_lr):
        self.actor = Actor(s_dim, n_actions, a_hidd_dims)
        self.critic = Critic(s_dim, c_hidd_dims)
        self.actor_optim = torch.optim.ASGD(params=self.actor.parameters(), lr=a_lr)
        self.critic_optim = torch.optim.ASGD(params=self.critic.parameters(), lr=c_lr)
        self.on_cuda = False

    def move_to_gpu(self):
        if torch.cuda.is_available() and not self.on_cuda:
            self.actor.cuda()
            self.critic.cuda()
            self.on_cuda = True

    def move_to_cpu(self):
        if self.on_cuda:
            self.actor.cpu()
            self.critic.cpu()
            self.on_cuda = False

    def update_model(self, t_batch, s_batch, lap_batch, e_batch, entropy_beta, max_grad_norm):
        v_batch = self.critic(s_batch)
        advantage = t_batch - v_batch
        # actor
        a_loss = (-lap_batch * advantage.detach()).mean() - entropy_beta * e_batch.mean()
        self.actor_optim.zero_grad()
        a_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm)
        self.actor_optim.step()
        # critic
        c_loss = F.mse_loss(t_batch, v_batch)
        self.critic_optim.zero_grad()
        c_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
        self.critic_optim.step()

    def save(self, filename):
        if self.on_cuda:
            self.move_to_cpu()
        save_dict = {'actor_params': self.actor.state_dict(),
                     'critic_params': self.critic.state_dict()}
        torch.save(save_dict, filename)

    def load(self, filename):
        save_dict = torch.load(filename)
        self.actor.load_state_dict(save_dict['actor_params'])
        self.critic.load_state_dict(save_dict['critic_params'])

    def set_eval(self):
        self.actor.eval()
        self.critic.eval()
