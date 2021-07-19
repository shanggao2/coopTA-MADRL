import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain
from embedding import Embedding
from attention import Attention
from actor import Actor
from critic import Critic
from misc import zero_model_grad


class MAA2CAM(object):
    def __init__(self, num_agents, num_actions, ob_dim, embd_dim, hidd_dim, a_lr, c_lr):
        self.on_cuda = False
        self.embeddings = [Embedding(ob_dim, embd_dim) for _ in range(num_agents)]
        self.attentions = [Attention(embd_dim) for _ in range(num_agents)]
        self.actors = [Actor(embd_dim * 2, hidd_dim, num_actions + 1) for _ in range(num_agents)]
        self.critics = [Critic(embd_dim * 2, hidd_dim) for _ in range(num_agents)]
        self.actor_params = [chain(embedding.parameters(), attention.parameters(), actor.parameters())
                             for embedding, attention, actor
                             in zip(self.embeddings, self.attentions, self.actors)]
        self.critic_params = [chain(embedding.parameters(), attention.parameters(), critic.parameters())
                              for embedding, attention, critic
                              in zip(self.embeddings, self.attentions, self.critics)]
        self.actor_optims = [torch.optim.ASGD(params=params, lr=a_lr) for params in self.actor_params]
        self.critic_optims = [torch.optim.ASGD(params=params, lr=c_lr) for params in self.critic_params]

    def pre_process(self, all_obs):
        all_embd_obs = [embd(ob) for embd, ob in zip(self.embeddings, all_obs)]
        all_c = [attention(all_embd_obs, idx) for idx, attention in enumerate(self.attentions)]
        return all_embd_obs, all_c

    def get_action_probs(self, all_obs):
        return [actor(hidd_input) for hidd_input, actor in zip(all_obs, self.actors)]

    def get_values(self, all_obs):
        return [critic(hidd_input) for hidd_input, critic in zip(all_obs, self.critics)]

    def update_actor(self, state, target, entropy, ltap, idx, entropy_beta, max_grad_norm):
        critic = self.critics[idx]
        v = critic(state)
        advantage = target - v
        loss = (-ltap * advantage.detach()).mean() - entropy * entropy_beta
        # loss = (-ltap * advantage.detach()).mean()
        # loss = -loss
        # loss = (-ltap * advantage.detach()).mean()
        actor_optim = self.actor_optims[idx]
        actor_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_params[idx], max_grad_norm)
        actor_optim.step()
        self.zero_grad()

    def update_critic(self, value, target, idx, max_grad_norm):
        loss = F.mse_loss(target, value)
        critic_optim = self.critic_optims[idx]
        critic_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_params[idx], max_grad_norm)
        critic_optim.step()
        self.zero_grad()

    def update_model(self, raw_obs_list, state_list, target_list, entropy_list, ltap_list, entropy_beta, max_grad_norm):
        # re-create compute graph for critics
        all_embd_obs, all_c = self.pre_process(raw_obs_list)
        all_obs = [torch.cat((ob, c), dim=1) for ob, c in zip(all_embd_obs, all_c)]
        value_list = self.get_values(all_obs)

        # [self.update_critic(value, target, idx, max_grad_norm) for value, target, idx in
        #  zip(value_list, target_list, range(len(self.critics)))]
        #
        # [self.update_actor(state, target, entropy, ltap, idx, entropy_beta, max_grad_norm) for
        #  state, target, entropy, ltap, idx in
        #  zip(state_list, target_list, entropy_list, ltap_list, range(len(self.actors)))]

        for idx in range(len(self.actors) - 1, -1, -1):
            self.update_actor(state_list[idx], target_list[idx], entropy_list[idx],
                              ltap_list[idx], idx, entropy_beta, max_grad_norm)
            self.update_critic(value_list[idx], target_list[idx], idx, max_grad_norm)

    def move_to_gpu(self):
        if torch.cuda.is_available() and not self.on_cuda:
            for embedding, attention, actor, critic in zip(self.embeddings, self.attentions,
                                                           self.actors, self.critics):
                embedding.cuda()
                attention.cuda()
                actor.cuda()
                critic.cuda()
            self.on_cuda = True

    def move_to_cpu(self):
        if self.on_cuda:
            for embedding, attention, actor, critic in zip(self.embeddings, self.attentions, self.actors, self.critics):
                embedding.cpu()
                attention.cpu()
                actor.cpu()
                critic.cpu()
            self.on_cuda = False

    def save(self, filename):
        if self.on_cuda:
            self.move_to_cpu()
        save_dict = {'embedding_params': [embedding.state_dict() for embedding in self.embeddings],
                     'attention_params': [attention.state_dict() for attention in self.attentions],
                     'actor_params': [actor.state_dict() for actor in self.actors],
                     'critic_params': [critic.state_dict() for critic in self.critics],
                     'actor_optim_params': [actor_optim.state_dict() for actor_optim in self.actor_optims],
                     'critic_optim_params': [critic_optim.state_dict() for critic_optim in self.critic_optims]}
        torch.save(save_dict, filename)

    def load(self, filename):
        save_dict = torch.load(filename)
        [embedding.load_state_dict(params) for embedding, params in zip(self.embeddings, save_dict['embedding_params'])]
        [attention.load_state_dict(params) for attention, params in zip(self.attentions, save_dict['attention_params'])]
        [actor.load_state_dict(params) for actor, params in zip(self.actors, save_dict['actor_params'])]
        [critic.load_state_dict(params) for critic, params in zip(self.critics, save_dict['critic_params'])]
        [actor_optim.load_state_dict(params) for actor_optim, params in
         zip(self.actor_optims, save_dict['actor_optim_params'])]
        [critic_optim.load_state_dict(params) for critic_optim, params in
         zip(self.critic_optims, save_dict['critic_optim_params'])]

    def set_eval(self):
        [embedding.eval() for embedding in self.embeddings]
        [attention.eval() for attention in self.attentions]
        [actor.eval() for actor in self.actors]
        [critic.eval() for critic in self.critics]

    def zero_grad(self):
        [zero_model_grad(embd) for embd in self.embeddings]
        [zero_model_grad(att) for att in self.attentions]
        [zero_model_grad(actor) for actor in self.actors]
        [zero_model_grad(critic) for critic in self.critics]
