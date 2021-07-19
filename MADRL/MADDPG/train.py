import json
import time
import torch
import numpy as np
from actor import Actor
from critic import Critic
from maddpg import MADDPG
from misc import get_actions_train
from memory import ReplayMemory, Experience
import sys

sys.path.append("..")
from env_m.environment import Environment


def run(**info):
    env = Environment(is_train=True, cfg_file_name=sys.argv[1])
    env.init_all()
    rm = ReplayMemory(info['mem_size'])
    # step = 0
    finished_step = 0
    # get init obs
    done, t_list, w_list, obs, masks = env.get_obs()
    # create model
    model = MADDPG(n_agents=env.num_workers, n_actions=env.num_tasks, ob_dim=obs.shape[1],
                   a_hidd_dims=info['actor_hiddle_layers'], c_hidd_dims=info['critic_hiddle_layers'],
                   a_lr=info['a_lr'], c_lr=info['c_lr'], tau=info['tau'])
    all_obs = [torch.from_numpy(obs[i]).view(1, -1).float() for i in range(obs.shape[0])]
    # move to gpu if can
    # if info['cuda_available']:
    #     model.move_to_gpu()
    #     all_obs = [ob.cuda() for ob in all_obs]

    print('begin training')
    start_time = time.time()
    while finished_step < info['total_step'] or not done:
        # no more new data generated
        if finished_step == info['total_step'] and env.file_task:
            env.num_published -= 1
            env.total_value -= env.file_task.budget
            env.file_task = None
        # get actions, take actions, get reward
        all_action_probs = [curr_actor(ob) for curr_actor, ob in zip(model.curr_actors, all_obs)]
        taken_actions, actions_one_hot = get_actions_train(all_action_probs, masks, len(w_list), env.num_tasks,
                                                           env.num_workers, finished_step, info['exp_it_thr'],
                                                           info['base_num'])
        reward = env.step(taken_actions, t_list, w_list)
        # get next obs
        done, t_list, w_list, next_obs, masks = env.get_obs()
        if done:
            if finished_step < info['total_step']:
                env.reset()
                if not t_list:
                    done, t_list, w_list, next_obs, masks = env.get_obs()
            else:
                break
        all_next_obs = [torch.from_numpy(next_obs[i]).view(1, -1).float() for i in range(next_obs.shape[0])]
        # if info['cuda_available']:
        #     all_next_obs = [next_ob.cuda() for next_ob in all_next_obs]
        # collect experience, add count, state transition
        rm.push(all_obs, actions_one_hot, reward, all_next_obs)
        finished_step += 1
        all_obs = all_next_obs
        # update networks
        if len(rm) >= info['batch_size']:
            # model.update_model(rm, info['batch_size'], info['cuda_available'], info['gamma'], info['max_grad_norm'])
            model.update_model(rm, info['batch_size'], info['gamma'], info['max_grad_norm'])
        # output
        if finished_step % 100 == 0:
            print('TCR={}   '.format(round(float(env.num_completed) / env.num_published, 3)), end='')
            print('WPR={}   '.format(round(env.total_benefit / env.total_value, 3)), end='')
            print('time={}   '.format(round((time.time() - start_time) / 60, 3)), end='')
            print('step={}'.format(finished_step))

    print('TCR={}   '.format(round(float(env.num_completed) / env.num_published, 3)), end='')
    print('WPR={}   '.format(round(env.total_benefit / env.total_value, 3)), end='')
    print('time={}'.format(round((time.time() - start_time) / 60, 3)))
    env.on_end()
    # save model
    model.save('/home/jlu/repo/model_save/' + sys.argv[1] + '.pt')


if __name__ == '__main__':
    # load config
    cfg_file = open('/home/jlu/repo/config/' + sys.argv[1] + '.json', 'r')
    cfg_data = cfg_file.read()
    cfg_file.close()
    config = json.loads(cfg_data)
    actor_hiddle_layers = config['actor_hiddle_layers']
    critic_hiddle_layers = config['critic_hiddle_layers']
    assert len(actor_hiddle_layers) >= 1
    assert len(critic_hiddle_layers) >= 1
    total_step = config['total_step']
    exp_it_thr = int(total_step * 0.75)  # explore iteration threshold
    base_num = total_step * 0.25

    info = {'actor_hiddle_layers': actor_hiddle_layers,
            'critic_hiddle_layers': critic_hiddle_layers,
            'a_lr': config['a_lr'],
            'c_lr': config['c_lr'],
            'tau': config['tau'],
            'total_step': total_step,
            'batch_size': config['batch_size'],
            'gamma': config['gamma'],
            'max_grad_norm': config['max_grad_norm'],
            'exp_it_thr': exp_it_thr,
            'base_num': base_num,
            'mem_size': config['mem_size'],
            'cuda_available': torch.cuda.is_available()}

    run(**info)
