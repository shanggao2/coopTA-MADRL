import json
import time
import torch
import numpy as np
from actor import Actor
from critic import Critic
from maddpg import MADDPG
from misc import get_actions_test
import sys

sys.path.append("..")
from env_m.environment import Environment


def run(**info):
    env = Environment(is_train=False, cfg_file_name=sys.argv[1])
    env.init_all()

    done, t_list, w_list, obs, masks = env.get_obs()
    model = MADDPG(n_agents=env.num_workers, n_actions=env.num_tasks, ob_dim=obs.shape[1],
                   a_hidd_dims=info['actor_hiddle_layers'], c_hidd_dims=info['critic_hiddle_layers'],
                   a_lr=info['a_lr'], c_lr=info['c_lr'], tau=info['tau'])
    all_obs = [torch.from_numpy(obs[i]).view(1, -1).float() for i in range(obs.shape[0])]
    model.load('/home/jlu/repo/model_save/' + sys.argv[1] + '.pt')
    if info['cuda_available']:
        model.move_to_gpu()
        all_obs = [ob.cuda() for ob in all_obs]
    model.set_eval()

    print('begin testing')
    start_time = time.time()
    finished_step = 0
    while not done:
        all_action_probs = [curr_actor(ob) for curr_actor, ob in zip(model.curr_actors, all_obs)]
        taken_actions = get_actions_test(all_action_probs, masks, len(w_list), env.num_tasks, env.num_workers)
        _ = env.step(taken_actions, t_list, w_list)
        done, t_list, w_list, obs, masks = env.get_obs()
        if done:
            break
        all_obs = [torch.from_numpy(obs[i]).view(1, -1).float() for i in range(obs.shape[0])]
        if info['cuda_available']:
            all_obs = [ob.cuda() for ob in all_obs]
        finished_step += 1
        if finished_step % 100 == 0:
            print('TCR={}   '.format(round(float(env.num_completed) / env.num_published, 3)), end='')
            print('WPR={}   '.format(round(env.total_benefit / env.total_value, 3)), end='')
            print('time={}   '.format(round((time.time() - start_time) / 60, 3)), end='')
            print('step={}   '.format(finished_step), end='')
            print('vir_t={}  '.format(env.cur_time), end=' ')
            if env.file_task:
                print('tid={}   '.format(env.file_task.tid))
            else:
                print('None')

    print('TCR={}   '.format(round(float(env.num_completed) / env.num_published, 3)), end='')
    print('WPR={}   '.format(round(env.total_benefit / env.total_value, 3)), end='')
    print('time={}   '.format(round((time.time() - start_time) / 60, 3)), end='')
    print('step={}   '.format(finished_step), end='')
    print('vir_t={}'.format(env.cur_time))
    env.on_end()


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

    info = {'actor_hiddle_layers': actor_hiddle_layers,
            'critic_hiddle_layers': critic_hiddle_layers,
            'a_lr': config['a_lr'],
            'c_lr': config['c_lr'],
            'tau': config['tau'],
            'cuda_available': torch.cuda.is_available()}

    run(**info)
