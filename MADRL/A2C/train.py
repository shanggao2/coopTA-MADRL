import json
import time
import torch
from a2c import A2C
from misc import get_action, deal_trajectory
import sys

sys.path.append("..")
from env_s.environment import Environment


def run(**info):
    # create environment
    env = Environment(is_train=True, cfg_file_name=sys.argv[1])
    env.init_all()
    # get init state
    done, t, w_list, state = env.get_state()
    # create model
    model = A2C(n_actions=env.max_n + 1, s_dim=state.shape[1], a_hidd_dims=info['a_hidd_dims'],
                c_hidd_dims=info['c_hidd_dims'], a_lr=info['a_lr'], c_lr=info['c_lr'])
    # numpy.array to torch.tensor
    state = torch.from_numpy(state).float()
    # move to GPU
    #    if info['cuda_available']:
    #        model.move_to_gpu()
    #        state = state.cuda()

    s_list, lap_list, e_list, r_list, d_list = [], [], [], [], []
    finished_step = 0
    print('begin training')
    start_time = time.time()
    while finished_step < info['total_step'] or not done:
        s_list.append(state)
        # no more new data generated
        if finished_step == info['total_step'] and env.file_task:
            env.num_published -= 1
            env.total_value -= env.file_task.budget
            env.file_task = None
        # take action and get reward
        action_probs = model.actor(state)
        action_entropy = (-action_probs * torch.log(action_probs)).sum(dim=1)
        taken_action, log_taken_action_prob = get_action(action_probs, len(w_list))
        reward = env.step(taken_action, t, w_list)
        # get next state, collect trajectories, add counter
        done, t, w_list, state = env.get_state()
        lap_list.append(log_taken_action_prob)
        e_list.append(action_entropy)
        r_list.append(reward)
        d_list.append(done)
        pre_done = done
        if done:
            if finished_step < info['total_step']:
                env.reset()
                if not t:
                    done, t, w_list, state = env.get_state()
            else:
                break
        state = torch.from_numpy(state).float()
        #        if info['cuda_available']:
        #            state = state.cuda()
        finished_step += 1
        # deal  trajectories and update model
        if len(s_list) == info['batch_size']:
            if not pre_done:
                next_v = model.critic(state)
            else:
                zero = [[0.0]]
                next_v = torch.FloatTensor(zero)
            #                if info['cuda_available']:
            #                    next_v = torch.cuda.FloatTensor(zero)
            #                else:
            #                    next_v = torch.FloatTensor(zero)
            t_batch, s_batch, lap_batch, e_batch = deal_trajectory(s_list, lap_list, e_list, r_list,
                                                                   d_list, next_v, info['gamma'])
            model.update_model(t_batch, s_batch, lap_batch, e_batch, info['entropy_beta'], info['max_grad_norm'])
            s_list, lap_list, e_list, r_list, d_list = [], [], [], [], []  # clear
        # output info
        if finished_step % 100 == 0:
            print('TCR={}   '.format(round(float(env.num_completed) / env.num_published, 3)), end='')
            print('WPR={}   '.format(round(env.total_benefit / env.total_value, 3)), end='')
            print('time={}   '.format(round((time.time() - start_time) / 60, 3)), end='')
            print('step={}'.format(finished_step))

    print('TCR={}   '.format(round(float(env.num_completed) / env.num_published, 3)), end='')
    print('WPR={}   '.format(round(env.total_benefit / env.total_value, 3)), end='')
    print('time={}'.format(round((time.time() - start_time) / 60, 3)))
    env.on_end()
    model.save('/home/jlu/repo/model_save/' + sys.argv[1] + '.pt')


if __name__ == '__main__':
    # load config
    cfg_file = open('/home/jlu/repo/config/' + sys.argv[1] + '.json', 'r')
    cfg_data = cfg_file.read()
    config = json.loads(cfg_data)
    info = {'a_hidd_dims': config['a_hidd_dims'],
            'c_hidd_dims': config['c_hidd_dims'],
            'a_lr': config['a_lr'],
            'c_lr': config['c_lr'],
            'total_step': config['total_step'],
            'batch_size': config['batch_size'],
            'gamma': config['gamma'],
            'entropy_beta': config['entropy_beta'],
            'max_grad_norm': config['max_grad_norm'],
            'cuda_available': torch.cuda.is_available()}

    run(**info)
