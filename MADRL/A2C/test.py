import json
import time
import torch
from a2c import A2C
from misc import get_action
import sys

sys.path.append("..")
from env_s.environment import Environment


def run(**info):
    env = Environment(is_train=False, cfg_file_name=sys.argv[1])
    env.init_all()

    # get init state
    done, t, w_list, state = env.get_state()
    # create model
    model = A2C(n_actions=env.max_n + 1, s_dim=state.shape[1], a_hidd_dims=info['a_hidd_dims'],
                c_hidd_dims=info['c_hidd_dims'], a_lr=info['a_lr'], c_lr=info['c_lr'])
    model.load('/home/jlu/repo/model_save/' + sys.argv[1] + '.pt')
    # numpy.array to torch.tensor
    state = torch.from_numpy(state).float()
    # move to GPU
    # if info['cuda_available']:
    #     model.move_to_gpu()
    #     state = state.cuda()
    model.set_eval()

    print('begin testing')
    start_time = time.time()
    finished_step = 0
    while not done:
        action_probs = model.actor(state)
        taken_action, _ = get_action(action_probs, len(w_list))
        _ = env.step(taken_action, t, w_list)
        done, t, w_list, state = env.get_state()
        if done:
            break
        state = torch.from_numpy(state).float()
        # if info['cuda_available']:
        #     state = state.cuda()
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
    config = json.loads(cfg_data)
    info = {'a_hidd_dims': config['a_hidd_dims'],
            'c_hidd_dims': config['c_hidd_dims'],
            'a_lr': config['a_lr'],
            'c_lr': config['c_lr'],
            'cuda_available': torch.cuda.is_available()}

    run(**info)
