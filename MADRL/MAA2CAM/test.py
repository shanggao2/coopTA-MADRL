import json
import time
import torch
from torch.autograd import Variable
from maa2cam import MAA2CAM
from misc import get_actions, deal_trajectory
import sys

sys.path.append("..")
from env_m.environment import Environment


def run(**info):
    env = Environment(is_train=False, cfg_file_name=sys.argv[1])
    env.init_all()

    done, t_list, w_list, obs, masks = env.get_obs()
    model = MAA2CAM(num_agents=env.num_workers, num_actions=env.num_tasks, ob_dim=obs.shape[1],
                    embd_dim=info['embd_dim'], hidd_dim=info['hidd_dim'], a_lr=info['a_lr'], c_lr=info['c_lr'])
    model.load('/home/jlu/repo/model_save/' + sys.argv[1] + '.pt')
    obs = [Variable(torch.from_numpy(obs[i]).view(1, -1).float()) for i in range(obs.shape[0])]
    # if info['cuda_available']:
    #     model.move_to_gpu()
    #     obs = [ob.cuda() for ob in obs]
    model.set_eval()
    all_embd_obs, all_c = model.pre_process(obs)
    all_obs = [torch.cat((ob, c), dim=1) for ob, c in zip(all_embd_obs, all_c)]

    print('begin testing')
    start_time = time.time()
    finished_step = 0
    while not done:
        all_action_probs = model.get_action_probs(all_obs)
        taken_actions, _ = get_actions(all_action_probs, masks, len(w_list),
                                       env.num_tasks, env.num_workers)
        _ = env.step(taken_actions, t_list, w_list)
        done, t_list, w_list, obs, masks = env.get_obs()
        if done:
            break
        obs = [torch.from_numpy(obs[i]).view(1, -1).float() for i in range(obs.shape[0])]
        # if info['cuda_available']:
        #     obs = [ob.cuda() for ob in obs]
        all_embd_obs, all_c = model.pre_process(obs)
        all_obs = [torch.cat((ob, c), dim=1) for ob, c in zip(all_embd_obs, all_c)]
        finished_step += 1
        if finished_step % 100 == 0:
            print('TCR={}   '.format(round(float(env.num_completed) / env.num_published, 3)), end='')
            print('WPR={}   '.format(round(env.total_benefit / env.total_value, 3)), end='')
            print('time={}   '.format(round((time.time() - start_time) / 60, 3)), end='')
            print('step={}'.format(finished_step))

    print('TCR={}   '.format(round(float(env.num_completed) / env.num_published, 3)), end='')
    print('WPR={}   '.format(round(env.total_benefit / env.total_value, 3)), end='')
    print('time={}'.format(round((time.time() - start_time) / 60, 3)))
    env.on_end()


if __name__ == '__main__':
    # load config
    cfg_file = open('/home/jlu/repo/config/' + sys.argv[1] + '.json', 'r')
    cfg_data = cfg_file.read()
    config = json.loads(cfg_data)
    info = {'embd_dim': config['embd_dim'],
            'hidd_dim': config['hidd_dim'],
            'a_lr': config['a_lr'],
            'c_lr': config['c_lr'],
            'cuda_available': torch.cuda.is_available()}

    run(**info)
