import json
import time
import torch
from maa2cam import MAA2CAM
from misc import get_actions, deal_trajectory
import sys

sys.path.append("..")
from env_m.environment import Environment


def run(**info):
    # some preparation
    env = Environment(is_train=True, cfg_file_name=sys.argv[1])
    env.init_all()
    finished_step = 0
    """
    3 steps:
    1) get raw abservations
    2) create the MA-A2C-AM model
    3) generate new observations (that can be fed directly into AC models) from raw observations.
       (by embedding raw observations and calculating context vectors through the attention mechanism)
    """
    # get original abservations for all agents
    done, t_list, w_list, obs, masks = env.get_obs()
    # create model
    model = MAA2CAM(num_agents=env.num_workers, num_actions=env.num_tasks, ob_dim=obs.shape[1],
                    embd_dim=info['embd_dim'], hidd_dim=info['hidd_dim'], a_lr=info['a_lr'], c_lr=info['c_lr'])
    # transform
    obs = [torch.from_numpy(obs[i]).view(1, -1).float() for i in range(obs.shape[0])]
    # move to GPU
    # if info['cuda_available']:
    #     model.move_to_gpu()
    #     obs = [ob.cuda() for ob in obs]

    # get embedding observations and context vectors (attention mechanism) for all agents
    all_embd_obs, all_c = model.pre_process(obs)
    # catenate
    all_obs = [torch.cat((ob, c), dim=1) for ob, c in zip(all_embd_obs, all_c)]

    raw_obs_list, state_list, reward_list, entropy_list, ltap_list, done_list = [], [], [], [], [], []
    print('begin training')
    start_time = time.time()
    while finished_step < info['total_step'] or not done:
        # no more new data generated
        if finished_step == info['total_step'] and env.file_task:
            env.num_published -= 1
            env.total_value -= env.file_task.budget
            env.file_task = None
        # get action probabilities for all agents
        all_action_probs = model.get_action_probs(all_obs)
        # the entropy of action probabilitoes for all agents
        all_entropy = [(-action_probs * torch.log(action_probs)).sum(dim=1) for action_probs in all_action_probs]
        # for all agents: get the the action taken by the agent and the corresponding action probability
        taken_actions, log_taken_action_probs = get_actions(all_action_probs, masks, len(w_list),
                                                            env.num_tasks, env.num_workers)

        # get shared reward
        reward = env.step(taken_actions, t_list, w_list)

        # collect trajectories
        if str(obs[0].device) == 'cpu':
            obs_copy = [torch.FloatTensor(ob) for ob in obs]
        else:
            obs_copy = [torch.cuda.FloatTensor(ob) for ob in obs]
        raw_obs_list.append(obs_copy)
        state_list.append(all_obs)
        reward_list.append(reward)
        entropy_list.append(all_entropy)
        ltap_list.append(log_taken_action_probs)

        # get the next observation
        done, t_list, w_list, obs, masks = env.get_obs()
        done_list.append(done)

        # calculate reward needed
        pre_done = done
        # reset or end the training
        if done:
            if finished_step < info['total_step']:
                env.reset()
                if not t_list:
                    done, t_list, w_list, obs, masks = env.get_obs()
            else:
                break
        # generate new observations (that can be fed directly into AC models) from raw observations.
        obs = [torch.from_numpy(obs[i]).view(1, -1).float() for i in range(obs.shape[0])]
        # obs = [ob.cuda() for ob in obs]
        all_embd_obs, all_c = model.pre_process(obs)
        all_obs = [torch.cat((ob, c), dim=1) for ob, c in zip(all_embd_obs, all_c)]

        # increase the count
        finished_step += 1

        # update nets by minibatch
        if finished_step % info['batch_size'] == 0:
            if not pre_done:
                all_obs_copy = [obs.detach() for obs in all_obs]
                next_values = model.get_values(all_obs_copy)
            else:
                next_values = [torch.FloatTensor([[0.0]]) for _ in range(env.num_workers)]
                # if info['cuda_available']:
                #     next_values = [torch.cuda.FloatTensor([[0.0]]) for _ in range(env.num_workers)]
                # else:
                #     next_values = [torch.FloatTensor([[0.0]]) for _ in range(env.num_workers)]
            raw_obs_list, state_list, target_list, entropy_list, ltap_list = \
                deal_trajectory(raw_obs_list, state_list, reward_list, entropy_list,
                                done_list, ltap_list, next_values, info['gamma'])
            model.update_model(raw_obs_list, state_list, target_list, entropy_list,
                               ltap_list, info['entropy_beta'], info['max_grad_norm'])
            # reset and clear
            raw_obs_list, state_list, reward_list, entropy_list, ltap_list, done_list = [], [], [], [], [], []

        if finished_step % 100 == 0:
            print('TPR={}   '.format(round(float(env.num_completed) / env.num_published, 3)), end='')
            print('WPR={}   '.format(round(env.total_benefit / env.total_value, 3)), end='')
            print('time={}   '.format(round((time.time() - start_time) / 60, 3)), end='')
            print('step={}'.format(finished_step))

    print('TPR={}   '.format(round(float(env.num_completed) / env.num_published, 3)), end='')
    print('WPR={}   '.format(round(env.total_benefit / env.total_value, 3)), end='')
    print('time={}'.format(round((time.time() - start_time) / 60, 3)))
    env.on_end()

    # save the model
    model.save('/home/jlu/repo/model_save/' + sys.argv[1] + '.pt')


if __name__ == '__main__':
    # load config
    cfg_file = open('/home/jlu/repo/config/' + sys.argv[1] + '.json', 'r')
    cfg_data = cfg_file.read()
    config = json.loads(cfg_data)
    total_step = config['total_step']
    info = {'embd_dim': config['embd_dim'],
            'hidd_dim': config['hidd_dim'],
            'a_lr': config['a_lr'],
            'c_lr': config['c_lr'],
            'total_step': config['total_step'],
            'batch_size': config['batch_size'],
            'alpha': config['alpha'],
            'gamma': config['gamma'],
            'entropy_beta': config['entropy_beta'],
            'max_grad_norm': config['max_grad_norm'],
            'cuda_available': torch.cuda.is_available(),
            'base_num': total_step / 28.0}

    run(**info)
