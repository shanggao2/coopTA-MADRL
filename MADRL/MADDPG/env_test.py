import time
import json
import numpy as np
import time
import sys

sys.path.append("..")
from env_m.environment import Environment


def gen_actions(masks):
    actions = []
    for i in range(masks.shape[0]):
        for j in range(masks.shape[1]):
            if masks[i][j] > 0.0:
                actions.append(j)
                break
    return actions


def run():
    env = Environment(is_train=False)
    env.init_all()
    done, t_list, w_list, obs, masks = env.get_obs()
    cnt = 1

    while not done:
        actions = gen_actions(masks)
        _ = env.step(actions, t_list, w_list)
        if cnt % 100 == 0:
            print('time={}  t_pub={}  t_com={}  t_ing={}  t_wai={}  t_exp={}  t_val={}  t_ben={}'.format(
                env.cur_time, env.num_published, env.num_completed, len(env.ing_tasks),
                len(env.resubmit_tasks), env.num_expired, env.total_value, env.total_benefit))
            print('TCR={}   '.format(round(float(env.num_completed) / env.num_published, 3)), end='')
            print('WPR={}   '.format(round(env.total_benefit / env.total_value, 3)), end='')
            print('ava_w={}'.format(len(env.available_workers)))
            print('------------------------------------')
        cnt += 1
        done, t_list, w_list, obs, masks = env.get_obs()

    print('time={}  t_pub={}  t_com={}  t_ing={}  t_wai={}  t_exp={}  t_val={}  t_ben={}'.format(
        env.cur_time, env.num_published, env.num_completed, len(env.ing_tasks),
        len(env.resubmit_tasks), env.num_expired, env.total_value, env.total_benefit))
    print('TCR={}   '.format(round(float(env.num_completed) / env.num_published, 3)), end='')
    print('WPR={}   '.format(round(env.total_benefit / env.total_value, 3)), end='')
    print('ava_w={}'.format(len(env.available_workers)))
    print('------------------------------------')

    env.on_end()


def run1():
    env = Environment(is_train=True, cfg_file_name=sys.argv[1])
    env.init_all()
    wn_list = []
    cnt = 1

    done, t_list, w_list, obs, masks = env.get_obs()
    wn_list.append(len(w_list))

    start_time = time.time()
    while not done:
        actions = gen_actions(masks)
        _ = env.step(actions, t_list, w_list)

        if cnt % 100 == 0:
            print('step={}  time={}'.format(cnt, round((time.time() - start_time) / 60, 3)))
        cnt += 1

        done, t_list, w_list, obs, masks = env.get_obs()
        if done:
            break
        wn_list.append(len(w_list))

    env.on_end()
    data = np.array(wn_list).reshape((-1, 1))
    np.savetxt('/home/jlu/repo/midres/' + sys.argv[1] + '.csv', data, fmt='%d')


def test():
    data = np.loadtxt('/home/jlu/repo/midres/' + sys.argv[1] + '.csv', dtype=int)
    cnt = 0
    for i in range(len(data)):
        if data[i] <= int(sys.argv[3]):
            cnt += 1
    print(cnt / len(data))


if __name__ == '__main__':
    # run()
    if sys.argv[2] == 'run':
        run1()
    else:
        test()
