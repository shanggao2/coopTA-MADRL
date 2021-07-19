import time
import random
import numpy as np
import sys

sys.path.append("..")
from env_s.environment import Environment


def run():
    env = Environment(is_train=True)
    env.init_all()
    done, t, w_list, state = env.get_state()
    cnt = 1

    while not done:
        assert state is not None
        _ = env.step(random.randint(0, len(w_list) - 1), t, w_list)
        if cnt % 100 == 0:
            print('time={}  t_pub={}  t_com={}  t_ing={}  t_wai={}  t_exp={}  t_val={}  t_ben={}'.format(
                env.cur_time, env.num_published, env.num_completed, len(env.ing_tasks),
                len(env.resubmit_tasks), env.num_expired, env.total_value, env.total_benefit))
            print('TPR={}   '.format(round(float(env.num_completed) / env.num_published, 3)), end='')
            print('WPR={}   '.format(round(env.total_benefit / env.total_value, 3)), end='')
            print('ava_w={}'.format(len(env.available_workers)))
            print('------------------------------------')
        cnt += 1
        done, t, w_list, state = env.get_state()

    print('time={}  t_pub={}  t_com={}  t_ing={}  t_wai={}  t_exp={}  t_val={}  t_ben={}'.format(
        env.cur_time, env.num_published, env.num_completed, len(env.ing_tasks),
        len(env.resubmit_tasks), env.num_expired, env.total_value, env.total_benefit))
    print('TPR={}   '.format(round(float(env.num_completed) / env.num_published, 3)), end='')
    print('WPR={}   '.format(round(env.total_benefit / env.total_value, 3)), end='')
    print('ava_w={}'.format(len(env.available_workers)))
    print('------------------------------------')

    env.on_end()


def find_in_list(t, tlist):
    find = False
    g_idx = 0
    for idx, tt in enumerate(tlist):
        if tt.tid == t.tid:
            find = True
            g_idx = idx
            break
    return find, g_idx


def run1():
    env = Environment(is_train=True, cfg_file_name=sys.argv[1])
    env.init_all()
    done, t, w_list, _ = env.get_state()
    wn_list = []
    cnt = 1

    start_time = time.time()
    while not done:
        wn_list.append(len(w_list))
        _ = env.step(random.randint(0, len(w_list) - 1), t, w_list)
        if cnt % 100 == 0:
            print('step={}  time={}'.format(cnt, round((time.time() - start_time) / 60, 3)))
        cnt += 1
        done, t, w_list, _ = env.get_state()

    env.on_end()
    data = np.array(wn_list).reshape((-1, 1))
    np.savetxt('/home/jlu/repo/midres/' + sys.argv[1] + '.csv', data, fmt='%d')

    env.on_end()


def test():
    data = np.loadtxt('/home/jlu/repo/midres/' + sys.argv[1] + '.csv', dtype=int)
    cnt = 0
    for i in range(len(data)):
        if data[i] <= int(sys.argv[3]):
            cnt += 1
    print(cnt / len(data))


if __name__ == '__main__':
    if sys.argv[2] == 'run':
        run1()
    else:
        test()
