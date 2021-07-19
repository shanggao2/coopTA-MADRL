import json
import csv
import math
import random
import numpy as np
from itertools import product
from env_m.task import Task
from env_m.worker import Worker
from env_m.env_misc import is_cover_all, is_cover_part, get_dist, get_cost, check_cost, check_skills, need_skills, \
    same_set, save_record, same_skills, second_check, gen_masks


class Environment(object):
    def __init__(self, is_train, cfg_file_name):
        self.cur_time = 0.0
        self.available_workers = []
        self.leave_workers = []
        self.resubmit_tasks = []
        self.ing_tasks = []
        self.cfg = None
        self.file_t_g = None
        self.file_t_b = None
        self.file_t_s = None
        self.file_task = None
        self.num_tasks = 0
        self.num_workers = 0
        self.num_skills = 0
        self.num_published = 0
        self.num_completed = 0
        self.num_expired = 0
        self.total_value = 0.0
        self.total_benefit = 0.0
        self.time_slot = 10.0
        self.num_find = 0
        self.is_train = is_train
        self.cfg_file_name = cfg_file_name
        self.base_path = '/home/jlu/repo/'
        self.big_num = 10000.0

    def load_config(self):
        cfg_file = open(self.base_path + 'config/' + self.cfg_file_name + '.json', 'r')
        cfg_data = cfg_file.read()
        self.cfg = json.loads(cfg_data)

    def open_file(self):
        base_path = self.base_path + 'data/part/'
        fn_head = self.cfg['dataset']
        if self.is_train:
            fn_head = fn_head + '_train_task_'
        else:
            fn_head = fn_head + '_test_task_'
        self.file_t_g = open(base_path + fn_head + 'general.csv', 'r', newline='')
        self.file_t_b = open(base_path + fn_head + 'budget_' + self.cfg['budget_level'] + '.csv', 'r', newline='')
        self.file_t_s = open(base_path + fn_head + 'skills.csv', 'r', newline='')

    def close_file(self):
        self.file_t_g.close()
        self.file_t_b.close()
        self.file_t_s.close()

    def gen_workers(self):
        base_path = self.base_path + 'data/part/'
        fn_head = self.cfg['dataset']
        if self.is_train:
            fn_head = fn_head + '_train_worker_'
        else:
            fn_head = fn_head + '_test_worker_'
        file_w_g = open(base_path + fn_head + 'general_' + str(self.cfg['nWorkers']) + '.csv', 'r', newline='')
        file_w_s = open(base_path + fn_head + 'skills_' + str(self.cfg['nWorkers']) + '.csv', 'r', newline='')
        reader_w_g = csv.reader(file_w_g)
        reader_w_s = csv.reader(file_w_s)
        data_w_g = list(reader_w_g)
        data_w_s = list(reader_w_s)
        assert len(data_w_g) == len(data_w_s)
        # wid, lon, lat, pf, coo, umc, speed, skills
        for i in range(0, len(data_w_g)):
            self.available_workers.append(
                Worker(int(data_w_g[i][0]), float(data_w_g[i][1]), float(data_w_g[i][2]), float(data_w_g[i][3]),
                       data_w_g[i][4], float(data_w_g[i][5]), float(data_w_g[i][6]), data_w_s[i]))
        file_w_g.close()
        file_w_s.close()

    def gen_one_task(self):
        line_g = self.file_t_g.readline()
        line_b = self.file_t_b.readline()
        line_s = self.file_t_s.readline()
        if not line_g or not line_b or not line_s:
            return None
        line_g = line_g.splitlines()[0]
        line_b = line_b.splitlines()[0]
        line_s = line_s.splitlines()[0]
        line_g = line_g.split(',')
        line_s = line_s.split(',')
        # tid, at, lon, lat, budget, coo, ttl, deadline, exetime, skills
        return Task(int(line_g[0]), float(line_g[1]), float(line_g[2]), float(line_g[3]), float(line_b),
                    line_g[4], float(line_g[5]), float(line_g[1]) + float(line_g[5]), float(line_g[6]), line_s)

    def init_all(self):
        self.load_config()
        self.num_tasks = self.cfg['num_tasks']
        self.num_workers = self.cfg['num_workers']
        self.time_slot = self.cfg['time_slot']
        skills_num_dict = {'real': 7, 'syn': 5}
        self.num_skills = skills_num_dict[self.cfg['dataset']]
        self.open_file()
        self.gen_workers()
        self.file_task = self.gen_one_task()
        if self.file_task:
            self.cur_time = self.file_task.at
            self.num_published = 1
            self.total_value += self.file_task.budget

    def reset(self):
        self.close_file()
        self.open_file()
        self.file_task = self.gen_one_task()
        self.cur_time = self.file_task.at
        if self.file_task:
            self.num_published += 1
            self.total_value += self.file_task.budget

    def on_end(self):
        self.close_file()

    def resubmit(self, t):
        idx = 0
        while idx < len(self.resubmit_tasks):
            if self.resubmit_tasks[idx].at > t.at:
                break
            idx += 1
        self.resubmit_tasks.insert(idx, t)

    def search_in_rsq(self, t):
        idx = 0
        find = False
        while idx < len(self.resubmit_tasks):
            if self.resubmit_tasks[idx].tid == t.tid:
                find = True
                break
            idx += 1
        return find, idx

    def finally_check(self, task_list):
        w_set = set()
        idx = 0
        while idx < len(task_list):
            temp_set = set()
            t = task_list[idx]
            for w in t.workers:
                dist = get_dist(t, w)
                if dist / w.speed * 60.0 <= t.deadline - self.cur_time:
                    temp_set.add(w)
            if temp_set and len(temp_set) < len(t.workers) and t.coo == '1' and need_skills(t.skills):
                # Must ensure that the remaining workers can complete the task t.
                temp_set = second_check(t, temp_set)
                t.workers = temp_set
            if not temp_set:
                if t.deadline <= self.cur_time:
                    t.at = self.cur_time
                    t.has_ass = True
                    self.resubmit(t)
                else:
                    self.num_expired += 1
                del task_list[idx]
            else:
                w_set = w_set | temp_set
                idx += 1
        return w_set

    def condition_check(self, t, w, r):
        dist = get_dist(t, w)
        if dist > r:
            return False
        if dist / w.speed * 60.0 > t.deadline - self.cur_time:  # t.ttl
            return False
        if w.umc * dist > t.budget:
            return False
        if is_cover_all(t.skills, w.skills):
            return True
        if is_cover_part(t.skills, w.skills) and t.coo == '1' and w.coo == '1':
            return True
        return False

    def find_workers(self, t):
        w_set = set()
        for i in range(0, len(self.available_workers)):
            w = self.available_workers[i]
            if self.condition_check(t, w, self.cfg['task_radius']):
                w_set.add(w)
        if t.coo == '1' and need_skills(t.skills):
            w_set = second_check(t, w_set)
        return w_set

    def find_task_in_rsq(self):
        for t in self.resubmit_tasks:
            if not t.has_ass:
                return t
        return None

    # return: the task has fond, if the task is self.file_task
    def select_one_task(self):
        if not self.resubmit_tasks and not self.file_task:
            self.cur_time += 3.0
            self.check_ing()
            self.check_rsq()
            return None, False
        elif not self.resubmit_tasks and self.file_task:
            self.cur_time = self.file_task.at
            self.check_ing()
            return self.file_task, True
        elif self.resubmit_tasks and not self.file_task:
            while True:
                self.cur_time += 3.0
                self.check_ing()
                self.check_rsq()
                t = self.find_task_in_rsq()
                if t and t.at > self.cur_time:
                    self.cur_time = t.at
                    self.check_ing()
                    self.check_rsq()
                    if t in self.resubmit_tasks:
                        return t, False
                else:
                    return t, False
        else:
            while True:
                t = self.find_task_in_rsq()
                if not t or t.at >= self.file_task.at:
                    self.cur_time = self.file_task.at
                    self.check_ing()
                    self.check_rsq()
                    return self.file_task, True
                elif t.at <= self.cur_time:
                    return t, False
                else:
                    self.cur_time = t.at
                    self.check_ing()
                    self.check_rsq()
                    if t in self.resubmit_tasks:
                        return t, False

    # swfst: select workers for special task,
    # select workers for t that t.coo=0 or need_skills(t.skills)=False
    def swfst(self, t, w_list):
        d_list = []
        for w in w_list:
            d_list.append(get_cost(t, w))
        d_list_idx = sorted(range(len(d_list)), key=lambda k: d_list[k])
        tmp_w_set = set()
        for i in range(self.num_workers):
            tmp_w_set.add(w_list[d_list_idx[i]])
        return tmp_w_set

    def find_mn(self):
        for t in self.resubmit_tasks:
            t.has_ass = False
        start_time = self.cur_time
        g_t_list = []
        g_w_set = set()
        while True:
            t, from_data = self.select_one_task()
            if not t or self.cur_time - start_time > self.cfg['time_slot']:
                g_w_set = self.finally_check(g_t_list)
                done = not self.file_task and not self.resubmit_tasks and not self.ing_tasks
                if self.file_task and self.cur_time < self.file_task.at:
                    self.cur_time = self.file_task.at
                return done, g_t_list, list(g_w_set)
            w_set = self.find_workers(t)
            if len(w_set) > self.num_workers:
                if t.coo == '0' or not need_skills(t.skills):
                    w_set = self.swfst(t, list(w_set))
                else:
                    can, _ = self.select_workers(list(w_set), t)
                    assert can
                    w_set = t.workers
            if not w_set:
                if from_data:
                    t.at = self.cur_time
                    t.has_ass = True
                    self.resubmit(t)
                    self.file_task = self.gen_one_task()
                    if self.file_task:
                        self.num_published += 1
                        self.total_value += self.file_task.budget
                else:
                    t.has_ass = True
                continue
            temp_set = g_w_set | w_set
            if len(temp_set) <= self.num_workers and len(g_t_list) + 1 <= self.num_tasks:
                t.workers = w_set
                g_w_set = temp_set
                g_t_list.append(t)
                if from_data:
                    self.file_task = self.gen_one_task()
                    if self.file_task:
                        self.num_published += 1
                        self.total_value += self.file_task.budget
                else:
                    find, idx = self.search_in_rsq(t)
                    assert find
                    del self.resubmit_tasks[idx]
            else:
                g_w_set = self.finally_check(g_t_list)
                temp_set = g_w_set | w_set
                if len(temp_set) <= self.num_workers and len(g_t_list) + 1 <= self.num_tasks:
                    t.workers = w_set
                    g_w_set = temp_set
                    g_t_list.append(t)
                    if from_data:
                        self.file_task = self.gen_one_task()
                        if self.file_task:
                            self.num_published += 1
                            self.total_value += self.file_task.budget
                    else:
                        find, idx = self.search_in_rsq(t)
                        assert find
                        del self.resubmit_tasks[idx]
                else:
                    done = not self.file_task and not self.resubmit_tasks and not self.ing_tasks
                    if self.file_task and self.cur_time < self.file_task.at:
                        self.cur_time = self.file_task.at
                    return done, g_t_list, list(g_w_set)

    def gen_vir_fea(self, obs, task_list, num_skills):  # generate virtual feature
        obs.append(0.0)  # whether task j allows cooperation
        t = task_list[random.randint(0, len(task_list) - 1)]
        obs.append(format(t.budget * random.uniform(0.9, 1.1), '.1f'))  # budget
        obs.append(0.0)  # Is it possible for worker i to complete task j
        obs.append(format(random.gauss(0.55, 0.13) * self.cfg['task_radius'], '.2f'))  # cost
        for i in range(0, num_skills):  # skills
            obs.append(0.0)

    def normalize_obs(self, arr):
        arr_t = arr[:, 1:]
        arr_t = arr_t.reshape((self.num_tasks * self.num_workers, 4 + self.num_skills))
        temp = arr_t[:, 1]
        temp_max = temp.max()
        temp_min = temp.min()
        if temp_max - temp_min > 1e-3:
            arr_t[:, 1] = 2.0 * ((arr_t[:, 1] - temp_min) / (temp_max - temp_min)) - 1.0
        else:
            arr_t[:, 1] = 0.0
        temp = arr_t[:, 3]
        temp_max = temp.max()
        temp_min = temp.min()
        if temp_max - temp_min > 1e-3:
            arr_t[:, 3] = 2.0 * ((arr_t[:, 3] - temp_min) / (temp_max - temp_min)) - 1.0
        else:
            arr_t[:, 3] = 0.0
        arr_t = arr_t.reshape((self.num_workers, self.num_tasks * (4 + self.num_skills)))
        arr[:, 1:] = arr_t

    def gen_obs(self, task_list, worker_list):
        obs_all = []
        num_real = len(task_list)
        for w in worker_list:
            obs = []
            if w.coo == '1':  # whether worker i is willing to cooperate
                obs.append(1.0)
            else:
                obs.append(0.0)
            for t in task_list:
                if t.coo == '1':  # whether task j allows cooperation
                    obs.append(1.0)
                else:
                    obs.append(0.0)
                obs.append(t.budget)  # budget
                # The values differs depending on whether w can complete t.
                if w in t.workers:
                    obs.append(1.0)  # Is it possible for worker i to complete task j
                    obs.append(format(get_cost(t, w), '.2f'))  # cost
                    for i in range(0, len(t.skills)):  # skills
                        if t.skills[i] == '0':
                            obs.append(-1.0)
                        elif t.skills[i] == '1' and w.skills[i] == '0':
                            obs.append(0.0)
                        else:
                            obs.append(1.0)
                else:
                    obs.append(0.0)
                    obs.append(format(w.umc * self.cfg['task_radius'], '.2f'))
                    for i in range(0, len(t.skills)):  # skills
                        obs.append(0.0)
            for i in range(num_real, self.num_tasks):
                self.gen_vir_fea(obs, task_list, self.num_skills)
            obs_all.append(obs)
        for i in range(len(worker_list), self.num_workers):
            obs = [0.0]
            for j in range(0, self.num_tasks):
                self.gen_vir_fea(obs, task_list, self.num_skills)
            obs_all.append(obs)
        obs_all = np.array(obs_all, dtype=float)
        return obs_all

    def select_one_worker(self, w_list, t, need_skills_check):
        min_cost = self.big_num
        max_tt = 0.0
        can = False
        for w in w_list:
            dist = get_dist(t, w)
            cost = w.umc * dist
            travel_time = dist / w.speed * 60.0
            if travel_time > max_tt:
                max_tt = travel_time
            if need_skills_check:
                if not is_cover_all(t.skills, w.skills):
                    continue
            if cost < min_cost:
                min_cost = cost
                t.wb = t.budget - min_cost
                t.begin_time = self.cur_time
                t.eft = self.cur_time + max_tt + t.exetime
                t.workers = {w}
                can = True
        return can, min_cost

    def select_workers(self, w_list, t):
        min_cost = self.big_num  # big enough
        if len(w_list) == 0:
            return False, min_cost
        if t.coo == '0' or not need_skills(t.skills):
            return self.select_one_worker(w_list, t, False)

        # if t.coo == '1' or need_skills(t.skills)
        # remove has added
        can, min_cost = self.select_one_worker(w_list, t, True)
        if can:
            assert len(t.workers) == 1
            meet = False
            for w in w_list:
                if w.wid == list(t.workers)[0].wid:
                    w_list.remove(w)
                    meet = True
                    break
            assert meet
        # remove uncooperative workers
        idx = 0
        while idx < len(w_list):
            if w_list[idx].coo == '0':
                del w_list[idx]
            else:
                idx += 1
        # if w_list is null, no need continue
        if not w_list:
            if min_cost < self.big_num:
                return True, min_cost
            return False, min_cost
        # if w_list is not null
        # classify by skills
        full_list = [[w_list[0]]]
        for w in w_list[1:]:
            meet = False
            for i in range(len(full_list)):
                if same_skills(w.skills, full_list[i][0].skills):
                    full_list[i].append(w)
                    meet = True
                    break
            if not meet:
                full_list.append([w])
        # generate index
        idx_list = []
        for wl in full_list:
            il = [-1]
            for idx in range(len(wl)):
                il.append(idx)
            idx_list.append(il)
        # generate combinations and test
        groups = list(product(*idx_list))
        dist_dict = {}
        for g in groups:
            sub_w = []
            assert len(g) == len(full_list)
            # generate a combination
            for idx, wl in zip(g, full_list):
                if idx > -1:
                    sub_w.append(wl[idx])
            # skills
            if not check_skills(t, sub_w):
                continue
            # cost
            cost = 0.0
            max_tt = 0.0
            for w in sub_w:
                if w in dist_dict:
                    dist = dist_dict[w]
                else:
                    dist = get_dist(t, w)
                    dist_dict[w] = dist
                cost = cost + w.umc * dist
                travel_time = dist / w.speed * 60.0
                if travel_time > max_tt:
                    max_tt = travel_time
            if cost > t.budget:  # cost check
                continue
            if cost < min_cost:
                min_cost = cost
                t.wb = t.budget - min_cost
                t.begin_time = self.cur_time
                t.eft = self.cur_time + max_tt + t.exetime
                t.workers = set(sub_w)
        # return
        if min_cost < self.big_num:
            return True, min_cost
        return False, min_cost

    def check_ing(self):
        idx = 0
        while idx < len(self.ing_tasks):
            t = self.ing_tasks[idx]
            fail = False
            if t.eft <= self.cur_time:
                # deal workers
                slot = self.cur_time - t.eft
                for w in t.workers:
                    w.lon = t.lon + random.uniform(-1.0, 1.0) * w.speed * slot / 60.0 / 85.0
                    w.lat = t.lat + random.uniform(-1.0, 1.0) * w.speed * slot / 60.0 / 111.0
                    w.lon = max(115.42, w.lon)
                    w.lon = min(w.lon, 117.51)
                    w.lat = max(39.44, w.lat)
                    w.lat = min(w.lat, 41.06)
                    if random.uniform(0.0, 1.0) < self.cfg['p_worker_leave']:  # worker leave
                        self.leave_workers.append(w)
                    else:
                        self.available_workers.append(w)
                # deal task
                if not t.has_check_fail:
                    for w in t.workers:
                        if random.uniform(0.0, 1.0) < w.pf:
                            fail = True
                            break
                if fail:
                    self.num_expired += 1
                else:
                    self.num_completed += 1
                    self.total_benefit += t.wb
                del self.ing_tasks[idx]
            else:
                if not t.has_check_fail and random.uniform(0.0, 1.0) < self.cfg['check_prob']:
                    for w in t.workers:
                        if random.uniform(0.0, 1.0) < w.pf:
                            fail = True
                            break
                    t.has_check_fail = True
                if fail:
                    # deal workers
                    fail_pct = (self.cur_time - t.begin_time) / (t.eft - t.begin_time)
                    for w in t.workers:
                        w.lon = fail_pct * (t.lon - w.lon) + w.lon
                        w.lat = fail_pct * (t.lat - w.lat) + w.lat
                        w.lon = max(115.42, w.lon)
                        w.lon = min(w.lon, 117.51)
                        w.lat = max(39.44, w.lat)
                        w.lat = min(w.lat, 41.06)
                        if random.uniform(0.0, 1.0) < self.cfg['p_worker_leave']:  # worker leave
                            self.leave_workers.append(w)
                        else:
                            self.available_workers.append(w)
                    # deal task
                    if self.cur_time < t.deadline:
                        t.at = self.cur_time
                        t.workers = set()
                        t.has_ass = False
                        self.resubmit(t)
                    else:
                        self.num_expired += 1
                    del self.ing_tasks[idx]
                else:
                    idx += 1

    def check_rsq(self):
        idx = 0
        while idx < len(self.resubmit_tasks):
            t = self.resubmit_tasks[idx]
            if t.deadline <= self.cur_time:
                del self.resubmit_tasks[idx]
                self.num_expired += 1
            else:
                idx += 1

    def worker_leave(self):
        idx = 0
        while idx < len(self.available_workers):
            if random.uniform(0.0, 1.0) < self.cfg['p_worker_leave']:
                self.leave_workers.append(self.available_workers[idx])
                del self.available_workers[idx]
            else:
                idx += 1

    def worker_arrive(self):
        idx = 0
        while idx < len(self.leave_workers):
            if random.uniform(0.0, 1.0) < self.cfg['p_worker_arrive']:
                w = self.leave_workers[idx]
                # w.lon = w.lon + random.uniform(-1.0, 1.0) * 0.141
                # w.lat = w.lat + random.uniform(-1.0, 1.0) * 0.108
                self.available_workers.append(w)
                del self.leave_workers[idx]
            else:
                idx += 1

    def get_obs(self):
        self.num_find += 1
        if self.num_find % 100 == 0:
            self.worker_leave()
            self.worker_arrive()
        done, t_list, w_list = self.find_mn()
        obs = None
        masks = None
        while not done:
            if t_list:
                obs = self.gen_obs(t_list, w_list)
                self.normalize_obs(obs)
                masks = gen_masks(t_list, w_list)
                break
            done, t_list, w_list = self.find_mn()
        return done, t_list, w_list, obs, masks

    def step(self, actions, t_list, w_list):
        assert len(t_list) > 0
        total_value = 0.0
        for t in t_list:
            t.workers = set()
            total_value += t.budget
        for idx_w, w in enumerate(w_list):
            for idx_t, t in enumerate(t_list):
                if actions[idx_w] == idx_t:
                    t.workers.add(w)
        num_ass = 0.0
        total_tasks = len(t_list)

        w_benefit = 0.0
        w_set = set()
        while len(t_list) > 0:
            t = t_list[0]
            find, _ = self.select_workers(list(t.workers), t)
            if not find:
                t.ass_num += 1
                if t.ass_num <= 5:
                    t.has_ass = False
                    self.resubmit(t)
                else:
                    self.num_expired += 1
            else:
                num_ass += 1.0
                w_benefit += t.wb
                for w in t.workers:
                    w_set.add(w)
                self.ing_tasks.append(t)
            del t_list[0]
        for w in w_set:
            w_idx = 0
            while w_idx < len(self.available_workers):
                if self.available_workers[w_idx].wid == w.wid:
                    break
                w_idx += 1
            assert w_idx < len(self.available_workers)
            del self.available_workers[w_idx]
        reward = self.cfg['alpha'] * (num_ass / total_tasks) + (1 - self.cfg['alpha']) * (w_benefit / total_value) - 1.0
        return reward
