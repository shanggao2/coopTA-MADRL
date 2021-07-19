import json
import csv
import math
import random
import numpy as np
from env_s.task import Task
from env_s.worker import Worker
from env_s.env_misc import is_cover_all, normalize_state
from itertools import combinations


class Environment(object):
    def __init__(self, is_train, cfg_file_name):
        self.last_time = 0.0
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
        self.max_n = 0
        self.num_skills = 0
        self.num_published = 0
        self.num_completed = 0
        self.num_expired = 0
        self.total_value = 0.0
        self.total_benefit = 0.0
        self.num_find = 0
        self.is_train = is_train
        self.cfg_file_name = cfg_file_name
        # You need to modify the path according to your own settings.
        self.base_path = '/home/jlu/repo/'

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
        skills_num_dict = {'real': 7, 'syn': 5}
        self.max_n = self.cfg['max_n']
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

    def check_ing(self):
        idx = 0
        while idx < len(self.ing_tasks):
            t = self.ing_tasks[idx]
            w = t.worker
            fail = False
            if t.eft <= self.cur_time:  # completed
                # deal worker
                slot = self.cur_time - t.eft
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
                    fail = random.uniform(0.0, 1.0) < w.pf
                if fail:
                    self.num_expired += 1
                else:
                    self.num_completed += 1
                    self.total_benefit += t.wb
                del self.ing_tasks[idx]
            else:
                # simulate whether the task fails
                if not t.has_check_fail and random.uniform(0.0, 1.0) < self.cfg['check_prob']:
                    fail = random.uniform(0.0, 1.0) < w.pf
                    t.has_check_fail = True
                if fail:
                    # deal worker
                    fail_pct = (self.cur_time - t.begin_time) / (t.eft - t.begin_time)
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
                        t.worker = None
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

    def find_task_in_rsq(self):
        for t in self.resubmit_tasks:
            if not t.has_ass:
                return t
        return None

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

    def condition_check(self, t, w, r):
        dist = math.sqrt(math.pow((t.lon - w.lon) * 85.0, 2) + math.pow((t.lat - w.lat) * 111.0, 2))
        if dist > r:
            return False
        if dist / w.speed * 60.0 > t.deadline - self.cur_time:  # t.ttl
            return False
        if w.umc * dist > t.budget:
            return False
        if not is_cover_all(t.skills, w.skills):
            return False
        return True

    def find_workers(self, t):
        w_list = []
        for w in self.available_workers:
            if self.condition_check(t, w, self.cfg['task_radius']):
                w_list.append(w)
        return w_list

    def swfst(self, t, w_list):
        c_list = []
        for w in w_list:
            dist = math.sqrt(math.pow((t.lon - w.lon) * 85.0, 2) + math.pow((t.lat - w.lat) * 111.0, 2))
            cost = w.umc * dist
            c_list.append(cost)
        c_list_idx = sorted(range(len(c_list)), key=lambda k: c_list[k])
        tmp_w_list = []
        for i in range(self.max_n):
            tmp_w_list.append(w_list[c_list_idx[i]])
        return tmp_w_list

    def search_in_rsq(self, t):
        idx = 0
        find = False
        while idx < len(self.resubmit_tasks):
            if self.resubmit_tasks[idx].tid == t.tid:
                find = True
                break
            idx += 1
        return find, idx

    def find_n(self):
        for t in self.resubmit_tasks:
            t.has_ass = False
        w_list = []
        while True:
            t, from_data = self.select_one_task()
            done = not self.file_task and not self.resubmit_tasks and not self.ing_tasks
            if not t:
                return done, t, w_list
            w_list = self.find_workers(t)
            if not w_list:
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
            elif len(w_list) > self.max_n:
                w_list = self.swfst(t, w_list)
            if from_data:
                self.file_task = self.gen_one_task()
                if self.file_task:
                    self.num_published += 1
                    self.total_value += self.file_task.budget
            else:
                find, idx = self.search_in_rsq(t)
                assert find
                del self.resubmit_tasks[idx]
            if self.file_task and self.cur_time < self.file_task.at:
                self.cur_time = self.file_task.at
            return done, t, w_list

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
                # self.available_workers.append(self.leave_workers[idx])
                del self.leave_workers[idx]
            else:
                idx += 1

    def gen_state(self, t, w_list):
        state_all = []
        for w in w_list:
            state = [1.0, w.pf]
            dist = math.sqrt(math.pow((t.lon - w.lon) * 85.0, 2) + math.pow((t.lat - w.lat) * 111.0, 2))
            cost = w.umc * dist
            state.append(cost)
            travel_time = dist / w.speed * 60.0
            state.append(travel_time)
            for s in w.skills:
                if s == '1':
                    state.append(1.0)
                else:
                    state.append(0.0)
            state_all.append(state)
        for i in range(len(w_list), self.max_n):
            state = [0.0]
            pf = random.gauss(0.1, 0.03)
            state.append(format(pf, '.2f'))
            idx = random.randint(0, len(w_list) - 1)
            cost = state_all[idx][2] * random.uniform(0.9, 1.1)
            state.append(cost)
            idx = random.randint(0, len(w_list) - 1)
            travel_time = state_all[idx][3] * random.uniform(0.9, 1.1)
            state.append(travel_time)
            for _ in range(self.num_skills):
                state.append(0.0)
            state_all.append(state)
        state_all = np.array(state_all, dtype=float)
        return state_all

    def get_state(self):
        self.num_find += 1
        if self.num_find % 100 == 0:
            self.worker_leave()
            self.worker_arrive()
        done, t, w_list = self.find_n()
        state = None
        while not done:
            if t:
                state = self.gen_state(t, w_list)
                normalize_state(state)
                state = state.reshape((1, -1))
                break
            done, t, w_list = self.find_n()
        return done, t, w_list, state

    def step(self, action, t, w_list):
        if action >= len(w_list):
            if t.daelline <= self.cur_time:
                t.at = self.cur_time
                t.worker = None
                t.has_ass = False
                self.resubmit(t)
            else:
                self.num_expired += 1
            return 0.0
        w = w_list[action]
        t.worker = w
        dist = math.sqrt(math.pow((t.lon - w.lon) * 85.0, 2) + math.pow((t.lat - w.lat) * 111.0, 2))
        cost = w.umc * dist
        t.wb = t.budget - cost
        t.begin_time = self.cur_time
        travel_time = dist / w.speed * 60.0
        t.eft = self.cur_time + travel_time + t.exetime
        self.ing_tasks.append(t)
        # deal worker
        w_idx = 0
        while w_idx < len(self.available_workers):
            if self.available_workers[w_idx].wid == w.wid:
                break
            w_idx += 1
        assert w_idx < len(self.available_workers)
        del self.available_workers[w_idx]
        # get reward and return
        reward = self.cfg['alpha'] + (1 - self.cfg['alpha']) * (t.wb / t.budget) - 1
        return reward
