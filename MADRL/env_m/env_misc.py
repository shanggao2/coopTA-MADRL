import math
import numpy as np
from itertools import product


def is_cover_all(t_skills, w_skills):
    assert len(t_skills) == len(w_skills)
    for i in range(0, len(t_skills)):
        if t_skills[i] == '1' and w_skills[i] == '0':
            return False
    return True


def is_cover_part(t_skills, w_skills):
    assert len(t_skills) == len(w_skills)
    is_need_skill = False
    for i in range(0, len(t_skills)):
        if t_skills[i] == '1':
            is_need_skill = True
            if w_skills[i] == '1':
                return True
    if is_need_skill is False:
        return True
    return False


def get_dist(t, w):
    return math.sqrt(math.pow((t.lon - w.lon) * 85.0, 2) + math.pow((t.lat - w.lat) * 111.0, 2))


def get_cost(t, w):
    dist = get_dist(t, w)
    return w.umc * dist


def check_cost(t, w_list, cost_dict):
    total_cost = 0.0
    for w in w_list:
        if w in cost_dict:
            cost = cost_dict[w]
        else:
            cost = get_cost(t, w)
            cost_dict[w] = cost
        total_cost += cost
    return total_cost < t.budget


def check_skills(t, w_list):
    for idx, ts in enumerate(t.skills):
        if ts == '1':
            meet = False
            for w in w_list:
                if w.skills[idx] == '1':
                    meet = True
                    break
            if not meet:
                return False
    return True


def need_skills(skills):
    for s in skills:
        if s == '1':
            return True
    return False


def same_set(t, w_set):
    assert len(t.record.list_wid) == len(t.record.list_lon) == len(t.record.list_lat)
    if not w_set or len(t.record.list_wid) != len(w_set):
        return False
    curr = list(w_set)
    curr = sorted(curr, key=lambda ww: ww.wid)
    r = t.record
    for idx, w in enumerate(curr):
        if r.list_wid[idx] != w.wid or abs(r.list_lon[idx] - w.lon) > 1e-6 or abs(r.list_lat[idx] - w.lat) > 1e-6:
            return False
    return True


def save_record(t, w_set):
    curr = list(w_set)
    curr = sorted(curr, key=lambda ww: ww.wid)
    t.record.list_wid = []
    t.record.list_lon = []
    t.record.list_lat = []
    for w in curr:
        t.record.list_wid.append(w.wid)
        t.record.list_lon.append(w.lon)
        t.record.list_lat.append(w.lat)


def same_skills(skills1, skills2):
    for s1, s2 in zip(skills1, skills2):
        if s1 != s2:
            return False
    return True


def second_check(t, w_set):
    # if hit record, return record, else, record current info
    if same_set(t, w_set):
        return t.record.set_res
    save_record(t, w_set)
    the_same = False
    # select workers who can perform tasks independently
    temp_w_set = set()
    tmp_w_list = []
    for w in w_set:
        if is_cover_all(t.skills, w.skills):
            temp_w_set = temp_w_set | {w}
            tmp_w_list.append(w)
    # remove has added
    for w in tmp_w_list:
        w_set.remove(w)
    # remove uncooperative workers
    tmp_w_list = []
    for w in w_set:
        if w.coo == '0':
            tmp_w_list.append(w)
    for w in tmp_w_list:
        w_set.remove(w)
    # if w_set is null, no need continue
    if not w_set:
        if not the_same:
            t.record.set_res = set(temp_w_set)
        return temp_w_set
    # else, keep checking
    w_list = list(w_set)
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
    # generate combinations and check
    groups = list(product(*idx_list))
    cost_dict = {}
    for g in groups:
        sw_list = []
        assert len(g) == len(full_list)
        for idx, wl in zip(g, full_list):
            if idx > -1:
                sw_list.append(wl[idx])
        if set(sw_list).issubset(temp_w_set) or not sw_list:
            continue
        if check_cost(t, sw_list, cost_dict) and check_skills(t, sw_list):
            temp_w_set = temp_w_set | set(sw_list)
        if len(temp_w_set) == len(w_list):
            break
    # return
    if not the_same:
        t.record.set_res = set(temp_w_set)
    return temp_w_set


def gen_masks(task_list, worker_list):
    masks = np.zeros((len(worker_list), len(task_list)), dtype=int)
    for w_idx, w in enumerate(worker_list):
        for t_idx, t in enumerate(task_list):
            if w in t.workers:
                masks[w_idx][t_idx] = 1.0
    return masks
