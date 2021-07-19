from env_m.record import Record


class Task(object):
    def __init__(self, tid, at, lon, lat, budget, coo, ttl, deadline, exetime, skills):
        self.tid = tid
        self.at = at
        self.lon = lon
        self.lat = lat
        self.budget = budget
        self.coo = coo
        self.ttl = ttl
        self.deadline = deadline
        self.exetime = exetime
        self.skills = skills
        self.workers = set()
        self.begin_time = 0.0
        self.eft = 0.0  # estimated finish time
        self.wb = 0.0  # workers' benefit
        self.has_ass = False
        self.ass_num = 0
        self.has_check_fail = False
        self.record = Record()
