class Worker(object):
    def __init__(self, wid, lon, lat, pf, coo, umc, speed, skills):
        self.wid = wid
        self.lon = lon
        self.lat = lat
        self.pf = pf  # probability of failure
        self.coo = coo  # willing to cooperate
        self.umc = umc  # unit mileage cost
        self.speed = speed
        self.skills = skills
