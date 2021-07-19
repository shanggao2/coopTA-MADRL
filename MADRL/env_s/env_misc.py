def is_cover_all(t_skills, w_skills):
    assert len(t_skills) == len(w_skills)
    for t_s, w_s in zip(t_skills, w_skills):
        if t_s == '1' and w_s == '0':
            return False
    return True


def normalize_state(arr):
    for i in range(1, 4):
        arr_t = arr[:, i]
        t_max = arr_t.max(axis=0)
        t_min = arr_t.min(axis=0)
        if t_max - t_min > 1e-3:
            arr_t = (arr_t - t_min) / (t_max - t_min)
            arr[:, i] = arr_t
        else:
            arr[:, i] = 0.5
