import numpy as np
from numba import njit



@njit(cache=True)
def calculate_speed(delta):
    b = 0.523
    g = 9.81
    l_d = 0.329
    f_s = 0.5
    max_v = 5

    if abs(delta) < 0.06:
        return max_v

    V = f_s * np.sqrt(b*g*l_d/np.tan(abs(delta)))

    return V


@njit(cache=True)
def calculate_safe_speed(delta, range_val):
    b = 0.523
    g = 9.81
    l_d = 0.329
    f_s = 0.5
    max_v = 6
    max_a = 8 * 0.8 # for safety factor
    min_range_for_max_v = max_v ** 2 / (2*max_a) 

    if range_val < min_range_for_max_v:
        max_v = range_val / min_range_for_max_v * max_v

    if abs(delta) < 0.06:
        return max_v

    V = f_s * np.sqrt(b*g*l_d/np.tan(abs(delta)))

    return V