
import numpy as np 
import csv

import RewardSignalDesign.LibFunctions as lib


class SteeringReward:
    def __init__(self, b_s):
        self.b_s = b_s
        self.name = f"Steering({b_s})"
        
    def __call__(self, state, s_prime):
        s_prime['reward'] = find_reward(s_prime)
        scaled_steering = abs(s_prime['steering_deltas'][0]/0.4)
        reward = (0.5 - scaled_steering) * self.b_s
        reward += s_prime['reward']

        return reward

def find_reward(s_p):
    if s_p['collisions'][0] == 1:
        return -1
    elif s_p['lap_counts'][0] == 1:
        return 1
    return 0


def find_closest_pt(pt, wpts):
    """
    Returns the two closes points in order along wpts
    """
    dists = [lib.get_distance(pt, wpt) for wpt in wpts]
    min_i = np.argmin(dists)
    d_i = dists[min_i] 
    if min_i == len(dists) - 1:
        min_i -= 1
    if dists[max(min_i -1, 0) ] > dists[min_i+1]:
        p_i = wpts[min_i]
        p_ii = wpts[min_i+1]
        d_i = dists[min_i] 
        d_ii = dists[min_i+1] 
    else:
        p_i = wpts[min_i-1]
        p_ii = wpts[min_i]
        d_i = dists[min_i-1] 
        d_ii = dists[min_i] 

    return p_i, p_ii, d_i, d_ii

def get_tiangle_h(a, b, c):
    s = (a + b+ c) / 2
    A = np.sqrt(s*(s-a)*(s-b)*(s-c))
    h = 2 * A / c

    return h

def distance_potential(s, s_p, end, beta=0.2, scale=0.5):
    prev_dist = lib.get_distance(s[0:2], end)
    cur_dist = lib.get_distance(s_p[0:2], end)
    d_dis = (prev_dist - cur_dist) / scale

    return d_dis * beta




# Track base
class TrackPtsBase:
    def __init__(self, config) -> None:
        self.wpts = None
        self.ss = None
        self.map_name = config.map_name
        self.total_s = None

    def load_center_pts(self):
        track_data = []
        filename = 'maps/' + self.map_name + '_std.csv'
        
        try:
            with open(filename, 'r') as csvfile:
                csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
        
                for lines in csvFile:  
                    track_data.append(lines)
        except FileNotFoundError:
            raise FileNotFoundError("No map file center pts")

        track = np.array(track_data)
        print(f"Track Loaded: {filename} in reward")

        N = len(track)
        self.wpts = track[:, 0:2]
        ss = np.array([lib.get_distance(self.wpts[i], self.wpts[i+1]) for i in range(N-1)])
        ss = np.cumsum(ss)
        self.ss = np.insert(ss, 0, 0)

        self.total_s = self.ss[-1]

        self.diffs = self.wpts[1:,:] - self.wpts[:-1,:]
        self.l2s   = self.diffs[:,0]**2 + self.diffs[:,1]**2 

    def load_reference_pts(self):
        track_data = []
        filename = 'maps/' + self.map_name + '_opti.csv'
        
        try:
            with open(filename, 'r') as csvfile:
                csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
        
                for lines in csvFile:  
                    track_data.append(lines)
        except FileNotFoundError:
            raise FileNotFoundError("No reference path")

        track = np.array(track_data)
        print(f"Track Loaded: {filename} in reward")

        self.ss = track[:, 0]
        self.wpts = track[:, 1:3]

        self.total_s = self.ss[-1]

        self.diffs = self.wpts[1:,:] - self.wpts[:-1,:]
        self.l2s   = self.diffs[:,0]**2 + self.diffs[:,1]**2 

    def find_s(self, point):
        dots = np.empty((self.wpts.shape[0]-1, ))
        for i in range(dots.shape[0]):
            dots[i] = np.dot((point - self.wpts[i, :]), self.diffs[i, :])
        t = dots / self.l2s

        t = np.clip(dots / self.l2s, 0.0, 1.0)
        projections = self.wpts[:-1,:] + (t*self.diffs.T).T
        dists = np.linalg.norm(point - projections, axis=1)

        min_dist_segment = np.argmin(dists)
        dist_from_cur_pt = dists[min_dist_segment]
        if dist_from_cur_pt > 1: #more than 2m from centerline
            return self.ss[min_dist_segment] - dist_from_cur_pt # big makes it go back

        s = self.ss[min_dist_segment] + dist_from_cur_pt

        return s 

    def get_distance_r(self, pt1, pt2, beta):
        s = self.find_s(pt1)
        ss = self.find_s(pt2)
        ds = ss - s
        scale_ds = ds / self.total_s
        r = scale_ds * beta
        shaped_r = np.clip(r, -0.5, 0.5)

        return shaped_r


class RefDistanceReward(TrackPtsBase):
    def __init__(self, config, b_distance) -> None:
        TrackPtsBase.__init__(self, config)

        self.load_reference_pts()
        self.b_distance = b_distance

    def __call__(self, state, s_prime):
        s_prime['reward'] = find_reward(s_prime)
        prime_pos = np.array([s_prime['poses_x'][0], s_prime['poses_y'][0]])
        pos = np.array([state['poses_x'][0], state['poses_y'][0]])
        reward = self.get_distance_r(pos, prime_pos, 1)

        reward += s_prime['reward']

        return reward

class RefCTHReward(TrackPtsBase):
    def __init__(self, conf, mh, md) -> None:
        TrackPtsBase.__init__(self, conf)
        self.max_v = conf.max_v
        self.dis_scale = 1

        self.load_reference_pts()
        self.mh = mh 
        self.md = md 

    def __call__(self, state, s_prime):
        s_prime['reward'] = find_reward(s_prime)
        prime_pos = np.array([s_prime['poses_x'][0], s_prime['poses_y'][0]])
        theta = s_prime['poses_theta'][0]
        velocity = s_prime['linear_vels_x'][0]

        pt_i, pt_ii, d_i, d_ii = find_closest_pt(prime_pos, self.wpts)
        d = lib.get_distance(pt_i, pt_ii)
        d_c = get_tiangle_h(d_i, d_ii, d) / self.dis_scale

        th_ref = lib.get_bearing(pt_i, pt_ii)
        th = theta
        d_th = abs(lib.sub_angles_complex(th_ref, th))
        v_scale = velocity / self.max_v

        new_r =  self.mh * np.cos(d_th) * v_scale - self.md * d_c

        return new_r + s_prime['reward']