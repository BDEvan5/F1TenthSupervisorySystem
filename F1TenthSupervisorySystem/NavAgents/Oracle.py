import numpy as np
from numba import njit
from matplotlib import pyplot as plt
import csv

import LearningLocalPlanning.LibFunctions as lib

from F1TenthSupervisorySystem.Utils.speed_utils import calculate_speed
from F1TenthSupervisorySystem.Utils import pure_pursuit_utils


class Oracle:
    def __init__(self, sim_conf) -> None:
        self.name = "Oracle Path Follower"
        self.path_name = None

        self.wheelbase = sim_conf.l_f + sim_conf.l_r

        self.v_gain = sim_conf.v_gain
        self.lookahead = sim_conf.lookahead
        self.max_reacquire = 20

        self.waypoints = None
        self.vs = None
        self.v_min_plan = sim_conf.v_min_plan

        self.aim_pts = []
        self.plan_track(sim_conf.map_name)

    def _get_current_waypoint(self, position):
        lookahead_distance = self.lookahead
    
        wpts = np.vstack((self.waypoints[:, 0], self.waypoints[:, 1])).T
        nearest_point, nearest_dist, t, i = pure_pursuit_utils.nearest_point_on_trajectory_py2(position, wpts)
        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = pure_pursuit_utils.first_point_on_trajectory_intersecting_circle(position, lookahead_distance, wpts, i+t, wrap=True)
            if i2 == None:
                return None
            current_waypoint = np.empty((3, ))
            # x, y
            current_waypoint[0:2] = wpts[i2, :]
            # speed
            current_waypoint[2] = self.waypoints[i, 2]
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return np.append(wpts[i, :], self.waypoints[i, 2])
        else:
            return None


    def extract_state(self, obs):
        ego_idx = obs['ego_idx']
        pose_th = obs['poses_theta'][ego_idx] 
        p_x = obs['poses_x'][ego_idx]
        p_y = obs['poses_y'][ego_idx]
        velocity = obs['linear_vels_x'][ego_idx]
        delta = obs['steering_deltas'][ego_idx]

        state = np.array([p_x, p_y, pose_th, velocity, delta])

        return state


    def plan(self, obs):
        if obs['linear_vels_x'][0] < self.v_min_plan:
            return np.array([0, 7])

        state = self.extract_state(obs)
        pose_th = state[2]
        pos = np.array(state[0:2], dtype=np.float)

        lookahead_point = self._get_current_waypoint(pos)

        self.aim_pts.append(lookahead_point[0:2])

        if lookahead_point is None:
            return [0, 4.0]

        speed, steering_angle = pure_pursuit_utils.get_actuation(pose_th, lookahead_point, pos, self.lookahead, self.wheelbase)

        # speed = calculate_speed(steering_angle)
        # speed = 2


        return np.array([steering_angle, speed])

    def plan_track(self, map_name):
        track = []
        filename = 'maps/' + map_name + "_opti.csv"
        with open(filename, 'r') as csvfile:
            csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
        
            for lines in csvFile:  
                track.append(lines)

        track = np.array(track)
        print(f"Track Loaded: {filename}")

        wpts = track[:, 1:3]
        vs = track[:, 5]

        self.waypoints = np.concatenate([wpts, vs[:, None]], axis=-1)
        self.expand_wpts()

        return self.waypoints[:, 0:2]

    def expand_wpts(self):
        n = 5 # number of pts per orig pt
        dz = 1 / n
        o_line = self.waypoints[:, 0:2]
        o_vs = self.waypoints[:, 2]
        new_line = []
        new_vs = []
        for i in range(len(self.waypoints)-1):
            dd = lib.sub_locations(o_line[i+1], o_line[i])
            for j in range(n):
                pt = lib.add_locations(o_line[i], dd, dz*j)
                new_line.append(pt)

                dv = o_vs[i+1] - o_vs[i]
                new_vs.append(o_vs[i] + dv * j * dz)

        wpts = np.array(new_line)
        vs = np.array(new_vs)
        self.waypoints = np.concatenate([wpts, vs[:, None]], axis=-1)

    def plot_plan(self):
        plt.figure(1)
        plt.plot(self.waypoints[:, 0], self.waypoints[:, 1], 'x-', markersize=16)
        plt.gca().set_aspect('equal')

        # plt.show()
        plt.pause(0.0001)

        
