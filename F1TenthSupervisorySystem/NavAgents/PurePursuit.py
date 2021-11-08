from matplotlib import pyplot as plt
from F1TenthSupervisorySystem.Utils.Trajectory import Trajectory 
import numpy as np
from F1TenthSupervisorySystem.Utils import pure_pursuit_utils as pp_utils


class PurePursuit:
    def __init__(self, conf):
        self.name = "PurePursuit"
        

        self.trajectory = Trajectory(conf.map_name)

        self.lookahead = conf.lookahead
        self.vgain = conf.v_gain
        self.wheelbase =  conf.l_f + conf.l_r
        self.max_steer = conf.max_steer

        self.progresses = []

    def plan_act(self, obs):
        return self.act(obs)

    def act(self, obs):
        ego_idx = obs['ego_idx']
        pose_th = obs['poses_theta'][ego_idx] 
        p_x = obs['poses_x'][ego_idx]
        p_y = obs['poses_y'][ego_idx]
        v_current = obs['linear_vels_x'][ego_idx]

        self.progresses.append(obs['progresses'][0])

        pos = np.array([p_x, p_y], dtype=np.float)

        v_min_plan = 1
        if v_current < v_min_plan:
            return np.array([[0, 7]])

        lookahead_point = self.trajectory.get_current_waypoint(pos, self.lookahead)

        speed, steering_angle = pp_utils.get_actuation(pose_th, lookahead_point, pos, self.lookahead, self.wheelbase)
        steering_angle = np.clip(steering_angle, -self.max_steer, self.max_steer)
        speed *= self.vgain

        # speed = calculate_speed(steering_angle)

        return np.array([[steering_angle, speed]])

    def plot_progress(self):
        plt.figure(2)
        plt.plot(self.progresses)

        plt.show()


