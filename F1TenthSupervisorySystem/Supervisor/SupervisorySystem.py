
import numpy as np
from numba import njit
from matplotlib import pyplot as plt
import yaml
from F1TenthSupervisorySystem.Supervisor.Dynamics import update_complex_state

class Supervisor:
    def __init__(self, planner, kernel, conf):
        """
        A wrapper class that can be used with any other planner.
        Requires a planner with:
            - a method called 'plan_act' that takes a state and returns an action

        """
        
        #TODO: make sure these parameters are defined in the planner an then remove them here. This is constructor dependency injection
        self.d_max = conf.max_steer
        self.v = 2
        # self.kernel = ForestKernel(conf)
        self.kernel = kernel
        self.planner = planner

        # aliases for the test functions
        try:
            self.n_beams = planner.n_beams
        except: pass
        self.plan_act = self.plan
        self.name = planner.name


        self.action = None
        self.loop_counter = 0 
        self.plan_f = conf.plan_frequency
        self.v_min_plan = conf.v_min_plan

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
        if self.action is None or self.loop_counter == self.plan_f:
            self.loop_counter = 0
            self.calculate_action(obs)
        self.loop_counter += 1
        return self.action

    def calculate_action(self, obs):
        init_action = self.planner.plan_act(obs)
        init_action[1] = self.v
        state = self.extract_state(obs)
        safe, next_state = check_init_action(state, init_action, self.kernel)
        if safe:
            self.action = init_action
            return

        dw = self.generate_dw()
        valids = simulate_and_classify(state, dw, self.kernel)
        if not valids.any():
            print('No Valid options')
            print(f"State: {state}")
            self.action = init_action
            return
        
        self.action = modify_action(valids, dw)
        # print(f"Valids: {valids} -> new action: {action}")


    def generate_dw(self):
        n_segments = 5
        dw = np.ones((5, 2))
        dw[:, 0] = np.linspace(-self.d_max, self.d_max, n_segments)
        dw[:, 1] *= self.v
        return dw

#TODO jit all of this.

def check_init_action(state, u0, kernel, time_step=0.2):
    next_state = update_complex_state(state, u0, time_step)
    # next_state = update_std_state(state, u0, 0.2)
    safe = kernel.check_state(next_state)
    
    return safe, next_state

def simulate_and_classify(state, dw, kernel, time_step=0.2):
    valid_ds = np.ones(len(dw))
    for i in range(len(dw)):
        next_state = update_complex_state(state, dw[i], time_step)
        # next_state = update_std_state(state, dw[i], 0.2)
        safe = kernel.check_state(next_state)
        valid_ds[i] = safe 

        # print(f"State: {state} + Action: {dw[i]} --> Expected: {next_state}  :: Safe: {safe}")

    return valid_ds 


class LearningSupervisor(Supervisor):
    def __init__(self, planner, kernel, conf):
        Supervisor.__init__(self, planner, kernel, conf)
        self.intervention_mag = 0
        self.intervene = False



    def intervene_reward(self):
        if self.intervene:
            self.intervene = False
            return -0.5
        return 0

    def magnitude_reward(self): #TODO: Implement this
        if self.intervene:
            self.intervene = False
            return - abs(self.intervention_mag)
        return 0

    def zero_reward(self):
        return 0

    def calculate_reward(self):
        # return self.zero_reward()
        return self.magnitude_reward()
        # return self.intervene_reward()

    def done_entry(self, s_prime):
        s_prime['reward'] = self.calculate_reward()
        self.planner.done_entry(s_prime)

    def calculate_action(self, obs):
        obs['reward'] = self.calculate_reward() # check this works.
        init_action = self.planner.plan_act(obs)
        state = self.extract_state(obs)

        safe, next_state = check_init_action(state, init_action, self.kernel)
        if safe:
            # self.safe_history.add_locations(init_action[0], init_action[0])
            self.action = init_action
            return

        self.intervene = True

        dw = self.generate_dw()
        valids = simulate_and_classify(state, dw, self.kernel)
        if not valids.any():
            print(f'No Valid options: state : {state}')
            self.action = init_action
            return
        
        self.action = modify_action(valids, dw)
        # print(f"Valids: {valids} -> new action: {action}")
        # self.safe_history.add_locations(init_action[0], action[0])

        self.intervention_mag = self.action[0] - init_action[0]


@njit(cache=True)
def modify_action(valid_window, dw):
    """ 
    By the time that I get here, I have already established that pp action is not ok so I cannot select it, I must modify the action. 
    """
    idx_search = int((len(dw)-1)/2)
    d_size = len(valid_window)
    for i in range(d_size):
        p_d = int(min(d_size-1, idx_search+i))
        if valid_window[p_d]:
            return dw[p_d]
        n_d = int(max(0, idx_search-i))
        if valid_window[n_d]:
            return dw[n_d]

    

class BaseKernel:
    def __init__(self, sim_conf):
        self.resolution = sim_conf.n_dx

    def view_kernel(self, theta):
        phi_range = np.pi
        theta_ind = int(round((theta + phi_range/2) / phi_range * (self.kernel.shape[2]-1)))
        plt.figure(5)
        plt.title(f"Kernel phi: {theta} (ind: {theta_ind})")
        img = self.kernel[:, :, theta_ind].T 
        plt.imshow(img, origin='lower')

        # plt.show()
        plt.pause(0.0001)

    def check_state(self, state=[0, 0, 0]):
        i, j, k = self.get_indices(state)

        # print(f"Expected Location: {state} -> Inds: {i}, {j}, {k} -> Value: {self.kernel[i, j, k]}")
        # self.plot_kernel_point(i, j, k)
        if self.kernel[i, j, k] == 1:
            return False # unsfae state
        return True # safe state

    def plot_kernel_point(self, i, j, k):
        plt.figure(5)
        plt.clf()
        plt.title(f"Kernel inds: {i}, {j}, {k}")
        img = self.kernel[:, :, k].T 
        plt.imshow(img, origin='lower')
        plt.plot(i, j, 'x', markersize=20, color='red')
        # plt.show()
        plt.pause(0.0001)

class ForestKernel(BaseKernel):
    def __init__(self, sim_conf):
        super().__init__(sim_conf)
        self.kernel = None
        self.side_kernel = np.load(f"{sim_conf.kernel_path}SideKernel_{sim_conf.kernel_name}.npy")
        self.obs_kernel = np.load(f"{sim_conf.kernel_path}ObsKernel_{sim_conf.kernel_name}.npy")
        img_size = int(sim_conf.obs_img_size * sim_conf.n_dx)
        obs_size = int(sim_conf.obs_size * sim_conf.n_dx)
        self.obs_offset = int((img_size - obs_size) / 2)

    def construct_kernel(self, track_size, obs_locations):
        self.kernel = construct_forest_kernel(track_size, obs_locations, self.resolution, self.side_kernel, self.obs_kernel, self.obs_offset)

    def get_indices(self, state):
        phi_range = np.pi
        x_ind = min(max(0, int(round((state[0])*self.resolution))), self.kernel.shape[0]-1)
        y_ind = min(max(0, int(round((state[1])*self.resolution))), self.kernel.shape[1]-1)
        theta_ind = int(round((state[2] + phi_range/2) / phi_range * (self.kernel.shape[2]-1)))
        theta_ind = min(max(0, theta_ind), self.kernel.shape[2]-1)

        return x_ind, y_ind, theta_ind


@njit(cache=True)
def construct_forest_kernel(track_size, obs_locations, resolution, side_kernel, obs_kernel, obs_offset):
    kernel = np.zeros((track_size[0], track_size[1], side_kernel.shape[2]))
    length = int(track_size[1] / resolution)
    for i in range(length):
        kernel[:, i*resolution:(i+1)*resolution] = side_kernel

    for obs in obs_locations:
        i = int(round(obs[0] * resolution)) - obs_offset
        j = int(round(obs[1] * resolution)) - obs_offset * 2
        if i < 0:
            kernel[0:i+obs_kernel.shape[0], j:j+obs_kernel.shape[1]] += obs_kernel[abs(i):kernel.shape[0], :]
            continue

        if kernel.shape[0] - i <= (obs_kernel.shape[0]):
            kernel[i:i+obs_kernel.shape[0], j:j+obs_kernel.shape[1]] += obs_kernel[0:kernel.shape[0]-i, :]
            continue


        kernel[i:i+obs_kernel.shape[0], j:j+obs_kernel.shape[1]] += obs_kernel
    
    return kernel


class TrackKernel(BaseKernel):
    def __init__(self, sim_conf):
        super().__init__(sim_conf)
        kernel_name = f"{sim_conf.kernel_path}TrackKernel_{sim_conf.track_kernel_path}_{sim_conf.map_name}.npy"
        self.kernel = np.load(kernel_name)

        file_name = 'maps/' + sim_conf.map_name + '.yaml'
        with open(file_name) as file:
            documents = yaml.full_load(file)
            yaml_file = dict(documents.items())
        self.origin = yaml_file['origin']

        # self.view_kernel(0)


    def construct_kernel(self, a, b):
        pass

    def get_indices(self, state):
        phi_range = np.pi * 2
        x_ind = min(max(0, int(round((state[0]-self.origin[0])*self.resolution))), self.kernel.shape[0]-1)
        y_ind = min(max(0, int(round((state[1]-self.origin[1])*self.resolution))), self.kernel.shape[1]-1)

        phi = state[2]
        if phi >= phi_range/2:
            phi = phi - phi_range
        elif phi < -phi_range/2:
            phi = phi + phi_range
        theta_ind = int(round((phi + phi_range/2) / phi_range * (self.kernel.shape[2]-1)))

        if theta_ind > 40 or theta_ind < -40:
            print(f"Theta ind: {theta_ind}")


        return x_ind, y_ind, theta_ind





