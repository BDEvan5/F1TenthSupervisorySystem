
import numpy as np
from numba import njit
from matplotlib import pyplot as plt
import yaml
from F1TenthSupervisorySystem.Supervisor.Dynamics import update_complex_state
from F1TenthSupervisorySystem.Supervisor.SimulatorDynamics import run_dynamics_update


class SupervisorHistory:
    def __init__(self):
        self.old_actions = []
        self.new_actions = []        

    def add_actions(self, old, new=None):
        if new is None:
            self.old_actions.append(old[0])
            self.new_actions.append(old[0])
        else:
            self.old_actions.append(old[0])
            self.new_actions.append(new[0])
            
        
    def plot_actions(self, wait=False):
        plt.figure(1)
        plt.clf()
        plt.plot(self.old_actions)
        plt.plot(self.new_actions)
        plt.legend(['Old', 'New'])
        plt.title('ActionHistory')
        plt.pause(0.0001)

        if wait:
            plt.show()

        self.old_actions = []
        self.new_actions = []


class Supervisor:
    def __init__(self, planner, kernel, conf):
        """
        A wrapper class that can be used with any other planner.
        Requires a planner with:
            - a method called 'plan_act' that takes a state and returns an action

        """
        
        self.d_max = conf.max_steer
        self.v = 2
        self.kernel = kernel
        self.planner = planner
        self.time_step = conf.kernel_time_step

        # aliases for the test functions
        try:
            self.n_beams = planner.n_beams
        except: pass
        self.plan_act = self.plan
        self.name = planner.name

        self.dw = np.ones((5, 2))
        self.dw[:, 0] = np.linspace(-self.d_max, self.d_max, conf.n_modes)
        self.dw[:, 1] *= self.v

        self.action = None
        self.loop_counter = 0 
        self.plan_f = conf.plan_frequency
        self.v_min_plan = conf.v_min_plan
        self.history = SupervisorHistory()

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
        init_action = self.planner.plan(obs)
        init_action[1] = self.v
        state = self.extract_state(obs)
        safe, next_state = check_init_action(state, init_action, self.kernel, self.time_step)
        if safe:
            self.action = init_action
            # self.history.add_actions(init_action)
            return self.action

        valids = simulate_and_classify(state, self.dw, self.kernel, self.time_step)
        if not valids.any():
            print('No Valid options')
            print(f"State: {state}")
            self.action = init_action
            return self.action
        
        self.action = modify_action(valids, self.dw)
        # self.history.add_actions(init_action, self.action)
        # print(f"Valids: {valids} -> new action: {action}")
        return self.action

#TODO jit all of this.

def check_init_action(state, u0, kernel, time_step):
    # next_state = update_complex_state(state, u0, time_step)
    next_state = run_dynamics_update(state, u0, time_step)
    # next_state = update_std_state(state, u0, 0.2)
    safe = kernel.check_state(next_state)
    
    return safe, next_state

def simulate_and_classify(state, dw, kernel, time_step):
    valid_ds = np.ones(len(dw))
    next_states = []
    for i in range(len(dw)):
        # next_state = update_complex_state(state, dw[i], time_step)
        next_state = run_dynamics_update(state, dw[i], time_step)
        # next_state = update_std_state(state, dw[i], 0.2)
        next_states.append(next_state)
        safe = kernel.check_state(next_state)
        valid_ds[i] = safe 

        # print(f"State: {state} + Action: {dw[i]} --> Expected: {next_state}  :: Safe: {safe}")
    # kernel.plot_kernel_modes(next_states)

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

    def plan(self, obs):
        obs['reward'] = self.calculate_reward() # check this works.
        init_action = self.planner.plan(obs)
        state = self.extract_state(obs)

        safe, next_state = check_init_action(state, init_action, self.kernel, self.time_step)
        if safe:
            # self.safe_history.add_locations(init_action[0], init_action[0])
            self.action = init_action
            return self.action

        self.intervene = True

        valids = simulate_and_classify(state, self.dw, self.kernel, self.time_step)
        if not valids.any():
            print(f'No Valid options: state : {state}')
            self.action = init_action
            return self.action
        
        self.action = modify_action(valids, self.dw)
        # print(f"Valids: {valids} -> new action: {action}")
        # self.safe_history.add_locations(init_action[0], action[0])

        self.intervention_mag = self.action[0] - init_action[0]
        return self.action

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

    def plot_kernel_modes(self, states):
        plt.figure(5)
        plt.clf()
        middle = states[2]
        i, j, k = self.get_indices(middle)
        plt.title(f"Kernel inds: {i}, {j}, {k}")
        img = self.kernel[:, :, k].T 
        plt.imshow(img, origin='lower')
        # plt.show()
        for state in states:
            i, j, k = self.get_indices(state)
            if self.kernel[i, j, k]:
                plt.plot(i, j, 'x', markersize=20, color='red')
            else:
                plt.plot(i, j, 'x', markersize=20, color='green')
        plt.pause(0.0001)


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
        while phi >= phi_range/2:
            phi = phi - phi_range
        while phi < -phi_range/2:
            phi = phi + phi_range
        theta_ind = int(round((phi + phi_range/2) / phi_range * (self.kernel.shape[2]-1)))

        if theta_ind > 40 or theta_ind < -40:
            print(f"Theta ind: {theta_ind}")


        return x_ind, y_ind, theta_ind





