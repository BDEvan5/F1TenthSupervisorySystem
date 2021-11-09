

import numpy as np
import csv
import gym
import time

import os, shutil
import yaml
from argparse import Namespace

def load_conf(fname):
    full_path =  "config/" + fname + '.yaml'
    with open(full_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)

    conf = Namespace(**conf_dict)

    return conf

def init_file_struct(path):
    if os.path.exists(path):
        try:
            os.rmdir(path)
        except:
            shutil.rmtree(path)
    os.mkdir(path)


"""Train"""
def TrainVehicle(conf, vehicle, add_obs=False):
    path = conf.vehicle_path + vehicle.name

    env = gym.make('f110_gym:f110-v0', map=conf.map_name, map_ext=conf.map_ext, num_agents=1)
    map_reset_pt = np.array([[conf.sx, conf.sy, conf.stheta]])
    state, step_reward, done, info = env.reset(map_reset_pt)

    done = False
    start = time.time()

    max_ep_time = 40 
    for n in range(conf.train_n):
        a = vehicle.plan(state)
        s_prime, r, done, info = env.step(a[None, :])

        state = s_prime
        vehicle.agent.train()
        
        # env.render('human_fast')
        # env.render('human')

        if s_prime['lap_times'][0] > max_ep_time:
            done = True
        
        if done or s_prime['collisions'][0] == 1:
            find_conclusion(s_prime, start)
            vehicle.done_entry(s_prime)
            # t_his.lap_done(True)
            # vehicle.show_vehicle_history()
            # history.show_history()
            # history.reset_history()
            # t_his.lap_done(True)

            start = time.time()

            state, step_reward, done, info = env.reset(map_reset_pt)
            if add_obs:
                env.add_obstacles(conf.n_obs, [conf.obs_size, conf.obs_size])
            # env.render()

    vehicle.agent.save(directory=path)

    print(f"Finished Training: {vehicle.name}")



def find_conclusion(s_p, start):
    laptime = s_p['lap_times'][0]
    if s_p['collisions'][0] == 1:
        print(f'Collision --> Sim time: {laptime:.2f} Real time: {(time.time()-start):.2f}')
        return -1
    elif s_p['lap_counts'][0] == 1:
        print(f'Complete --> Sim time: {laptime:.2f} Real time: {(time.time()-start):.2f}')
        return 1
    else:
        print("No conclusion: Awkward palm trees")
        print(s_p)
    return 0





def run_evaluation(conf, vehicle, render=False):
    env = gym.make('f110_gym:f110-v0', map=conf.map_name, map_ext=conf.map_ext, num_agents=1)

    for i in range(conf.test_n):

        obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))

        laptime = 0.0
        start = time.time()
        obses = []
        while not done and laptime < conf.max_time:
            action = vehicle.plan(obs)
            obs, r, done, info = env.step(action[None, :])
            
            laptime += step_reward
            # env.render(mode='human')
            if render:
                env.render(mode='human_fast')
        r = find_conclusion(obs, start)

        #TODO: keep going here.
        if r == -1 or r == 0:
            crashes[j] += 1
        else:
            self.completes[j] += 1
            self.lap_times[j].append(laptime)


    print(f"Crashes: {crashes}")
    print(f"Completes: {completes} --> {success_rate:.2f} %")
    print(f"Lap times Avg: {avg_times} --> Std: {std_dev}")

    eval_dict = {}
    eval_dict['name'] = vehicle.name
    eval_dict['success_rate'] = float(success_rate)
    eval_dict['avg_times'] = float(avg_times)
    eval_dict['std_dev'] = float(std_dev)
    eval_dict['no_obs_time'] = float(no_obs_time)

    print(f"Finished running test and saving file with results.")

    return eval_dict



"""Testing Function"""
class TestData:
    def __init__(self) -> None:
        self.endings = None
        self.crashes = None
        self.completes = None
        self.lap_times = None

        self.names = []
        self.lap_histories = None

        self.N = None

    def init_arrays(self, N, laps):
        self.completes = np.zeros((N))
        self.crashes = np.zeros((N))
        self.lap_times = np.zeros((laps, N))
        self.endings = np.zeros((laps, N)) #store env reward
        self.lap_times = [[] for i in range(N)]
        self.N = N
 
    def save_txt_results(self):
        test_name = 'EvalResults/' + self.eval_name + '.txt'
        with open(test_name, 'w') as file_obj:
            file_obj.write(f"\nTesting Complete \n")
            file_obj.write(f"Map name:  \n")
            file_obj.write(f"-----------------------------------------------------\n")
            file_obj.write(f"-----------------------------------------------------\n")
            for i in range(self.N):
                file_obj.write(f"Vehicle: {self.vehicle_list[i].name}\n")
                file_obj.write(f"Crashes: {self.crashes[i]} --> Completes {self.completes[i]}\n")
                percent = (self.completes[i] / (self.completes[i] + self.crashes[i]) * 100)
                file_obj.write(f"% Finished = {percent:.2f}\n")
                file_obj.write(f"Avg lap times: {np.mean(self.lap_times[i])}\n")
                file_obj.write(f"-----------------------------------------------------\n")

    def print_results(self):
        print(f"\nTesting Complete ")
        print(f"-----------------------------------------------------")
        print(f"-----------------------------------------------------")
        for i in range(self.N):
            if len(self.lap_times[i]) == 0:
                self.lap_times[i].append(0)
            print(f"Vehicle: {self.vehicle_list[i].name}")
            print(f"Crashes: {self.crashes[i]} --> Completes {self.completes[i]}")
            percent = (self.completes[i] / (self.completes[i] + self.crashes[i]) * 100)
            print(f"% Finished = {percent:.2f}")
            print(f"Avg lap times: {np.mean(self.lap_times[i])}")
            print(f"-----------------------------------------------------")
        
    def save_csv_results(self):
        test_name = 'EvalResults/'  + self.eval_name + '.csv'

        data = [["#", "Name", "%Complete", "AvgTime", "Std"]]
        for i in range(self.N):
            v_data = [i]
            v_data.append(self.vehicle_list[i].name)
            v_data.append((self.completes[i] / (self.completes[i] + self.crashes[i]) * 100))
            v_data.append(np.mean(self.lap_times[i]))
            v_data.append(np.std(self.lap_times[i]))
            data.append(v_data)

        with open(test_name, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(data)

    # def load_csv_data(self, eval_name):
    #     file_name = 'Vehicles/Evals' + eval_name + '.csv'

    #     with open(file_name, 'r') as csvfile:
    #         csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
            
    #         for lines in csvFile:  
    #             self.
    #             rewards.append(lines)


    # def plot_eval(self):
    #     pass


class TestVehicles(TestData):
    def __init__(self, conf, eval_name) -> None:
        self.conf = conf
        self.eval_name = eval_name
        self.vehicle_list = []
        self.N = None

        TestData.__init__(self)

    def add_vehicle(self, vehicle):
        self.vehicle_list.append(vehicle)

    def run_eval(self):
        N = self.N = len(self.vehicle_list)
        self.init_arrays(N, self.conf.test_n)
        conf = self.conf

        env = gym.make('f110_gym:f110-v0', map=conf.map_name, map_ext=conf.map_ext, num_agents=1)
        map_reset_pt = np.array([[conf.sx, conf.sy, conf.stheta]])

        for i in range(self.conf.test_n):
            for j in range(N):
                vehicle = self.vehicle_list[j]

                r, laptime = self.run_lap(vehicle, env, map_reset_pt)
                print(f"#{i}: Lap time for ({vehicle.name}): {laptime} --> Reward: {r}")
                self.endings[i, j] = r
                if r == -1 or r == 0:
                    self.crashes[j] += 1
                else:
                    self.completes[j] += 1
                    self.lap_times[j].append(laptime)

        self.print_results()
        self.save_txt_results()
        self.save_csv_results()

    def run_lap(self, vehicle, env, map_reset_pt):
        state, step_reward, done, info = env.reset(map_reset_pt)
        done = False
        start = time.time()
        while not done:
            a = vehicle.plan(state)
            s_p, r, done, _ = env.step(a[None, :])
            state = s_p
            # env.render()

            if s_p['collisions'][0] == 1:
                break

        r = find_conclusion(s_p, start)
        lap_time = state['lap_times'][0]

        return r, lap_time

