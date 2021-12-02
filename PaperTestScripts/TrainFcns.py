

import numpy as np
import csv
import gym
import time

import os, shutil
import yaml
from argparse import Namespace

# Admin functions
def save_conf_dict(dictionary, save_name=None):
    if save_name is None:
        save_name  = dictionary["name"]
    path = dictionary["vehicle_path"] + dictionary["name"] + f"/{save_name}_record.yaml"
    with open(path, 'w') as file:
        yaml.dump(dictionary, file)

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


# Training functions
def train_kernel_episodic(vehicle, conf, show=False):
    print(f"Starting Episodic Training: {vehicle.planner.name}")

    env = gym.make('f110_gym:f110-v0', map=conf.map_name, map_ext=conf.map_ext, num_agents=1)
    start_time = time.time()
    state, done = env.reset(False), False

    for n in range(sim_conf.train_n):
        a, fake_done = vehicle.plan(state)
        s_prime, r, done, _ = env.step_plan(a)

        state = s_prime
        vehicle.planner.agent.train(2)
        
        if done or fake_done:
            vehicle.done_entry(s_prime, env.steps)
            if show:
                env.render(wait=False)
                vehicle.safe_history.plot_safe_history()

            state = env.reset(False)

    vehicle.planner.t_his.print_update(True)
    vehicle.planner.t_his.save_csv_data()
    vehicle.planner.agent.save(vehicle.planner.path)
    vehicle.save_intervention_list()

    train_time = time.time() - start_time
    print(f"Finished Episodic Training: {vehicle.planner.name} in {train_time} seconds")

    return train_time 

def train_kernel_continuous(env, vehicle, sim_conf, show=False):
    print(f"Starting Continuous Training: {vehicle.planner.name}")
    start_time = time.time()
    state, done = env.reset(False), False

    for n in range(sim_conf.train_n):
        a, fake_done = vehicle.plan(state)
        s_prime, r, done, _ = env.step_plan(a)

        state = s_prime
        vehicle.planner.agent.train(2)
        
        if done:
            vehicle.fake_done(env.steps)
            if show:
                env.render(wait=False)
                vehicle.safe_history.plot_safe_history()

            done = False
            state = env.fake_reset()

    vehicle.planner.t_his.print_update(True)
    vehicle.planner.t_his.save_csv_data()
    vehicle.planner.agent.save(vehicle.planner.path)
    vehicle.save_intervention_list()

    train_time = time.time() - start_time
    print(f"Finished Continuous Training: {vehicle.planner.name} in {train_time} seconds")

    return train_time 

def train_baseline_vehicle(env, vehicle, sim_conf, show=False):
    start_time = time.time()
    state, done = env.reset(False), False
    print(f"Starting Baseline Training: {vehicle.name}")
    crash_counter = 0

    for n in range(sim_conf.train_n):
        a = vehicle.plan_act(state)
        s_prime, r, done, _ = env.step_plan(a)

        state = s_prime
        vehicle.agent.train(2)
        
        if done:
            vehicle.done_entry(s_prime)
            if show:
                env.render(wait=False)
            if state['reward'] == -1:
                crash_counter += 1

            state = env.reset(False)

    vehicle.t_his.print_update(True)
    vehicle.t_his.save_csv_data()
    vehicle.agent.save(vehicle.path)

    train_time = time.time() - start_time
    print(f"Finished Training: {vehicle.name} in {train_time} seconds")
    print(f"Crashes: {crash_counter}")

    return train_time, crash_counter


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


