import numpy as np
import csv
import gym
import time

import os, shutil
import yaml
from argparse import Namespace


# Test Functions
def evaluate_vehicle(vehicle, conf, show=False):
    env = gym.make('f110_gym:f110-v0', map=conf.map_name, map_ext=conf.map_ext, num_agents=1)

    crashes = 0
    completes = 0
    lap_times = [] 
    laptime = 0.0
    start = time.time()

    for i in range(conf.test_n):
        obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
        lap_time = 0.0
        while not done and obs['lap_counts'][0] == 0:
            action = vehicle.plan_act(obs)
            obs, step_reward, done, _ = env.step(action[None, :])
            laptime += step_reward
            if show:
                env.render(mode='human_fast')

        r = find_conclusion(obs, start)

        if r == -1:
            crashes += 1
        else:
            completes += 1
            lap_times.append(laptime)

    success_rate = (completes / (completes + crashes) * 100)
    if len(lap_times) > 0:
        avg_times, std_dev = np.mean(lap_times), np.std(lap_times)
    else:
        avg_times, std_dev = 0, 0


    print(f"Crashes: {crashes}")
    print(f"Completes: {completes} --> {success_rate:.2f} %")
    print(f"Lap times Avg: {avg_times} --> Std: {std_dev}")

    eval_dict = {}
    eval_dict['name'] = vehicle.name
    eval_dict['success_rate'] = float(success_rate)
    eval_dict['avg_times'] = float(avg_times)
    eval_dict['std_dev'] = float(std_dev)

    print(f"Finished running test and saving file with results.")

    return eval_dict

def render_kernel(env, vehicle, sim_conf, show=False):
    lap_times = [] 

    state = env.reset(False)
    done, score = False, 0.0

    state = env.reset(False)
    done, score = False, 0.0
    for i in range(sim_conf.test_n):
        while not done:
            a = vehicle.plan_act(state)
            s_p, r, done, _ = env.step_plan(a)
            state = s_p
        if show:
            env.render(wait=False, name=vehicle.name)


        print(f"({i}) Complete -> time: {env.steps}")
        lap_times.append(env.steps)
        env.render_trajectory(vehicle.planner.path, f"Traj_{i}", vehicle.safe_history)
        vehicle.safe_history.save_safe_history(vehicle.planner.path, f"Traj_{i}")
        state = env.reset(False)
        
        done = False

    avg_times, std_dev = np.mean(lap_times), np.std(lap_times)
    print(f"Lap times Avg: {avg_times} --> Std: {std_dev}")

    eval_dict = {}
    eval_dict['name'] = vehicle.name
    eval_dict['avg_times'] = float(avg_times)
    eval_dict['std_dev'] = float(std_dev)

    print(f"Finished running test and saving file with results.")

    return eval_dict

def render_baseline(env, vehicle, sim_conf, show=False):
    lap_times = [] 

    state = env.reset(False)
    done, score = False, 0.0

    state = env.reset(False)
    done, score = False, 0.0
    for i in range(sim_conf.test_n):
        while not done:
            a = vehicle.plan_act(state)
            s_p, r, done, _ = env.step_plan(a)
            state = s_p
        if show:
            env.render(wait=False, name=vehicle.name)


        print(f"({i}) Complete -> time: {env.steps}")
        lap_times.append(env.steps)
        env.render_trajectory(vehicle.path, f"Traj_{i}")
        # state = env.reset(True)
        state = env.reset(False)
        
        done = False

    avg_times, std_dev = np.mean(lap_times), np.std(lap_times)
    print(f"Lap times Avg: {avg_times} --> Std: {std_dev}")

    eval_dict = {}
    eval_dict['name'] = vehicle.name
    eval_dict['avg_times'] = float(avg_times)
    eval_dict['std_dev'] = float(std_dev)

    print(f"Finished running test and saving file with results.")

    return eval_dict




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


