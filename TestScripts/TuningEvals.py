
from F1TenthSupervisorySystem.NavAgents.EndAgent import EndVehicleTrain, EndVehicleTest

from GeneralTrainTest import *

from F1TenthSupervisorySystem.Supervisor.SupervisorySystem import Supervisor, TrackKernel, LearningSupervisor

from F1TenthSupervisorySystem.NavAgents.follow_the_gap import FollowTheGap
from F1TenthSupervisorySystem.NavAgents.Oracle import Oracle


import numpy as np
from matplotlib import pyplot as plt

config_file = "track_kernel"


test_n = 100
run_n = 2
baseline_name = f"std_end_baseline_{run_n}"

eval_name = f"end_comparison_{run_n}"
sim_conf = load_conf("track_kernel")


def train_final_kenel():
    n = 1
    test_name = f"final_{n}"
    agent_name = f"kernel_{test_name}_{run_n}"

    planner = EndVehicleTrain(agent_name, sim_conf)
    kernel = TrackKernel(sim_conf)
    safety_planner = LearningSupervisor(planner, kernel, sim_conf)
    safety_planner.calculate_reward = safety_planner.magnitude_reward
    #TODO: set up rewards as separate classes

    train_time = TrainVehicle(sim_conf, safety_planner)

    planner = EndVehicleTest(agent_name, sim_conf)
    kernel = TrackKernel(sim_conf)
    safety_planner = Supervisor(planner, kernel, sim_conf)

    eval_dict = run_evaluation(sim_conf, safety_planner, render=False)

    config_dict = vars(sim_conf)
    config_dict['EvalName'] = test_name 
    config_dict['train_time'] = train_time
    config_dict['test_number'] = n
    config_dict['reward'] = f"magnitude_{1}"
    config_dict.update(eval_dict)

    save_conf_dict(config_dict)

def eval_kernel_final():
    n = 1
    test_name = f"final_{n}"
    agent_name = f"kernel_{test_name}_{run_n}"
    agent_name = "kernel_end_RewardMag_2"
    sim_conf.test_n = 1

    planner = EndVehicleTest(agent_name, sim_conf)
    kernel = TrackKernel(sim_conf)
    safety_planner = Supervisor(planner, kernel, sim_conf)
    eval_dict = run_evaluation(sim_conf, safety_planner, render=False)

    config_dict = vars(sim_conf)
    config_dict['EvalName'] = test_name 
    config_dict['train_time'] = 0
    config_dict['test_number'] = n
    config_dict['reward'] = f"magnitude_{1}"
    config_dict.update(eval_dict)

    save_conf_dict(config_dict)

if __name__ == "__main__":
    # train_final_kenel()
    eval_kernel_final()

