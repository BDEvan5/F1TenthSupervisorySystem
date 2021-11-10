
from F1TenthSupervisorySystem.NavAgents.EndAgent import EndVehicleTrain, EndVehicleTest

from GeneralTrainTest import *

from F1TenthSupervisorySystem.Supervisor.SupervisorySystem import Supervisor, TrackKernel, LearningSupervisor

from F1TenthSupervisorySystem.Utils.RewardFunctions import *


import numpy as np
from matplotlib import pyplot as plt

config_file = "track_kernel"
sim_conf = load_conf("track_kernel")


def tune_baseline_reward(reward_name, RewardClass):
    n = 1
    test_name = f"reward_{n}"
    agent_name = f"bTuning_{test_name}_{reward_name}_{n}"

    planner = EndVehicleTrain(agent_name, sim_conf)
    planner.calculate_reward = RewardClass
    # planner = EndVehicleTrain(agent_name, sim_conf, True)
    train_time = TrainVehicle(sim_conf, planner, True)

    planner = EndVehicleTest(agent_name, sim_conf)
    eval_dict = run_evaluation(sim_conf, planner, render=False)

    config_dict = vars(sim_conf)
    config_dict['EvalName'] = test_name 
    config_dict['train_time'] = train_time
    config_dict['test_number'] = n
    config_dict['reward'] = reward_name
    # config_dict['b_heading'] = 0.004
    # config_dict['b_vel'] = 0.004
    config_dict.update(eval_dict)

    save_conf_dict(config_dict)


def cth_reward():
    reward_name = f"CthRef_1"
    reward_class = RefCTHReward(sim_conf, 0.04, 0.004)

    tune_baseline_reward(reward_name, reward_class)

def dist_reward():
    reward_name = f"DistRef_1"
    reward_class = RefDistanceReward(sim_conf, 1)

    tune_baseline_reward(reward_name, reward_class)
    
def steer_reward():
    reward_name = f"Steer_1"
    reward_class = SteeringReward(0.01)

    tune_baseline_reward(reward_name, reward_class)
    



if __name__ == "__main__":
    # cth_reward()
    # dist_reward()
    steer_reward()

