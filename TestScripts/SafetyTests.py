
from F1TenthSupervisorySystem.NavAgents.EndAgent import EndVehicleTrain, EndVehicleTest

from GeneralTrainTest import *

from F1TenthSupervisorySystem.NavAgents.PurePursuit import PurePursuit
from F1TenthSupervisorySystem.NavAgents.RandoPlanner import RandomPlanner

from F1TenthSupervisorySystem.Supervisor.SupervisorySystem import Supervisor, TrackKernel, LearningSupervisor


import numpy as np
from matplotlib import pyplot as plt

config_file = "track_kernel"


def rando_test():
    conf = load_conf("track_kernel")

    planner = RandomPlanner(conf)
    kernel = TrackKernel(conf)
    safety_planner = Supervisor(planner, kernel, conf)

    run_kernel_test(conf, safety_planner, 5, obstacles=0)


test_n = 100
run_n = 2
baseline_name = f"std_end_baseline_{run_n}"
kernel_name = f"kernel_end_RewardMag_{run_n}"
# kernel_name = f"kernel_end_RewardZero_{run_n}"
# kernel_name = f"kernel_end_RewardInter_0.5_{run_n}"

eval_name = f"end_comparison_{run_n}"
sim_conf = load_conf("track_kernel")



def train_baseline(agent_name):
    planner = EndVehicleTrain(agent_name, sim_conf)
    # planner = EndVehicleTrain(agent_name, sim_conf, True)

    TrainVehicle(sim_conf, planner)

def test_baseline(agent_name):
    planner = EndVehicleTest(agent_name, sim_conf)

    run_multi_test(sim_conf, planner)

def test_oracle():
    # vehicle = Oracle(sim_conf)
    vehicle = FollowTheGap(sim_conf)

    run_multi_test(sim_conf, vehicle)

def train_kenel(agent_name):
    planner = EndVehicleTrain(agent_name, sim_conf)
    kernel = TrackKernel(sim_conf)
    safety_planner = LearningSupervisor(planner, kernel, sim_conf)

    TrainVehicle(sim_conf, safety_planner)

def test_kernel_sss(vehicle_name):
    planner = EndVehicleTest(vehicle_name, sim_conf)
    kernel = TrackKernel(sim_conf)
    safety_planner = Supervisor(planner, kernel, sim_conf)

    run_kernel_test(sim_conf, safety_planner, test_n)

def baseline_vs_kernel(baseline_name, kernel_name):
    test = TestVehicles(sim_conf, eval_name)
    
    baseline = EndVehicleTest(baseline_name, sim_conf)
    test.add_vehicle(baseline)

    planner = EndVehicleTest(kernel_name, sim_conf)
    kernel = TrackKernel(sim_conf)
    safety_planner = Supervisor(planner, kernel, sim_conf)
    test.add_vehicle(safety_planner)

    # test.run_eval(True, wait=False)
    test.run_eval(False, wait=False)

from F1TenthSupervisorySystem.NavAgents.follow_the_gap import FollowTheGap
from F1TenthSupervisorySystem.NavAgents.Oracle import Oracle

def full_comparison(baseline_name, kernel_name):
    test = TestVehicles(sim_conf, eval_name)
    
    baseline = EndVehicleTest(baseline_name, sim_conf)
    test.add_vehicle(baseline)

    planner = EndVehicleTest(kernel_name, sim_conf)
    kernel = TrackKernel(sim_conf)
    safety_planner = Supervisor(planner, kernel, sim_conf)
    test.add_vehicle(safety_planner)

    vehicle = FollowTheGap(sim_conf)
    test.add_vehicle(vehicle)

    vehicle = Oracle(sim_conf)
    test.add_vehicle(vehicle)

    test.run_eval(False, wait=False)




if __name__ == "__main__":
    # train_baseline(baseline_name)
    # test_baseline(baseline_name)
    # test_oracle()

    train_kenel(kernel_name)
    # test_kernel_sss(kernel_name)
    # test_kernel_sss(baseline_name)
    # test_baseline(kernel_name)

    # baseline_vs_kernel(baseline_name, kernel_name)
    # sim_conf.test_n = 1
    # full_comparison(baseline_name, kernel_name)


    # rando_test()

