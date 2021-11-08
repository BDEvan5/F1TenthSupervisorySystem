
from F1TenthSupervisorySystem.Supervisor.SupervisorySystem import Supervisor, TrackKernel
import numpy as np
from matplotlib import pyplot as plt

from F1TenthSupervisorySystem.NavAgents.PurePursuit import PurePursuit
from F1TenthSupervisorySystem.NavAgents.RandoPlanner import RandomPlanner


from GeneralTrainTest import run_multi_test, test_vehicle, run_kernel_test, load_conf


config_file = "track_kernel"
conf = load_conf(config_file)
# config_file = "config_test"

def test_pure_pursuit():
    vehicle = PurePursuit(conf)

    run_multi_test(conf, vehicle, 5, obstacles=0)


def test_kernel_pp():
    planner = PurePursuit(conf)
    kernel = TrackKernel(conf)
    safety_planner = Supervisor(planner, kernel, conf)

    run_kernel_test(conf, safety_planner, 5, obstacles=0)


def test_rando_kernel():
    planner = RandomPlanner(conf)
    kernel = TrackKernel(conf)
    safety_planner = Supervisor(planner, kernel, conf)

    run_kernel_test(conf, safety_planner, 5)

if __name__ == "__main__":
    # test_pure_pursuit()
    # test_kernel_pp()
    test_rando_kernel()


