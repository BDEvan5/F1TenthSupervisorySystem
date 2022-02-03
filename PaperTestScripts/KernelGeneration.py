

# from SupervisorySafetySystem.KernelGenerator import prepare_track_img, ViabilityGenerator
# from SupervisorySafetySystem.DiscrimKernel import DiscrimGenerator

import time 

from TestScripts import *
from TrainFcns import *
from F1TenthSupervisorySystem.NavAgents.RandoPlanner import RandomPlanner
from F1TenthSupervisorySystem.Supervisor.SupervisorySystem import Supervisor, TrackKernel
from F1TenthSupervisorySystem.Supervisor.KernelGenerator import prepare_track_img, ViabilityGenerator, shrink_img


def generate_viability_kernel(conf, val, make_picture=False):
    assert conf.kernel_mode == "viab"
    start_time = time.time()
    img = prepare_track_img(conf) 
    o, img = shrink_img(img, 22)
    kernel = ViabilityGenerator(img, conf)
    filled = kernel.calculate_kernel(50)
    kernel.save_kernel(f"Kernel_{conf.kernel_mode}_{val}_{conf.map_name}")
    time_taken = time.time() - start_time
    print(f"Time taken: {time_taken}")
    kernel.view_build(False)
    # if make_picture:
    #     kernel.make_picture(False)

    return time_taken, filled

def generate_discriminating_kernel(conf, val, make_picture=False):
    assert conf.kernel_mode == "disc"
    start_time = time.time()
    img = prepare_track_img(conf) 
    kernel = DiscrimGenerator(img, conf)
    filled = kernel.calculate_kernel(50)
    kernel.save_kernel(f"Kernel_disc_{val}_{conf.map_name}")
    time_taken = time.time() - start_time
    print(f"Time taken: {time_taken}")
    kernel.view_build(False)
    if make_picture:
        kernel.make_picture(False)

    return time_taken, filled
    
def generate_standard_kernels():
    conf = load_conf("TestKernelGen")

    conf.kernel_mode = "viab"
    generate_viability_kernel(conf, "std", True)
    # conf.kernel_mode = "disc"
    # generate_discriminating_kernel(conf, "std", True)


def kernel_discretization_ndx():
    conf = load_conf("SafetyTests")
    conf.EvalName = "KernelDiscret_ndx" 
    conf.test_n = 100
    env = TrackSim(conf)

    discretes = [60, 70, 80, 90, 100, 120]
    # discretes = [80, 100]
    success_rates = []
    for value in discretes:
        conf.n_dx = value

        for mode in ["viab", "disc"]:
            conf.kernel_mode = mode
            if mode == "viab":
                kernel_time, kernel_filled = generate_viability_kernel(conf, value)
            elif mode == "disc":
                kernel_time, kernel_filled = generate_discriminating_kernel(conf, value)

            kernel_name = f"Kernel_{conf.kernel_mode}_{value}_{conf.map_name}.npy"    
            planner = RandomPlanner(f"RandoKernelTest_{conf.kernel_mode}_{value}")

            kernel = TrackKernel(conf, False, kernel_name)
            safety_planner = Supervisor(planner, kernel, conf)

            eval_dict = evaluate_vehicle(env, safety_planner, conf, False)
            success_rates.append(eval_dict['success_rate'])
            
            config_dict = vars(conf)
            config_dict['kernel_time'] = kernel_time
            config_dict['kernel_filled'] = kernel_filled
            config_dict.update(eval_dict)

            save_conf_dict(config_dict)

    print(f"Discretizations: {discretes}")
    print(f"Success rates: {success_rates}")


def kernel_discretization_time():
    conf = load_conf("SafetyTests")
    conf.EvalName = "KernelDiscret_time" 
    conf.test_n = 100
    env = TrackSim(conf)

   # times = [0.09, 0.1, 0.12, 0.15]
    times = [0.1, 0.15]
    success_rates = []
    for value in times:
        conf.kernel_time_step = value
        conf.lookahead_time_step = value * 2  

        conf.kernel_mode = "viab"
        kernel_time = generate_viability_kernel(conf, value)
        conf.kernel_mode = "disc"
        kernel_time = generate_discriminating_kernel(conf, value)

        for mode in ["viab", "disc"]:
            conf.kernel_mode = mode
            kernel_name = f"Kernel_{conf.kernel_mode}_{value}_{conf.map_name}.npy"    
            planner = RandomPlanner(f"RandoKernelTest_{conf.kernel_mode}_{value}")

            kernel = TrackKernel(conf, False, kernel_name)
            safety_planner = Supervisor(planner, kernel, conf)

            eval_dict = evaluate_vehicle(env, safety_planner, conf, False)
            success_rates.append(eval_dict['success_rate'])
            
            config_dict = vars(conf)
            config_dict['kernel_time'] = kernel_time
            config_dict.update(eval_dict)

            save_conf_dict(config_dict)

    print(f"Discretizations: {times}")
    print(f"Success rates: {success_rates}")
    
    

def rando_pictures():
    conf = load_conf("SafetyTests")
    planner = RandomPlanner("RandoPictures")

    env = TrackSim(conf)
    kernel = TrackKernel(conf, False)
    safety_planner = Supervisor(planner, kernel, conf)

    conf.test_n = 5
    render_kernel(env, safety_planner, conf, True)

def rando_results(n):
    conf = load_conf("SafetyTests")
    conf.vehicle = "random"
    conf.test_n = 10

    # for mode in ["viab", "disc"]:
    for mode in ["viab"]:
        conf.kernel_mode = mode
        planner = RandomPlanner(conf, f"RandoResult_{conf.kernel_mode}_{n}")

        kernel = TrackKernel(conf)
        safety_planner = Supervisor(planner, kernel, conf)

        eval_dict = evaluate_vehicle(safety_planner, conf, True)
        
        config_dict = vars(conf)
        config_dict['test_number'] = n
        config_dict.update(eval_dict)

        save_conf_dict(config_dict)

def run_constant_tests(n):
    conf = load_conf("SafetyTests")
    conf.vehicle = "constant"

    env = TrackSim(conf)
    for mode in ["viab", "disc"]:
        conf.kernel_mode = mode
        kernel = TrackKernel(conf, False)
        for value in [0, -0.4, 0.4]:
            conf.constant_value = value
            planner = ConstantPlanner(f"ConstPlanner_{value}_{conf.kernel_mode}_{n}", conf.constant_value)
            safety_planner = Supervisor(planner, kernel, conf)

            eval_dict = render_kernel(env, safety_planner, conf, True)
            
            config_dict = vars(conf)
            config_dict['test_number'] = n
            config_dict.update(eval_dict)

            save_conf_dict(config_dict)






if __name__ == "__main__":
    # kernel_discretization_ndx()
    # kernel_discretization_time()

    # generate_standard_kernels()

    # rando_pictures()
    rando_results(1)
    # run_constant_tests(1)


