


"""Train"""
def TrainVehicle(conf, vehicle, render=False):
    start_time = time.time()
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
        
        if render:
            env.render('human_fast')

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
            # env.render()

    vehicle.agent.save(directory=path)

    print(f"Finished Training: {vehicle.name}")

    train_time = time.time() - start_time

    return train_time


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

    crashes = 0
    completes = 0 
    lap_times = []

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



