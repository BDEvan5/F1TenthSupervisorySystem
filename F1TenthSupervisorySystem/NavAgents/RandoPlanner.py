import numpy as np



class RandomPlanner:
    def __init__(self, conf):
        self.d_max = conf.max_steer # radians  
        self.v = 2        
        self.name = "RandoPlanner"

    def plan_act(self, obs):
        v_current = obs['linear_vels_x'][0]
        v_min_plan = 1
        if v_current < v_min_plan:
            return np.array([[0, 2]])

        np.random.seed()
        steering = np.random.normal(0, 0.1)
        steering = np.clip(steering, -self.d_max, self.d_max)
        return np.array([steering, self.v])
