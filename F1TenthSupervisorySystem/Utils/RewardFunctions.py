
class SteeringReward:
    def __init__(self, b_s):
        self.b_s = b_s
        self.name = f"Steering({b_s})"
        
    def __call__(self, state, s_prime):
        s_prime['reward'] = find_reward(s_prime)
        scaled_steering = abs(s_prime['steering_deltas'][0]/0.4)
        reward = (0.5 - scaled_steering) * self.b_s
        reward += s_prime['reward']

        return reward

def find_reward(s_p):
    if s_p['collisions'][0] == 1:
        return -1
    elif s_p['lap_counts'][0] == 1:
        return 1
    return 0


