import numpy as np
from envs.grid_core import World
from envs.scenarios.pursuit_battery import Scenario as BaseScenario
from envs.scenarios.pursuit_battery import Prey as BasePrey
from envs.scenarios.pursuit_base import Predator as Agent
import config
import time

FLAGS = config.flags.FLAGS

n_predator = FLAGS.n_predator
n_prey = FLAGS.n_prey
map_size = FLAGS.map_size

max_step_per_ep = FLAGS.max_step_per_ep
history_len = FLAGS.history_len

power_dec = 1
power_threshold = 0


class Predator(Agent):
    def __init__(self, power_threshold=0, power_dec=1):
        super(Predator, self).__init__(obs_range=3)
        self.power_threshold = power_threshold
        self.power_dec = power_dec
        self.power = 100
        self.step = 0.0
        self.gathered = 0
        self.obs_dim = (((self.obs_range*2 + 1)**2)*3 + 3)*history_len
        self.involved = 0
        self.previous_power_levels = np.full(20, 100)
        self.alpha = 0.6  # Smoothing factor for moving average
        self.gathered_moving_avg = 0.0  # Initialize moving average
        self.start_time = time.time()  # Record start time

    @property
    def p_pos(self):
        return self._x, self._y

    def get_obs(self):
        return np.array(self._obs).flatten()

    def update_obs(self, obs):
        self.step += 1.0
        self.grid = obs
        obs = np.append(obs.bin_encode().flatten(),\
            [self.power/100.0, self.step/100.0, self.gathered/10.0])
        self._obs.append(obs)

    def set_previous_power_levels(self, agent_id, power):
        self.previous_power_levels[agent_id] = power

    def get_previous_power_levels(self, agent_id):
        return self.previous_power_levels[agent_id]

    def base_reward(self, capture, involved, is_terminal):
        self.gathered += capture
        self.involved += involved

        reward = 0

        if self.action.u != 2:
            self.power -= self.power_dec

        if FLAGS.rwd_form == "siglin":
            if is_terminal: # siglin
                if self.power > 0:
                    reward += 100./(1. + np.exp(-(self.power-30)/10.0))
                reward += self.gathered*12

        if FLAGS.rwd_form == "picsq":
            if is_terminal: #picsq
                if self.power > 0:
                    reward += 100*(self.power > 50)
                reward +=5*self.gathered**2

        if FLAGS.rwd_form == "sigsig":
            if is_terminal: #sigsig
                if self.power > 0:
                    reward += 100./(1. + np.exp(-(self.power-70)/10.0))
                reward += 150./(1. + np.exp(-(self.gathered-5)))

        return reward

    def is_done(self):
        return False

    def reset(self):
        self.power = 100
        self.gathered = 0
        self.involved = 0
        self.step = 0.0

class Prey(BasePrey):
    def __init__(self):
        super(Prey, self).__init__()
        self.death_timer = 0

    @property
    def p_pos(self):
        return self._x, self._y

    def reset(self):
        super(Prey, self).reset()
        self.death_timer = 0

    def should_reincarnate(self):
        self.death_timer += 1
        return self.death_timer >= 15

class Scenario(BaseScenario):
    consumers = []

    def __init__(self):
        super().__init__()
        self.step = 0
        self.consumers = Scenario.consumers
        self.n_agents = 10
        self.lambda_value = 0.5
        self.trust_matrix = np.zeros((10,10))
        self.predator = Predator()       
        #self.agent_ids = [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
        self.agent_ids = [0,  1,  2,  3,  4,  5,  6,  7,  8,  9]
        self.encounters_all, self.encounters_step = [], []
        self.capture_times = np.zeros(n_predator, dtype=np.float64)  # To store capture times for each predator
        self.start_time = time.time()

    def make_world(self):
        world = World(width=map_size, height=map_size)

        agents = []
        self.atype_to_idx = {
            "predator": [],
            "prey": []
        }

        # add predators
        for i in range(n_predator):
            agents.append(Predator(power_threshold=power_threshold, power_dec=power_dec))
            self.atype_to_idx["predator"].append(i)

        # add preys
        for i in range(n_prey):
            agents.append(Prey())
            self.atype_to_idx["prey"].append(n_predator + i)

        world.agents = agents
        for i, agent in enumerate(world.agents):
            agent.id = i + 1

        # make initial conditions
        self.reset_world(world)
        self.world = world
        return world

    def reward(self, agent, world):
        self.step += 1
        self.prey_captured = 0
        self.consumers = []
        base_rewards = []

        for i in self.atype_to_idx["prey"]:
            prey = world.agents[i]

            if not prey.exists:
                if prey.should_reincarnate():
                    prey.reset()
                    world.placeObj(prey)
                    continue

            if prey.exists and prey.captured:
                world.removeObj(prey)
                self.consumers.extend(prey.consumers)
                self.prey_captured += 1

        for predator_id in world.agents:  
            involved = self.consumers.count(predator_id)
            base_rewards.append(predator_id.base_reward(self.prey_captured, involved, (self.step == max_step_per_ep)))

        if len(base_rewards) > 0:
            return sum(base_rewards)/len(base_rewards)
        else:
            return 0
        
    def done(self, agent, world):
        return (self.step == max_step_per_ep)
    
    def update_trust(self):
        self.encouters_step = []
        for helper_id in self.consumers:
            for recipient_id in self.consumers:
                if helper_id < len(self.agent_ids) and recipient_id < len(self.agent_ids):
                    old_trust_value = self.trust_matrix[helper_id, recipient_id]
                    updated_trust_value = (1 - self.lambda_value) * old_trust_value + self.lambda_value * 1
                    self.trust_matrix[helper_id, recipient_id] = updated_trust_value
                    self.encounters_all.append([helper_id, recipient_id])
                    self.encounters_step.append([helper_id, recipient_id])

        for agent in self.world.agents:
            if isinstance(agent, Predator):
                current_power = agent.power
                agent_id = agent.id - 1  # assuming agent.id starts from 1
                previous_power = self.predator.get_previous_power_levels(agent_id)

                if current_power > previous_power:
                    for prey in self.world.agents:
                        if isinstance(prey, Prey) and prey.exists and not prey.captured:
                            distance_to_prey1 = np.linalg.norm(np.array(agent.p_pos) - np.array(prey.p_pos))

                            if distance_to_prey1 == 1 and agent.id not in self.consumers:
                                for neighbor_id in range(len(self.agent_ids)):
                                    distance_to_prey2 = np.linalg.norm(np.array(agent.p_pos) - np.array(prey.p_pos))

                                    if neighbor_id != agent_id and distance_to_prey2 == 1:
                                        old_trust_value = self.trust_matrix[agent_id, neighbor_id]
                                        updated_trust_value = (1 - self.lambda_value) * old_trust_value + self.lambda_value * -1
                                        self.trust_matrix[agent_id, neighbor_id] = updated_trust_value
                
                # Update previous power levels
                self.predator.set_previous_power_levels(agent_id, current_power)
                
    def get_trust(self):
        return self.trust_matrix
    
    def encounters_count(self):
        return self.encounters_all, self.encounters_step 
    
    def get_capture_times(self):
        return self.capture_times