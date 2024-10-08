import numpy as np
from envs.grid_core import Grid

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
N = 0
E = 1
O = 2
W = 3
S = 4

class MultiAgentEnv():
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, rx_callback=None, tx_callback=None, pre_encode=False):

        self.world = world
        self.grid = Grid(12,12)
        self.agents = self.world.agents
        self.n = len(world.agents)
        self.episode_length = 0 
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        self.rx_callback = rx_callback
        self.tx_callback = tx_callback

        self.pre_encode = pre_encode

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.agent_precedence = []

        for agent in self.agents:
            self.agent_precedence.append(agent.itype)
            self.observation_space.append(agent.obs_dim)
            self.action_space.append(agent.act_spc)

        self.bin_encoding = None

    def get_predator_positions(self):
        """
        Get the positions of all predators in the environment.

        Returns:
        - list of tuples: Positions of all predators [(x1, y1), (x2, y2), ...]
        """
        predator_positions = []
        for agent in self.agents:
            if agent.itype == "predator":
                predator_positions.append((agent._x, agent._y))
        return predator_positions

    def get_prey_positions(self):
        """
        Get the positions of all preys in the environment.

        Returns:
        - list of tuples: Positions of all preys [(x1, y1), (x2, y2), ...]
        """
        prey_positions = []
        for agent in self.agents:
            if agent.itype == "prey":
                prey_positions.append((agent._x, agent._y))
        return prey_positions
    
    def get_agent_profile(self):
        agent_profile = {}

        for i, agent in enumerate(self.agents):
            if agent.itype in agent_profile:
                agent_profile[agent.itype]['n_agent'] += 1
                agent_profile[agent.itype]['idx'].append(i)
            else:
                agent_profile[agent.itype] = {
                    'n_agent': 1,
                    'idx': [i],
                    'act_spc': agent.act_spc,
                    'obs_dim': agent.obs_dim
                }

        return agent_profile

    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = []

        self.agents = self.world.agents
        self.world.physical_step(action_n)

        if self.pre_encode:
            self.bin_encoding = self.world.get_bin_encoding()

        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))
            info_n.append(self._get_info(agent))

        #affecting average base reward to all agents 
        a = reward_n[21]
        for i in range(0, 20):
            reward_n[i] = a

        self.episode_length += 1

        return obs_n, np.asarray(reward_n), np.asarray(done_n), info_n

    def ep_length(self):
        return self.episode_length
    
    def incentivize(self, incentives_n):
        rx_inc_n = []

        self.world.incentive_step(incentives_n)
        for agent in self.agents:
            agent.assign_incentive()
        for agent in self.agents:
            rx_inc_n.append(self._get_received(agent))

        return rx_inc_n

    def get_expenses(self, incentives_n):
        tx_inc_n = []
        
        self.world.incentive_step(incentives_n)
        for agent in self.agents:
            tx_inc_n.append(self._get_expenses(agent))

        return tx_inc_n

    def reset(self, args=None):
        # reset world
        self.reset_callback(self.world, args)

        if self.pre_encode:
            self.bin_encoding = self.world.get_bin_encoding()

        obs_n = []
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        self.episode_length = 0
        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    def _get_done(self, agent):
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        return self.reward_callback(agent, self.world)

    def _get_received(self, agent):
        if self.rx_callback == None:
            return 0
        return self.rx_callback(agent, self.world)

    def _get_expenses(self, agent):
        if self.tx_callback == None:
            return 0
        return self.tx_callback(agent, self.world)

    def get_full_encoding(self, encoding="binary"):
        if encoding == "binary":
            if self.bin_encoding is None:
                return self.world.get_bin_encoding()
            else:
                return self.bin_encoding
        elif encoding == "id":
            return self.world.get_id_encoding()
        else:
            return self.world.get_full_encoding()

    def get_coordinates(self, agent_idxs):
        x = []
        y = []
        for i in agent_idxs:
            x.append(self.agents[i]._x)
            y.append(self.agents[i]._y)

        return x, y
     
    def simulate_step(self, action_n):
        """
        Simulate a step to get the observations, rewards, done flags, and info without changing the environment's state.
        """
        # Save the current state
        saved_agents = []
        for agent in self.agents:
            saved_state = {
                'x': agent._x,
                'y': agent._y,
                'done_moving': agent.done_moving,
                'waiting': agent.waiting,
                'collided': agent.collided,
                'exists': agent.exists,
                'power': agent.power,
                'obs': agent.get_obs()  # Save the current observation
            }
            saved_agents.append(saved_state)

        # Apply the actions to compute the results
        obs_n = []
        reward_n = []
        done_n = []
        info_n = []

        if self.pre_encode:
            self.bin_encoding = self.world.get_bin_encoding()

        for i, agent in enumerate(self.agents):
            if not agent.exists or agent.power <= 0:
                reward_n.append(0)
                done_n.append(self._get_done(agent))
                info_n.append(self._get_info(agent))
                obs_n.append(agent.get_obs())
                continue

            original_pos = (agent._x, agent._y)
            action = action_n[i]
            new_x, new_y = self._get_new_position(original_pos, action)

            if not (0 <= new_x < self.grid.width and 0 <= new_y < self.grid.height):
                # Invalid move: out of bounds
                reward_n.append(self._get_reward(agent))
            else:
                intended_cell = self.grid.get(new_x, new_y)
                if intended_cell is not None:
                    # Invalid move: cell is occupied
                    reward_n.append(self._get_reward(agent))
                else:
                    # Temporarily move the agent
                    self.grid.set(agent._x, agent._y, None)
                    self.grid.set(new_x, new_y, agent)
                    agent.set_pos(new_x, new_y)

                    # Get the reward for the new position
                    reward_n.append(self._get_reward(agent))

                    # Revert the move
                    self.grid.set(new_x, new_y, None)
                    self.grid.set(agent._x, agent._y, agent)
                    agent.set_pos(original_pos[0], original_pos[1])

            # Get observation, done flag, and info for the simulated action
            obs_n.append(self._get_obs(agent))
            done_n.append(self._get_done(agent))
            info_n.append(self._get_info(agent))

        # Revert the environment to its original state
        for agent, saved_state in zip(self.agents, saved_agents):
            agent._x = saved_state['x']
            agent._y = saved_state['y']
            agent.done_moving = saved_state['done_moving']
            agent.waiting = saved_state['waiting']
            agent.collided = saved_state['collided']
            agent.exists = saved_state['exists']
            agent.power = saved_state['power']
            agent.obs = saved_state['obs']
            #agent.update_obs(saved_state['obs'])  # Restore the original observation
            #we are returning the same observation for the 
            #reward that the agent got from it ???
        return obs_n, np.asarray(reward_n), np.asarray(done_n), info_n

    def _get_new_position(self, pos, action):
        x, y = pos
        if action == N:
            y -= 1
        elif action == E:
            x += 1
        elif action == W:
            x -= 1
        elif action == S:
            y += 1
        return x, y

    def _is_position_valid(self, x, y):
        return 0 <= x < self.grid.width and 0 <= y < self.grid.height
