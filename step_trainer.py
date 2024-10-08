from agents.qmix_full.agent import Agent
from agents.base_trainer import Trainer as BaseTrainer
from agents.base_trainer import stringify
from agents.simple_agent import RunningAgent as NonLearningAgent
import numpy as np
import tensorflow as tf
import config
from datetime import datetime

np.set_printoptions(precision=2)

FLAGS = config.flags.FLAGS
minibatch_size = FLAGS.minibatch_size
n_predator = FLAGS.n_predator
n_prey = FLAGS.n_prey
test_interval = FLAGS.test_interval
train_interval = FLAGS.train_interval
map_size = FLAGS.map_size



class Trainer(BaseTrainer):
    def __init__(self, environment, logger):
        self.env = environment
        self.logger = logger
        self.n_agents = n_predator + n_prey

        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)))

        self._agent_profile = self.env.get_agent_profile()
        agent_precedence = self.env.agent_precedence

        self.predator_singleton = Agent(
            act_space=self._agent_profile["predator"]["act_spc"],
            obs_space=self._agent_profile["predator"]["obs_dim"],
            sess=self.sess,
            n_agents=n_predator,
            name="predator"
        )

        self.agents = []
        for i, atype in enumerate(agent_precedence):
            if atype == "predator":
                agent = self.predator_singleton
            else:
                agent = NonLearningAgent(self._agent_profile[atype]["act_spc"])

            self.agents.append(agent)

        # Initialize tf variables
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.saver = tf.compat.v1.train.Saver()

        if FLAGS.load_nn:
            if FLAGS.nn_file == "":
                logger.error("No file for loading Neural Network parameter")
                exit()
            self.saver.restore(self.sess, FLAGS.nn_file)
        else:
            self.predator_singleton.sync_target()

    def get_neighbors_within_observable_space(self, position, predator_positions, observable_range=(7, 7)):
        neighbors = []
        x, y = position
        x_min, x_max = x - observable_range[0] // 2, x + observable_range[0] // 2
        y_min, y_max = y - observable_range[1] // 2, y + observable_range[1] // 2
        for i, (nx, ny) in enumerate(predator_positions):
            if x_min <= nx <= x_max and y_min <= ny <= y_max and (nx, ny) != position:
                neighbors.append((i, nx, ny))
        return neighbors

    def get_closest_agents(self, position, predator_positions, k=6):
        distances = []
        for i, pos in enumerate(predator_positions):
            if pos != position:
                dist = np.sqrt((pos[0] - position[0]) ** 2 + (pos[1] - position[1]) ** 2)
                distances.append((i, dist))
        distances.sort(key=lambda x: x[1])
        closest_indices = [i for i, _ in distances[:k]]
        return closest_indices


    def learn(self, max_global_steps, max_step_per_ep):
        epsilon = 1.0
        epsilon_dec = 1.0/(FLAGS.explore)
        epsilon_min = 0.1

        start_time = datetime.now()

        if max_global_steps % test_interval != 0:
            max_global_steps += test_interval - (max_global_steps % test_interval)

        steps_before_train = min(FLAGS.minibatch_size*4, FLAGS.rb_capacity)

        tds = []
        ep = 0
        global_step = 0
        
        while global_step < max_global_steps:
            ep += 1
            obs_n = self.env.reset()
            ax, ay = self.env.get_coordinates(range(self.n_agents))
            coords = np.asarray([ax, ay], dtype=np.float32).reshape(-1) / (map_size - 1)

            for step in range(max_step_per_ep):
                global_step += 1
                act_n = self.get_actions(obs_n, epsilon)

                predator_positions = self.env.get_predator_positions()
                closest_indices_lists = []
                for predator in range(0, n_predator):
                    closest_indices = self.get_closest_agents(predator_positions[predator], predator_positions)
                    closest_indices_lists.append(closest_indices)

                obs_n_next, reward_n, done_n, _ = self.env.step(act_n)
                done = done_n[:n_predator].all()
                done_n[:n_predator] = done
                ax, ay = self.env.get_coordinates(range(self.n_agents))
                next_coords = np.asarray([ax, ay], dtype=np.float32).reshape(-1) / (map_size - 1)

                next_predator_positions = self.env.get_predator_positions()
                next_closest_indices_lists = []
                for predator in range(0, n_predator):
                    next_closest_indices = self.get_closest_agents(next_predator_positions[predator], next_predator_positions)
                    next_closest_indices_lists.append(next_closest_indices)
                
                exp = [obs_n[:n_predator], act_n[:n_predator], reward_n[:n_predator],
                    obs_n_next[:n_predator], done_n[:n_predator], coords, next_coords, closest_indices_lists, next_closest_indices_lists]
                
                
                self.predator_singleton.add_to_memory(exp)

                if global_step > steps_before_train and global_step % train_interval == 0:
                    td = self.predator_singleton.train()
                    tds.append(td)

                if global_step % test_interval == 0:
                    
                    mean_steps, mean_b_reward, mean_captured, success_rate, rem_bat = self.test(25, max_step_per_ep)
                
                    time_diff = datetime.now() - start_time
                    start_time = datetime.now()

                    est = (max_global_steps - global_step)*time_diff/test_interval 
                    etd = est + start_time

                    print(global_step, ep, "%0.2f"%(mean_steps), mean_b_reward[:n_predator], "%0.2f"%mean_b_reward[:n_predator].mean(), "%0.2f"%epsilon)
                    print("estimated time remaining %02d:%02d (%02d:%02d)"%(est.seconds//3600,(est.seconds%3600)//60,etd.hour,etd.minute))
                
                    self.logger.info("%d\tsteps\t%0.2f" %(global_step, mean_steps))
                    self.logger.info("%d\tb_rwd\t%s" %(global_step, stringify(mean_b_reward[:n_predator],"\t")))
                    self.logger.info("%d\tcaptr\t%s" %(global_step, stringify(mean_captured[:n_predator], "\t")))
                    self.logger.info("%d\tsuccs\t%s" %(global_step, stringify(success_rate[:n_predator], "\t")))
                    self.logger.info("%d\tbttry\t%s" %(global_step, stringify(rem_bat, "\t")))

                    td = np.asarray(tds).mean()
                    self.logger.info("%d\ttd_er\t%0.2f" %(global_step, td))
                    tds = []

                if done or global_step == max_global_steps: 
                    break

                obs_n = obs_n_next
                coords = next_coords
                epsilon = max(epsilon_min, epsilon - epsilon_dec)
