"""import numpy as np
import tensorflow as tf
import config

FLAGS = config.flags.FLAGS
gamma = FLAGS.gamma  # reward discount factor
history_len = FLAGS.history_len
lr = FLAGS.lr    # learning rate
h_nodes = 64
m_nodes = 32
n_prey = FLAGS.n_prey

class DQN:
    def __init__(self, sess, state_dim, sup_state_dim, action_dim, n_agents, nn_id):
        self.sess = sess
        self.state_dim = state_dim
        self.sup_state_dim = sup_state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents

        scope = 'dqn_' + str(nn_id)

        tf.compat.v1.disable_eager_execution()

        self.coords_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 2*(n_agents + n_prey)], name='coords_ph')
        self.next_coords_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 2*(n_agents + n_prey)], name='next_coords_ph')

        self.state_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, n_agents, state_dim])
        self.next_state_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, n_agents, state_dim])

        self.action_ph = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, n_agents])

        self.reward_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None])
        self.is_not_terminal_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None])
        self.lr = tf.compat.v1.placeholder(dtype=tf.float32)

        self.n_in = tf.compat.v1.placeholder(dtype=tf.int32)

        a_onehot = tf.one_hot(self.action_ph, action_dim, 1.0, 0.0, axis=-1)

        self.closest_indices_lists = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, n_agents, 4])  

        with tf.compat.v1.variable_scope(scope):
            self.ind_dqns = []
            q_values = []

            for i in range(n_agents):
                ind_dqn = self.generate_indiv_dqn(self.state_ph[:, i], i)
                self.ind_dqns.append(ind_dqn)

                action_qval = tf.reshape(tf.reduce_sum(tf.multiply(ind_dqn, a_onehot[:, i]), axis=1), (-1, 1))
                q_values.append(action_qval)

            self.concat_dqns = tf.reshape(tf.concat(self.ind_dqns, 1), (-1, self.n_agents, self.action_dim))
            q_values = tf.concat(q_values, axis=1)

            self.mixing_networks = []
            
            for i in range(n_agents):
                closest_agents = self.closest_indices_lists[:, i, :]  
                selected_q_values = tf.gather(q_values, closest_agents, axis=1)  

                # Randomly sample from these 4 Q-values to fill the 10 required Q-values
                num_needed = self.n_agents
                random_samples = tf.random.uniform([tf.shape(selected_q_values)[0], num_needed], maxval=4, dtype=tf.int32)
                sampled_q_values = tf.gather(tf.reshape(selected_q_values, [-1, 4]), random_samples, axis=1)
                sampled_q_values = tf.reshape(sampled_q_values, [-1, num_needed])

                mixing_network = self.generate_mixing_network(self.state_ph, self.coords_ph, sampled_q_values)
                self.mixing_networks.append(mixing_network)
            
            self.q_total = tf.reduce_sum(tf.concat(self.mixing_networks, axis=1))
            print('self.q_total', self.q_total, self.q_total.shape)

        with tf.compat.v1.variable_scope('slow_target_' + scope):
            self.ind_target_dqns = []
            next_q_values = []

            for i in range(n_agents):
                ind_target_dqn = self.generate_indiv_dqn(self.next_state_ph[:, i], i, trainable=False)
                self.ind_target_dqns.append(ind_target_dqn)

                max_qval_next = tf.reshape(tf.reduce_max(ind_target_dqn, axis=1), (-1, 1))
                next_q_values.append(max_qval_next)

            next_q_values = tf.concat(next_q_values, axis=1)
            self.mixing_networks_targets = []
                
            for i in range(n_agents):
                closest_agents = self.closest_indices_lists[:, i, :]  
                selected_q_values = tf.gather(next_q_values, closest_agents, axis=1) 

                num_needed = self.n_agents
                random_samples = tf.random.uniform([tf.shape(selected_q_values)[0], num_needed], maxval=4, dtype=tf.int32)
                sampled_q_values = tf.gather(tf.reshape(selected_q_values, [-1, 4]), random_samples, axis=1)
                sampled_q_values = tf.reshape(sampled_q_values, [-1, num_needed])

                mixing_network = self.generate_mixing_network(self.next_state_ph, self.next_coords_ph, sampled_q_values, trainable=False)
                self.mixing_networks_targets.append(mixing_network)

            self.slow_q_total = tf.reduce_sum(tf.concat(self.mixing_networks_targets, axis=1))

        q_network_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        target_q_network_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='slow_target_' + scope)

        discount = self.is_not_terminal_ph * gamma
        target = self.reward_ph + discount * self.slow_q_total
        self.td_errors = tf.reduce_sum(tf.square(target - self.q_total))
        print(self.td_errors.shape)
        print(len(q_network_vars))
        self.train_network = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(self.td_errors, var_list=q_network_vars)

        update_slow_target_ops = []
        for i in range(len(q_network_vars)):
            assign_op = tf.compat.v1.assign(target_q_network_vars[i], q_network_vars[i])
            update_slow_target_ops.append(assign_op)
        self.update_slow_target_dqn = tf.group(*update_slow_target_ops)

    def generate_indiv_dqn(self, s, a_id, trainable=True):
        side = int(np.sqrt((self.state_dim - self.sup_state_dim*history_len)//(history_len*3)))

        if self.sup_state_dim > 0:
            obs = tf.reshape(s, (-1, history_len, self.state_dim//history_len))
            sup = tf.reshape(obs[:,:,-1*self.sup_state_dim:], (-1, history_len*self.sup_state_dim))
            obs = tf.reshape(obs[:,:,:-1*self.sup_state_dim], (-1, history_len, side*side*3))
            obs = tf.transpose(obs, perm=[0,2,1])
            obs = tf.reshape(obs, (-1,side,side,history_len*3))
        else:
            obs = tf.reshape(s, (-1, history_len, side*side*3))
            obs = tf.transpose(obs, perm=[0,2,1])
            obs = tf.reshape(obs, (-1,side,side,history_len*3))

        conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation=tf.nn.relu)(obs)
        conv1 = tf.keras.layers.Flatten()(conv1)

        if self.sup_state_dim > 0:
            concat = tf.concat([conv1, sup], axis=1)
        else:
            concat = conv1

        hidden = tf.keras.layers.Dense(h_nodes, activation=tf.nn.relu)(concat)
        hidden2 = tf.keras.layers.Dense(h_nodes, activation=tf.nn.relu)(hidden)
        q_values = tf.keras.layers.Dense(self.action_dim)(hidden2)

        return q_values


    def generate_mixing_network(self, state, coords, q_values, trainable=True):
        if self.sup_state_dim > 0:
            sup_state = state[:,:,-1*self.sup_state_dim:]
            sup_state = tf.reshape(sup_state, (-1,self.sup_state_dim*self.n_agents))
            hyper_in = tf.concat([coords, sup_state], axis=1)
        else:
            hyper_in = coords

        # For mixing network layer 1 (linear)
        w1 = tf.keras.layers.Dense(self.n_agents*m_nodes, use_bias=True)(hyper_in)

        # For mixing network layer 2
        w2 = tf.keras.layers.Dense(m_nodes, use_bias=True)(hyper_in)

        # For mixing network hidden layer (linear)
        b1 = tf.keras.layers.Dense(m_nodes, use_bias=True)(hyper_in)

        # For mixing network output layer (2-layer hypernetwork with ReLU)
        b2_h = tf.keras.layers.Dense(m_nodes, activation=tf.nn.relu, use_bias=True)(hyper_in)
        b2 = tf.keras.layers.Dense(1, use_bias=True)(b2_h)

        w1 = tf.reshape(tf.abs(w1), [-1, self.n_agents, m_nodes])
        w2 = tf.reshape(tf.abs(w2), [-1, m_nodes, 1])

        q_values = tf.reshape(q_values, [-1, 1, self.n_agents])
        q_hidden = tf.nn.elu(tf.reshape(tf.matmul(q_values, w1), [-1, m_nodes]) + b1)
        q_hidden = tf.reshape(q_hidden, [-1, 1, m_nodes])
        q_total = tf.reshape(tf.matmul(q_hidden, w2), [-1, 1]) + b2

        return q_total

    def get_q_values(self, state_ph):
        return self.sess.run(self.concat_dqns, feed_dict={self.state_ph: state_ph, self.n_in: len(state_ph)})

    def training_qnet(self, coords_ph, state_ph, action_ph, reward_ph, is_not_terminal_ph, next_coords_ph, next_state_ph, closest_indices_lists, lr=lr):
        return self.sess.run([self.td_errors, self.train_network], 
            feed_dict={
                self.coords_ph: coords_ph,
                self.state_ph: state_ph,
                self.next_coords_ph: next_coords_ph,
                self.next_state_ph: next_state_ph,
                self.action_ph: action_ph,
                self.reward_ph: reward_ph,
                self.is_not_terminal_ph: is_not_terminal_ph,
                self.n_in: len(coords_ph),
                self.closest_indices_lists : closest_indices_lists, 
                self.lr: lr
            })

    def training_target_qnet(self):
        self.sess.run(self.update_slow_target_dqn)
"""


import numpy as np
import tensorflow as tf
import config


FLAGS = config.flags.FLAGS
gamma = FLAGS.gamma  
history_len = FLAGS.history_len
lr = FLAGS.lr 
h_nodes = 64
m_nodes = 32
n_prey = FLAGS.n_prey
 

class DQN:
    def __init__(self, sess, state_dim, sup_state_dim, action_dim, n_agents, nn_id):
        self.sess = sess
        self.state_dim = state_dim
        self.sup_state_dim = sup_state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents

        scope = 'dqn_' + str(nn_id)

        tf.compat.v1.disable_eager_execution()

        self.coords_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 2*(n_agents + n_prey)], name='coords_ph')
        self.next_coords_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 2*(n_agents + n_prey)], name='next_coords_ph')

        self.state_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, n_agents, state_dim])
        self.next_state_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, n_agents, state_dim])

        self.action_ph = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, n_agents])

        self.reward_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, n_agents])
        self.is_not_terminal_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, n_agents])
        self.lr = tf.compat.v1.placeholder(dtype=tf.float32)

        self.n_in = tf.compat.v1.placeholder(dtype=tf.int32)

        a_onehot = tf.one_hot(self.action_ph, action_dim, 1.0, 0.0, axis=-1)

        self.closest_indices_lists = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, n_agents, 6])  
        self.next_closest_indices_lists = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, n_agents,6])  

        self.slow_q_totals, self.q_totals = [], []

        with tf.compat.v1.variable_scope(scope):
            self.ind_dqns = []
            q_values = []

            for i in range(n_agents):
                ind_dqn = self.generate_indiv_dqn(self.state_ph[:, i], i)
                self.ind_dqns.append(ind_dqn)

                action_qval = tf.reshape(tf.reduce_sum(tf.multiply(ind_dqn, a_onehot[:, i]), axis=1), (-1, 1))
                q_values.append(action_qval)

            self.concat_dqns = tf.reshape(tf.concat(self.ind_dqns, 1), (-1, self.n_agents, self.action_dim))
            q_values = tf.concat(q_values, axis=1)

            for i in range(n_agents):
                selected_q_values = []
                closest_agents = self.closest_indices_lists[:, i, :]  
                selected_q_values = tf.gather(q_values, closest_agents, axis=1, batch_dims=1)

                x_coords = tf.gather(self.coords_ph, closest_agents * 2, axis=1, batch_dims=1)

                y_coords = tf.gather(self.coords_ph, closest_agents * 2 + 1, axis=1, batch_dims=1)

                coords_ph = tf.stack([x_coords, y_coords], axis=2)
                coords_ph = tf.reshape(coords_ph, [-1, 12])

                selected_q_values = tf.reshape(selected_q_values, [-1, 6])

                '''num_needed = self.n_agents
                random_samples = tf.random.uniform([tf.shape(selected_q_values)[0], num_needed], maxval=4, dtype=tf.int32)
                sampled_q_values = tf.gather(tf.reshape(selected_q_values, [-1, 4]), random_samples, axis=1)
                sampled_q_values = tf.concat(sampled_q_values, axis=1)'''

                self.q_total = self.generate_mixing_network(self.state_ph, coords_ph, selected_q_values)

                self.q_total = tf.concat(self.q_total, axis=1)

                self.q_totals.append(self.q_total)

        with tf.compat.v1.variable_scope('slow_target_' + scope):
            self.ind_target_dqns = []
            next_q_values = []

            for i in range(n_agents):
                ind_target_dqn = self.generate_indiv_dqn(self.next_state_ph[:, i], i, trainable=False)
                self.ind_target_dqns.append(ind_target_dqn)

                max_qval_next = tf.reshape(tf.reduce_max(tf.multiply(ind_target_dqn, a_onehot[:, i]), axis=1), (-1, 1))
                next_q_values.append(max_qval_next)

            next_q_values = tf.concat(next_q_values, axis=1)
                
            for i in range(n_agents):
                next_closest_agents = self.next_closest_indices_lists[:, i, :]  
                selected_slow_q_values = tf.gather(next_q_values, next_closest_agents, axis=1, batch_dims=1)
                x_next_coords_ph = tf.gather(self.next_coords_ph, next_closest_agents * 2, axis=1, batch_dims=1)

                y_next_coords_ph = tf.gather(self.next_coords_ph, next_closest_agents * 2 + 1, axis=1, batch_dims=1)

                next_coords_ph = tf.stack([x_next_coords_ph, y_next_coords_ph], axis=2)
                next_coords_ph = tf.reshape(next_coords_ph, [-1, 12])

                selected_slow_q_values = tf.reshape(selected_slow_q_values, [-1, 6])
                
                '''selected_slow_q_values = tf.gather(next_q_values, next_closest_agents, axis=1) 
                next_coords_ph = tf.gather(self.next_coords_ph, 2*next_closest_agents, axis=1) 

                random_samples = tf.random.uniform([tf.shape(selected_slow_q_values)[0], num_needed], maxval=4, dtype=tf.int32)
                sampled_slow_q_values = tf.gather(tf.reshape(selected_slow_q_values, [-1, 4]), random_samples, axis=1)
            
                sampled_slow_q_values = tf.concat(sampled_slow_q_values, axis=1)
                '''

                self.slow_q_total = self.generate_mixing_network(self.next_state_ph, next_coords_ph, selected_slow_q_values, trainable=False)
                self.slow_q_totals.append(self.slow_q_total)

            q_network_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=scope)
            target_q_network_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='slow_target_' + scope)

            discount = self.is_not_terminal_ph * gamma
            target = self.reward_ph + discount * self.slow_q_totals

            self.td_errors = tf.reduce_sum(tf.square(target - self.q_totals))

            self.train_networks = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(self.td_errors, var_list=q_network_vars)


        update_slow_target_ops = []
        for i in range(len(q_network_vars)):
            assign_op = tf.compat.v1.assign(target_q_network_vars[i], q_network_vars[i])
            update_slow_target_ops.append(assign_op)
        self.update_slow_target_dqn = tf.group(*update_slow_target_ops)


    def generate_indiv_dqn(self, s, a_id, trainable=True):
        side = int(np.sqrt((self.state_dim - self.sup_state_dim*history_len)//(history_len*3)))

        if self.sup_state_dim > 0:
            obs = tf.reshape(s, (-1, history_len, self.state_dim//history_len))
            sup = tf.reshape(obs[:,:,-1*self.sup_state_dim:], (-1, history_len*self.sup_state_dim))
            obs = tf.reshape(obs[:,:,:-1*self.sup_state_dim], (-1, history_len, side*side*3))
            obs = tf.transpose(obs, perm=[0,2,1])
            obs = tf.reshape(obs, (-1,side,side,history_len*3))
        else:
            obs = tf.reshape(s, (-1, history_len, side*side*3))
            obs = tf.transpose(obs, perm=[0,2,1])
            obs = tf.reshape(obs, (-1,side,side,history_len*3))

        conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation=tf.nn.relu)(obs)
        conv1 = tf.keras.layers.Flatten()(conv1)

        if self.sup_state_dim > 0:
            concat = tf.concat([conv1, sup], axis=1)
        else:
            concat = conv1

        hidden = tf.keras.layers.Dense(h_nodes, activation=tf.nn.relu)(concat)
        hidden2 = tf.keras.layers.Dense(h_nodes, activation=tf.nn.relu)(hidden)
        q_values = tf.keras.layers.Dense(self.action_dim)(hidden2)

        return q_values


    ''''def generate_mixing_network(self, state, coords, q_values, trainable=True):
        hyper_in = tf.concat([state], axis=1)  

        w1 = tf.keras.layers.Dense(self.n_agents*m_nodes, use_bias=True)(hyper_in)

        w2 = tf.keras.layers.Dense(m_nodes, use_bias=True)(hyper_in)

        b1 = tf.keras.layers.Dense(m_nodes, use_bias=True)(hyper_in)

        b2_h = tf.keras.layers.Dense(m_nodes, activation=tf.nn.relu, use_bias=True)(hyper_in)
        b2 = tf.keras.layers.Dense(1, use_bias=True)(b2_h)

        w1 = tf.reshape(tf.abs(w1), [-1, self.n_agents, m_nodes])
        w2 = tf.reshape(tf.abs(w2), [-1, m_nodes, 1])
        
        q_values = tf.concat([q_values], axis=1)   

        q_values = tf.reshape(q_values, [-1, m_nodes, self.n_agents])
        q_hidden = tf.nn.elu(tf.reshape(tf.matmul(q_values, w1), [-1, m_nodes]) + b1)
        q_hidden = tf.reshape(q_hidden, [-1, 1, m_nodes])
        q_total = tf.reshape(tf.matmul(q_hidden, w2), [-1, 1]) + b2

        return q_total'''
 

    def generate_mixing_network(self, state, coords, q_values, trainable=True):
        
        if self.sup_state_dim > 0:
            sup_state = state[:,:,-1*self.sup_state_dim:]
            sup_state = tf.reshape(sup_state, (-1,self.sup_state_dim*self.n_agents))
            hyper_in = tf.concat([coords, sup_state], axis=1)
        else:
            hyper_in = coords

        num_units = 6 * m_nodes
        w1 = tf.keras.layers.Dense(num_units, use_bias=True, trainable=trainable, name='dense_w1')(hyper_in)
        num_units = m_nodes
        w2 = tf.keras.layers.Dense(num_units, use_bias=True, trainable=trainable, name='dense_w2')(hyper_in)
        b1 = tf.keras.layers.Dense(m_nodes, use_bias=True, trainable=trainable, name='dense_b1')(hyper_in)

        b2_h = tf.keras.layers.Dense(m_nodes, activation=tf.nn.relu, use_bias=True, trainable=trainable, name='dense_b2_h')(hyper_in)
        b2 = tf.keras.layers.Dense(1, use_bias=True, trainable=trainable, name='dense_b2')(b2_h)

        w1 = tf.reshape(tf.abs(w1), [-1, 6, m_nodes])
        w2 = tf.reshape(tf.abs(w2), [-1, m_nodes, 1])

        q_values = tf.reshape(q_values, [-1,1,6])
        q_hidden = tf.nn.elu(tf.reshape(tf.matmul(q_values, w1),[-1,m_nodes]) + b1)
        q_hidden = tf.reshape(q_hidden, [-1,1,m_nodes])
        q_total = tf.reshape(tf.matmul(q_hidden, w2),[-1,1]) + b2

        return q_total


    def get_q_values(self, state_ph):
        return self.sess.run(self.concat_dqns, feed_dict={self.state_ph: state_ph, self.n_in: len(state_ph)})


    def training_qnet(self, coords_ph, state_ph, action_ph, reward_ph, is_not_terminal_ph, next_coords_ph, next_state_ph, closest_indices_lists, next_closest_indices_lists, lr=lr):
        return self.sess.run([self.td_errors, self.train_networks], 
            feed_dict={
                self.coords_ph: coords_ph,
                self.state_ph: state_ph,
                self.next_coords_ph: next_coords_ph,
                self.next_state_ph: next_state_ph,
                self.action_ph: action_ph,
                self.reward_ph: reward_ph,
                self.is_not_terminal_ph: is_not_terminal_ph,
                self.n_in: len(coords_ph),
                self.closest_indices_lists : closest_indices_lists, 
                self.next_closest_indices_lists : next_closest_indices_lists,
                self.lr: lr
            })

    def training_target_qnet(self):
        self.sess.run(self.update_slow_target_dqn)



