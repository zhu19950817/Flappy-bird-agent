import tensorflow as tf
import numpy as np
from collections import deque
import random


relu = tf.nn.relu
leaky_relu = tf.nn.leaky_relu
softmax = tf.nn.softmax

class DQN:
    def __init__(self,
                 n_action,
                 n_feature,
                 learning_rate=0.00025,
                 initial_epsilon=0.01,
                 final_epsilon=0.001,
                 exploration_frame=1000000,
                 discount_factor=0.99,
                 minibatch_size=32,
                 memory_size=50000,
                 update_frequency=100,
                 frame_per_action=1):

        self.n_action = n_action
        self.n_feature = n_feature
        self.learning_rate = learning_rate
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.exploration_frame = exploration_frame
        self.cur_epsilon = initial_epsilon
        self.discount_factor = discount_factor
        self.minibatch_size = minibatch_size
        self.memory_size = memory_size
        self.frame_per_action = frame_per_action
        self.state = tf.placeholder("float", [None, 80, 80, 4])
        self.replay_memory = deque()

        self.obsvr_size = 100
        self.cur_state = np.zeros([n_feature,n_feature,4])
        self.update_frequency = update_frequency

        self.time_step = 0
        with tf.variable_scope('behavior_net'):
            self.Q_val_b = self.network(self.state)
        with tf.variable_scope('target_net'):
            self.Q_val_t = self.network(self.state)

        self.action_input = tf.placeholder("float", [None, self.n_action])
        self.y_input = tf.placeholder("float", [None])
        self.Q_action = tf.reduce_sum(tf.multiply(self.Q_val_b, self.action_input), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - self.Q_action))
        self.train_op = tf.train.AdamOptimizer(1e-6).minimize(self.cost)

        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        self.b_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='behavior_net')
        self.target_replace_op = [tf.assign(t, b) for t, b in zip(self.t_params, self.b_params)]
        self.saver = tf.train.Saver()
        self.sess = tf.InteractiveSession()

        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter("logs/", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find any previous network weights")


    def initial_state(self, observation):
        self.cur_state = np.stack((observation, observation, observation, observation), axis=2)


    def conv2d_layer(self, inputs, filters, kernel_size, strides, activation, name):
        return tf.layers.conv2d(
            inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
            padding=('SAME'), use_bias=True,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
            activation=activation,
            reuse=tf.AUTO_REUSE, name=name)


    def fc_layer(self, inputs, units,activation, name):
        return tf.layers.dense(
            inputs=inputs, units=units, activation=activation,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
            name=name)


    def max_pool(self, inputs):
        return tf.nn.max_pool(inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


    def network(self, inputs):
        cv1 = self.conv2d_layer(inputs, filters=32, kernel_size=8, strides=4, activation=relu, name='cv1')
        tf.summary.histogram('cv1/outputs', cv1)
        p1 = self.max_pool(cv1)
        cv2 = self.conv2d_layer(p1, filters=64, kernel_size=4, strides=2, activation=relu, name='cv2')
        tf.summary.histogram('cv2/outputs', cv2)
        cv3 = self.conv2d_layer(cv2, filters=64, kernel_size=3, strides=1, activation=relu, name='cv3')
        tf.summary.histogram('cv3/outputs', cv3)
        cv3 = tf.reshape(cv3,[-1,1600])
        fc1 = self.fc_layer(cv3, units=512, activation=relu, name='fc1')
        tf.summary.histogram('fc1/outputs', fc1)
        fc2 = self.fc_layer(fc1, units=2, activation=None, name='fc2')
        tf.summary.histogram('fc2/outputs', fc2)
        return fc2


    def get_action(self):
        q_val = 0
        if self.time_step % self.frame_per_action == 0:
            if np.random.uniform() > self.cur_epsilon:
                q_val = self.sess.run(self.Q_val_b,feed_dict= {self.state:[self.cur_state]})
                action = np.argmax(q_val)
            else:
                action = np.random.randint(self.n_action)
                
        else:
            action = 0
        print("/ Q_VALUE", q_val)
        if self.cur_epsilon > self.final_epsilon and self.time_step > self.obsvr_size:
            self.cur_epsilon -= (self.initial_epsilon - self.final_epsilon) / self.exploration_frame
        return action


    def memory_store(self,nextObservation, action, reward, terminal):
        new_state = np.append(self.cur_state[:,:,1:],nextObservation,axis = 2)
        self.replay_memory.append((self.cur_state, action, reward, new_state, terminal))
        if len(self.replay_memory) > self.memory_size:
            self.replay_memory.popleft()
        if self.time_step > self.obsvr_size:
            self.train()



        print ("TIMESTEP", self.time_step, "/ EPSILON", self.cur_epsilon,
               "/ REWARD", reward, "/ ACTION", np.argmax(action), end="")

        self.cur_state = new_state
        self.time_step += 1



    def train(self):
        minibatch = random.sample(self.replay_memory, self.minibatch_size)
        cur_state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        q_val_batch = self.Q_val_t.eval(feed_dict={self.state: next_state_batch})

        y_batch = []
        for i in range(self.minibatch_size):
            terminal = minibatch[i][4]
            if terminal:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + self.discount_factor * np.max(q_val_batch[i]))

        self.sess.run(self.train_op,feed_dict={self.action_input: action_batch,
                                               self.y_input: y_batch,
                                     self.state: cur_state_batch})
        rs = self.sess.run(self.merged, feed_dict={self.action_input: action_batch,
                                               self.y_input: y_batch,
                                     self.state: cur_state_batch})
        self.writer.add_summary(rs, self.time_step)

        if self.time_step % self.update_frequency == 0:
            self.sess.run(self.target_replace_op)     

        if self.time_step % 10000 == 0:
            self.saver.save(self.sess, 'saved_networks/' + 'network' + '-dqn', global_step=self.time_step)