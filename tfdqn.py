# Cartpole DQN

# Deep Q-Learning Network with Keras and OpenAI Gym, based on Keon Kim's code](https://github.com/keon/deep-q-learning/blob/master/dqn.py).

import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import tensorflow as tf
import os

env = gym.make('CartPole-v0')  # initialise environment
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
batch_size = 32

n_episodes = 1001  # n games we want agent to play (default 1001)

output_dir = 'model_output/cartpole/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


#### Define agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        # double-ended queue; acts like list, but elements can be added/removed from either end
        self.memory = deque(maxlen=2000)
        # decay or discount rate: enables agent to take into account future actions in addition to the immediate ones, but discounted at this rate
        self.gamma = 0.95
        # exploration rate: how much to act randomly; more initially than later due to epsilon decay
        self.epsilon = 1.0
        # decrease number of random explorations as the agent's performance (hopefully) improves over time
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01  # minimum amount of random exploration permitted
        self.learning_rate = 0.001  # rate at which NN adjusts models parameters via SGD to reduce cost
        self.model = self._build_model()  # private method

    def _build_model(self):

        self.sess = tf.Session()

        # neural net to approximate Q-value function:
        c_names = ['train_net', tf.GraphKeys.GLOBAL_VARIABLES]
        self.s = tf.placeholder(
            tf.float32, [None, self.state_size], name='s')  # input
        self.a = tf.placeholder(
            tf.float32, [None, self.action_size], name='a')  # input

        w1 = tf.get_variable(
            'w1', [self.state_size, 24],
            initializer=tf.random_normal_initializer(0., 0.3),
            collections=c_names)
        b1 = tf.get_variable(
            'b1', [1, 24],
            initializer=tf.constant_initializer(0.1),
            collections=c_names)
        l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)
        w2 = tf.get_variable(
            'w2', [24, 24],
            initializer=tf.random_normal_initializer(0., 0.3),
            collections=c_names)
        b2 = tf.get_variable(
            'b2', [1, 24],
            initializer=tf.constant_initializer(0.1),
            collections=c_names)
        l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

        w3 = tf.get_variable(
            'w3', [24, action_size],
            initializer=tf.random_normal_initializer(0., 0.3),
            collections=c_names)
        b3 = tf.get_variable(
            'b3', [1, action_size],
            initializer=tf.constant_initializer(0.1),
            collections=c_names)
        self.l3 = tf.matmul(l2, w3) + b3

        #model = Sequential()
        #model.add(
        #   Dense(24, input_dim=self.state_size,
        #        activation='relu'))  # 1st hidden layer; states as input
        #model.add(Dense(24, activation='relu'))  # 2nd hidden layer
        #model.add(Dense(self.action_size, activation='linear')
        #         )  # 2 actions, so 2 output neurons: 0 and 1 (L/R)
        #model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        self.loss = tf.reduce_mean(tf.squared_difference(self.l3, self.a))

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss)

        self.sess.run(tf.global_variables_initializer())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(
            (state, action, reward, next_state,
             done))  # list of previous experiences, enabling re-training later

    def act(self, state):
        if np.random.rand(
        ) <= self.epsilon:  # if acting randomly, take random action
            return random.randrange(self.action_size)
        #act_values = self.model.predict(state)
        act_values = self.sess.run(self.l3, feed_dict={self.s: state})
        a = np.argmax(act_values[0])
        # print('act_values:', act_values[0])
        return a

    # method that trains NN with experiences sampled from memory
    def replay(self, batch_size):
        minibatch = random.sample(self.memory,
                                  batch_size)  # sample a minibatch from memory

        for state, action, reward, next_state, done in minibatch:  # extract data for each minibatch sample
            target = reward  # if done (boolean whether game ended or not, i.e., whether final state or not), then target = reward
            if not done:  # if not done, then predict future discounted reward
                # (target) = reward + (discount rate gamma) * max(next_state value)
                res = self.sess.run(self.l3, feed_dict={self.s: next_state})[0]
                target = (reward + self.gamma * np.amax(
                    self.sess.run(self.l3, feed_dict={self.s: next_state})[0]))

            # target_f = self.model.predict(state)
            target_f = self.sess.run(self.l3, feed_dict={self.s: state})
            target_f[0][action] = target
            # target_f[0][action] = target
            self.sess.run(
                [self.optimizer, self.loss],
                feed_dict={
                    self.s: state,
                    self.a: target_f
                })

            # self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        print('save name:', name)
        ##self.model.save_weights(name)
        #### Interact with environment


agent = DQNAgent(state_size, action_size)  # initialise agent
done = False
for e in range(n_episodes):  # iterate over new episodes of the game
    state = env.reset()  # reset state at start of each new episode of the game
    state = np.reshape(state, [1, state_size])

    for time in range(5000):
        # env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print("episode: {}/{}, score: {}, e: {:.2}".format(
                e, n_episodes, time, agent.epsilon))
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
    if e % 50 == 0:
        agent.save(output_dir + "weights_" + '{:04d}'.format(e) + '.hdf5')
