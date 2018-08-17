# Cartpole DQN

# Deep Q-Learning Network with Keras and OpenAI Gym, based on Keon Kim's code](https://github.com/keon/deep-q-learning/blob/master/dqn.py).

import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import os


env = gym.make('CartPole-v0') # initialise environment
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
batch_size = 32

n_episodes = 1001 # n games we want agent to play (default 1001)

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
        self.epsilon_min = 0.01 # minimum amount of random exploration permitted
        self.learning_rate = 0.001 # rate at which NN adjusts models parameters via SGD to reduce cost 
        self.model = self._build_model() # private method 
    def _build_model(self):
        # neural net to approximate Q-value function:
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu')) # 1st hidden layer; states as input
        model.add(Dense(24, activation='relu')) # 2nd hidden layer
        model.add(Dense(self.action_size, activation='linear')) # 2 actions, so 2 output neurons: 0 and 1 (L/R)
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model
            
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # list of previous experiences, enabling re-training later
                
    def act(self, state):
        if np.random.rand() <= self.epsilon: # if acting randomly, take random action
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        a = np.argmax(act_values[0])
        # print('act_values:', act_values[0])
        return a

    # method that trains NN with experiences sampled from memory
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size) # sample a minibatch from memory
        for state, action, reward, next_state, done in minibatch: # extract data for each minibatch sample
            target = reward # if done (boolean whether game ended or not, i.e., whether final state or not), then target = reward
            if not done: # if not done, then predict future discounted reward
                # (target) = reward + (discount rate gamma) * max(next_state value)
                target = (reward + self.gamma * 
                              np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0) 
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
                    
    def load(self, name):
        self.model.load_weights(name)
        
    def save(self, name):
        print('save name:', name)
        self.model.save_weights(name)
        #### Interact with environment

agent = DQNAgent(state_size, action_size) # initialise agent
done = False
for e in range(n_episodes): # iterate over new episodes of the game
    state = env.reset() # reset state at start of each new episode of the game
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
            print("episode: {}/{}, score: {}, e: {:.2}"
                  .format(e, n_episodes, time, agent.epsilon))
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
    if e % 50 == 0:
        agent.save(output_dir + "weights_" + '{:04d}'.format(e) + '.hdf5')
