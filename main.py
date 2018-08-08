import gym
import random
# rule and interface for cartpole-v0: https://github.com/openai/gym/wiki/CartPole-v0
env = gym.make('CartPole-v0')

def toint(f):
    return round(f*1000)

Q = dict()
gamma = 0.6
for i_episode in range(100):
    observation = env.reset()
    for t in range(100):
        env.render()
        #print(observation)
        state = toint(observation[2])
        r = random.randint(0, 9)
        if state in Q and r <= 4:
            if Q[state][0] > Q[state][1]:
                action = 0
            else:
                action = 1
        else:
            if state < 0:
                action = 0
            else:
                action = 1                    
                
        observation, reward, done, info = env.step(action)
        next_state = toint(observation[2])

        next_max = 0
        if next_state in Q:
            if Q[next_state][0] >= Q[next_state][1]:
                next_max = Q[next_state][0]
            else:
                next_max = Q[next_state][1]

        if state not in Q:
            Q[state] = [0, 0]        
            
        Q[state][action] = reward + gamma * next_max                
                
        if done:
            #print(Q)
            print("Episode finished after {} timesteps".format(t+1))
            break
