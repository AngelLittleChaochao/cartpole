import gym
import random
# rule and interface for cartpole-v0: https://github.com/openai/gym/wiki/CartPole-v0
env = gym.make('CartPole-v0')

def toint(f):
    return round(f*1000)

# --------------------------------------------
# Get unique state value for observed state.
# @para x: x position
# --------------------------------------------
def get_state(x, x_dot, theta, theta_dot):
    one_degree = 0.0174532
    six_degrees = 0.1047192
    twelve_degrees = 0.2094384
    fifty_degrees = 0.87266
    
    if x < -2.4 or x > 2.4 or theta < -0.2094384 or theta > 0.2094384:
        return -1
    state = 0

    if x < -0.8:
        state = 0
    elif x < 0.8:
        state = 1
    else:
        state = 2

    if x_dot < -0.5:
        state = state
    elif x_dot < 0.5:
        state += 3
    else:
        state += 6

    if theta < -six_degrees:
        state = state
    elif theta < -one_degree:
        state += 9
    elif theta < 0:
        state += 18
    elif theta < one_degree:
        state += 27
    elif theta < six_degrees:
        state += 36
    else:
        state += 45

    if theta_dot < -fifty_degrees:
        state = state
    elif theta_dot < fifty_degrees:
        state += 54
    else:
        state += 108

    return state    

Q = dict()
gamma = 0.9
prev_state = -1
prev_action = -1
for i_episode in range(1000):
    observation = env.reset()
    for t in range(100):
        env.render()
        #print(observation)        
        state = get_state(observation[0], observation[1], observation[2], observation[3])

        -- update prev_state Q value
        if prev_state ~= -1 and prev_action ~= -1:
            if prev_state not in Q:
                Q[prev_state] = [0,0]
            if state not in Q:
                Q[state] = [0,0]
            Q[prev_state][action] = reward + gamma * next_max
        
        
        r = random.randint(0, 9)
        if state in Q and r <= 6:
            if Q[state][0] > Q[state][1]:
                action = 0
            else:
                action = 1
        else:
            action = random.randint(0, 1)            
                
        observation, reward, done, info = env.step(action)
        next_state = get_state(observation[0], observation[1], observation[2], observation[3])

        next_max = 0
        if next_state in Q:
            if Q[next_state][0] >= Q[next_state][1]:
                next_max = Q[next_state][0]
            else:
                next_max = Q[next_state][1]

        if state not in Q:
            Q[state] = [0, 0]        
            
        
                
        if done:
            #print(Q)
            print("Episode finished after {} timesteps".format(t+1))
            break
