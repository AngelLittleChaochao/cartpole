## Q-learning

The concept of Q-learning is simple, but when we use it to solve problems, there are different tricks. When decide action, we both consider exploration and exploitation, it is considering old experience and also has chance to find new ways.

### Q-table

Through learning, Q-table has the (state, action) value, it is referenced to make decisions. Q-table value is updated by the formula:

Q[state, action] = R(state, action) + Gamma * max(Q[next_state, action])

* R(state, action) is the reward for taking action in state 'state'. It usually returns by the environment.
* Gamma is learning rate, from 0 to 1. 
* max(Q[next_state, action]) is the max Q value in next state no matter taking which action.

There are also some update formula that may be complicated than this one.

### Cartpole in Q-learning

[Open AI gym](https://gym.openai.com/docs/) is a toolkit for reinforcement learning, it implements many environments. Cartpole may be the most simple one.
I reference the blog [here](https://medium.com/@tuzzer/cart-pole-balancing-with-q-learning-b54c6068d947) and a [C solution](http://pages.cs.wisc.edu/~finton/qcontroller.html). From this excersice, I learned that action policy is very important, and how to mapping the state is also critical.

[Cartpole environment rules](https://github.com/openai/gym/wiki/CartPole-v0)
For each state, the environment gives 4 values, x, x_dot, theta, theta_dot. At first, I only used theta as the representation of the state, the result is bad. The steps cannot be larger than 60. So let's consider using the 4 values. Since each value is a consecutive float. If we just use the value as the key, it's hard to get the same value for other states. So how we represent the state?

Mapping the 4 values to one.

One method is ...

Another method is to generate parameters for each value, like y = k1*x + k2*x_dot + k3*theta + k4*theta_dot. You can find this implementation [here](http://kvfrans.com/simple-algoritms-for-solving-cartpole/).
