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
One method is to generate parameters for each value, like y = k1*x + k2*x_dot + k3*theta + k4*theta_dot. You can find this implementation [here](http://kvfrans.com/simple-algoritms-for-solving-cartpole/).

Another method is to split each value for several buckets. You can define the buckets for yourself. In the [C solution](http://pages.cs.wisc.edu/~finton/qcontroller.html), for x, x_dot, theta_dot it has three bucket; for theta, it splits 5 buckets. How to define the value for each buckets? We need to map the four dimensions(x, x_dot, theta, theta_dot) to one value, and the one value can also unique represent the four dimensions(one bucket is one value). For the C solution case:
	
	x: three buckets, 0, 1, 2
	x_dot: three buckets, 0, 3, 6 (step > 2, step = 3)
	theta: six buckets, 0, 9, 18, 27, 36, 45 (step > 6 + 2, step = 9)
	theta_dot: three buckets, 0, 54, 108 (step > 45 + 6 + 2, step = 54)
	
Since we need the value to represent the state(the state is adding all four dimensional values.), it has requirements to the value. If the buckets changes, the value changes. For example:

Another method is to generate parameters for each value, like y = k1*x + k2*x_dot + k3*theta + k4*theta_dot. You can find this implementation [here](http://kvfrans.com/simple-algoritms-for-solving-cartpole/).
	x: three buckets, 0, 1, 2
	x_dot: three buckets, 0, 3, 6 (step > 2, step = 3)
	theta: eight buckets, 0, 9, 18, 27, 36, 45, 54, 63 (step > 6 + 2, step = 9)
	theta_dot: three buckets, 0, 72, 144 (step > 63 + 6 + 2, step = 72)

Another important thing is how to assign the value in which condition. Here we use some rule, theta is bigger, value is bigger, as with other dimensional values.

So in this case, we need to define the action. If all the four dimension value are in the middle, whether the pole moves left or not doens't matter, if the value is smaller, it needs to move left; otherwise right. So we can use this method to determine the action. You can also change the value to have your own test.

## Deep Q-learning

### Using tensorflow docker

To avoid installing dependencies, I used tensorflow docker image.
After installing docker, next we need to login in the command line.
*Attension: docker login is using by your docker id, not the id like url.
To get your docker id log into [docker hub](https://hub.docker.com/) with your email and password. On the top right is docker id. Use that in the CLI and you'll probably be fine.*

``` sh
docker login
docker pull tensorflow/tensorflow:latest

```
