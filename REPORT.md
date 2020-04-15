[Udacity's Deep Reinforcement Learning Nanodegree]: https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893
[scores]: images/scores.png
[PPO]: images/PPO.png

# Udacity Deep Reinforcement Learning Nanodegree Project 2: Continuous Control

## Introduction
The purpose of this project is to train 20 agents in parallel to keep a double jointed arms within goal regions of space.

## Environment
In this environment, a double-jointed arm can move to spherical target locations around it. A reward of +0.1 is provided for each step that an agent's "hand" is in its goal location. Thus, the goal of each agent is to maintain its position at its target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of an arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints of an arm. Every entry in the action vector isIU a number between -1 and 1.

The task is episodic, and in order to solve the environment, the agents must get an average score of +30 over 100 consecutive episodes.

## Proximal Policy Optimization (PPO) Algorithm
Proximal Policy Optimization (PPO) with an actor critic network is an appropriate learning algorithm for this use case because the state and action spaces are both continuous.

The critic's neural network maps states to state values. The actor's neural network maps states to mean action vectors. The actor also has a parameter that is used as the standard deviation of a multivariate normal distribution of action vectors. The actor samples an action from a normal distribution parameterized by this mean and standard deviation. The actor also computes the log probability of sampling the action from the distribution. The action is taken by the agent to step in the environment. The PPO algorithm uses the log probabilities of all actions and the state values from all states in the episode to optimize a clipped surrogate function and update the networks' weights.

The screenshot below is a high level description of the algorithm. The screenshot was taken from the lecture materials of [Udacity's Deep Reinforcement Learning Nanodegree].

![PPO]

More information about the PPO algorithm, including a precise definition of the clipped surrogate function, can be found [here](https://arxiv.org/abs/1707.06347).

## Implementation
### Descriptions of each file
#### `ppo_agent.py`
Implements the PPO, actor-critic style Agent class, which provides the following methods:
* `__init__()`
  * The environment is provided as an argument
  * Initializes agent's actor and critic neural networks

* `_extract_salp()`
  * Extracts the states, actions, and log probabilities of actions for each agent from an episode trajectory

* `_generate_episode()`
  * Generates an episode trajectory for each agent until an agent reaches a terminal state

* `load()`
  * Loads weights from an existing weight file. Also sets agent to eval mode.

* `_process_episode()`
  * Processes an episode trajectory backwards in time to calculate returns and advantages for each agent

* `save()`
  * Saves agent's weights to a weight file

* `train_for_episode()`
  * Calls `_generate_episode()`, `_process_episode()`, and `_train_network()` to train the agent for one episode

* `_train_network()`
  * Uses the PPO algorithm to compute policy loss using the clipped surrogate function.
  * Calculates value loss using mean squared error between old policy returns and new policy returns
  * Trains the actor and critic networks by minimizing the sum of the policy loss and value loss

#### `model.py`
Implements a simple neural network with one hidden layer. These networks are used to approximate actions (actor) and state values (critic), given states as inputs.

#### `Continuous_Control.ipynb`
Main notebook for running the code. The notebook loads the Reacher environment, instantiates the agent, trains the agent, saves the agent's weight file, and plots the agent's score per episode during training. The notebook can also be used to load a weight file into an agent and play the environment to see how well the agent performs.

### Hyperparameters

All hyperparameters are defined in `ppo_agent.py`.

| Hyperparameter | Value | Description | Defined In|
|-               |-      | -           | -         |
|`LR` | 0.0002 | Learning rate for Adam optimizer | `ppo_agent.py` |
|`EPS` | 0.00001 | Epsilon value for Adam optimizer | `ppo_agent.py` |
|`DISCOUNT_RATE` | 0.99 | Discount rate for bootstrapping future returns | `ppo_agent.py` |
|`LEARNING_ROUNDS` | 3 | Number of times to iterate over the episode during learning | `ppo_agent.py` |
|`NUM_MINIBATCHES` | 64 | Number of minibatches to split an episode trajectory into during learning | `ppo_agent.py` |
| `PPO_CLIP` | 0.2 | Clipping value for clipped surrogate function | `ppo_agent.py` |
| `GRADIENT_CLIP` | 5 | Clipping value for gradient clipping during optimization | `ppo_agent.py` |


### Network Architecture

The network architecture is defined fully in `model.py`.

#### Actor Network
| Layer    | Input Dim | Output Dim | Activation |                    Notes                      |
| ---------| --------- | ---------- | ---------- | --------------------------------------------- |
| `FC1`    |     33    |     256     |    ReLU    | Input dimension is the dimension of the state space.   |
| `FC2`     |     256    |     256     |    ReLU    |                                               |
| `FC3`     |     256   |     4      |    tanh    | Output dimension is the dimension of the action space. Tanh activation is used to keep the actions between -1 and 1, as required by the Reacher environment. |

#### Critic Network
| Layer    | Input Dim | Output Dim | Activation |                    Notes                      |
| ---------| --------- | ---------- | ---------- | ---------------------------------------------|
| `FC1`    |     33    |     256     |    ReLU    | Input dimension is the state space dimension  |
| `FC2`     |     256    |     256     |    ReLU    |                                               |
| `FC3`     |     256   |     1      |    None    | Output dimension is 1 because the action value is a scalar for each state|

### Running `Continuous_Control.ipynb`
To run the notebook, follow these steps:

1. Ensure that the following Python libraries are installed:
  * `numpy`
  * `pandas`
  * `matplotlib`
  * `pytorch`

  Also ensure that the Reacher environment is installed following the instructions in the README.

1. Run the first code cell to import all necessary libraries.

1. In the second code cell, update the `file_name` argument to the `UnityEnvironment` function with the location of the Reacher environment (`env = UnityEnvironment(file_name=...)`).

1. Run the second code cell to load the Reacher environment and get information about the state and action space. This information is used to instantiate the agent. This cell will print out the dimensions of the state and action spaces, and the number of agents. It will also print out an example state.

1. Run the third code cell to train the agent. The code will loop until the maximum number of episodes have been played (specified by `n_episodes`) or the agent achieves an average score of 30.0 or greater over the 100 most recent episodes. If the agent achieves such a score, this cell will save the agent's weights to a file called `solution.pth` and the list of scores to a file called `scores.npy`.

1. After training the agent, run the next code cell to plot the score and the running average of the 100 most recent scores.

1. The next two code cells are used to load existing `solution.pth` file into the agent and watch it perform in the Reacher environment.

1. The final code cell closes the environment.

## Results
### Score Plot
![scores]

The above image shows the plot of the agent's score for each episode (blue) and the running average of the scores of the previous 100 episodes (red). **The agent achieves an average score greater than or equal to 30.0 for 100 episodes after episode 268.**

## Future work

### Incorporating entropy into the loss function

As discussed in the [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) paper, the combined loss function may include an additional entropy term to encourage exploration. I did not incorporate the entropy term in this project because it was not covered in the lectures.


### Making trajectories shorter than entire episodes

In my implementation, I defined a trajectory using every time step of an episode. These trajectories could be shorter potentially to keep each policy update small, as required by the PPO algorithm.

### Using a different network architecture
I did not experiment significantly with different neural network architectures. A variety of network architectures could be tested to find a good tradeoff between bias and variance.
