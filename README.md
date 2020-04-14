[Trained agent]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"

[Udacity repository]: https://github.com/udacity/deep-reinforcement-learning#dependencies

# Project 2: Continuous Control

### Introduction

For this project, 20 agents were trained in parallel to keep a double jointed arm within a goal region of space.

 ![Trained Agent][Trained agent]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that an agent's hand is in its goal location. Thus, the goal of each agent is to maintain its position at its target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of an arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints of an arm. Every entry in the action vector should be a number between -1 and 1.

The task is episodic, and in order to solve the environment, the agents must get an average score of +30 over 100 consecutive episodes.


### Getting Started

1. Follow the instructions on the [Udacity repository] to configure a Python environment with the dependencies and Unity environments.

1. Clone this project and ensure it can be ran with the Python environment.

1. Download the Reacher environment from one of the links below.  You need only select the environment that matches your operating system:
  - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
  - macOS: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
  - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
  - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

  (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

  (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

1. Place the file in the root directory of this repo and unzip the file.

1. Follow the instructions in the project [report](https://github.com/jknagin/Udacity-DRLND-P1-Navigation/blob/master/Report.md#running-navigationipynb) and the main Jupyter [notebook](https://github.com/jknagin/Udacity-DRLND-P1-Navigation/blob/master/Navigation.ipynb) to get started with training the agent.
