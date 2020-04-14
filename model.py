import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Tuple


class Network(nn.Module):
    """ 3-layer fully connected neural network."""

    def __init__(self, input_size: int, output_size: int, output_activation: Callable = None) -> None:
        super(Network, self).__init__()

        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_size)
        self.output_activation = output_activation

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(inp))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if self.output_activation is not None:
            x = self.output_activation(x)

        return x


class AgentNetwork(nn.Module):

    def __init__(self, num_states: int, num_actions: int) -> None:
        """ Network class used by agent to map states to actions, action values, and log probabilities of actions

        :param num_states: dimension of state space
        :param num_actions: dimension of action space
        """

        super(AgentNetwork, self).__init__()

        # Network mapping from state vectors to mean values of multivariate normal distributions of action vectors
        self.actor = Network(num_states, num_actions, F.tanh)

        # Network mapping from state vectors to state values
        self.critic = Network(num_states, 1)

        # Set device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        # Trainable parameter of actor network to generate normal distributions from which to sample action vectors
        self.std = nn.Parameter(torch.ones(1, num_actions)).to(self.device)

    def forward(self, state: torch.Tensor, action: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map state to action, log probability, and state value

        :param state: state to be mapped
        :param action: if an action is provided, calculate the log probability of that action. Else, calculate action from state.
        :return: action, log probability of action, and state value
        """

        # Map state to mean action value
        a = self.actor(state)

        # Generate normal distribution using mapped mean and learned std to sample actual action values
        distribution = torch.distributions.Normal(a, self.std)

        # Calculate log probability of action. Action is either mapped from the state or provided as an argument.
        if action is None:
            action = distribution.sample()
        log_prob = distribution.log_prob(action)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)

        # Use the critic network to map the state to a state value
        value = self.critic(state)

        return action, log_prob, value
