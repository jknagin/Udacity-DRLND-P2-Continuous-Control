from model import AgentNetwork
import torch
import numpy as np
import torch.nn as nn
from unityagents import UnityEnvironment
import torch.nn.functional as F

LR = 2e-4
EPS = 1e-5
DISCOUNT_RATE = 0.99
LEARNING_ROUNDS = 3
NUM_MINIBATCHES = 64
PPO_CLIP = 0.2
GRADIENT_CLIP = 5


class Agent:
    """PPO agent which learns from interacting with the environment. """

    def __init__(self, env: UnityEnvironment, seed=0) -> None:
        """ Receive environment and extract relevant information to initialize actor critic model.

        :param env: UnityEnvironment environment to learn in
        """

        # Seed RNG
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Save environment to agent and extract relevant information from it
        self.env = env
        self.brain_name = env.brain_names[0]
        self.brain = env.brains[self.brain_name]
        self.env_info = env.reset(train_mode=False)[self.brain_name]
        self.num_agents = len(self.env_info.agents)  # 20
        self.state_size = self.brain.vector_observation_space_size  # 33
        self.action_size = self.brain.vector_action_space_size  # 4

        # Create agent network and optimizer using hyper-parameters
        self.network = AgentNetwork(self.state_size, self.action_size)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LR, eps=EPS)

        # Assign all other hyper-parameters
        self.discount_rate = DISCOUNT_RATE
        self.learning_rounds = LEARNING_ROUNDS
        self.num_minibatches = NUM_MINIBATCHES
        self.ppo_clip = PPO_CLIP
        self.gradient_clip = GRADIENT_CLIP

    def act(self, state) -> np.ndarray:
        return self.network(state)[0].cpu().detach().numpy()

    def _extract_salp(self, episode):
        """ Collect states, actions, and log probs from episode into tensors."""

        horizon = len(episode) - 1
        states = torch.zeros((horizon * self.num_agents, self.state_size)).cpu()
        actions = torch.zeros((horizon * self.num_agents, self.action_size)).cpu()
        log_probs = torch.zeros((horizon * self.num_agents, 1)).cpu()
        for idx in range(horizon):
            state, _, action, log_prob, _, _ = episode[idx]  # Extract time step data from episode

            # Reshape tensors at each time step and add them to return tensors
            states[self.num_agents * idx:self.num_agents * (idx + 1), :] = torch.reshape(state, (
                self.num_agents, self.state_size))
            actions[self.num_agents * idx:self.num_agents * (idx + 1), :] = torch.reshape(action, (
                self.num_agents, self.action_size))
            log_probs[self.num_agents * idx:self.num_agents * (idx + 1), :] = torch.reshape(log_prob,
                                                                                            (self.num_agents, 1))

        return states, actions, log_probs

    def _generate_episode(self):
        """ Generate an episode until reaching a terminal state with any of the parallel agents. """
        
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        state = torch.Tensor(env_info.vector_observations).cpu()
        episode = []
        episode_rewards = np.zeros(self.num_agents)
        while True:
            action, log_prob, value = self.network(state)
            env_info = self.env.step(action.cpu().detach().numpy())[self.brain_name]
            reward = env_info.rewards
            done = np.array(env_info.local_done)
            episode_rewards += reward
            episode.append([state, value.detach(), action.detach(), log_prob.detach(), reward, 1 - done])
            state = torch.Tensor(env_info.vector_observations).cpu()
            if np.any(done):
                break

        _, _, last_value = self.network(state)
        episode.append([state, last_value, None, None, None, None])
        return episode, last_value, episode_rewards

    def load(self, solution_filename: str = 'solution.pth') -> None:

        """Load weights from existing weights file and sets networks to eval mode.

        :param solution_filename: weights filename to load weights from
        """

        self.network.load_state_dict(torch.load(solution_filename))
        self.network.eval()

    def _process_episode(self, episode, last_value):
        """ Calculate cumulative returns and advantages of the episode.

        Iterate backwards over the episode, starting from the last state's value as last_value.
        """

        # Initialize advantage to 0 for each agent
        advantage = torch.zeros((self.num_agents, 1)).cpu()

        horizon = len(episode) - 1

        # Collect all advantages for each agent for each time step except the last
        advantages = torch.zeros((horizon * self.num_agents, 1)).cpu()

        # Initialize the return as the state value of the last state of the episode.
        # Collect all cumulative returns for each agent for each time step except the last
        return_ = last_value.detach()
        returns = torch.zeros((horizon * self.num_agents, 1)).cpu()
        returns[-self.num_agents:] = return_

        # Iterate over the episode backwards, calculating TD errors, advantages, and returns for each agent
        for i in reversed(range(horizon)):
            state, value, action, log_prob, reward, done = episode[i]
            reward = torch.Tensor(reward).unsqueeze(1).cpu()
            done = torch.Tensor(done).unsqueeze(1).cpu()
            next_value = episode[i + 1][1]
            return_ = reward + self.discount_rate * return_ * done
            td_error = reward + self.discount_rate * done * next_value.detach() - value.detach()
            advantage = advantage * self.discount_rate * done + td_error
            returns[self.num_agents * i:self.num_agents * (i + 1), :] = return_.detach()
            advantages[self.num_agents * i:self.num_agents * (i + 1), :] = advantage.detach()

        # Normalize advantages by the mean and standard deviation
        advantages = (advantages - advantages.mean()) / advantages.std()

        states, actions, log_probs = self._extract_salp(episode)
        return states, actions, log_probs, returns, advantages

    def save(self, solution_filename: str = 'solution.pth') -> None:
        """Save local network weights.

        :param solution_filename: weights filename to save weights to
        """

        # torch.save(self.network.actor.state_dict(), solution_filename_actor)
        # torch.save(self.network.critic.state_dict(), solution_filename_critic)
        torch.save(self.network.state_dict(), solution_filename)

    def train_for_episode(self) -> float:
        """Train the agent for one episode.
        Generate an episode, process the episode, and train the actor and critic using PPO.

        :return: mean of cumulative rewards of each agent for the episode
        """

        episode, last_value, episode_rewards = self._generate_episode()
        states, actions, log_probs, returns, advantages = self._process_episode(episode, last_value)
        self._train_network(states, actions, log_probs, returns, advantages)

        return float(np.mean(episode_rewards))

    def _train_network(self, states, actions, log_probs, returns, advantages) -> None:
        """ Train the actor and critic networks.

        Batch the episode into random minibatches to mitigate effects due to temporal correlation.
        The states, actions, and log_probs are calculating while the agent performs the episode.
        The returns and advantages are calculated with bootstrapping by iterating over the episode backwards.
        The states and actions are used to calculate new log probs and new state values.
        The new log probs, the old log probs, and the advantages are used to calculate the policy loss
            using the clipped surrogate function.
        The returns and new state values are used to calculate the value loss using mean squared error.
        The policy and value loss are optimized simultaneously as one loss function defined as their sum.
        """

        horizon = states.size(0) // self.num_agents
        for _ in range(self.learning_rounds):
            indices_shuffled = np.arange(horizon * self.num_agents)
            np.random.shuffle(indices_shuffled)
            batch_start = 0
            batch_size = states.size(0) // self.num_minibatches

            while True:
                batch_indices = indices_shuffled[batch_start:min(batch_start + batch_size, horizon * self.num_agents)]
                batch_indices = torch.Tensor(batch_indices).long()
                batch_start += batch_size

                sampled_states = states[batch_indices].squeeze()
                sampled_actions = actions[batch_indices].squeeze()
                sampled_log_probs_old = log_probs[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                _, log_probs_new, values = self.network(sampled_states, sampled_actions)
                ratio = (log_probs_new - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1 - self.ppo_clip, 1 + self.ppo_clip) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean(0)
                value_loss = F.mse_loss(sampled_returns, values)

                self.optimizer.zero_grad()
                (policy_loss + value_loss).backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clip)
                self.optimizer.step()

                if batch_start + batch_size >= horizon * self.num_agents:
                    break

