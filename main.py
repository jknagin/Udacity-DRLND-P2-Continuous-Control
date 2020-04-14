from unityagents import UnityEnvironment
import numpy as np
from collections import deque
from ppo_agent import Agent
import matplotlib.pyplot as plt
from typing import List
import pandas as pd


def ppo(agent: Agent, solution_score: float = 30, n_episodes: int = 1000) -> List[float]:
    """ Train the PPO agent.

    :param agent: agent to be trained
    :param solution_score: score at which agent's environment is considered to be solved
    :param n_episodes: maximum number of episodes for which to train the agent
    :return: list of agent's scores for all episodes of training
    """

    all_scores = []  # List of all scores collected in training
    latest_scores = deque(maxlen=100)  # Deque of 100 most recent scores to determine if environment is solved
    for i in range(n_episodes):
        score = agent.train_for_episode()  # Train the agent for one episode and return the agent's score for the episode
        latest_scores.append(score)  # Store score
        all_scores.append(score)  # Store score

        # Print status updates
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i, np.mean(latest_scores)), end="")
        if i % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i, np.mean(latest_scores)))

        # Notify when environment is solved and save agent model parameters and score
        if np.mean(latest_scores) >= solution_score:
            print("\nEnvironment solved in {} episodes".format(i + 1))
            agent.save()  # Save local model weights to solution.pth
            np.save('scores.npy', np.array(all_scores))
            break

    return all_scores


def main():
    # Create environment
    env = UnityEnvironment(file_name='Reacher.app')

    # Create agent using environment
    agent = Agent(env)

    # Train the agent, and collect scores during training
    scores = ppo(agent)

    # Calculate rolling average of scores over specified window of episodes
    window = 100
    scores_w = pd.Series(scores).rolling(window=window).mean().iloc[window - 1:].values

    # Plot all scores and rolling average of scores
    plt.figure()
    plt.plot(scores, color='b', marker='o', label='All Scores')
    plt.plot(np.arange(len(scores_w)) + window, scores_w, color='r', marker='o', label='Average of Last {} Scores'.format(window))
    plt.title('Agent Score vs. Episode #')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

    env.close()


if __name__ == '__main__':
    main()
