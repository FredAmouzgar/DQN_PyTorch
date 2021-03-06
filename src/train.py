from src.agent import Agent
from collections import deque
import numpy as np
import torch
from tqdm import tqdm

def train(agent, env, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    prior_score = -5
    scores = []  # list containing scores from each episode
    brain_name = env.brain_names[0]
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    episode_loop = tqdm(range(1, n_episodes + 1), desc="Episode 0 | Avg Score: None", leave=False)
    for i_episode in episode_loop:
        # reset the environment
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action, memory=None)[brain_name]
            next_state, reward, done = env_info.vector_observations[0], env_info.rewards[0], env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        #print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        #if episode_loop.n % 100 == 0:
        # print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        episode_loop.set_description_str('Episode {} | Avg Score: {:.2f}'.format(episode_loop.n, np.mean(scores_window)))

        if np.mean(scores_window) > prior_score:
            # print('\nEnvironment reached in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
            #                                                                             np.mean(scores_window)))
            episode_loop.set_description_str('Achieved in {:d} episode | Avg Score: {:.2f}'.format(episode_loop.n - 100,np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            prior_score = np.mean(scores_window)
    return scores