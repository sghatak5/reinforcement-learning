"""
This script sets up and trains a Q-learning agent to solve the Tower of Hanoi problem 
using the specified parameters, and visualizes the results.

Functions:
    create_env(numDisks): Creates the Tower of Hanoi environment with the specified number of disks.
    train_q_learning(env, no_episodes, epsilon, epsilon_min, epsilon_decay, alpha, gamma): 
        Trains the Q-learning agent using the specified parameters.
    visualize_q_table(q_table_path): Visualizes the Q-table for the trained agent.

Parameters:
    train (bool): If True, trains the Q-learning agent. Default is True.
    visualize_results (bool): If True, visualizes the Q-table after training. Default is False.
    alpha (float): Learning rate for the Q-learning algorithm. Default is 0.1.
    gamma (float): Discount factor for the Q-learning algorithm. Default is 0.99.
    epsilon (float): Initial exploration rate for the Q-learning algorithm. Default is 0.9.
    epsilon_min (float): Minimum exploration rate for the Q-learning algorithm. Default is 0.1.
    epsilon_decay (float): Decay rate for the exploration rate. Default is 0.009.
    no_episodes (int): Number of episodes for training the Q-learning agent. Default is 1000.
    numDisks (int): Number of disks in the Tower of Hanoi problem. Default is 3.
    redner (bool): Renders the environment. Default False

Execution:
    If `train` is set to True, the script will create the environment with the specified number 
    of disks and train the Q-learning agent using the specified parameters.
    
    If `visualize_results` is set to True, the script will visualize the Q-table after training.
"""
from padm_env import create_env
from q_learning import train_q_learning

train = True

alpha = 0.1
gamma = 0.99
epsilon = 0.9
epsilon_min = 0.1
epsilon_decay = 0.009
no_episodes = 1000
render = False

numDisks = 3

if train:
    env = create_env(numDisks)

    train_q_learning(env=env, 
                     no_episodes=no_episodes,
                     epsilon=epsilon,
                     epsilon_min=epsilon_min,
                     epsilon_decay=epsilon_decay,
                     alpha=alpha,
                     gamma=gamma, 
                     render=render)