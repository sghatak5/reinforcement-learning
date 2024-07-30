"""
This script defines functions for training a Q-learning agent on the Tower of Hanoi problem,
visualizing the resulting Q-table, and plotting the training metrics.

Excploration vs Exploitation Used : Epsilon-Greedy with Decay
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def train_q_learning(env, 
                     no_episodes,
                     epsilon, 
                     epsilon_min,
                     epsilon_decay, 
                     alpha, 
                     gamma, 
                     render,
                     q_table_save_path="q_table.npy"):
    """
    Trains a Q-learning agent to solve the Tower of Hanoi problem using the specified parameters.

    Parameters:
    -----------
    env : Tower of Hanoi environment
        The environment instance where the agent learns.
    no_episodes : int
        Number of episodes for training.
    epsilon : float
        Initial exploration rate.
    epsilon_min : float
        Minimum exploration rate.
    epsilon_decay : float
        Decay rate for the exploration rate.
    alpha : float
        Learning rate.
    gamma : float
        Discount factor.
    q_table_save_path : str, optional
        File path to save the Q-table after training. Default is "q_table.npy".

    Returns:
    --------
    None
    """
    # Initialize Q-table with zeros
    shape = (3**env.numDisks, env.actionSpace.n)
    q_table = np.zeros(shape)
    total_reward = 0
    steps = 0
    steps_list = []
    rewards_list = []
    prev_step = 0
    last_reward = 0
    
    
    for episode in range(no_episodes):
        state = env.reset()
        actionToMove = [(0, 1), (0, 2), (1, 0),
                    (1, 2), (2, 0), (2, 1)]
        Exploration = 0
        Exploitation = 0


        while True:
            if np.random.rand() < epsilon:
                action = env.actionSpace.sample() #Exploration
                Exploration += 1
            else:
                action = np.argmax(q_table[state_to_index(state)]) #Exploitation
                Exploitation += 1
            next_state, reward, done, info = env.step(actionToMove[action])
            if render:
                env.render() 

            total_reward += reward
            steps += 1
            current_index = state_to_index(state)
            next_index = state_to_index(next_state)

            q_table[current_index][action] = q_table[current_index][action] + alpha * \
                (reward + gamma * (np.max(q_table[next_index])) - q_table[current_index][action])

            state = next_state

            if done:
                steps_list.append(steps - prev_step)
                prev_step = steps
                rewards_list.append(total_reward - last_reward)
                last_reward = total_reward
                q_table[next_index][action] = q_table[next_index][action] + alpha * (reward - q_table[next_index][action])
                print()
                break
    
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print(f"Episode {episode + 1}, Total Reward: {total_reward}, Explored: {Exploration}, Exploited: {Exploitation}")

    env.close()
    print("------------------------")
    print(" Learning is completed.")
    print("------------------------")
    print(f"Average Rewards per Episode: {total_reward/no_episodes},Average Number of Steps to Solve: {steps/no_episodes}, Minimum Steps: {min(steps_list)}")
    np.save(q_table_save_path, q_table)
    print("------------------------")
    print(f"Q-table saved to {q_table_save_path}")


def state_to_index(state):
    """
    Converts the state representation of the Tower of Hanoi environment into an index for the Q-table.

    Parameters:
    -----------
    state : tuple
        The state of the environment, representing the configuration of disks on pegs.

    Returns:
    --------
    index : int
        Index corresponding to the state in the Q-table.
    """
    index = 0
    num_pegs = len(state)
    
    for i, disk in enumerate(state):
        index += disk * (3 ** (num_pegs - i - 1))
    
    return index