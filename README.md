# Tower of Hanoi Q-Learning Agent

This repository contains a Q-learning agent that solves the Tower of Hanoi problem. The agent is trained using reinforcement learning techniques and can visualize the results of the training process.

## Overview

The repository consists of three main scripts:

1. `main.py`: Sets up and trains the Q-learning agent, and visualizes the results.
2. `padm_env.py`: Defines the Tower of Hanoi environment using the Gymnasium framework.
3. `q_learning.py`: Contains the Q-learning algorithm and training logic.

## Usage

### main.py

This script sets up the environment and trains the Q-learning agent with the specified parameters. It also has an option to visualize the results.

#### Functions

- `create_env(numDisks)`: Creates the Tower of Hanoi environment with the specified number of disks.
- `train_q_learning(env, no_episodes, epsilon, epsilon_min, epsilon_decay, alpha, gamma)`: Trains the Q-learning agent using the specified parameters.
- `visualize_q_table(q_table_path)`: Visualizes the Q-table for the trained agent.

If `train` is set to `True`, the script will create the environment with the specified number of disks and train the Q-learning agent using the specified parameters.

If `visualize_results` is set to `True`, the script will visualize the Q-table after training.

### padm_env.py

This script defines the Tower of Hanoi environment using the Gymnasium framework.

#### Class: HanoiTower

- `__init__(self, numDisks=3)`: Initializes the Tower of Hanoi environment.
- `reset(self)`: Resets the environment to its initial state.
- `step(self, action)`: Executes a step in the environment given an action.
- `disksOnPeg(self, peg)`: Returns a list of disks on a given peg.
- `moveAllowed(self, action)`: Checks if a given move is allowed.
- `render(self)`: Renders the current state of the Tower of Hanoi.
- `close(self)`: Closes the Pygame window.

### q_learning.py

This script defines functions for training a Q-learning agent on the Tower of Hanoi problem and visualizing the resulting Q-table.

#### Functions

- `train_q_learning(env, no_episodes, epsilon, epsilon_min, epsilon_decay, alpha, gamma, render, q_table_save_path="q_table.npy")`: Trains a Q-learning agent using the specified parameters.
- `state_to_index(state)`: Converts the state representation of the Tower of Hanoi environment into an index for the Q-table.

## Acknowledgements

This project utilizes the Gymnasium framework for creating the environment and Pygame for rendering the visualization.
