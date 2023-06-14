# Snake Game with Deep Q-Learning

This is a Python implementation of the classic Snake game using Pygame, where the snake is controlled by an AI agent trained using Deep Q-Learning.

## Requirements

This implementation requires the following Python libraries:

- Pygame
- Numpy
- PyTorch

## Usage

Just run `snake_dqn.py` to start the game. The game window should open and you'll see the snake moving around and learning as it plays more games.

## Overview

The AI agent is trained using Deep Q-Learning where the state is the grid of the game, the actions are the four possible directions the snake can go to (up, right, down, left), and the reward is based on the result of the action (1 for eating a fruit, -1 for hitting itself and 0 otherwise).

The game uses an epsilon-greedy strategy for action selection, which balances between exploitation (choosing the best known action) and exploration (choosing a random action). Currently, the epsilon value is set at 0.1, which means the AI agent will choose a random action 10% of the time and the best known action 90% of the time.

## Model

The AI agent uses a Deep Q-Network (DQN) model with two convolutional layers and two fully connected layers. The Q-values for each action are calculated by passing the game state through the model.

The model's parameters are updated based on the loss calculated from the difference between the current Q-values and the target Q-values. 

## Saving and Loading

The model parameters are saved to 'model.pth' when the game is quit. If 'model.pth' exists when the script is run, the model parameters are loaded from this file. If the file doesn't exist, the model starts learning from scratch.

## Customization

You can modify various parameters in the script to change the behavior and performance of the AI agent. For example, you can change the size of the game grid, the learning rate, the discount factor, the batch size, etc.

## Future Improvements

- Implement a replay memory for storing and sampling previous experiences to improve the learning process.
- Use a decaying epsilon for the epsilon-greedy strategy, starting with a high value for more exploration and gradually decreasing it for more exploitation.
- Consider more sophisticated reward structures for better training results.
