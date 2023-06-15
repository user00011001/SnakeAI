# Snake Game with AI using Deep Q-Learning

This is a simple implementation of the classic Snake game using Pygame library, along with an AI agent trained with Deep Q-Learning algorithm to play the game.

## Game Overview

The Snake game is a single-player game where the player controls a snake on a grid. The objective is to eat fruits that appear on the grid while avoiding collisions with the walls and the snake's own body. The snake grows in length whenever it eats a fruit.

## AI Agent

The AI agent is trained using Deep Q-Learning, a reinforcement learning algorithm. The agent uses a Deep Neural Network (DQN) model to approximate the Q-values, which represent the expected future rewards for each action in a given state. The model is trained by playing the game and updating the Q-values based on the observed rewards and state transitions.

## Requirements

- Python 3.x
- Pygame library
- PyTorch library

## Usage

1. Install the required libraries using the following command:

   ```
   pip install pygame torch
   ```

2. Run the game using the following command:

   ```
   python snake_game.py
   ```

   The game window will open, and the AI agent will start playing automatically. You can observe the game progress and the snake's movements.

3. To exit the game, click the close button on the game window.

## Customization

You can modify various parameters in the code to customize the game and the AI agent's behavior. Some of the parameters you can change include:

- Game parameters:
  - `WINDOW_SIZE`: The size of the game window.
  - `GRID_SIZE`: The size of the grid on which the game is played.
  - `GRID_OFFSET`: The offset between grid cells for visual appearance.

- AI parameters:
  - `DISCOUNT_FACTOR`: The discount factor used in the Q-Learning algorithm.
  - `EPSILON_START`, `EPSILON_END`, `EPSILON_DECAY`: Parameters for controlling the exploration-exploitation trade-off in the AI agent's behavior.
  - `LEARNING_RATE`: The learning rate used in the optimization of the DQN model.
  - `BATCH_SIZE`: The batch size used for training the DQN model.
  - `REPLAY_MEMORY_SIZE`: The size of the replay memory buffer used for experience replay.
