# Snake AI Game

This program, `snake.py`, is an implementation of the classic Snake game using the Pygame library. The game features an AI player that learns to navigate the grid and collect fruits while avoiding self-collisions.

## Prerequisites

To run this program, you need to have Python installed on your machine. Additionally, you need to have the following libraries installed:

- Pygame
- NumPy

You can install the required libraries using the following command:

```
pip install pygame numpy
```

## Usage

To start the game, simply run the Python script `snake.py` using the following command:

```
python snake.py
```

Once the game window opens, you can observe the AI player's performance. The AI player learns by updating a Q-table based on its actions and rewards. It uses a basic Q-learning algorithm to improve its performance over time.

## Game Controls

The AI player controls the movement of the snake automatically. You can observe the snake moving on the grid and collecting fruits. The game runs in an infinite loop until you close the game window.

## Customization

You can modify certain parameters in the script to customize the game behavior. Here are the available parameters:

- `WINDOW_SIZE`: The size of the game window in pixels (square window).
- `GRID_SIZE`: The number of grid cells in each row and column.
- `GRID_OFFSET`: The size of the offset around each grid cell for drawing.
- `LEARNING_RATE`: The learning rate of the Q-learning algorithm.
- `DISCOUNT_FACTOR`: The discount factor for future rewards in the Q-learning algorithm.
- `EPSILON`: The exploration rate, determining the likelihood of the AI player taking random actions.

Feel free to experiment with different parameter values to observe their effects on the AI player's performance.

## Acknowledgements

This program is inspired by the classic Snake game and Q-learning algorithm. It utilizes the Pygame library for graphics and user interface, as well as the NumPy library for efficient array operations.
