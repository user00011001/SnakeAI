# Snake AI with Shared Apple

This is a Python program that implements a snake game with AI-controlled snakes. The goal of the AI is to control the snakes to eat the shared apple while avoiding collisions with other snakes and themselves.

## Requirements

- Python 3.x
- Pygame library
- Torch library

## Instructions

1. Install the required libraries using the following command:
   ```
   pip install pygame torch
   ```

2. Run the program by executing the Python file:
   ```
   python snake_ai_shared_apple.py
   ```

3. The game window will open, and the AI-controlled snakes will start playing the game. The snakes are represented by different colors on the grid.

4. Each snake moves independently and tries to eat the shared apple, which is represented by a colored square. The AI uses Deep Q-Networks (DQN) to learn and make decisions.

5. The game will continue until you close the window. You can exit the game by clicking the close button.

## Configuration

The program provides some configurable parameters that you can adjust:

- `WINDOW_SIZE`: The size of the game window in pixels.
- `GRID_SIZE`: The number of cells in the grid.
- `GRID_OFFSET`: The offset for drawing the grid lines.
- `MODEL_PARAMS`: A list of dictionaries specifying the AI model parameters. Each dictionary contains the following keys:
  - `name`: The name of the model.
  - `color`: The color of the snake controlled by the model.
  - `discount_factor`: The discount factor for future rewards in the Q-learning algorithm.
  - `learning_rate`: The learning rate for the neural network optimizer.
- `EPSILON_START`, `EPSILON_END`, `EPSILON_DECAY`: The parameters for epsilon-greedy exploration.
- `BATCH_SIZE`: The batch size for training the neural network.
- `REPLAY_MEMORY_SIZE`: The maximum size of the replay memory for experience replay.

## Saving and Loading Models

The program automatically saves and loads the model parameters to/from disk. The model parameters are saved in separate files named `model_name.pth` for each model. The saved files can be found in the same directory as the Python script.

To load the saved model parameters, the program checks for existing files at the beginning. If the files exist, the program loads the parameters into the corresponding models. If the files do not exist or the parameters are incompatible, the models start training from scratch.
