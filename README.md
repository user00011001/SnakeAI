# Snake Game with Deep Q-Network (DQN) Reinforcement Learning

This program implements the classic Snake game using the Deep Q-Network (DQN) algorithm for reinforcement learning. The DQN model learns to navigate the game grid and collect fruits while avoiding self-collisions.

## Requirements

- Python 3.x
- Pygame library
- PyTorch library

## Installation

1. Clone the repository or download the code files.
2. Install the required dependencies using the following command:

   ```
   pip install pygame torch
   ```

## Usage

1. Run the program using the following command:

   ```
   python3 snake_game.py
   ```

2. The game window will open, and you will see the snake navigating the grid.
3. The game will start automatically, and the snake will move on its own according to the learned policy.
4. If you want to manually control the snake, you can modify the code to accept user input and control the direction of the snake.
5. The DQN model will learn and improve its performance over time through reinforcement learning.
6. The program will save the learned model parameters to the "model.pth" file periodically, allowing you to resume training or play the game using the saved model in subsequent runs.

## Customization

You can customize the game parameters and AI parameters by modifying the constants defined at the beginning of the code:

- `WINDOW_SIZE`: Specifies the size of the game window in pixels.
- `GRID_SIZE`: Specifies the number of cells in each row and column of the game grid.
- `DISCOUNT_FACTOR`: Controls the importance of future rewards in the Q-learning update.
- `EPSILON`: Determines the exploration-exploitation trade-off during action selection.
- `LEARNING_RATE`: Controls the learning rate for updating the Q-values.
- `BATCH_SIZE`: Specifies the batch size for mini-batch updates.

Feel free to adjust these parameters to experiment with different settings and observe the effects on the AI's learning and gameplay.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- The Pygame library: https://www.pygame.org/
- The PyTorch library: https://pytorch.org/
- OpenAI for the DQN algorithm: https://openai.com/
