# Snake Game with Multiple AI Models

This is a snake game implemented using Pygame library, where multiple AI models are trained to play the game using the Deep Q-Network (DQN) algorithm.

## Requirements

- Python 3.x
- Pygame
- Torch

## Getting Started

1. Clone the repository:

2. Install the dependencies:

   ```bash
   pip install pygame torch
   ```

3. Run the game:

   ```bash
   python3 snake.py
   ```

4. The game window will appear, and the AI models will start training and playing the game simultaneously.

## Customizing the AI Models

The AI models' parameters can be customized in the `MODEL_PARAMS` list defined in the `main.py` file. Each model can have its own set of parameters. Here are the available parameters for each model:

- `name`: The name of the model.
- `color`: The color of the snake controlled by the model (in RGB format).
- `discount_factor`: The discount factor for future rewards in the DQN algorithm.
- `learning_rate`: The learning rate used in the DQN algorithm.
- Additional parameters: Add any additional parameters specific to each model.

Feel free to modify these parameters according to your needs.

## Saving and Loading Models

The trained model parameters can be saved and loaded from disk. The model parameters will be saved as `.pth` files in the current working directory. The file name will be based on the model's name.

To save the model parameters, simply close the game window. The model parameters will be saved automatically.

To load existing model parameters, make sure the `.pth` files corresponding to each model's name exist in the current working directory.

## Further Customizations

The game's window size, grid size, epsilon decay, batch size, replay memory size, and other parameters can be adjusted by modifying the constants defined at the beginning of the `snake.py` file.
