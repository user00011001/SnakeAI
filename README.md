# Deep Q-Learning Snake Game with Pygame and PyTorch

This Python script implements a classic game of Snake using the Pygame library. It utilizes Deep Q-Learning (DQN) for training an agent to play the game. The agent uses a neural network implemented using the PyTorch library to make decisions based on the game state.

## Prerequisites

To run this script, you need to have the following Python libraries installed:

- pygame
- numpy
- torch

You can install them using pip:
```
pip install pygame numpy torch
```

## How to Run the Script

You can run the script using the following command in the terminal:

```
python3 snake.py
```

If an existing model file `model.pth` is found in the current directory, the script will load the saved model parameters. Otherwise, it will start training from scratch.

The script will continue running the game and training the model until you manually stop it (e.g., by closing the pygame window).

## Model Training

The DQN agent is trained with a replay memory to improve the learning process. Experiences (state, action, reward, and new state) are stored in the replay memory and sampled randomly for training the model, which breaks the correlation between consecutive experiences and improves the stability and performance of the DQN.

The agent uses an epsilon-greedy strategy with a decaying epsilon for action selection. At the beginning of training, the epsilon is high (EPSILON_START) to encourage more exploration. The epsilon value decays after each game loop (multiplied by EPSILON_DECAY) until it reaches a minimum value (EPSILON_END), allowing more exploitation of the learned policy.

The agent gets a reward of -10 if it hits itself, a reward of -1 for a normal step, and a reward of 10 when it gets the fruit. This reward structure encourages the agent to get the fruit while avoiding hitting itself and to reach the fruit as fast as possible.

## Model Saving

The script automatically saves the trained model parameters to a file named 'model.pth' when the pygame window is manually closed. When you run the script next time, the saved model parameters will be loaded.

This allows you to stop and resume the training process at any time without losing the learned knowledge. You can also use the trained model for playing the game without further training.

Please note that training the model to play the game well can take a long time (often requiring millions of game steps), depending on the complexity of the game and the capacity and architecture of the model. The training process can be accelerated by using a more powerful computer or a cloud-based machine learning platform that supports PyTorch.
