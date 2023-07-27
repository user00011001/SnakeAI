import pygame
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os

# Pygame parameters
WINDOW_SIZE = 600
GRID_SIZE = 20
GRID_OFFSET = 2

# AI parameters
MODEL_PARAMS = [
    {
        "name": "model",
        "color": (255, 0, 0),
        "discount_factor": 0.8,
        "learning_rate": 0.01,
        # Additional parameters for model1
        "parameter1": 0.5,
        "parameter2": "value1"
    },
    {
        "name": "model1",
        "color": (0, 255, 0),
        "discount_factor": 0.9,
        "learning_rate": 0.001,
        # Additional parameters for model2
        "parameter1": 0.2,
        "parameter2": "value2"
    },
    {
        "name": "model2",
        "color": (0, 0, 255),
        "discount_factor": 0.5,
        "learning_rate": 0.01,
        # Additional parameters for model3
        "parameter1": 0.7,
        "parameter2": "value3"
    },
        {
        "name": "model3",
        "color": (125, 125, 0),
        "discount_factor": 0.75,
        "learning_rate": 0.005,
        # Additional parameters for model3
        "parameter1": 0.8,
        "parameter2": "value4"
    },
    {
        "name": "model4",
        "color": (0, 125, 125),
        "discount_factor": 0.65,
        "learning_rate": 0.02,
        # Additional parameters for model4
        "parameter1": 0.6,
        "parameter2": "value5"
    },
    {
        "name": "model5",
        "color": (125, 0, 125),
        "discount_factor": 0.85,
        "learning_rate": 0.03,
        # Additional parameters for model5
        "parameter1": 0.9,
        "parameter2": "value6"
    }, 
{
    "name": "model6",
    "color": (0, 0, 125),
    "discount_factor": 0.45,
    "learning_rate": 0.05,
    # Additional parameters for model6
    "parameter1": 0.9,
    "parameter2": "value7"
},  # <-- This is the missing comma
{
    "name": "model7",
    "color": (125, 125, 125),
    "discount_factor": 0.7,
    "learning_rate": 0.015,
    # Additional parameters for model7
    "parameter1": 0.4,
    "parameter2": "value8"
},
{
    "name": "model8",
    "color": (255, 255, 255),
    "discount_factor": 0.55,
    "learning_rate": 0.02,
    # Additional parameters for model8
    "parameter1": 0.3,
    "parameter2": "value9"
}


    # Add more models here
]

EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 32
REPLAY_MEMORY_SIZE = 5000

# Pygame initialization
pygame.init()

# Game surface
win = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))

# Clock for controlling the frame rate
clock = pygame.time.Clock()

# AI DQN model
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1)  # Change the input channel to 3
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 4)  # output for each action

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 32 * 16 * 16)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Initialize the models, criterions, optimizers, replay memories, and epsilon values
models = []
criterions = []
optimizers = []
replay_memories = []
epsilons = []

for model_params in MODEL_PARAMS:
    model = DQN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=model_params["learning_rate"])
    replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
    epsilon = EPSILON_START

    models.append(model)
    criterions.append(criterion)
    optimizers.append(optimizer)
    replay_memories.append(replay_memory)
    epsilons.append(epsilon)

# Function to spawn fruit
def spawn_fruit(snake):
    while True:
        x, y = np.random.randint(GRID_SIZE), np.random.randint(GRID_SIZE)
        if (x, y) not in snake:
            return (x, y)

# AI game state
snakes = []
directions = ["UP", "RIGHT", "DOWN", "LEFT"]
fruit = None  # Single shared apple position

# Function to reset the game
def reset():
    global snakes, directions, fruit
    snakes = [deque([(GRID_SIZE//2, GRID_SIZE//2)]) for _ in range(len(MODEL_PARAMS))]
    directions = [0] * len(MODEL_PARAMS)  # start moving up
    fruit = spawn_fruit(snakes[0])  # Spawn a single shared fruit for all models

# Initialize models, criterions, optimizers, and replay memories
models = []
criterions = []
optimizers = []
replay_memories = []

for model_params in MODEL_PARAMS:
    model = DQN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=model_params["learning_rate"])
    replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

    models.append(model)
    criterions.append(criterion)
    optimizers.append(optimizer)
    replay_memories.append(replay_memory)

# Epsilon values for each model
epsilons = [EPSILON_START] * len(MODEL_PARAMS)

# Get the path for saving the models
model_paths = [os.path.join(os.getcwd(), model_params["name"] + ".pth") for model_params in MODEL_PARAMS]

# Load the model parameters if the files exist
for i, model_path in enumerate(model_paths):
    if os.path.isfile(model_path):
        try:
            models[i].load_state_dict(torch.load(model_path))
            print("Loaded model parameters from:", model_path)
        except (FileNotFoundError, RuntimeError):
            print("No existing model parameters found or incompatible parameters. Starting training from scratch.")
    else:
        print("No existing model parameters found. Starting training from scratch.")

# Update the epsilon value for each model
def update_epsilons():
    global epsilons
    epsilons = [max(epsilon * EPSILON_DECAY, EPSILON_END) for epsilon in epsilons]

def step(action, model_index):
    global directions, snakes, fruit
    head_x, head_y = snakes[model_index][0]
    if action == 0:  # UP
        head_y -= 1
    elif action == 1:  # RIGHT
        head_x += 1
    elif action == 2:  # DOWN
        head_y += 1
    elif action == 3:  # LEFT
        head_x -= 1
    head_x %= GRID_SIZE
    head_y %= GRID_SIZE
    if (head_x, head_y) in snakes[model_index] or any((head_x, head_y) in s for i, s in enumerate(snakes) if i != model_index):
        # Reset the snake's size to its initial state
        snakes[model_index] = deque([(GRID_SIZE//2, GRID_SIZE//2)])
        return -10  # hit self or other model
    snakes[model_index].appendleft((head_x, head_y))
    if (head_x, head_y) == fruit:
        fruit = None  # remove the eaten fruit
        fruit = spawn_fruit(snakes[model_index])  # spawn a new fruit
        return 10  # got fruit
    else:
        snakes[model_index].pop()  # didn't get fruit, remove tail
        if fruit is None:
            fruit = spawn_fruit(snakes[model_index])
        reward = -1 + 1 / distance(head_x, head_y, fruit[0], fruit[1])  # normal step + smaller reward for getting closer
        if len(snakes[model_index]) > max(len(snake) for snake in snakes):
            reward += 5  # reward for being the longest snake
        return reward

# Update the model for each model
def update_models(batch, model_index):
    states, actions, rewards, new_states, dones = zip(*batch)

    states = torch.from_numpy(np.array(states)).float()  # Do not add an extra dimension
    actions = torch.tensor(actions)
    rewards = torch.tensor(rewards, dtype=torch.float)
    new_states = torch.from_numpy(np.array(new_states)).float()  # Do not add an extra dimension
    dones = torch.tensor(np.array(dones, dtype=bool), dtype=torch.bool)

    model = models[model_index]
    criterion = criterions[model_index]
    optimizer = optimizers[model_index]

    q_values = model(states)
    next_q_values = model(new_states)

    q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()
    next_q_value = next_q_values.max(1)[0]
    expected_q_values = rewards + MODEL_PARAMS[model_index]["discount_factor"] * next_q_value * (~dones)

    loss = criterion(q_values, expected_q_values.detach())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Update the epsilon for each model
def update_epsilon(model_index):
    epsilons[model_index] = max(epsilons[model_index] * EPSILON_DECAY, EPSILON_END)

# Function to calculate the distance between two points
def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# Function to draw the grid
def draw_grid():
    win.fill((0, 0, 0))
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            rect = pygame.Rect(x*(WINDOW_SIZE//GRID_SIZE), y*(WINDOW_SIZE//GRID_SIZE), WINDOW_SIZE//GRID_SIZE, WINDOW_SIZE//GRID_SIZE)
            pygame.draw.rect(win, (0, 255, 0), rect, GRID_OFFSET)

# Function to draw the snake for each model
def draw_snakes():
    for i, snake in enumerate(snakes):
        color = MODEL_PARAMS[i]["color"]
        for j, position in enumerate(snake):
            x, y = position
            rect = pygame.Rect(x*(WINDOW_SIZE//GRID_SIZE), y*(WINDOW_SIZE//GRID_SIZE), WINDOW_SIZE//GRID_SIZE, WINDOW_SIZE//GRID_SIZE)
            pygame.draw.rect(win, color, rect)
            # Draw the head larger
            if j == 0:
                rect = pygame.Rect(x*(WINDOW_SIZE//GRID_SIZE)+GRID_OFFSET, y*(WINDOW_SIZE//GRID_SIZE)+GRID_OFFSET, WINDOW_SIZE//GRID_SIZE-2*GRID_OFFSET, WINDOW_SIZE//GRID_SIZE-2*GRID_OFFSET)
                pygame.draw.rect(win, color, rect)


# Function to draw the fruits for each model
def draw_fruits():
    if fruit is not None:
        for i in range(len(MODEL_PARAMS)):
            color = MODEL_PARAMS[i]["color"]
            x, y = fruit
            rect = pygame.Rect(x*(WINDOW_SIZE//GRID_SIZE), y*(WINDOW_SIZE//GRID_SIZE), WINDOW_SIZE//GRID_SIZE, WINDOW_SIZE//GRID_SIZE)
            pygame.draw.rect(win, color, rect)

# Reset the game
reset()

# Game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            # Save the model parameters before exiting
            for i, model_path in enumerate(model_paths):
                torch.save(models[i].state_dict(), model_path)
                print("Saved model parameters to:", os.path.abspath(model_path))
            pygame.quit()
            sys.exit()

    # Get states for each model
    states = []
    for i in range(len(models)):
        state = np.zeros((3, GRID_SIZE, GRID_SIZE))  # Change from 2 to 3
        head_x, head_y = snakes[i][0]
        state[0][head_y][head_x] = 1  # mark head position
        for j, (x, y) in enumerate(snakes[i]):
            if j > 0:
                state[0][y][x] = -1  # mark snake body
        for j, other_snake in enumerate(snakes):
            if j != i:
                for (x, y) in other_snake:
                    state[1][y][x] = -1  # mark other snake body
        if fruit is not None:
            state[2] = distance(head_x, head_y, fruit[0], fruit[1])  # Add distance to fruit as a feature
        states.append(state)

    # AI decision for each model
    actions = []
    for i, model in enumerate(models):
        q_values = model(torch.from_numpy(np.array([states[i]])).float())
        if random.random() > epsilons[i]:  # Exploitation
            action = torch.argmax(q_values).item()
        else:  # Exploration
            action = random.randint(0, 3)  # Random action
        actions.append(action)

    # Perform the actions and get the new states
    new_states = []
    rewards = []
    for i, action in enumerate(actions):
        reward = step(action, i)
        rewards.append(reward)

        new_state = np.zeros((3, GRID_SIZE, GRID_SIZE))  # Change from 2 to 3
        head_x, head_y = snakes[i][0]
        new_state[0][head_y][head_x] = 1  # mark head position
        for j, (x, y) in enumerate(snakes[i]):
            if j > 0:
                new_state[0][y][x] = -1  # mark snake body
        for j, other_snake in enumerate(snakes):
            if j != i:
                for (x, y) in other_snake:
                    new_state[1][y][x] = -1  # mark other snake body
        if fruit is not None:
            new_state[2] = distance(head_x, head_y, fruit[0], fruit[1])  # Add distance to fruit as a feature
        new_states.append(new_state)

    # Store the transition in replay memory
    for i in range(len(models)):
        replay_memories[i].append((states[i], actions[i], rewards[i], new_states[i], False))  # False for 'done' as the game does not end

    # Train the models
    for i in range(len(models)):
        if len(replay_memories[i]) < BATCH_SIZE:
            continue
        batch = random.sample(replay_memories[i], BATCH_SIZE)
        update_models(batch, i)
        update_epsilon(i)

    # Draw everything
    draw_grid()
    draw_snakes()
    draw_fruits()

    pygame.display.update()
    clock.tick(60)  # Limit the frame rate to 60 FPS
