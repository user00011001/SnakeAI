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
DISCOUNT_FACTOR = 0.8
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.01
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
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=1)  # Change the input channel to 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 4)  # output for each action

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 32 * 16 * 16)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Initialize the model
model = DQN()

# Use Mean Squared Error loss for Q-Learning
criterion = nn.MSELoss()

# Use Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Set the path for saving the model
model_path = os.path.join(os.getcwd(), "model.pth")

# Check if a saved model exists
if os.path.isfile(model_path):
    try:
        model.load_state_dict(torch.load(model_path))
        print("Loaded model parameters from:", model_path)
    except (FileNotFoundError, RuntimeError):  # Added RuntimeError for size mismatch
        print("No existing model parameters found or incompatible parameters. Starting training from scratch.")
else:
    print("No existing model parameters found. Starting training from scratch.")

# Replay memory
replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

# AI game state
snake = deque([(GRID_SIZE//2, GRID_SIZE//2)])
directions = ["UP", "RIGHT", "DOWN", "LEFT"]
direction = 0  # start moving up
fruit = None

# Epsilon value
epsilon = EPSILON_START

# Function to spawn fruit
def spawn_fruit():
    global fruit
    while True:
        x, y = np.random.randint(GRID_SIZE), np.random.randint(GRID_SIZE)
        if (x, y) not in snake:
            fruit = (x, y)
            break

# Spawn a fruit at the beginning
spawn_fruit()

def draw_grid():
    win.fill((0, 0, 0))
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            rect = pygame.Rect(x*(WINDOW_SIZE//GRID_SIZE), y*(WINDOW_SIZE//GRID_SIZE), WINDOW_SIZE//GRID_SIZE, WINDOW_SIZE//GRID_SIZE)
            pygame.draw.rect(win, (0, 255, 0), rect, GRID_OFFSET)

def draw_snake():
    for position in snake:
        x, y = position
        rect = pygame.Rect(x*(WINDOW_SIZE//GRID_SIZE), y*(WINDOW_SIZE//GRID_SIZE), WINDOW_SIZE//GRID_SIZE, WINDOW_SIZE//GRID_SIZE)
        pygame.draw.rect(win, (255, 255, 255), rect)

def draw_fruit():
    if fruit is not None:
        x, y = fruit
        rect = pygame.Rect(x*(WINDOW_SIZE//GRID_SIZE), y*(WINDOW_SIZE//GRID_SIZE), WINDOW_SIZE//GRID_SIZE, WINDOW_SIZE//GRID_SIZE)
        pygame.draw.rect(win, (255, 0, 0), rect)

def reset():
    global snake, direction, fruit
    snake = deque([(GRID_SIZE//2, GRID_SIZE//2)])
    direction = 0
    fruit = None
    spawn_fruit()  

def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def step(action):
    global direction, snake, fruit
    head_x, head_y = snake[0]
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
    if (head_x, head_y) in snake:
        snake = deque([(head_x, head_y)])  # Reset snake's size to 1
        return -10  # hit self
    snake.appendleft((head_x, head_y))
    if (head_x, head_y) == fruit:
        fruit = None  # remove the eaten fruit
        spawn_fruit()  # spawn a new fruit
        return 10  # got fruit
    else:
        snake.pop()  # didn't get fruit, remove tail
        if fruit is None:
            spawn_fruit()
        return -1 + 1 / distance(head_x, head_y, fruit[0], fruit[1])  # normal step + smaller reward for getting closer

def update_model(batch):
    states, actions, rewards, new_states, dones = zip(*batch)

    states = torch.from_numpy(np.array(states)).float()  # Do not add an extra dimension
    actions = torch.tensor(actions)
    rewards = torch.tensor(rewards, dtype=torch.float)
    new_states = torch.from_numpy(np.array(new_states)).float()  # Do not add an extra dimension
    dones = torch.tensor(np.array(dones, dtype=bool), dtype=torch.bool)


    q_values = model(states)
    next_q_values = model(new_states)

    q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()
    next_q_value = next_q_values.max(1)[0]
    expected_q_values = rewards + DISCOUNT_FACTOR * next_q_value * (~dones)

    loss = criterion(q_values, expected_q_values.detach())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def update_epsilon():
    global epsilon
    epsilon = max(epsilon * EPSILON_DECAY, EPSILON_END)

# Load the model parameters if the file exists
model_path = "model.pth"
try:
    model.load_state_dict(torch.load(model_path))
    print("Loaded model parameters from:", os.path.abspath(model_path))
except (FileNotFoundError, RuntimeError):  # Added RuntimeError for size mismatch
    print("No existing model parameters found or incompatible parameters. Starting training from scratch.")

print("Current working directory:", os.getcwd())

# Game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            # Save the model parameters before exiting
            torch.save(model.state_dict(), model_path)
            print("Saved model parameters to:", os.path.abspath(model_path))
            pygame.quit()
            sys.exit()

    # Get state
    state = np.zeros((2, GRID_SIZE, GRID_SIZE))  # Change from 1 to 2
    head_x, head_y = snake[0]
    state[0][head_y][head_x] = 1  # mark head position
    for i, (x, y) in enumerate(snake):
        if i > 0:
            state[0][y][x] = -1  # mark snake body
    state[1] = distance(head_x, head_y, fruit[0], fruit[1])  # Add distance to fruit as a feature

    # AI decision
    q_values = model(torch.from_numpy(state).float().unsqueeze(0))
    if random.random() < epsilon:
        action = random.randint(0, 3)
    else:
        _, action = q_values.max(1)


    # Perform step
    reward = step(action)

    # Get new state
    new_state = np.zeros((2, GRID_SIZE, GRID_SIZE))  # Change from 1 to 2
    head_x, head_y = snake[0]
    new_state[0][head_y][head_x] = 1  # mark head position
    for i, (x, y) in enumerate(snake):
        if i > 0:
            new_state[0][y][x] = -1  # mark snake body
    new_state[1] = distance(head_x, head_y, fruit[0], fruit[1])  # Add distance to fruit as a feature

    # Add to replay memory
    replay_memory.append((state, action, reward, new_state, reward < 0))

    # Update model
    if len(replay_memory) >= BATCH_SIZE:
        batch = random.sample(replay_memory, BATCH_SIZE)
        update_model(batch)

    # Update epsilon
    update_epsilon()

    # Draw everything
    draw_grid()
    draw_snake()
    draw_fruit()

    # Update the window
    pygame.display.update()

    # Control the frame rate
    clock.tick(10)
