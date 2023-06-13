import pygame
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Pygame parameters
WINDOW_SIZE = 600
GRID_SIZE = 20
GRID_OFFSET = 2

# AI parameters
DISCOUNT_FACTOR = 0.8
EPSILON = 0.1
LEARNING_RATE = 0.01
BATCH_SIZE = 32

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
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1)
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

# AI game state
snake = deque([(GRID_SIZE//2, GRID_SIZE//2)])
directions = ["UP", "RIGHT", "DOWN", "LEFT"]
direction = 0  # start moving up
fruit = None

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

def spawn_fruit():
    global fruit
    while True:
        x, y = np.random.randint(GRID_SIZE), np.random.randint(GRID_SIZE)
        if (x, y) not in snake:
            fruit = (x, y)
            break

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
        return -1  # hit self
    snake.appendleft((head_x, head_y))
    if fruit is None or (head_x, head_y) != fruit:
        snake.pop()  # didn't get fruit, remove tail
        if fruit is None:
            spawn_fruit()
        return 0  # normal step
    else:
        spawn_fruit()
        return 1  # got fruit

def update_model(state, action, new_state, reward, done):
    # Convert state to tensor
    state_tensor = torch.from_numpy(state).float().unsqueeze(0)
    new_state_tensor = torch.from_numpy(new_state).float().unsqueeze(0)

    # Get Q-values
    q_values = model(state_tensor)
    with torch.no_grad():
        new_q_values = model(new_state_tensor)

    # Get target Q-value
    target_q_value = reward
    if not done:
        target_q_value += DISCOUNT_FACTOR * torch.max(new_q_values).item()

    # Convert target Q-value to tensor with the same size as q_values
    target_q_values_tensor = q_values.clone()
    target_q_values_tensor[0][action] = target_q_value

    # Compute loss
    loss = criterion(q_values, target_q_values_tensor)

    # Update model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



# Load the model parameters if the file exists
try:
    model.load_state_dict(torch.load("model.pth"))
    print("Loaded model parameters from 'model.pth'")
except FileNotFoundError:
    print("No existing model parameters found. Starting training from scratch.")

# Game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            # Save the model parameters before exiting
            torch.save(model.state_dict(), "model.pth")
            print("Saved model parameters to 'model.pth'")
            pygame.quit()
            sys.exit()

    # Get state
    state = np.zeros((GRID_SIZE, GRID_SIZE))
    head_x, head_y = snake[0]
    state[head_y][head_x] = 1  # mark head position
    for i, (x, y) in enumerate(snake):
        if i > 0:
            state[y][x] = -1  # mark snake body

    # Select action
    if random.random() < EPSILON:
        action = random.randint(0, 3)  # random action
    else:
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            action = torch.argmax(model(state_tensor)).item()  # best action

    # Perform action
    reward = step(action)
    if reward == -1:  # hit self
        reset()
    else:
        new_state = np.zeros((GRID_SIZE, GRID_SIZE))
        head_x, head_y = snake[0]
        new_state[head_y][head_x] = 1  # mark head position
        for i, (x, y) in enumerate(snake):
            if i > 0:
                new_state[y][x] = -1  # mark snake body
        update_model(state, action, new_state, reward, reward == -1)

    # Drawing
    draw_grid()
    draw_snake()
    draw_fruit()

    pygame.display.update()
    clock.tick(10)
