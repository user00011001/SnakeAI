import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Game settings
WINDOW_SIZE = 600
GRID_SIZE = 20
GRID_OFFSET = 2

# DQN settings
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 32
REPLAY_MEMORY_SIZE = 5000
DISCOUNT_FACTOR = 0.95
LEARNING_RATE = 0.001

# Initialize pygame
pygame.init()
win = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
clock = pygame.time.Clock()

# DQN Model
class DQN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Initialize DQN model and settings
model = DQN(6, 4)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
epsilon = EPSILON_START

# Snake game settings
snake = deque([(GRID_SIZE//2, GRID_SIZE//2)])
direction = 0
fruit = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)

def reset():
    global snake, direction, fruit
    snake = deque([(GRID_SIZE//2, GRID_SIZE//2)])
    direction = 0
    fruit = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)

def step(action):
    global snake, direction, fruit

    if action == 0:  # UP
        head = snake[0][0], snake[0][1] - 1
    elif action == 1:  # RIGHT
        head = snake[0][0] + 1, snake[0][1]
    elif action == 2:  # DOWN
        head = snake[0][0], snake[0][1] + 1
    elif action == 3:  # LEFT
        head = snake[0][0] - 1, snake[0][1]

    if head in snake or head[0] < 0 or head[0] >= GRID_SIZE or head[1] < 0 or head[1] >= GRID_SIZE:
        reset()
        return -10

    snake.appendleft(head)

    if head == fruit:
        fruit = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
        return 10

    snake.pop()
    return -1

def game_loop():
    global direction, epsilon
    running = True
    score = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # DQN chooses action
        state = extract_state()
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            q_values = model(state)
            action = torch.argmax(q_values).item()

        reward = step(action)
        score += reward

        # Save to replay memory
        next_state = extract_state()
        replay_memory.append((state, action, reward, next_state))

        # Train the model using replay memory
        if len(replay_memory) >= BATCH_SIZE:
            minibatch = random.sample(replay_memory, BATCH_SIZE)
            train(minibatch)

        # Decrease epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        draw_window(score)

    pygame.quit()

def draw_window(score):
    win.fill((0, 0, 0))
    for segment in snake:
        pygame.draw.rect(win, (0, 255, 0), (segment[0]*GRID_SIZE, segment[1]*GRID_SIZE, GRID_SIZE-GRID_OFFSET, GRID_SIZE-GRID_OFFSET))
    pygame.draw.rect(win, (255, 0, 0), (fruit[0]*GRID_SIZE, fruit[1]*GRID_SIZE, GRID_SIZE-GRID_OFFSET, GRID_SIZE-GRID_OFFSET))
    pygame.display.set_caption(f'Score: {score}')
    pygame.display.flip()

def extract_state():
    head = snake[0]
    dx = fruit[0] - head[0]
    dy = fruit[1] - head[1]
    return torch.tensor([head[0], head[1], dx, dy, len(snake), direction], dtype=torch.float32).unsqueeze(0)

def train(minibatch):
    states, actions, rewards, next_states = zip(*minibatch)
    states = torch.cat(states)
    next_states = torch.cat(next_states)
    actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(-1)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)

    current_q_values = model(states).gather(1, actions)
    max_next_q_values = model(next_states).max(1)[0].unsqueeze(-1)
    expected_q_values = rewards + DISCOUNT_FACTOR * max_next_q_values

    loss = criterion(current_q_values, expected_q_values.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

if __name__ == "__main__":
    game_loop()
