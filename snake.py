import pygame
import sys
import numpy as np
from collections import deque

# Pygame parameters
WINDOW_SIZE = 600
GRID_SIZE = 20
GRID_OFFSET = 2

# AI parameters
LEARNING_RATE = 0.5
DISCOUNT_FACTOR = 0.8
EPSILON = 0.1

# Pygame initialization
pygame.init()

# Game surface
win = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))

# Clock for controlling the frame rate
clock = pygame.time.Clock()

# AI Q-table
q_table = np.zeros((GRID_SIZE, GRID_SIZE, 4))

# Game state
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

def step():
    global direction, snake, fruit
    head_x, head_y = snake[0]
    if direction == 0:  # UP
        head_y -= 1
    elif direction == 1:  # RIGHT
        head_x += 1
    elif direction == 2:  # DOWN
        head_y += 1
    elif direction == 3:  # LEFT
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


def update_q_table(old_state, action, reward, new_state):
    old_q_value = q_table[old_state[0], old_state[1], action]
    next_max_q_value = np.max(q_table[new_state[0], new_state[1]])
    q_table[old_state[0], old_state[1], action] = (1 - LEARNING_RATE) * old_q_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max_q_value)

# Game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # AI logic
    old_state = snake[0]
    if np.random.rand() < EPSILON:
        direction = np.random.randint(4)  # random direction
    else:
        direction = np.argmax(q_table[old_state[0], old_state[1]])  # best direction

    # Step the game
    reward = step()
    if reward == -1:  # hit self
        reset()
    else:
        new_state = snake[0]
        update_q_table(old_state, direction, reward, new_state)

    # Drawing
    draw_grid()
    draw_snake()
    draw_fruit()

    pygame.display.update()
    clock.tick(10)
