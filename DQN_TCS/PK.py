import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import tkinter as tk

# 游戏设置
GAME_WIDTH = 800
GAME_HEIGHT = 600
SNAKE_SIZE = 20
FPS = 10
# DQN设置
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
LEARNING_RATE = 0.001
MEMORY_SIZE = 10000
BATCH_SIZE = 64
MODEL_PATH = "snake_model.pth"


# 贪吃蛇游戏
class SnakeGame:
    def __init__(self):
        self.width = GAME_WIDTH
        self.height = GAME_HEIGHT
        self.snake1 = [(self.width // 4, self.height // 2)]
        self.snake2 = [(self.width * 3 // 4, self.height // 2)]
        self.direction1 = (0, -SNAKE_SIZE)
        self.direction2 = (0, -SNAKE_SIZE)
        self.food = self.place_food()
        self.score1 = 0
        self.score2 = 0

    def place_food(self):
        return (random.randint(0, (self.width - SNAKE_SIZE) // SNAKE_SIZE) * SNAKE_SIZE,
                random.randint(0, (self.height - SNAKE_SIZE) // SNAKE_SIZE) * SNAKE_SIZE)

    def step(self, action1, action2):
        self.direction1 = self.get_new_direction(action1, self.direction1)
        self.direction2 = self.get_new_direction(action2, self.direction2)

        new_head1 = (self.snake1[0][0] + self.direction1[0], self.snake1[0][1] + self.direction1[1])
        new_head2 = (self.snake2[0][0] + self.direction2[0], self.snake2[0][1] + self.direction2[1])
        
        self.snake1.insert(0, new_head1)
        self.snake2.insert(0, new_head2)

        if new_head1 == self.food:
            self.food = self.place_food()
            self.score1 += 1
        else:
            self.snake1.pop()

        if new_head2 == self.food:
            self.food = self.place_food()
            self.score2 += 1
        else:
            self.snake2.pop()

        reward1, done1 = self.check_collision(new_head1, self.snake1)
        reward2, done2 = self.check_collision(new_head2, self.snake2)

        return (new_head1, reward1, done1), (new_head2, reward2, done2)

    def get_new_direction(self, action, current_direction):
        if action == 0:  # Up
            new_direction = (0, -SNAKE_SIZE)
        elif action == 1:  # Down
            new_direction = (0, SNAKE_SIZE)
        elif action == 2:  # Left
            new_direction = (-SNAKE_SIZE, 0)
        elif action == 3:  # Right
            new_direction = (SNAKE_SIZE, 0)
        else:
            new_direction = current_direction

        if (current_direction[0] + new_direction[0] == 0 and current_direction[1] + new_direction[1] == 0):
            new_direction = current_direction
        
        return new_direction

    def check_collision(self, new_head, snake):
        reward = 0
        done = False

        if (new_head[0] < 0 or new_head[0] >= self.width or
                new_head[1] < 0 or new_head[1] >= self.height or
                new_head in snake[1:]):
            done = True
            reward = -10
        else:
            reward = 1

        return reward, done

# 神经网络
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        return self.net(x)

# 代理
class Agent:
    def __init__(self):
        self.model = self.load_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()
        self.memory = []
        self.epsilon = EPSILON

    def load_model(self):
        model = DQN()
        try:
            state_dict = torch.load(MODEL_PATH)
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict[k.replace("net.", "")] = v
            model.load_state_dict(new_state_dict)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
        return model

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state)
            return torch.argmax(q_values).item()

    def remember(self, transition):
        if len(self.memory) > MEMORY_SIZE:
            self.memory.pop(0)
        self.memory.append(transition)

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        state, action, reward, next_state, done = zip(*batch)

        state = torch.FloatTensor(state)
        action = torch.LongTensor(action).unsqueeze(1)
        reward = torch.FloatTensor(reward)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done)

        q_values = self.model(state).gather(1, action).squeeze(1)
        next_q_values = self.model(next_state).max(1)[0]
        target = reward + GAMMA * next_q_values * (1 - done)

        loss = self.criterion(q_values, target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def start_game():
    game = SnakeGame()
    agent = Agent()
    clock = pygame.time.Clock()
    pygame.init()
    screen = pygame.display.set_mode((GAME_WIDTH, GAME_HEIGHT))
    pygame.display.set_caption("贪吃蛇人机对抗")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action1 = 0
        elif keys[pygame.K_DOWN]:
            action1 = 1
        elif keys[pygame.K_LEFT]:
            action1 = 2
        elif keys[pygame.K_RIGHT]:
            action1 = 3
        else:
            action1 = -1  # No action

        state2 = game.snake2[0]
        action2 = agent.get_action(state2)
        (next_state1, reward1, done1), (next_state2, reward2, done2) = game.step(action1, action2)

        if action1 != -1:
            agent.remember((state2, action2, reward2, next_state2, done2))
            agent.train()

        screen.fill((0, 0, 0))
        for segment in game.snake1:
            pygame.draw.rect(screen, (0, 255, 0), (*segment, SNAKE_SIZE, SNAKE_SIZE))
        for segment in game.snake2:
            pygame.draw.rect(screen, (0, 0, 255), (*segment, SNAKE_SIZE, SNAKE_SIZE))
        pygame.draw.rect(screen, (255, 0, 0), (*game.food, SNAKE_SIZE, SNAKE_SIZE))
        pygame.display.flip()
        clock.tick(FPS)

        if done1 or done2:
            running = False

    pygame.quit()

def quit_game():
    root.quit()
    root.destroy()

root = tk.Tk()
root.title("贪吃蛇人机对抗")

game_frame = tk.Frame(root, width=GAME_WIDTH, height=GAME_HEIGHT)
game_frame.pack()

start_button = tk.Button(root, text="开始游戏", command=start_game)
start_button.pack()

root.protocol("WM_DELETE_WINDOW", quit_game)
root.mainloop()
