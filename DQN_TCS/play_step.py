from re import S  # 导入re模块的S标志
from matplotlib import collections  # 导入matplotlib的collections模块
import pygame  # 导入pygame库
from enum import Enum  # 导入Enum枚举类
import random  # 导入random模块
from collections import namedtuple, deque  # 导入namedtuple和deque类
import numpy as np  # 导入numpy库

# 初始化pygame
pygame.init()
BLOCK_SIZE = 20  # 方块大小
BLACK = (0, 0, 0)  # 颜色常量
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
INITIAL_SPEED = 10  # 初始速度
FONT = pygame.font.Font('arial.ttf', 25)  # 设置字体

# 定义一个Point命名元组，表示坐标点
Point = namedtuple('Point', 'x, y')

# 定义一个Direction枚举类，表示移动方向
class Direction(Enum):
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4

# 定义SnakeGameAI类，实现贪吃蛇游戏
class SnakeGameAI:
    def __init__(self, w=640, h=480):
        self.W = w  # 窗口宽度
        self.H = h  # 窗口高度
        self.direction = Direction.RIGHT  # 初始移动方向为右
        self.display = pygame.display.set_mode((self.W, self.H))  # 创建游戏窗口
        self.clock = pygame.time.Clock()  # 创建时钟对象
        pygame.display.set_caption('Snake')  # 设置窗口标题
        self.paused = False  # 初始化为未暂停状态
        self.reset()  # 初始化游戏状态

    def reset(self, keep_score=False):
        # 初始化蛇头和蛇身
        self.head = Point(x=self.W / 2, y=self.H / 2)
        self.snake = [
            self.head,
            Point(x=self.head.x - BLOCK_SIZE, y=self.head.y),
            Point(x=2 * self.head.x - BLOCK_SIZE, y=self.head.y),
        ]
        self.food = None  # 初始化食物位置
        self._place_food()  # 放置食物
        self.frame_iteration = 0  # 帧数
        if not keep_score:
            self.score = 0  # 分数
        self.speed = INITIAL_SPEED  # 初始速度

    def _place_food(self):
        # 随机放置食物，不能放在蛇身上
        x = random.randint(0, (self.W - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.H - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if Point(x, y) in self.snake:
            self._place_food()

    def play_step(self):
    # 游戏每一步的逻辑
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and self.direction != Direction.RIGHT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT and self.direction != Direction.LEFT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP and self.direction != Direction.DOWN:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN and self.direction != Direction.UP:
                    self.direction = Direction.DOWN
                elif event.key == pygame.K_SPACE:
                    if not self.paused:
                        self.paused = True
                        self._show_pause_screen()
                    else:
                        self.paused = False

        if self.paused:
            return 0, False, self.score

        # 移动蛇头
        self._move(self.direction)
        self.snake.insert(0, self.head)

        # 检查游戏是否结束
        is_done = False
        reward = 0
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            is_done = True
            reward -= 10
            self._show_game_over_screen()
            return reward, is_done, self.score

        # 判断是否吃到食物
        if self.head == self.food:
            self._place_food()
            self.score += 1
            reward = 10
        else:
            self.snake.pop()

        # 更新速度
        self.speed = (INITIAL_SPEED *0.5 + len(self.snake) ) * 1.1

        # 更新UI
        self._update_ui()
        self.clock.tick(self.speed)

        # 返回奖励、游戏是否结束和分数
        return reward, is_done, self.score


    def is_collision(self, pt=None):
        # 判断是否发生碰撞
        if pt is None:
            pt = self.head
        if pt in self.snake[1:]:
            return True
        if pt.x < 0 or pt.x > self.W - BLOCK_SIZE or pt.y < 0 or pt.y > self.H - BLOCK_SIZE:
            return True
        return False

    def _update_ui(self):
        # 更新游戏UI
        # 绘制背景图片
        background_image = pygame.image.load('image.jpg')
        background_image = pygame.transform.scale(background_image, (self.W, self.H))
        self.display.blit(background_image, (0, 0))

        # 绘制蛇和食物
        for i, pt in enumerate(self.snake):
            if i == 0:
                pygame.draw.rect(self.display, (0, 255, 0), pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, (0, 155, 0), pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
            else:
                pygame.draw.rect(self.display, (0, 255, 0), pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        text = FONT.render('Score:' + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, direction):
        # 移动蛇头
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        self.head = Point(x, y)
        self.speed = INITIAL_SPEED + len(self.snake) // 5

    def _show_game_over_screen(self):
        # 显示游戏结束选项
        self.display.fill(BLACK)
        game_over_text = FONT.render('Game Over', True, WHITE)
        self.display.blit(game_over_text, [self.W / 2 - game_over_text.get_width() / 2, self.H / 2 - 100])
        
        restart_text = FONT.render('Press R to Restart', True, WHITE)
        self.display.blit(restart_text, [self.W / 2 - restart_text.get_width() / 2, self.H / 2])
        
        continue_text = FONT.render('Press C to Continue', True, WHITE)
        self.display.blit(continue_text, [self.W / 2 - continue_text.get_width() / 2, self.H / 2 + 50])
        
        quit_text = FONT.render('Press Q to Quit', True, WHITE)
        self.display.blit(quit_text, [self.W / 2 - quit_text.get_width() / 2, self.H / 2 + 100])
        
        pygame.display.flip()

        waiting_for_input = True
        while waiting_for_input:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.reset()
                        waiting_for_input = False
                    elif event.key == pygame.K_c:
                        self.reset(keep_score=True)
                        waiting_for_input = False
                    elif event.key == pygame.K_q:
                        pygame.quit()
                        quit()

    def _show_pause_screen(self):
    # 显示暂停选项
        pause_text = FONT.render('Game Paused', True, WHITE)
        self.display.blit(pause_text, [self.W / 2 - pause_text.get_width() / 2, self.H / 2])
        pygame.display.flip()

        waiting_for_input = True
        while waiting_for_input:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        waiting_for_input = False
        self.paused = False


if __name__ == '__main__':
    game = SnakeGameAI()
    while True:
        game.play_step()