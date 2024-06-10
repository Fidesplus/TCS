from game import BLOCK_SIZE, Direction, Point, SnakeGameAI  # 导入游戏相关的模块和类
import torch  # 导入PyTorch库
import numpy as np  # 导入NumPy库
from model import Linear_QNet, QTrainer  # 导入神经网络模型和训练器类
from collections import deque  # 导入双向队列类
import random  # 导入随机数模块
import os  # 导入操作系统模块

LR = 0.001  # 学习率
MEMORY_SIZE = 100_1000  # 记忆库大小
BATCH_SIZE = 100  # 批处理大小

# 定义Agent类，实现蛇游戏AI的训练
class Agent:
    def __init__(self):
        self.model = Linear_QNet(11, 256, 3)  # 创建线性Q网络模型
        self.gamma = 0.9  # 折扣因子
        self.epsilon = 0  # 探索率
        self.n_games = 0  # 游戏次数
        self.memory = deque(maxlen=MEMORY_SIZE)  # 创建双向队列作为记忆库
        self.trainer = QTrainer(self.model, LR, self.gamma)  # 创建Q网络训练器

    def get_action(self, state):
        self.epsilon = 80 - self.n_games  # 更新探索率
        final_move = [0, 0, 0]  # 最终动作
        if random.randint(0, 200) < self.epsilon:  # epsilon贪婪策略
            move = random.randint(0, 2)  # 随机选择动作
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)  # 转换状态为PyTorch张量
            prediction = self.model(state0)  # 获取模型预测
            move = torch.argmax(prediction).item()  # 获取最大值的索引
            final_move[move] = 1

        return final_move  # 返回最终动作

    def get_state(self, game):
        head = game.snake[0]  # 获取蛇头位置

        # 定义相对于蛇头位置的四个方向的点
        pt_left = Point(head.x - BLOCK_SIZE, head.y)
        pt_right = Point(head.x + BLOCK_SIZE, head.y)
        pt_up = Point(head.x, head.y - BLOCK_SIZE)
        pt_down = Point(head.x, head.y + BLOCK_SIZE)

        # 判断四个方向上的危险性和移动方向
        dir_left = game.direction == Direction.LEFT
        dir_right = game.direction == Direction.RIGHT
        dir_up = game.direction == Direction.UP
        dir_down = game.direction == Direction.DOWN

        # 构建状态列表
        state = [
            # danger straight
            (dir_up and game.is_collision(pt_up)) or
            (dir_down and game.is_collision(pt_down)) or
            (dir_left and game.is_collision(pt_left)) or
            (dir_right and game.is_collision(pt_right)),

            # danger left
            (dir_up and game.is_collision(pt_left)) or
            (dir_down and game.is_collision(pt_right)) or
            (dir_left and game.is_collision(pt_down)) or
            (dir_right and game.is_collision(pt_up)),

            # danger right
            (dir_up and game.is_collision(pt_right)) or
            (dir_down and game.is_collision(pt_left)) or
            (dir_left and game.is_collision(pt_up)) or
            (dir_right and game.is_collision(pt_down)),

            # move direction
            dir_up,
            dir_down,
            dir_left,
            dir_right,

            # food location
            game.food.x < head.x,  # food in left
            game.food.x > head.x,  # food in right
            game.food.y < head.y,  # food in up
            game.food.y > head.y,  # food in down

        ]
        return np.array(state, dtype=int)  # 返回状态数组

    def remember(self, state, action, reward, next_state, is_done):
        self.memory.append((state, action, reward, next_state, is_done))  # 添加经验到记忆库

    def train_short_memory(self, state, action, reward, next_state, is_done):
        self.trainer.train_step(state, action, reward, next_state, is_done)  # 训练短期记忆

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # 从记忆库中随机采样
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, is_dones = zip(*mini_sample)  # 拆分样本
        self.trainer.train_step(states, actions, rewards, next_states, is_dones)  # 训练长期记忆

    def save_model(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.mkdir(model_folder_path)  # 创建模型文件夹
        file_name = os.path.join(model_folder_path, file_name)  # 拼接文件路径
        torch.save(self.model.state_dict(), file_name)  # 保存模型权重

    def load_model(self, file_path='model.pth'):
        file_path = os.path.join('./model', file_path)  # 拼接文件路径
        self.model.load_state_dict(torch.load(file_path))  # 加载模型权重


# 定义训练函数
def train():
    plot_scores = []  # 记录分数
    plot_mean_scores = []  # 记录平均分数
    total_score = 0  # 总分
    record = 0  # 记录最高分
    agent = Agent()  # 创建Agent对象

    # 不加载之前保存的模型，直接开始训练
    game = SnakeGameAI()  # 创建游戏对象
    while True:
        state_old = agent.get_state(game)  # 获取旧状态
        final_move = agent.get_action(state_old)  # 获取动作
        reward, is_done, score = game.play_step(final_move)  # 执行动作并返回奖励、游戏是否结束和分数
        state_next = agent.get_state(game)  # 获取新状态

        agent.train_short_memory(state_old, final_move, reward, state_next, is_done)  # 训练短期记忆
        agent.remember(state_old, final_move, reward, state_next, is_done)  # 添加经验到记忆库

        if is_done:  # 如果游戏结束
            agent.n_games += 1  # 游戏次数加一
            game.reset()  # 重置游戏
            agent.train_long_memory()  # 训练长期记忆
            if score > record:  # 如果分数高于记录的最高分
                record = score  # 更新最高分
                agent.save_model()  # 保存模型权重
            print('Game', agent.n_games, 'Score', score, 'Record:', record)  # 打印游戏次数、分数和最高分
            total_score += score  # 更新总分
            mean_scores = total_score / agent.n_games  # 计算平均分数
            plot_mean_scores.append(mean_scores)  # 添加平均分数到列表


if __name__ == '__main__':
    train()  # 开始训练
