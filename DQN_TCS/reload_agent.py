from game import BLOCK_SIZE, Direction, Point, SnakeGameAI  # 导入游戏相关的模块和类
import torch  # 导入PyTorch库
import numpy as np  # 导入NumPy库
from model import Linear_QNet, QTrainer  # 导入神经网络模型和训练器类
from collections import deque  # 导入双端队列类
import random  # 导入随机数模块

# 修改颜色
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (200, 0, 0)
GREEN = (0, 255, 0)  # 新的绿色

LR = 0.001  # 学习率
MEMORY_SIZE = 100_1000  # 记忆大小
BATCH_SIZE = 100  # 批次大小

# Agent 类，用于控制智能体的行为
class Agent:
    def __init__(self):
        self.model = Linear_QNet(11, 256, 3)  # 创建神经网络模型
        self.gama = 0.9  # 折扣因子
        self.epsilon = 0  # 探索率
        self.n_games = 0  # 游戏次数
        self.memory = deque(maxlen=MEMORY_SIZE)  # 经验回放记忆
        self.trainer = QTrainer(self.model, LR, self.gama)  # Q学习训练器

    # 根据状态获取动作
    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

    # 获取游戏状态
    def get_state(self, game):
        head = game.snake[0]

        pt_left = Point(head.x - BLOCK_SIZE, head.y)
        pt_right = Point(head.x + BLOCK_SIZE, head.y)
        pt_up = Point(head.x, head.y - BLOCK_SIZE)
        pt_down = Point(head.x, head.y + BLOCK_SIZE)

        dir_left = game.direction == Direction.LEFT
        dir_right = game.direction == Direction.RIGHT
        dir_up = game.direction == Direction.UP
        dir_down = game.direction == Direction.DOWN

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
        return np.array(state, dtype=int)

    # 记忆
    def remember(self, state, action, reward, next_state, is_done):
        self.memory.append((state, action, reward, next_state, is_done))

    # 短期记忆训练
    def train_short_memory(self, state, action, reward, next_state, is_done):
        self.trainer.train_step(state, action, reward, next_state, is_done)

    # 长期记忆训练
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, is_dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, is_dones)

# 创建一个新的 Agent 实例
agent = Agent()
# 加载已保存的模型权重
agent.model.load_state_dict(torch.load(r'D:\毕业设计\DQN_TCS\model\model.pth'))

# 训练函数
def train():
    plot_scores = []  # 绘制分数
    plot_mean_scores = []  # 绘制平均分数
    total_score = 0  # 总分数
    record = 0  # 最高分数
    game = SnakeGameAI()  # 创建游戏实例
    while True:
        state_old = agent.get_state(game)  # 获取当前状态
        final_move = agent.get_action(state_old)  # 获取当前动作
        reward, is_done, score = game.play_step(final_move)  # 执行动作并获取奖励和分数
        state_next = agent.get_state(game)  # 获取下一个状态

        agent.train_short_memory(state_old, final_move, reward, state_next, is_done)  # 训练短期记忆
        agent.remember(state_old, final_move, reward, state_next, is_done)  # 记忆

        if is_done:  # 如果游戏结束
            agent.n_games += 1  # 游戏次数加一
            game.reset()  # 重置游戏
            agent.train_long_memory()  # 训练长期记忆
            if score > record:  # 如果分数超过最高记录
                record = score  # 更新最高记录
                agent.model.save_model()  # 保存模型权重
            print('Game', agent.n_games, 'Score', score, 'Record:', record)  # 打印当前游戏信息
            total_score += score  # 更新总分数
            mean_scores = total_score / agent.n_games  # 计算平均分数
            plot_mean_scores.append(mean_scores)  # 添加到绘制平均分数列表中

if __name__ == '__main__':
    train()  # 执行训练函数
