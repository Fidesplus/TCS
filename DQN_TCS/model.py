import torch  # 导入PyTorch库
import torch.optim as optim  # 导入PyTorch优化器模块
import torch.nn as nn  # 导入PyTorch神经网络模块
import os  # 导入操作系统模块

# 定义一个线性神经网络模型
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(  # 定义神经网络结构
            nn.Linear(input_size, hidden_size),  # 输入层到隐藏层的线性变换
            nn.ReLU(),  # 激活函数ReLU
            nn.Linear(hidden_size, output_size)  # 隐藏层到输出层的线性变换
        )

    def forward(self, x):
        return self.net(x)  # 前向传播

    def save_model(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):  # 如果模型文件夹不存在，则创建
            os.mkdir(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)  # 构建模型文件路径
        torch.save(self.state_dict(), file_name)  # 保存模型参数到文件

# 定义一个Q学习训练器
class QTrainer:
    def __init__(self, model, lr, gama):
        self.model = model  # 模型
        self.lr = lr  # 学习率
        self.gama = gama  # 折扣因子
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)  # Adam优化器
        self.criterion = nn.MSELoss()  # 损失函数MSELoss

    def train_step(self, state, action, reward, next_state, is_done):
        state = torch.tensor(state, dtype=torch.float)  # 转换为张量并设置数据类型
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)

        if len(state.shape) == 1:  # 如果状态形状为一维，则扩展维度
            is_done = (is_done,)
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_state = torch.unsqueeze(next_state, 0)

        pred = self.model(state)  # 前向传播，获取预测值
        target = pred.clone()  # 复制预测值作为目标值
        for idx in range(len(is_done)):
            Q_new = reward[idx]  # 新的Q值等于奖励值
            if not is_done[idx]:  # 如果游戏未结束
                Q_new = Q_new + self.gama * torch.max(self.model(next_state[idx]))  # 更新Q值
            target[idx][torch.argmax(action[idx]).item()] = Q_new  # 更新目标值
        self.optimizer.zero_grad()  # 梯度清零
        loss = self.criterion(target, pred)  # 计算损失
        loss.backward()  # 反向传播
        self.optimizer.step()  # 更新模型参数
