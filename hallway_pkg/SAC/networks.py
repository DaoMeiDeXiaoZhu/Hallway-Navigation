import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

# 定义常量，与参考代码保持一致
LOG_STD_MAX = 2
LOG_STD_MIN = -5

# 初始化权重函数 (保留你原本的用法，如果没有定义，PyTorch默认初始化也够用)
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

# ----------------------------------------------------------------
# Critic 网络 (保持 Twin Q 结构，但优化了初始化和结构)
# ----------------------------------------------------------------
class SoftQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(SoftQNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], dim=1)

        # Q1 calculation
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        # Q2 calculation
        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


# ----------------------------------------------------------------
# Actor 网络 (参考 CleanRL 进行了核心修改)
# ----------------------------------------------------------------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, action_bound=1.0):
        super(Actor, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

        # 【修改 1】动作缩放处理
        # 假设动作范围是 [-action_bound, action_bound]
        # 如果你的动作空间不对称，建议传入 env.action_space 来计算
        self.register_buffer(
            "action_scale",
            torch.tensor(action_bound, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(0.0, dtype=torch.float32)
        )

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)

        # 【修改 2】使用 tanh 平滑映射 log_std，而不是硬截断 clamp
        # 这种方式梯度更平滑，来自 SpinningUp / CleanRL 的最佳实践
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

        return mean, log_std

    def sample(self, state):
        # 获取未经过 tanh 的均值和对数标准差
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # 创建正态分布
        normal = Normal(mean, std)

        # 重参数化采样 (Reparameterization Trick)
        x_t = normal.rsample()  # x_t 是无界的
        y_t = torch.tanh(x_t)   # y_t 在 [-1, 1]

        # 【修改 3】应用动作缩放
        # action = tanh(x) * scale + bias
        action = y_t * self.action_scale + self.action_bias

        # 计算对数概率 (Log Prob)
        log_prob = normal.log_prob(x_t)
        
        # 【修改 4】应用雅可比行列式修正 (Enforcing Action Bound)
        # 公式：log_prob -= log(scale * (1 - tanh^2(x)))
        # 这里的 1e-6 是为了数值稳定性
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        # 【修改 5】计算用于评估的确定性动作
        # 必须先 tanh 再缩放，否则测试时动作可能越界
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean