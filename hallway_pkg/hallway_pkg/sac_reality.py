import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from ament_index_python import get_package_share_directory
from .nodes_reality import RobotEnvNode

LOG_STD_MAX = 2
LOG_STD_MIN = -5
target_pos = (7.3, 0)

# 初始化权重函数
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


# ----------------------------------------------------------------
# Actor 网络
# ----------------------------------------------------------------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, action_bound=1.0):
        super(Actor, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

        # 动作缩放处理
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

        # action = tanh(x) * scale + bias
        action = y_t * self.action_scale + self.action_bias

        # 计算对数概率 (Log Prob)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean


def get_state(node: RobotEnvNode):
    # 获取最新同步数据
    pos_x, pos_y, pos_z, roll, pitch, yaw = node.latest_pose
    linear_x, linear_y, angular_z = node.latest_twist

    # 归一化雷达数据
    normalize_scan_data = (node.latest_scan - node.range_min) / (node.range_max - node.range_min)
    
    # 计算相对于当前目标的距离
    diff_x = target_pos[0] - pos_x
    diff_y = target_pos[1] - pos_y
    norm_diff_x = diff_x / 8.3
    norm_diff_y = diff_y / 8.3

    # 角度偏差
    goal_angle = math.atan2(diff_y, diff_x)
    heading_error = goal_angle - yaw
    while heading_error > math.pi: heading_error -= 2 * math.pi
    while heading_error < -math.pi: heading_error += 2 * math.pi

    # 最终状态向量
    state = normalize_scan_data.tolist() + [norm_diff_x, norm_diff_y, heading_error, linear_x, angular_z]

    return state


def load_actor(actor, checkpoint_pth='actor_prama.pth'):
    share_dir = get_package_share_directory('real_env_navigation_pkg')
    load_path = os.path.join(share_dir, 'checkpoint_pth', checkpoint_pth)
    if os.path.exists(load_path):
        print(f"开始加载检查点...")
        checkpoint = torch.load(load_path)
        actor.load_state_dict(checkpoint['actor_state_dict'])
        print("成功加载检查点，开始导航!")
        return True
    else:
        print("检查点不存在，无法导航!")
        return False
