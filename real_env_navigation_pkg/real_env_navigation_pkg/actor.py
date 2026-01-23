import math
import os
import threading
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

from ament_index_python import get_package_share_directory
from .node import RobotEnvNode
import rclpy


# ----------------------------------------------------------------
# Actor 网络
# ----------------------------------------------------------------
LOG_STD_MAX = 2
LOG_STD_MIN = -5
target_pos = (7.3, 0)

# 初始化权重函数
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


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


def main(args=None):
    rclpy.init(args=args)
    
    # 1. 初始化主节点
    node = RobotEnvNode()
    
    # 2. 开启后台线程处理回调
    # 注意：这里 daemon=True 意味着主线程一退，它就会立刻死掉，所以我们在 finally 里要小心
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # 3. 加载模型
    actor = Actor(state_dim=95, action_dim=2) 
    actor.eval()
    
    if not load_actor(actor, 'actor_prama.pth'):
        print("请检查模型路径！")
        return

    print("等待传感器初始化...")
    try:
        # 4. 等待传感器就绪
        while rclpy.ok():
            ready, msg = node.is_ready()
            if ready:
                print("传感器就绪，开始推理！")
                break
            else:
                print(f"等待中: {msg}")
                time.sleep(0.5)

        # 5. 推理循环
        print(">>> 开始推理循环...")
        rate = node.create_rate(10)
        while rclpy.ok():
            # --- 增加空值检查 ---
            if node.latest_pose is None or node.latest_scan is None:
                print("警告: 传感器数据丢失，跳过本次推理")
                rate.sleep()
                continue
            
            # A. 获取状态
            state = get_state(node)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            # B. 推理动作
            with torch.no_grad():
                _, _, action_mean = actor.sample(state_tensor)
                action = action_mean.cpu().numpy()[0]

            # C. 发送指令
            # print(f"Action: v={action[0]:.2f}, w={action[1]:.2f}") # 调试时可以打开
            node.move(action[0] * 0.5, action[1])
            
            # D. 维持频率
            rate.sleep()

    except KeyboardInterrupt:
        print("\n!!! 检测到键盘中断 (Ctrl+C) !!!")
    
    # === 新增：捕获其他所有报错，让你知道为什么挂了 ===
    except Exception as e:
        import traceback
        print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(">>> 程序发生运行错误 (Runtime Error) <<<")
        print(f"错误信息: {e}")
        print("详细堆栈:")
        traceback.print_exc()
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

    finally:
        print("--------------------------------")
        print(">>> 正在执行系统级强制停车...")

        # 【核武器】直接调用系统终端命令
        # 这相当于你在终端里手敲了一行命令，不依赖当前 Python 脚本的任何状态
        # 即使 Python 挂了，这行命令也会由操作系统独立执行
        
        cmd = "ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist \"{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}\""
        
        # 连发 3 次，确保停车
        for i in range(3):
            print(f"   [System Call] 发送停车指令 ({i+1}/3)...")
            os.system(cmd) 
            time.sleep(0.1)

        print(">>> 正在清理资源...")
        try:
            node.destroy_node()
        except:
            pass
            
        if rclpy.ok():
            rclpy.shutdown()
            
        # 强制杀掉所有后台线程，不再等待
        print("!!! 程序已退出 !!!")
        # 强制退出 Python 进程，防止 spin 线程卡死
        os._exit(0)

if __name__ == '__main__':
    main()
