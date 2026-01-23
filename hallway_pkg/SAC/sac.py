import threading
import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
from networks import *
from reply_buffer import *
import rclpy
from nodes import *


class SACAgent:
    def __init__(self, state_dim=95, action_dim=2, actor_lr=3e-4, critic_lr=5e-4, alpha_lr=3e-4, gamma=0.99, tau=0.003, alpha=0.3, 
                 buffer_capacity=int(1e6), checkpoint_pth='checkpoint.pth'):
        
        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.checkpoint_pth = checkpoint_pth
        self.targets = [(7.3, 0)]
        self.target_index = 0
        self.last_distance = 0
        self.max_distance = (self.targets[0][0]**2 + self.targets[0][1]**2)

        # 初始化最新消息
        self.latest_scan = None
        self.latest_pose = None
        self.latest_twist = None

        # 初始化经验池
        self.replay_buffer = ReplayBuffer(
            capacity=buffer_capacity, 
            state_dim=self.state_dim,
            action_dim=action_dim
        )

        # 初始化 Critic (Twin Q)
        self.critic = SoftQNetwork(state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.critic_target = SoftQNetwork(state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # 初始化 Actor
        self.actor = Actor(state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        # 初始值使用传入的 alpha（标量），并把 log_alpha 作为可训练参数
        self.log_alpha = torch.nn.Parameter(torch.tensor(float(np.log(self.alpha)), device=self.device))
        # 优化器（学习率可以单独设置，这里用 1e-3，按需调整）
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        # 目标熵（常用经验值：-action_dim）
        self.target_entropy = -0.5 * float(self.action_dim)

        # 尝试加载检查点
        self.load_checkpoint()

        # 初始化ROS话题、服务、机器人位置
        rclpy.init(args=None)
        self.node = RobotEnvNode()
        spin_thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        spin_thread.start()
        self.node.wait_for_system_ready()  # 等待系统就绪
        self.node.reset()  # 初始化机器人位置

        self.laser_range_min = self.node.range_min
        self.laser_range_max = self.node.range_max
    
    def reset_agent(self):
        self.target_index = 0
        self.last_distance = math.sqrt(self.targets[0][0]**2 + self.targets[0][1]**2)

    # 保证数据同步
    def get_latest_data(self):
        self.latest_scan = np.array(self.node.latest_scan, copy=True)
        self.latest_pose = np.array(self.node.latest_pose, copy=True)
        self.latest_twist = np.array(self.node.latest_twist, copy=True)

    def get_state(self):
        # 获取最新同步数据
        pos_x, pos_y, pos_z, roll, pitch, yaw = self.latest_pose
        linear_x, linear_y, angular_z = self.latest_twist

        # 归一化雷达数据
        normalize_scan_data = (self.latest_scan - self.laser_range_min) / (self.node.range_max - self.laser_range_min)
        
        # 计算相对于当前目标的距离
        diff_x = self.targets[self.target_index][0] - pos_x 
        diff_y = self.targets[self.target_index][1] - pos_y
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

    def get_reward(self):
        if self.latest_pose is None or self.latest_twist is None:
            return 0.0

        # --- 1. 数据解包 ---
        pos_x, pos_y, _, roll, pitch, yaw = self.latest_pose
        target_x, target_y = self.targets[self.target_index]
        linear_x, linear_y, angular_z = self.latest_twist
        
        # --- 2. 基础计算 ---
        diff_x = target_x - pos_x
        diff_y = target_y - pos_y
        distance = math.sqrt(diff_x**2 + diff_y**2)

        # 初始化上一帧距离
        if not hasattr(self, 'last_distance'):
            self.last_distance = distance

        # --- 3. 朝向角度计算 (新增核心逻辑) ---
        # 计算目标相对于机器人的角度 (全局坐标系下)
        target_angle = math.atan2(diff_y, diff_x)
        
        # 计算朝向误差 (归一化到 -pi 到 pi)
        heading_error = target_angle - yaw
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi
        
        # --- 4. 雷达扇区分析 ---
        num_points = len(self.latest_scan)
        one_third = num_points // 3
        
        valid_scan = self.latest_scan.copy()
        
        # 分区 (右 -> 中 -> 左)
        right_sector = valid_scan[0 : one_third]
        front_sector = valid_scan[one_third : 2*one_third]
        left_sector  = valid_scan[2*one_third : ]
        
        dist_r = np.mean(right_sector)
        dist_f = np.mean(front_sector)
        dist_l = np.mean(left_sector)

        # 判定前方是否受阻
        front_blocked = dist_f < 0.8

        # ==================== 奖励项计算 ====================

        # --- A. 距离趋近奖励 (Distance Reward) ---
        dist_improve = self.last_distance - distance
        
        # 策略：被堵住时，大幅降低对"远离目标"的惩罚，允许其绕行
        if front_blocked and dist_improve < 0:
            r_distance = dist_improve * 2.0 
        else:
            # 正常情况下，权重稍微调低一点，避免完全主导
            r_distance = 15.0 * dist_improve 

        # --- B. 朝向奖励 (Heading Reward) - [新增] ---
        # 鼓励车头对准目标。heading_error 越小，奖励越高（负数越少）
        # 只有在非避障状态下（front_blocked 为 False），我们才强制要求对准目标
        # 如果前方堵死，我们允许它转头去寻找路径，暂时不惩罚朝向错误
        if not front_blocked:
            r_heading = -abs(heading_error)
        else:
            r_heading = 0.0 # 避障时，允许车头偏离目标

        # --- C. 寻隙引导奖励 (Gap Reward) ---
        r_gap = 0.0
        if front_blocked:
            if dist_l > dist_r:
                # 左边空，应该左转 (angular_z > 0)
                r_gap = 1.0 * angular_z if angular_z > 0 else -0.5 # 只有做对动作才给大分
            else:
                # 右边空，应该右转 (angular_z < 0)
                r_gap = -1.0 * angular_z if angular_z < 0 else -0.5

        # --- D. 速度与倒车惩罚 (Velocity Reward) - [修改] ---
        # 1. 基础前进奖励
        r_speed = 0.1 * linear_x 

        # --- E. 避障与生存 (Safety Reward) ---
        min_scan = np.min(self.latest_scan)
        r_near = 0.0
        safe_dist = 0.5 # 触发避障惩罚的距离
        if min_scan < safe_dist:
            # 这里的系数保持适中，目的是告诉它"不舒服"，而不是"立刻结束"
            r_near = -2.0 * (safe_dist - min_scan)

        # --- F. 碰撞惩罚 (Collision Penalty) - [数值优化] ---
        # 原来的 -50 太大了，建议降到 -20 左右，相当于损失了20步的正常行走奖励
        is_collision = (min_scan <= self.node.range_min + 0.05)
        r_collision = -20.0 if is_collision else 0.0

        # --- G. 成功奖励 (Success Reward) - [数值优化] ---
        # 原来的 100 太大，建议降到 30 左右
        r_success = 0.0
        if distance <= 0.2:
            print(f"Reached Target {self.target_index}!")
            if self.target_index < len(self.targets) - 1:
                self.target_index += 1
                r_success = 10.0 # 阶段性目标奖励不用太大
                
                # 更新距离，避免因为目标跳变导致下一帧 dist_improve 剧烈波动
                new_tx, new_ty = self.targets[self.target_index]
                self.last_distance = math.sqrt((new_tx - pos_x)**2 + (new_ty - pos_y)**2)
                # 重新计算本次的 distance 避免逻辑错误（虽然函数马上结束了）
                distance = self.last_distance 
            else:
                r_success = 30.0 # 最终目标

        # --- H. 步数惩罚 (Time Penalty) ---
        # 鼓励动作快一点
        r_step = -0.05

        # --- 汇总 ---
        reward = (r_distance + 
                  0.03 * r_heading +    # 修正倒车问题的关键
                  r_gap +        
                  r_speed + 
                  r_near + 
                  r_collision + 
                  r_success + 
                  r_step)
        
        self.last_distance = distance
        
        return reward
    

    def get_done(self, step):
        pos_x, pos_y, _, roll, pitch, yaw = self.latest_pose
        
        # 计算到"最终目标"的距离 (用于判断任务彻底完成)
        final_tx, final_ty = self.targets[-1]
        dist_to_final = math.sqrt((final_tx - pos_x)**2 + (final_ty - pos_y)**2)
        
        min_scan = np.min(self.latest_scan)

        done = 0.0
        reset_flag = False

        # 1. 撞墙判定
        # 建议与 get_reward 保持一致，或者稍微严一点
        if min_scan <= self.node.range_min + 0.1:
            reset_flag = True
            done = 1.0 # 撞墙是终止状态
            print("发生碰撞")

        # 2. 翻车判定 (稍微放宽一点，防止加速时误判)
        elif abs(roll) > 0.3 or abs(pitch) > 0.3:
            reset_flag = True
            done = 1.0 # 翻车是终止状态
            print("发生翻车")

        # 3. 成功判定 (到达最终目标)
        elif dist_to_final <= 0.2:
            reset_flag = True
            done = 1.0 # 成功是终止状态
            print("任务完成")

        # 4. 超时判定
        elif step >= 500:
            reset_flag = True
            done = 1.0 
            print("时间结束")
        
        return done, reset_flag

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if evaluate:
            _, _, mean = self.actor.sample(state)
            return mean.detach().cpu().numpy()[0]
        else:
            action, _, _ = self.actor.sample(state)
            return action.detach().cpu().numpy()[0]

    def update_networks(self, batch_size):
        # 检查经验池数据量
        if len(self.replay_buffer) < batch_size:
            return

        # 1. 采样
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(batch_size)

        # ==================== 更新 Critic ====================
        with torch.no_grad():
            # 下一个时刻动作与 log_pi
            next_state_action, next_state_log_pi, _ = self.actor.sample(next_state_batch)

            # 目标 Q 值（使用 target critic）
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)

            # 当前可训练的 alpha（tensor）
            alpha = self.log_alpha.exp()

            # min Q - alpha * log_pi
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi

            # TD target
            next_q_value = reward_batch + (1.0 - done_batch) * self.gamma * min_qf_next_target

        # 当前 Q 值
        qf1, qf2 = self.critic(state_batch, action_batch)

        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        # print(f'critic网络梯度范数={get_grad_norm(self.critic)}')
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_optimizer.step()

        # ==================== 更新 Actor ====================
        pi, log_pi, _ = self.actor.sample(state_batch)

        # Actor 对应的 Q 值
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        # 使用当前 alpha（tensor）
        alpha = self.log_alpha.exp()

        # Actor loss: 最小化 (alpha * log_pi - Q)
        actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # print(f'actor网络梯度范数={get_grad_norm(self.actor)}')
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_optimizer.step()

        # ==================== 更新 alpha（温度参数） ====================
        log_pi_detached = log_pi.detach()
        alpha_loss = -(self.log_alpha * (log_pi_detached + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # 更新 self.alpha 的可读值（可选，用于日志或保存）
        self.alpha = self.log_alpha.exp().item()

        # ==================== 软更新目标网络 ====================
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        # 清理
        del qf1_loss, qf2_loss, qf_loss, actor_loss, alpha_loss
        del state_batch, action_batch, reward_batch, next_state_batch, done_batch
        torch.cuda.empty_cache()

    def save_checkpoint(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(script_dir, self.checkpoint_pth)
        
        # 新增 log_alpha 和 alpha_optimizer 的保存
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha,                         # 保存 alpha 的参数值
            'alpha_optimizer': self.alpha_optimizer.state_dict() # 保存 alpha 优化器的状态
        }, save_path)
        print(f"----------------- 检查点保存成功 -----------------")

    def load_checkpoint(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        load_path = os.path.join(script_dir, self.checkpoint_pth)
        if os.path.exists(load_path):
            print(f"开始加载检查点...")
            checkpoint = torch.load(load_path, map_location=self.device)
            
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

            # 加载 log_alpha 和 alpha_optimizer
            # 注意：必须使用 .data 或 copy_() 来更新值，保持 optimizer 对参数的引用有效
            if 'log_alpha' in checkpoint:
                self.log_alpha.data = checkpoint['log_alpha'].data
            
            if 'alpha_optimizer' in checkpoint:
                self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])

            print("成功加载检查点!")
        else:
            print("检查点不存在，使用随即权重初始化网络模型!")


def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)   # L2 范数
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm