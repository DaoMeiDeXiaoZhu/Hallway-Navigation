import time
import torch
import numpy as np
from .sac_simulation import *
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    robot = SACAgent()
    episodes = 5000

    # 开始主循环训练
    all_rewards = []
    for episode in range(1, episodes + 1):
        step = 0
        reset_flag = False
        robot.node.reset()
        robot.reset_agent()
        print(f'============================第{episode}回合训练开始!============================')
        
        # 开始当前回合循环
        episode_total_reward = 0
        while not reset_flag:
            # 获取当前状态
            robot.get_latest_data()
            state = robot.get_state()

            # 根据当前状态获取动作 (Actor 输出范围 [-1, 1])
            action = robot.select_action(state)

            # 放缩到固定区间
            linear_x, angular_z = action[0] * 0.5, action[1]

            # 执行动作持续时长
            robot.node.move(linear_x, angular_z)
            step += 1
            time.sleep(0.1)

            # 获取下一时间步状态
            robot.get_latest_data()
            next_state = robot.get_state()

            # 获取奖励
            reward = robot.get_reward()
            episode_total_reward += round(reward, 2)

            # 获取终止符done
            done, reset_flag = robot.get_done(step)

            # 存储经验
            robot.replay_buffer.add(state, action, reward, next_state, done)

            # 更新智能体
            robot.update_networks(batch_size=512)

            # 清理数据防止内存溢出
            del state, next_state, action, reward, done

        # 训练一定回合保存网络参数    
        if episode % 100 == 0:
            robot.save_checkpoint()
        
        # 记录每个回合的总奖励
        all_rewards.append(episode_total_reward)

        # 绘制回合总奖励曲线
        plot_rewards(all_rewards)

def plot_rewards(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    
    plt.title('rewards')
    plt.xlabel('Episodes')  # 或者 Steps，取决于你的数据含义
    plt.ylabel('Reward')
    plt.grid(True)          # 添加网格，方便观察
    
    # 保存图像到本地文件
    save_path = os.path.join('../checkpoint', 'rewards.png')
    plt.savefig(save_path)
    plt.close()             # 关闭图形，释放内存


if __name__ == '__main__':
    main()