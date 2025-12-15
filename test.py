#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import random
from collections import defaultdict


BOARD_LEN = 3


class TicTacToeEnv(object):
    def __init__(self):
        self.data = np.zeros((BOARD_LEN, BOARD_LEN))  # data 表示棋盘当前状态，1和-1分别表示x和o，0表示空位
        self.winner = None  # 1/0/-1表示玩家一胜/平局/玩家二胜，None表示未分出胜负
        self.terminal = False  # true表示游戏结束
        self.current_player = 1  # 当前正在下棋的人是玩家1还是-1

    def reset(self):
        # 游戏重新开始，返回状态
        self.data = np.zeros((BOARD_LEN, BOARD_LEN))
        self.winner = None
        self.terminal = False
        self.current_player = 1
        state = self.getState()
        return state

    def getState(self):
        # 注意到很多时候，存储数据不等同与状态，状态的定义可以有很多种，比如将棋的位置作一些哈希编码等
        # 这里直接返回data数据作为状态
        return self.data

    def getReward(self):
        """Return (reward_1, reward_2)
        """
        if self.terminal:
            if self.winner == 1:
                return 1, -1
            elif self.winner == -1:
                return -1, 1
        return 0, 0
    
    def checkThreat(self, player):
        """
        检查对手是否即将获胜（需要防守）
        Args:
            player: 要检查的玩家（1或-1）
        Returns:
            如果对手即将获胜，返回需要防守的位置，否则返回None
        """
        opponent = -player
        # 检查所有行
        for i in range(BOARD_LEN):
            row = self.data[i, :]
            if np.sum(row == opponent) == 2 and np.sum(row == 0) == 1:
                return [i, np.where(row == 0)[0][0]]
        
        # 检查所有列
        for j in range(BOARD_LEN):
            col = self.data[:, j]
            if np.sum(col == opponent) == 2 and np.sum(col == 0) == 1:
                return [np.where(col == 0)[0][0], j]
        
        # 检查主对角线
        diag1 = np.diag(self.data)
        if np.sum(diag1 == opponent) == 2 and np.sum(diag1 == 0) == 1:
            idx = np.where(diag1 == 0)[0][0]
            return [idx, idx]
        
        # 检查副对角线
        diag2 = np.array([self.data[i, BOARD_LEN-1-i] for i in range(BOARD_LEN)])
        if np.sum(diag2 == opponent) == 2 and np.sum(diag2 == 0) == 1:
            idx = np.where(diag2 == 0)[0][0]
            return [idx, BOARD_LEN-1-idx]
        
        return None
    
    def checkWinOpportunity(self, player):
        """
        检查是否有获胜机会
        Args:
            player: 要检查的玩家（1或-1）
        Returns:
            如果可以获胜，返回获胜位置，否则返回None
        """
        # 检查所有行
        for i in range(BOARD_LEN):
            row = self.data[i, :]
            if np.sum(row == player) == 2 and np.sum(row == 0) == 1:
                return [i, np.where(row == 0)[0][0]]
        
        # 检查所有列
        for j in range(BOARD_LEN):
            col = self.data[:, j]
            if np.sum(col == player) == 2 and np.sum(col == 0) == 1:
                return [np.where(col == 0)[0][0], j]
        
        # 检查主对角线
        diag1 = np.diag(self.data)
        if np.sum(diag1 == player) == 2 and np.sum(diag1 == 0) == 1:
            idx = np.where(diag1 == 0)[0][0]
            return [idx, idx]
        
        # 检查副对角线
        diag2 = np.array([self.data[i, BOARD_LEN-1-i] for i in range(BOARD_LEN)])
        if np.sum(diag2 == player) == 2 and np.sum(diag2 == 0) == 1:
            idx = np.where(diag2 == 0)[0][0]
            return [idx, BOARD_LEN-1-idx]
        
        return None

    def getCurrentPlayer(self):
        return self.current_player

    def getWinner(self):
        return self.winner

    def switchPlayer(self):
        if self.current_player == 1:
            self.current_player = -1
        else:
            self.current_player = 1

    def checkState(self):
        # 每次有人下棋，都要检查游戏是否结束
        # 从而更新self.terminal和self.winner
        # ----------------------------------
        # 实现自己的代码
        # ----------------------------------
        # 检查所有行
        for i in range(BOARD_LEN):
            if abs(self.data[i, :].sum()) == BOARD_LEN:
                self.terminal = True
                self.winner = int(self.data[i, 0])
                return
        
        # 检查所有列
        for j in range(BOARD_LEN):
            if abs(self.data[:, j].sum()) == BOARD_LEN:
                self.terminal = True
                self.winner = int(self.data[0, j])
                return
        
        # 检查主对角线
        diag1_sum = np.trace(self.data)
        if abs(diag1_sum) == BOARD_LEN:
            self.terminal = True
            self.winner = int(self.data[0, 0])
            return
        
        # 检查副对角线（从右上到左下）
        diag2_sum = sum(self.data[i, BOARD_LEN-1-i] for i in range(BOARD_LEN))
        if abs(diag2_sum) == BOARD_LEN:
            self.terminal = True
            self.winner = int(self.data[0, BOARD_LEN-1])
            return
        
        # 检查是否平局（棋盘已满且无人获胜）
        if np.all(self.data != 0):
            self.terminal = True
            self.winner = 0

    def step(self, action):
        """action: is a tuple or list [x, y]
        Return:
            state, reward, terminal
        """
        # ----------------------------------
        # 实现自己的代码
        # ----------------------------------
        x, y = action[0], action[1]
        
        # 检查动作是否合法（位置是否为空）
        if self.data[x, y] != 0:
            raise ValueError("Invalid action: position ({}, {}) is already occupied".format(x, y))
        
        # 保存当前玩家（在执行动作之前）
        acting_player = self.current_player
        
        # 执行动作：在当前玩家位置放置棋子
        self.data[x, y] = self.current_player
        
        # 检查游戏状态
        self.checkState()
        
        # 获取基础奖励
        reward = self.getReward()
        
        # 添加中间奖励（如果游戏未结束）
        if not self.terminal:
            # 检查是否阻止了对手获胜（防守）
            threat_pos = self.checkThreat(acting_player)
            if threat_pos is not None and threat_pos == [x, y]:
                # 成功防守，给予小奖励
                if acting_player == 1:
                    reward = (0.1, reward[1])
                else:
                    reward = (reward[0], 0.1)
            
            # 检查是否创造了获胜机会（进攻）
            win_pos = self.checkWinOpportunity(acting_player)
            if win_pos is not None and win_pos == [x, y]:
                # 创造了获胜机会，给予奖励
                if acting_player == 1:
                    reward = (0.2, reward[1])
                else:
                    reward = (reward[0], 0.2)
        
        # 切换玩家（如果游戏未结束）
        if not self.terminal:
            self.switchPlayer()
        
        # 返回状态、奖励和是否结束
        state = self.getState()
        # step返回的是单个reward值，这里返回执行动作的玩家的reward
        if acting_player == 1:
            current_reward = reward[0]
        else:
            current_reward = reward[1]
        
        return state, current_reward, self.terminal


class RandAgent(object):
    def policy(self, state):
        """
        Return: action
        """
        # ----------------------------------
        # 实现自己的代码
        # ----------------------------------
        # 找到所有空位置
        empty_positions = []
        for i in range(BOARD_LEN):
            for j in range(BOARD_LEN):
                if state[i, j] == 0:
                    empty_positions.append([i, j])
        
        # 如果还有空位，随机选择一个
        if len(empty_positions) > 0:
            idx = np.random.randint(len(empty_positions))
            return empty_positions[idx]
        else:
            # 棋盘已满，返回无效动作（这种情况不应该发生）
            return None


class QLearningAgent(object):
    """Q-Learning智能体"""
    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        """
        Args:
            learning_rate: 学习率
            discount_factor: 折扣因子（gamma）
            epsilon: 探索率（epsilon-greedy策略）
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: defaultdict(float))  # Q表：state -> action -> Q值
        
    def state_to_key(self, state):
        """将状态转换为字符串key（用于Q表）"""
        return str(state.tolist())
    
    def get_legal_actions(self, state):
        """获取所有合法动作"""
        legal_actions = []
        for i in range(BOARD_LEN):
            for j in range(BOARD_LEN):
                if state[i, j] == 0:
                    legal_actions.append([i, j])
        return legal_actions
    
    def policy(self, state, training=True, current_player=1):
        """
        选择动作（epsilon-greedy策略，带防守/进攻优先级）
        Args:
            state: 当前状态
            training: 是否在训练模式（训练时使用epsilon-greedy，测试时使用greedy）
            current_player: 当前玩家（1或-1）
        Return: action
        """
        legal_actions = self.get_legal_actions(state)
        if len(legal_actions) == 0:
            return None
        
        # 创建临时环境来检查威胁和获胜机会
        temp_env = TicTacToeEnv()
        temp_env.data = state.copy()
        temp_env.current_player = current_player
        
        # 优先级1：检查是否有获胜机会（必须下）
        win_pos = temp_env.checkWinOpportunity(current_player)
        if win_pos is not None and win_pos in legal_actions:
            return win_pos
        
        # 优先级2：检查是否需要防守（必须堵）
        threat_pos = temp_env.checkThreat(current_player)
        if threat_pos is not None and threat_pos in legal_actions:
            return threat_pos
        
        state_key = self.state_to_key(state)
        
        # epsilon-greedy策略
        if training and random.random() < self.epsilon:
            # 探索：随机选择
            return random.choice(legal_actions)
        else:
            # 利用：选择Q值最大的动作
            q_values = [self.q_table[state_key][tuple(action)] for action in legal_actions]
            max_q = max(q_values)
            # 如果有多个动作具有相同的最大Q值，随机选择一个
            best_actions = [action for action, q in zip(legal_actions, q_values) if q == max_q]
            return random.choice(best_actions)
    
    def update(self, state, action, reward, next_state, terminal):
        """
        更新Q值
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            terminal: 是否结束
        """
        state_key = self.state_to_key(state)
        action_key = tuple(action)
        
        current_q = self.q_table[state_key][action_key]
        
        if terminal:
            # 如果游戏结束，Q值就是奖励
            target_q = reward
        else:
            # 否则，使用Bellman方程更新
            next_state_key = self.state_to_key(next_state)
            next_legal_actions = self.get_legal_actions(next_state)
            if len(next_legal_actions) > 0:
                next_q_values = [self.q_table[next_state_key][tuple(a)] for a in next_legal_actions]
                max_next_q = max(next_q_values)
            else:
                max_next_q = 0
            target_q = reward + self.discount_factor * max_next_q
        
        # Q-learning更新公式
        self.q_table[state_key][action_key] = current_q + self.learning_rate * (target_q - current_q)
    
    def set_epsilon(self, epsilon):
        """设置探索率"""
        self.epsilon = epsilon
    
    def save_model(self, filepath):
        """
        保存Q表到文件
        Args:
            filepath: 保存路径
        """
        import json
        import pickle
        
        # 将defaultdict转换为普通dict以便序列化
        q_table_dict = {}
        for state_key, actions in self.q_table.items():
            q_table_dict[state_key] = dict(actions)
        
        model_data = {
            'q_table': q_table_dict,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath):
        """
        从文件加载Q表
        Args:
            filepath: 模型文件路径
        """
        import pickle
        from collections import defaultdict
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # 恢复defaultdict结构
        self.q_table = defaultdict(lambda: defaultdict(float))
        for state_key, actions in model_data['q_table'].items():
            self.q_table[state_key] = defaultdict(float, actions)
        
        self.learning_rate = model_data.get('learning_rate', 0.1)
        self.discount_factor = model_data.get('discount_factor', 0.9)
        self.epsilon = model_data.get('epsilon', 0.0)
        
        print(f"模型已从 {filepath} 加载，Q表大小: {len(self.q_table)} 个状态")


def train_qlearning(episodes=50000, opponent='random'):
    """
    训练Q-Learning智能体（同时训练先手和后手）
    Args:
        episodes: 训练轮数
        opponent: 对手类型 ('random' 或 'qlearning')
    """
    env = TicTacToeEnv()
    agent1 = QLearningAgent(learning_rate=0.2, discount_factor=1.0, epsilon=0.3)
    
    if opponent == 'random':
        agent2 = RandAgent()
    else:
        agent2 = QLearningAgent(learning_rate=0.2, discount_factor=1.0, epsilon=0.3)
    
    wins = 0
    losses = 0
    draws = 0
    
    print("开始训练...")
    print(f"训练参数: 学习率=0.2, 折扣因子=1.0, 初始探索率=0.3")
    print(f"训练模式: 智能体将同时学习先手和后手策略")
    
    for episode in range(episodes):
        state = env.reset()
        # 随机决定智能体是先手还是后手（50%概率）
        agent_is_first = random.random() < 0.5
        
        # 记录完整的游戏历史：(player, state, action, next_state, reward)
        game_history = []
        
        # 探索率衰减：从0.3线性衰减到0.01
        epsilon_decay = 0.3 - (0.3 - 0.01) * (episode / episodes)
        agent1.set_epsilon(epsilon_decay)
        if opponent == 'qlearning':
            agent2.set_epsilon(epsilon_decay)
        
        while True:
            current_player = env.getCurrentPlayer()
            current_state = state.copy()
            
            # 判断当前是智能体还是对手的回合
            is_agent_turn = (current_player == 1 and agent_is_first) or (current_player == -1 and not agent_is_first)
            
            if is_agent_turn:
                action = agent1.policy(state, training=True, current_player=current_player)
            else:
                if opponent == 'random':
                    action = agent2.policy(state)
                else:
                    action = agent2.policy(state, training=True, current_player=current_player)
            
            next_state, reward, terminal = env.step(action)
            game_history.append((current_player, current_state, action, next_state.copy(), reward, terminal))
            
            if terminal:
                # 游戏结束，获取最终奖励
                final_reward = env.getReward()
                
                # 找到智能体的所有步骤（可能是先手或后手）
                agent_steps = []
                for i, h in enumerate(game_history):
                    player, s, a, ns, r, t = h
                    is_agent_step = (player == 1 and agent_is_first) or (player == -1 and not agent_is_first)
                    if is_agent_step:
                        agent_steps.append((i, h))
                
                # 更新agent1的Q值（从后往前更新）
                for idx in range(len(agent_steps) - 1, -1, -1):
                    step_idx, (player, s, a, ns, r, t) = agent_steps[idx]
                    
                    if idx == len(agent_steps) - 1:
                        # 最后一步：使用最终奖励（根据智能体是哪个玩家）
                        if agent_is_first:
                            final_r = final_reward[0]
                        else:
                            final_r = final_reward[1]
                        agent1.update(s, a, final_r, ns, True)
                    else:
                        # 中间步骤：找到下一个智能体的状态（跳过对手的回合）
                        next_step_idx, (next_player, next_s, next_a, next_ns, next_r, next_t) = agent_steps[idx + 1]
                        # 使用这个状态的最大Q值来更新当前Q值
                        next_state_key = agent1.state_to_key(next_s)
                        next_legal_actions = agent1.get_legal_actions(next_s)
                        if len(next_legal_actions) > 0:
                            next_q_values = [agent1.q_table[next_state_key][tuple(a)] for a in next_legal_actions]
                            max_next_q = max(next_q_values) if next_q_values else 0
                        else:
                            max_next_q = 0
                        # 使用中间奖励r（可能包含防守/进攻奖励）
                        state_key = agent1.state_to_key(s)
                        action_key = tuple(a)
                        current_q = agent1.q_table[state_key][action_key]
                        target_q = r + agent1.discount_factor * max_next_q
                        agent1.q_table[state_key][action_key] = current_q + agent1.learning_rate * (target_q - current_q)
                
                # 更新agent2的Q值（如果是对手也是Q-learning）
                if opponent == 'qlearning':
                    agent2_steps = [(i, h) for i, h in enumerate(game_history) if not ((h[0] == 1 and agent_is_first) or (h[0] == -1 and not agent_is_first))]
                    for idx in range(len(agent2_steps) - 1, -1, -1):
                        step_idx, (player, s, a, ns, r, t) = agent2_steps[idx]
                        
                        if idx == len(agent2_steps) - 1:
                            if agent_is_first:
                                final_r = final_reward[1]
                            else:
                                final_r = final_reward[0]
                            agent2.update(s, a, final_r, ns, True)
                        else:
                            next_step_idx, (next_player, next_s, next_a, next_ns, next_r, next_t) = agent2_steps[idx + 1]
                            next_state_key = agent2.state_to_key(next_s)
                            next_legal_actions = agent2.get_legal_actions(next_s)
                            if len(next_legal_actions) > 0:
                                next_q_values = [agent2.q_table[next_state_key][tuple(a)] for a in next_legal_actions]
                                max_next_q = max(next_q_values) if next_q_values else 0
                            else:
                                max_next_q = 0
                            state_key = agent2.state_to_key(s)
                            action_key = tuple(a)
                            current_q = agent2.q_table[state_key][action_key]
                            target_q = r + agent2.discount_factor * max_next_q
                            agent2.q_table[state_key][action_key] = current_q + agent2.learning_rate * (target_q - current_q)
                
                # 统计结果（从智能体的视角）
                winner = env.getWinner()
                if (winner == 1 and agent_is_first) or (winner == -1 and not agent_is_first):
                    wins += 1
                elif (winner == -1 and agent_is_first) or (winner == 1 and not agent_is_first):
                    losses += 1
                else:
                    draws += 1
                
                break
            
            state = next_state
        
        # 每5000轮打印一次进度
        if (episode + 1) % 5000 == 0:
            win_rate = wins / (episode + 1) * 100
            current_epsilon = agent1.epsilon
            print(f"Episode {episode + 1}/{episodes} - 胜率: {win_rate:.2f}%, 负率: {losses/(episode+1)*100:.2f}%, 平局率: {draws/(episode+1)*100:.2f}%, 探索率: {current_epsilon:.3f}")
    
    print(f"\n训练完成！")
    print(f"总胜率: {wins/episodes*100:.2f}%")
    print(f"总负率: {losses/episodes*100:.2f}%")
    print(f"总平局率: {draws/episodes*100:.2f}%")
    print(f"Q表大小: {len(agent1.q_table)} 个状态")
    
    # 自动保存模型
    model_path = "tictactoe_model.pkl"
    agent1.save_model(model_path)
    print(f"训练好的模型已保存到: {model_path}")
    
    return agent1, agent2 if opponent == 'qlearning' else None


def test_agent(trained_agent, opponent='random', num_games=100):
    """
    测试训练好的智能体
    Args:
        trained_agent: 训练好的智能体
        opponent: 对手类型
        num_games: 测试游戏数量
    """
    env = TicTacToeEnv()
    if opponent == 'random':
        opponent_agent = RandAgent()
    else:
        opponent_agent = opponent
    
    trained_agent.set_epsilon(0)  # 测试时关闭探索
    
    wins = 0
    losses = 0
    draws = 0
    
    print(f"\n开始测试（{num_games}局）...")
    for game in range(num_games):
        state = env.reset()
        
        while True:
            current_player = env.getCurrentPlayer()
            
            if current_player == 1:
                action = trained_agent.policy(state, training=False, current_player=current_player)
            else:
                action = opponent_agent.policy(state)
            
            next_state, reward, terminal = env.step(action)
            
            if terminal:
                winner = env.getWinner()
                if winner == 1:
                    wins += 1
                elif winner == -1:
                    losses += 1
                else:
                    draws += 1
                break
            
            state = next_state
    
    print(f"测试结果:")
    print(f"  胜率: {wins/num_games*100:.2f}% ({wins}/{num_games})")
    print(f"  负率: {losses/num_games*100:.2f}% ({losses}/{num_games})")
    print(f"  平局率: {draws/num_games*100:.2f}% ({draws}/{num_games})")


def main():
    """主函数：训练、测试或随机演示"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        # 训练模式
        episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 50000
        opponent = sys.argv[3] if len(sys.argv) > 3 else 'random'
        trained_agent, _ = train_qlearning(episodes=episodes, opponent=opponent)
        
        # 训练后测试
        test_agent(trained_agent, opponent='random', num_games=100)
    
    else:
        # 默认：运行一次随机对局（原始功能）
        env = TicTacToeEnv()
        agent1 = RandAgent()
        agent2 = RandAgent()
        state = env.reset()

        print("随机对局演示：")
        while True:
            current_player = env.getCurrentPlayer()
            if current_player == 1:
                action = agent1.policy(state)
            else:
                action = agent2.policy(state)
            next_state, reward, terminal = env.step(action)
            print(f"\n玩家 {current_player} 下在位置 {action}")
            print(next_state)
            if terminal:
                winner_val = env.getWinner()
                if winner_val == 1:
                    winner = 'Player1'
                elif winner_val == -1:
                    winner = 'Player2'
                else:
                    winner = 'Draw'
                print(f'\nWinner: {winner}')
                break
            state = next_state


if __name__ == "__main__":
    main()
