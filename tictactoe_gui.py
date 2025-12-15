#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
井字棋人机对战GUI界面
使用PyQt5实现
"""

import sys
import os
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, 
                             QVBoxLayout, QHBoxLayout, QLabel, QMessageBox, QFileDialog)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QIcon

# 导入训练好的智能体
from code1 import QLearningAgent, TicTacToeEnv, BOARD_LEN


class TicTacToeGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.env = TicTacToeEnv()
        self.agent = None
        self.human_first = True
        self.game_over = False
        self.init_ui()
        self.load_agent()
        
    def init_ui(self):
        """初始化UI界面"""
        self.setWindowTitle('井字棋 - 人机对战')
        self.setFixedSize(500, 600)
        
        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # 标题
        title = QLabel('井字棋 - 人机对战')
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title.setFont(title_font)
        main_layout.addWidget(title)
        
        # 状态标签
        self.status_label = QLabel('等待开始...')
        self.status_label.setAlignment(Qt.AlignCenter)
        status_font = QFont()
        status_font.setPointSize(12)
        self.status_label.setFont(status_font)
        main_layout.addWidget(self.status_label)
        
        # 棋盘布局
        board_layout = QVBoxLayout()
        self.buttons = []
        for i in range(BOARD_LEN):
            row_layout = QHBoxLayout()
            row_buttons = []
            for j in range(BOARD_LEN):
                btn = QPushButton('')
                btn.setFixedSize(120, 120)
                btn.setFont(QFont('Arial', 36, QFont.Bold))
                btn.clicked.connect(lambda checked, row=i, col=j: self.on_button_click(row, col))
                row_layout.addWidget(btn)
                row_buttons.append(btn)
            board_layout.addLayout(row_layout)
            self.buttons.append(row_buttons)
        
        # 棋盘容器
        board_widget = QWidget()
        board_widget.setLayout(board_layout)
        board_widget.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 2px solid #333;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
            QPushButton:pressed {
                background-color: #d0d0d0;
            }
        """)
        main_layout.addWidget(board_widget)
        
        # 控制按钮布局
        control_layout = QHBoxLayout()
        
        # 新游戏按钮
        self.new_game_btn = QPushButton('新游戏')
        self.new_game_btn.setFixedHeight(40)
        self.new_game_btn.clicked.connect(self.new_game)
        control_layout.addWidget(self.new_game_btn)
        
        # 切换先手按钮
        self.switch_first_btn = QPushButton('切换先手')
        self.switch_first_btn.setFixedHeight(40)
        self.switch_first_btn.clicked.connect(self.switch_first)
        control_layout.addWidget(self.switch_first_btn)
        
        # 加载模型按钮
        self.load_model_btn = QPushButton('加载模型')
        self.load_model_btn.setFixedHeight(40)
        self.load_model_btn.clicked.connect(self.load_agent_from_file)
        control_layout.addWidget(self.load_model_btn)
        
        main_layout.addLayout(control_layout)
        
        # 底部信息
        info_label = QLabel('提示：点击棋盘下棋，X为先手，O为后手')
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("color: #666;")
        main_layout.addWidget(info_label)
        
    def load_agent(self):
        """加载智能体模型"""
        model_path = "tictactoe_model.pkl"
        if os.path.exists(model_path):
            try:
                self.agent = QLearningAgent()
                self.agent.load_model(model_path)
                self.agent.set_epsilon(0)  # 测试模式，不探索
                self.status_label.setText('模型加载成功！点击"新游戏"开始')
                self.status_label.setStyleSheet("color: green;")
            except Exception as e:
                self.status_label.setText(f'模型加载失败: {str(e)}')
                self.status_label.setStyleSheet("color: red;")
                self.agent = None
        else:
            self.status_label.setText('未找到模型文件，请先训练模型')
            self.status_label.setStyleSheet("color: orange;")
            self.agent = None
    
    def load_agent_from_file(self):
        """从文件选择对话框加载模型"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, '选择模型文件', '', 'Pickle Files (*.pkl);;All Files (*)'
        )
        if filepath:
            try:
                self.agent = QLearningAgent()
                self.agent.load_model(filepath)
                self.agent.set_epsilon(0)
                self.status_label.setText('模型加载成功！')
                self.status_label.setStyleSheet("color: green;")
                QMessageBox.information(self, '成功', '模型加载成功！')
            except Exception as e:
                QMessageBox.critical(self, '错误', f'模型加载失败:\n{str(e)}')
                self.agent = None
    
    def new_game(self):
        """开始新游戏"""
        if self.agent is None:
            QMessageBox.warning(self, '警告', '请先加载模型文件！')
            return
        
        self.env.reset()
        self.game_over = False
        self.update_board()
        
        if not self.human_first:
            # 如果智能体先手，让它先下
            self.agent_move()
        else:
            self.status_label.setText('你的回合（X）')
            self.status_label.setStyleSheet("color: blue;")
    
    def switch_first(self):
        """切换先手"""
        if not self.game_over:
            reply = QMessageBox.question(
                self, '确认', '切换先手将重新开始游戏，是否继续？',
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
        
        self.human_first = not self.human_first
        self.new_game()
    
    def update_board(self):
        """更新棋盘显示"""
        state = self.env.getState()
        for i in range(BOARD_LEN):
            for j in range(BOARD_LEN):
                btn = self.buttons[i][j]
                if state[i, j] == 1:
                    btn.setText('X')
                    btn.setStyleSheet("""
                        QPushButton {
                            background-color: #e3f2fd;
                            border: 2px solid #1976d2;
                            border-radius: 5px;
                            color: #1976d2;
                        }
                    """)
                elif state[i, j] == -1:
                    btn.setText('O')
                    btn.setStyleSheet("""
                        QPushButton {
                            background-color: #fff3e0;
                            border: 2px solid #f57c00;
                            border-radius: 5px;
                            color: #f57c00;
                        }
                    """)
                else:
                    btn.setText('')
                    btn.setStyleSheet("""
                        QPushButton {
                            background-color: #f0f0f0;
                            border: 2px solid #333;
                            border-radius: 5px;
                        }
                        QPushButton:hover {
                            background-color: #e0e0e0;
                        }
                    """)
    
    def on_button_click(self, row, col):
        """处理按钮点击事件"""
        if self.agent is None:
            QMessageBox.warning(self, '警告', '请先加载模型文件！')
            return
        
        if self.game_over:
            return
        
        # 检查是否是人类回合
        current_player = self.env.getCurrentPlayer()
        human_symbol = 1 if self.human_first else -1
        
        if current_player != human_symbol:
            QMessageBox.information(self, '提示', '现在不是你的回合！')
            return
        
        # 检查位置是否为空
        state = self.env.getState()
        if state[row, col] != 0:
            QMessageBox.information(self, '提示', '该位置已被占用！')
            return
        
        # 人类下棋
        action = [row, col]
        next_state, reward, terminal = self.env.step(action)
        self.update_board()
        
        if terminal:
            self.game_over = True
            winner = self.env.getWinner()
            if winner == human_symbol:
                self.status_label.setText('🎉 恭喜！你赢了！')
                self.status_label.setStyleSheet("color: green; font-weight: bold;")
                QMessageBox.information(self, '游戏结束', '🎉 恭喜！你赢了！')
            elif winner == -human_symbol:
                self.status_label.setText('😔 智能体赢了！')
                self.status_label.setStyleSheet("color: red; font-weight: bold;")
                QMessageBox.information(self, '游戏结束', '😔 智能体赢了！')
            else:
                self.status_label.setText('🤝 平局！')
                self.status_label.setStyleSheet("color: orange; font-weight: bold;")
                QMessageBox.information(self, '游戏结束', '🤝 平局！')
        else:
            # 智能体回合
            self.status_label.setText('智能体思考中...')
            self.status_label.setStyleSheet("color: purple;")
            QApplication.processEvents()  # 更新UI
            
            # 延迟一下，让用户看到状态变化
            import time
            time.sleep(0.3)
            
            self.agent_move()
    
    def agent_move(self):
        """智能体下棋"""
        if self.game_over:
            return
        
        state = self.env.getState()
        current_player = self.env.getCurrentPlayer()
        agent_symbol = -1 if self.human_first else 1
        
        if current_player != agent_symbol:
            return
        
        # 智能体选择动作
        action = self.agent.policy(state, training=False, current_player=current_player)
        
        if action is None:
            QMessageBox.warning(self, '错误', '智能体无法选择动作')
            return
        
        # 执行动作
        next_state, reward, terminal = self.env.step(action)
        self.update_board()
        
        if terminal:
            self.game_over = True
            winner = self.env.getWinner()
            if winner == agent_symbol:
                self.status_label.setText('😔 智能体赢了！')
                self.status_label.setStyleSheet("color: red; font-weight: bold;")
                QMessageBox.information(self, '游戏结束', '😔 智能体赢了！')
            elif winner == -agent_symbol:
                self.status_label.setText('🎉 恭喜！你赢了！')
                self.status_label.setStyleSheet("color: green; font-weight: bold;")
                QMessageBox.information(self, '游戏结束', '🎉 恭喜！你赢了！')
            else:
                self.status_label.setText('🤝 平局！')
                self.status_label.setStyleSheet("color: orange; font-weight: bold;")
                QMessageBox.information(self, '游戏结束', '🤝 平局！')
        else:
            self.status_label.setText('你的回合（' + ('X' if self.human_first else 'O') + '）')
            self.status_label.setStyleSheet("color: blue;")


def main():
    app = QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyle('Fusion')
    
    window = TicTacToeGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

