# 替换：RGB_INTERLACED->RGB_INTERLEAVED 根据：https://github.com/deepmind/lab/releases
from __future__ import division
from __future__ import print_function

import numpy as np
from game import Game, GameState, DinoAgent
from utilities import grab_screen


class DinoEnv(object):

    def __init__(self, game, game_agent, game_state):
        # 初始化
        self.game = game
        self.state = game_state
        self.agent = game_agent
        self.actions = ['nothing', 'up']
        self.num_actions = 2  # 能够采取的行动的数量
        self.num_features = 5  # 不知道是什么意思
        self.score = 0

    def reset(self):
        """
        重新开始一局游戏清空所有的统计信息
        :return:
        """
        self.score = 0
        self.game.stop()
        self.game.restart()

    def step(self, action):
        """
        采取一次行动
        :param action: 行动的编号 [0,num_actions)
        :return: state(这一帧的灰度图像), reward, done
        """
        score = self.game.get_score()
        reward = 0.1
        is_over = False  # game over
        if action == 1:
            self.agent.jump()
        image = self.agent.observe()
        if self.agent.is_crashed():
            reward = -1.0
            is_over = True
        return image, reward, is_over  # return the Experience tuple
