# 替换：RGB_INTERLACED->RGB_INTERLEAVED 根据：https://github.com/deepmind/lab/releases
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
from game import Game, GameState, DinoAgent
from utilities import grab_screen


class EnvLab(object):
    def __init__(self, chrome_driver, show_image=False):
        game = Game(chrome_driver=chrome_driver)
        dino = DinoAgent(game, start_immediately=True)
        game_state = GameState(dino, game, show=show_image)
        # 初始化
        self.env = game_state
        self.agent = dino
        self.game = game
        self.num_actions = 2

    def NumActions(self):
        return 2  # 2 个动作。0 保持 1 跳起

    def Reset(self):
        self.game.restart()

    def Act(self, action):
        """
        执行相关的动作
        :param action: 行为的编号 {0, 1}
        :return: 这一帧执行一个动作的返回的图像(RGB)，回报，是否有结束的信息
        """
        action_one_hot = np.zeros([self.num_actions])
        action_one_hot[action] = 1
        return self.env.get_state(action_one_hot)

    def IsRunning(self):
        return self.agent.is_running()

    def GetGameImage(self):
        image = grab_screen(self.game._driver,
                            getbase64Script="canvasRunner = document.getElementById('runner-canvas');return canvasRunner.toDataURL().substring(22)")
        return image

    def Restart(self):
        self.game.restart()
