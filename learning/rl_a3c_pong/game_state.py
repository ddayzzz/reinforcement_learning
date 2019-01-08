# -*- coding: utf-8 -*-
import sys
import numpy as np
import cv2
# gym version
import gym
from constants import ACTION_SIZE

class GameState(object):

    def __init__(self, seed, init_actions_max=7, display=False):
        """
        # 封装游戏环境
        # 游戏有6个动作空间，但是只有三个是主要的游戏实际的动作
        # https://ai.stackexchange.com/questions/2449/what-are-different-actions-in-action-space-of-environment-of-pong-v0-game-from
        :param seed: 种子
        :param init_actions_max: reset agent 重复的动作. 0 表示不执行重复的初始动作
        :param display: 是否绘图, 一般测试的时候开启
        """
        # OpenAI gym
        self._env = gym.make('Pong-v0')
        self._env.seed(seed=seed)
        # 是否显示游戏的图像
        self._display = display
        self._init_actions_max = init_actions_max
        # 定义 Pong 游戏的映射，把6个动作映射为一个{2，3，4}的3个动作
        # API 定义：Each action is repeatedly performed for a duration of kk frames, where kk is uniformly sampled from {2,3,4}.
        self.real_actions_space = np.array([2, 3, 4])
        # 首先清空游戏的信息
        self.reset()

    @staticmethod
    def _process_frame(frame, reshape):
        """
        处理一帧的图像，用于训练模型
        :param frame: RGB 的一帧图像
        :param reshape: 转换后的维度， 默认为原始的不含边界、分数的图像；为 True，是 84x84 的灰度图
        :return: 处理后图像
        """
        # x_t 是 RGB (210, 160, 3) -> 灰度
        x_t = np.dot(frame[...,:3], [0.299, 0.587, 0.144])
        # 灰度转换为， 去掉最后一个维度 (210, 160)
        reshaped_screen = np.reshape(x_t, (210, 160))

        # 裁剪图像， W,H = (84,110)
        resized_screen = cv2.resize(reshaped_screen, (84, 110))
        # 不要比分 边框信息
        x_t = resized_screen[18:102, :]
        # 是不是作为 NN 的输入
        if reshape:
            x_t = np.reshape(x_t, (84, 84, 1))
        x_t = x_t.astype(np.float32)
        # 标准化
        x_t *= (1.0 / 255.0)
        return x_t

    def reset(self):
        """
        重置当前的环境
        :return:
        """
        reset_x = self._env.reset()
        # agent 重复动作
        if self._init_actions_max > 0:
            no_op = np.random.randint(0, self._init_actions_max + 1)
            for _ in range(no_op):
                reset_x, _, _, _ = self._env.step(self.real_actions_space[0])  # 执行初始动作
        # 处理初始状态的图像帧
        x_t = self._process_frame(reset_x, reshape=False)

        self.reward = 0
        self.terminal = False
        # 一个状态，连续四帧
        self.s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    def process(self, action):
        """
        与环境交互，产生新的状态，回报和结束的相关信息
        :param action: 状态空间 [0,2]
        :return:
        """
        # 映射到 2，3，4
        real_action = self.real_actions_space[action]
        # 注意转换成映射后的行为
        x_t1, reward, done, info = self._env.step(real_action)
        x_t1 = self._process_frame(frame=x_t1, reshape=True)

        self.reward = reward
        self.terminal = done
        # 移除最远的一个帧，并加入新的帧
        self.s_t1 = np.append(self.s_t[:, :, 1:], x_t1, axis=2)
        # 显示帧
        if self._display:
            self._env.render()

    def update(self):
        """
        更新状态
        :return:
        """
        self.s_t = self.s_t1
