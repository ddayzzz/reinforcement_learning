# -*- coding: utf-8 -*-
# Atair 游戏
import numpy as np
import gym

import cv2

from constants import GYM_ENV

from gym.wrappers import Monitor


class AtariEnvSkipping(gym.Wrapper):

    def __init__(self, env, frameskip=4):
        super(AtariEnvSkipping, self).__init__(env=env)
        # 设置 ALE 的参数
        self.env.unwrapped.ale.setInt(b'frame_skip', frameskip)  # 同一个动作, 执行连续四帧
        self.env.unwrapped.ale.setFloat(b'repeat_action_probability', 0.0)  # 不要自作主张连续 {2,3,4} 重复帧, 固定连续四帧
        self.env.seed()
        # 显示相关信息
        print("lives={}".format(self.env.unwrapped.ale.lives()))
        print("frameskip={}".format(self.env.unwrapped.ale.getInt(b'frame_skip')))
        print("repeat_action_probability={}".format(self.env.unwrapped.ale.getFloat(b'repeat_action_probability')))
        print("action space={}".format(self.env.action_space.n))

    def _step(self, a):
        """
        覆盖掉原来 Atair 类执行动作的函数, 动作 {0,1,...,5} => {x,...,y} 的映射动作
        :param a:
        :return:
        """
        reward = 0.0
        action = self.env.unwrapped._action_set[a]

        reward += self.env.unwrapped.ale.act(action)
        ob = self.env.unwrapped._get_obs()

        return ob, reward, self.env.unwrapped.ale.game_over(), {"ale.lives": self.env.unwrapped.ale.lives()}


class GameState(object):

    def __init__(self, display=False, crop_screen=True, frame_skip=4, no_op_max=30, output_video=False):
        """
        定义 agent
        :param display: 是否绘图
        :param crop_screen: 是否截取屏幕
        :param frame_skip: 跳过的帧数
        :param no_op_max: reset 时默认执行的动作
        :param output_video: 输出视频
        """
        self._display = display
        self._crop_screen = crop_screen
        self._frame_skip = frame_skip
        if self._frame_skip < 1:
            self._frame_skip = 1
        self._no_op_max = no_op_max
        self.env_id = GYM_ENV

        self.env = gym.make(GYM_ENV)
        # 引入监控器

        self.env = AtariEnvSkipping(self.env, frameskip=self._frame_skip)
        if output_video:
            self.env = Monitor(env=self.env, directory='./videos', video_callable=lambda episode_id: True)  # 每一次episode保存
        self.reset()

    def _process_frame(self, action, reshape):
        """
        与环境交互一次
        :param action: 采取的动作
        :param reshape: 是否裁剪图片
        :return:
        """
        reward = 0
        observation, r, terminal, _ = self.env.step(action)
        reward += r

        grayscale_observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)

        if self._crop_screen:
            resized_observation = grayscale_observation.astype(np.float32)
            # 去掉分数和边缘
            x_t = resized_observation[34:34 + 160, :160]
            x_t = cv2.resize(x_t, (84, 84))
        else:
            # 仅仅调整大小
            resized_observation = cv2.resize(grayscale_observation, (84, 84))
            x_t = resized_observation.astype(np.float32)

        if reshape:
            x_t = np.reshape(x_t, (84, 84, 1))

        # 图像转为浮点形式
        x_t *= (1.0 / 255.0)
        return reward, terminal, x_t

    def reset(self):
        self.env.reset()

        # 随机进行一定的动作
        if self._no_op_max > 0:
            no_op = np.random.randint(0, self._no_op_max + 1)
            for _ in range(no_op):
                self.env.step(0)

        _, _, x_t = self._process_frame(0, False)

        self.reward = 0
        self.terminal = False
        self.s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    def process(self, action):
        """
        与环境交互
        :param action: 动作
        :return:
        """
        if self._display:
            self.env.render()

        r, t, x_t1 = self._process_frame(action, True)

        self.reward = r
        self.terminal = t
        self.s_t1 = np.append(self.s_t[:, :, 1:], x_t1, axis=2)

    def update(self):
        """
        更新下一个状态
        :return:
        """
        self.s_t = self.s_t1
