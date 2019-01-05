# 定义 pong 的环境
import gym
import numpy as np


class PongEnv():

    def __init__(self,
                 crop_image_width,
                 crop_image_height):
        self.crop_image_width = crop_image_width
        self.crop_image_height = crop_image_height
        self._env = gym.make("Pong-v0")
        #
        self.num_actions = self._env.action_space.n

    @staticmethod
    def process_img(I):
        I = I[35:195]
        I = I[::2, ::2, 0]
        I[I == 144] = 0
        I[I == 109] = 0
        I[I != 0] = 1
        return I.astype(np.float).ravel()

    def reset(self):
        """
        返回当前的 observation(处理后的)
        :return:
        """
        return self.process_img(self._env.reset())

    def step(self, action):
        """
        执行指定的行为
        :param action: 行为的标号
        :param render_observation:
        :return: 处理后的一帧 observation, reward, done
        """
        obs, rew, done, _ = self._env.step(action)
        return self.process_img(obs), rew, done

    def render(self):
        self._env.render()



