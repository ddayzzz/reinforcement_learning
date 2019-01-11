# -*- coding: utf-8 -*-
import sys
import numpy as np
import gym

import cv2

from constants import GYM_ENV

from gym.wrappers import Monitor


class AtariEnvSkipping(gym.Wrapper):

    def __init__(self, env, frameskip=4):
        super(AtariEnvSkipping, self).__init__(env=env)

        self.env.unwrapped.ale.setInt(b'frame_skip', frameskip)
        self.env.unwrapped.ale.setFloat(b'repeat_action_probability', 0.0)
        self.env.seed()

        print("lives={}".format(self.env.unwrapped.ale.lives()))
        print("frameskip={}".format(self.env.unwrapped.ale.getInt(b'frame_skip')))
        print("repeat_action_probability={}".format(self.env.unwrapped.ale.getFloat(b'repeat_action_probability')))
        print("action space={}".format(self.env.action_space.n))

    def _step(self, a):
        reward = 0.0
        action = self.env.unwrapped._action_set[a]

        reward += self.env.unwrapped.ale.act(action)
        ob = self.env.unwrapped._get_obs()

        return ob, reward, self.env.unwrapped.ale.game_over(), {"ale.lives": self.env.unwrapped.ale.lives()}


class GameState(object):

    def __init__(self, display=False, crop_screen=True, frame_skip=4, no_op_max=30, output_video=False):
        self._display = display
        self._crop_screen = crop_screen
        self._frame_skip = frame_skip
        if self._frame_skip < 1:
            self._frame_skip = 1
        self._no_op_max = no_op_max
        self.env_id = GYM_ENV

        self.env = gym.make(GYM_ENV)
        # 引入监控器
        if output_video:
            self.env = Monitor(env=self.env, directory='./videos', video_callable=lambda episode_id: True)  # 每一次episode保存
        self.env = AtariEnvSkipping(self.env, frameskip=self._frame_skip)

        self.reset()

    def _process_frame(self, action, reshape):
        reward = 0
        observation, r, terminal, _ = self.env.step(action)
        reward += r

        grayscale_observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)

        if self._crop_screen:
            resized_observation = grayscale_observation.astype(np.float32)
            # crop to fit 84x84
            x_t = resized_observation[34:34 + 160, :160]
            x_t = cv2.resize(x_t, (84, 84))
        else:
            # resize to height=84, width=84
            resized_observation = cv2.resize(grayscale_observation, (84, 84))
            x_t = resized_observation.astype(np.float32)

        if reshape:
            x_t = np.reshape(x_t, (84, 84, 1))

        # normalize
        x_t *= (1.0 / 255.0)
        return reward, terminal, x_t

    def reset(self):
        self.env.reset()

        # randomize initial state
        if self._no_op_max > 0:
            no_op = np.random.randint(0, self._no_op_max + 1)
            for _ in range(no_op):
                self.env.step(0)

        _, _, x_t = self._process_frame(0, False)

        self.reward = 0
        self.terminal = False
        self.s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    def process(self, action):
        if self._display:
            self.env.render()

        r, t, x_t1 = self._process_frame(action, True)

        self.reward = r
        self.terminal = t
        self.s_t1 = np.append(self.s_t[:, :, 1:], x_t1, axis=2)

    def update(self):
        self.s_t = self.s_t1
