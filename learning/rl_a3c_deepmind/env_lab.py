# 替换：RGB_INTERLACED->RGB_INTERLEAVED 根据：https://github.com/deepmind/lab/releases
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2

import deepmind_lab

class EnvLab(object):
    def __init__(self, width, height, fps, level):
        lab = deepmind_lab.Lab(level, [])
        # 定义了 Deepmind 的对象
        # 加载的游戏脚本为 level
        # observations 中的元素分别给环境中的observation 命名
        self.env = deepmind_lab.Lab(
            level, ["RGB_INTERLEAVED"],
            config = {
                "fps": str(fps),
                "width": str(width),
                "height": str(height)
            })

        self.env.reset()  # 在每个 epsido 结束后 调用会设置 is_running == False

        import pprint
        observation_spec = lab.observation_spec()  # 所有的 observations 的名称、类型、维度
        print("Observation spec:")
        pprint.pprint(observation_spec)
        self.action_spec = self.env.action_spec()  # 同理
        print("Action spec:")
        pprint.pprint(self.action_spec)

        self.indices = {a["name"]: i for i, a in enumerate(self.action_spec)}  # 变成所有 actions 的行为的索引
        self.mins = np.array([a["min"] for a in self.action_spec])  # action 的 最小值序列
        self.maxs = np.array([a["max"] for a in self.action_spec])  # action 的最大值序列
        self.num_actions = len(self.action_spec)  # 动作的数量
        print(self.num_actions)  # 7个动作

        self.action = None

    def NumActions(self):
        return 3 #self.num_actions*2。使用的动作的最大数量。

    def Reset(self):
        self.env.reset()

    def Act(self, action, frame_repeat):
        action = self.MapActions(action)
        return self.env.step(action, num_steps=frame_repeat)

    def IsRunning(self):
        return self.env.is_running()

    def Observation(self):
        obs = self.env.observations()
        img = obs["RGB_INTERLEAVED"]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def MapActions(self, action_raw):
        # 定义了所有的行为，构成 行为为 num_actions * num_actions 的矩阵
        self.action = np.zeros([self.num_actions])

        if (action_raw == 0):
            self.action[self.indices["LOOK_LEFT_RIGHT_PIXELS_PER_FRAME"]] = -25
        elif (action_raw == 1):
            self.action[self.indices["LOOK_LEFT_RIGHT_PIXELS_PER_FRAME"]] = 25

        """if (action_raw==2):
            self.action[self.indices["LOOK_DOWN_UP_PIXELS_PER_FRAME"]] = -25
        elif (action_raw==3):
            self.action[self.indices["LOOK_DOWN_UP_PIXELS_PER_FRAME"]] = 25

        if (action_raw==4):
            self.action[self.indices["STRAFE_LEFT_RIGHT"]] = -1
        elif (action_raw==5):
            self.action[self.indices["STRAFE_LEFT_RIGHT"]] = 1

        if (action_raw==6):
            self.action[self.indices["MOVE_BACK_FORWARD"]] = -1
        el"""
        if (action_raw == 2):  # 7
            self.action[self.indices["MOVE_BACK_FORWARD"]] = 1

        # all binary actions need reset
        """if (action_raw==8):
            self.action[self.indices["FIRE"]] = 0
        elif (action_raw==9):
            self.action[self.indices["FIRE"]] = 1

        if (action_raw==10):
            self.action[self.indices["JUMP"]] = 0
        elif (action_raw==11):
            self.action[self.indices["JUMP"]] = 1

        if (action_raw==12):
            self.action[self.indices["CROUCH"]] = 0
        elif (action_raw==13):
            self.action[self.indices["CROUCH"]] = 1"""

        return np.clip(self.action, self.mins, self.maxs).astype(np.intc)
