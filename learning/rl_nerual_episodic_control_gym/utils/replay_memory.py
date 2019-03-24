# DQN 的经验池
import random
from collections import namedtuple


# 定义元组必须要添加的属性
Transition = namedtuple('Transition', ('state', 'action', 'reward'))
ReplayMemoryUnit = namedtuple('MemoryUnity', ('state', 'action', 'Q_N'))

class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = ReplayMemoryUnit(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)