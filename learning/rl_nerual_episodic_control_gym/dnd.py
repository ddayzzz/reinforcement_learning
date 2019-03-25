# DND 的实现
import torch
import torch.optim as optim
from torch.nn import Parameter
from pyflann import FLANN  # 最邻近的库


class DND:

    def __init__(self, kernel, num_neighbors, max_memory, lr):
        """
        定义 DND 的结构
        :param kernel:
        :param num_neighbors:
        :param max_memory:
        :param lr:
        """
        self.kernel = kernel
        self.num_neighbors = num_neighbors
        self.max_memory = max_memory
        self.lr = lr
        self.keys = None
        self.values = None
        self.kdtree = FLANN()

        # key_cache stores a cache of all keys that exist in the DND
        # This makes DND updates efficient
        # 这个应该是存储所有存在于 DND 中的  keys 的集合
        self.key_cache = {}
        # stale_index is a flag that indicates whether or not the index in self.kdtree is stale
        # This allows us to only rebuild the kdtree index when necessary
        # 这个是标志 KD 树是否已经需要退化操作
        self.stale_index = True
        # indexes_to_be_updated is the set of indexes to be updated on a call to update_params
        # This allows us to rebuild only the keys of key_cache that need to be rebuilt when necessary
        # 用于在仅仅需要重新建立树的情况下被更新的索引记录
        self.indexes_to_be_updated = set()

        # Keys and value to be inserted into self.keys and self.values when commit_insert is called
        # 当 commit_insert 调用的时候, 用于整体更新 keys 和 相关的 values
        self.keys_to_be_inserted = None
        self.values_to_be_inserted = None

        # Move recently used lookup indexes
        # These should be moved to the back of self.keys and self.values to get LRU property
        # LRU 置换算法的要被移到尾后的 keys 和 values
        self.move_to_back = set()

    def get_index(self, key):
        """
        If key exists in the DND, return its index
        Otherwise, return None
        """
        if self.key_cache.get(tuple(key.data.cpu().numpy()[0])) is not None:
            if self.stale_index:
                self.commit_insert()
            return int(self.kdtree.nn_index(key.data.cpu().numpy(), 1)[0][0])
        else:
            return None

    def update(self, value, index):
        """
        Set self.values[index] = value
        """
        # print("update() called")
        values = self.values.data
        values[index] = value[0].data
        self.values = Parameter(values)
        self.optimizer = optim.RMSprop([self.keys, self.values], lr=self.lr)

    def insert(self, key, value):

        """
        Insert key, value pair into DND
        """
        # print("insert(value={0}) called".format(value))
        if self.keys_to_be_inserted is None:
            # Initial insert
            self.keys_to_be_inserted = key.data
            self.values_to_be_inserted = value.data
        else:
            # 将需要插入的 key-value 对一起插入
            # 对应的 keys 和 value 要进行连接

            self.keys_to_be_inserted = torch.cat(
                [self.keys_to_be_inserted, key.data], 0)
            self.values_to_be_inserted = torch.cat(
                [self.values_to_be_inserted, value.data], 0)
        self.key_cache[tuple(key.data.cpu().numpy()[0])] = 0
        self.stale_index = True

    def commit_insert(self):
        # print('commit_insert() called')
        if self.keys is None:
            self.keys = Parameter(self.keys_to_be_inserted)
            self.values = Parameter(self.values_to_be_inserted)
        elif self.keys_to_be_inserted is not None:
            # print('keys_to_be_inserted:', self.keys_to_be_inserted.size())
            # print('keys:', self.keys.size())
            # print('values_to_be_inserted:', self.values_to_be_inserted.size())
            # print('values:', self.values.size())
            self.keys = Parameter(
                torch.cat([self.keys.data, self.keys_to_be_inserted], 0))
            self.values = Parameter(
                torch.cat([self.values.data, self.values_to_be_inserted], 0))

        # Move most recently used key-value pairs to the back
        if len(self.move_to_back) != 0:
            # print(self.keys.size())
            a = self.keys.data[list(set(range(len(
                self.keys))) - self.move_to_back)]
            b = self.keys.data[list(self.move_to_back)]
            self.keys = Parameter(torch.cat([a, b], 0))
            self.values = Parameter(torch.cat([self.values.data[list(set(range(len(self.values))) - self.move_to_back)], self.values.data[list(self.move_to_back)]], 0))
            self.move_to_back = set()

        if len(self.keys) > self.max_memory:
            # Expel oldest key to maintain total memory
            for key in self.keys[:-self.max_memory]:
                del self.key_cache[tuple(key.data.cpu().numpy())]
            self.keys = Parameter(self.keys[-self.max_memory:].data)
            self.values = Parameter(self.values[-self.max_memory:].data)
        self.keys_to_be_inserted = None
        self.values_to_be_inserted = None
        self.optimizer = optim.RMSprop([self.keys, self.values], lr=self.lr)
        self.kdtree.build_index(self.keys.data.cpu().numpy())
        self.stale_index = False

    def lookup(self, lookup_key, update_flag=False):
        """
        Perform DND lookup
        If update_flag == True, add the nearest neighbor indexes to self.indexes_to_be_updated
        """
        # print("lookup() called")
        lookup_indexes = self.kdtree.nn_index(
            lookup_key.data.cpu().numpy(), min(self.num_neighbors, len(self.keys)))[0][0]
        output = 0
        kernel_sum = 0
        for i, index in enumerate(lookup_indexes):
            if i == 0 and self.key_cache.get(tuple(lookup_key[0].data.cpu().numpy())) is not None:
                # If a key exactly equal to lookup_key is used in the DND lookup calculation
                # then the loss becomes non-differentiable. Just skip this case to avoid the issue.
                continue
            if update_flag:
                self.indexes_to_be_updated.add(int(index))
            else:
                self.move_to_back.add(int(index))
            kernel_val = self.kernel(self.keys[int(index)], lookup_key[0])
            output += kernel_val * self.values[int(index)]
            kernel_sum += kernel_val
        output = output / kernel_sum
        return output

    def update_params(self):
        """
        Update self.keys and self.values via backprop
        Use self.indexes_to_be_updated to update self.key_cache accordingly and rebuild the index of self.kdtree
        """
        # print("update_params() called")
        for index in self.indexes_to_be_updated:
            del self.key_cache[tuple(self.keys[index].data.cpu().numpy())]
        self.optimizer.step()
        self.optimizer.zero_grad()
        for index in self.indexes_to_be_updated:
            self.key_cache[tuple(self.keys[index].data.cpu().numpy())] = 0
        self.indexes_to_be_updated = set()
        self.kdtree.build_index(self.keys.data.cpu().numpy())
        self.stale_index = False