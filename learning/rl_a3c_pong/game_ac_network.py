# -*- coding: utf-8 -*-
# 定义网络
import tensorflow as tf
import numpy as np


class GameACNetwork(object):

    def __init__(self,
                 action_size,
                 thread_index,  # -1 for global
                 device="/cpu:0"):
        """
        基本 AC 网络定义，定义了:
        1. Actor: states -> actions, 根据状态, 选择一个动作
        2. Actor-target: states -> actions, 根据 critic 的打分, 调整自身的策略(更新参数), 下次做的更好
        3. Critic: states, actions -> values, 根据 actor 的表现打分
        4. Critic-target: states, actions -> values, 根据环境的 reward 调整自己给分的策略
        LSTM 的作用主要在于实现类似于 DQN 中的 Experience Replay
        Critic 用 TD 进行训练
        :param action_size: 状态空间的大小
        :param thread_index: 线程编号，主要区别不同的 local 和 global AC 网络
        :param device: 会话运行的设备
        """
        self._action_size = action_size
        self._thread_index = thread_index
        self._device = device

    def prepare_loss(self, entropy_beta):
        """
        定义损失函数
        :param entropy_beta:
        :return:
        """
        with tf.device(self._device):
            # 输出的动作的各个概率
            self.a = tf.placeholder("float", [None, self._action_size])

            # temporary difference (R-V) (input for policy)
            # 时间差分 （Reward-Value）， 作为策略 pi 的输入
            self.td = tf.placeholder("float", [None])

            # 通过截取策略小于0的输出的值，避免 NaN
            log_pi = tf.log(tf.clip_by_value(self.pi, 1e-20, 1.0))

            # 论文第4页， 在策略添加 熵 可以避免过早收敛到局部最优的情况。 公式为 H(pi(s;theta))=log(pi(s;theta)) * pi(s;theta)
            entropy = -tf.reduce_sum(self.pi * log_pi, reduction_indices=1)

            # 策略的损失，论文使用梯度上升算法，需要取反.
            # 计算的公式在论文第四页
            policy_loss = - tf.reduce_sum(tf.reduce_sum(tf.multiply(log_pi, self.a), reduction_indices=1) * self.td + entropy * entropy_beta)

            # 用于累积(多步)回报， 使用时间差分，即  R_t - V(s;theta_v)
            self.r = tf.placeholder("float", [None])

            # Critic 的学习率一般是 Actor 学习率的一半。 Value 的损失
            value_loss = 0.5 * tf.nn.l2_loss(self.r - self.v)

            # 将策略的损失和至函数的损失合并起来一起优化
            self.total_loss = policy_loss + value_loss

    def run_policy_and_value(self, sess, s_t):
        raise NotImplementedError()

    def run_policy(self, sess, s_t):
        raise NotImplementedError()

    def run_value(self, sess, s_t):
        raise NotImplementedError()

    def get_vars(self):
        raise NotImplementedError()

    def sync_from(self, src_netowrk, name=None):
        """
        对于 AC 网络如果需要同步，则必须指定啦去参数的 AC 网络
        :param src_netowrk: 来源的 AC 网络
        :param name: 变量所在的作用域，防止本网络对自己赋值
        :return: 返回赋值 tensorflow 操作对象
        """
        src_vars = src_netowrk.get_vars()
        dst_vars = self.get_vars()

        sync_ops = []

        with tf.device(self._device):
            with tf.name_scope(name, "GameACNetwork", []) as name:
                for (src_var, dst_var) in zip(src_vars, dst_vars):
                    sync_op = tf.assign(dst_var, src_var)
                    sync_ops.append(sync_op)

                return tf.group(*sync_ops, name=name)

    # 使用 ALE 游戏专用的构建神经网络的代码
    # https://github.com/muupan/async-rl/blob/master/a3c_ale.py
    def _fc_variable(self, weight_shape):
        input_channels = weight_shape[0]
        output_channels = weight_shape[1]
        d = 1.0 / np.sqrt(input_channels)
        bias_shape = [output_channels]
        weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
        bias = tf.Variable(tf.random_uniform(bias_shape, minval=-d, maxval=d))
        return weight, bias

    def _conv_variable(self, weight_shape):
        w = weight_shape[0]
        h = weight_shape[1]
        input_channels = weight_shape[2]
        output_channels = weight_shape[3]
        d = 1.0 / np.sqrt(input_channels * w * h)
        bias_shape = [output_channels]
        weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
        bias = tf.Variable(tf.random_uniform(bias_shape, minval=-d, maxval=d))
        return weight, bias

    def _conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="VALID")


class GameACLSTMNetwork(GameACNetwork):

    def __init__(self,
                 action_size,
                 thread_index,
                 device="/cpu:0"):
        GameACNetwork.__init__(self, action_size, thread_index, device)
        super(GameACLSTMNetwork, self).__init__(action_size=action_size,
                                                thread_index=thread_index,
                                                device=device)
        # 定义默认的网络名称，防止参数重名
        scope_name = "net_" + str(self._thread_index)
        with tf.device(self._device), tf.variable_scope(scope_name) as scope:
            # Actor 网络部分
            # CNN 的参数，对图像进行处理，[kernel, kernel,in_channel, out_channel]
            self.W_conv1, self.b_conv1 = self._conv_variable([8, 8, 4, 16])  # stride=4
            self.W_conv2, self.b_conv2 = self._conv_variable([4, 4, 16, 32])  # stride=2
            # 最后一层全链接的权重的偏移
            self.W_fc1, self.b_fc1 = self._fc_variable([2592, 256])

            # LSTM 的形式是一个元组（cell状态, h）。有 256 个隐藏层
            self.lstm = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)

            # 策略网络的参数。（需要获取 LSTM 的输入）
            self.W_fc2, self.b_fc2 = self._fc_variable([256, action_size])

            # 值函数输出层的参数
            self.W_fc3, self.b_fc3 = self._fc_variable([256, 1])

            # 从 环境中获取得到的游戏的当前状态，保存了 4 个帧
            self.s = tf.placeholder("float", [None, 84, 84, 4])

            h_conv1 = tf.nn.relu(self._conv2d(self.s, self.W_conv1, 4) + self.b_conv1)
            h_conv2 = tf.nn.relu(self._conv2d(h_conv1, self.W_conv2, 2) + self.b_conv2)

            h_conv2_flat = tf.reshape(h_conv2, [-1, 2592])
            h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, self.W_fc1) + self.b_fc1)


            h_fc1_reshaped = tf.reshape(h_fc1, [1, -1, 256])

            # 展开 LSTM 单元的 placeholder，按照时间步的大小（step_size）展开
            self.step_size = tf.placeholder(tf.float32, [1])

            self.initial_lstm_state0 = tf.placeholder(tf.float32, [1, 256])
            self.initial_lstm_state1 = tf.placeholder(tf.float32, [1, 256])
            # LSTM 元组的形式（cell状态，s）
            self.initial_lstm_state = tf.contrib.rnn.LSTMStateTuple(self.initial_lstm_state0,
                                                                    self.initial_lstm_state1)


            # 展开 LSTM, 在 local AC网络中定义了最大的时间状态数量， 也就是公式中的 T
            # 当一局对抗结束后，展开的所有的时间步到会小于 LOCAL_TIME_STEP
            # dynamic_rnn 可以做到一次调用，输出 step_size 个数的记忆的信息
            # 构建时候的参数 time_major = False， 输出的维度为 [batch_size, max_time, cell.output_size]
            # dynamic_rnn 扮演了类似与 call 函数的功能
            lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(cell=self.lstm,
                                                              inputs=h_fc1_reshaped,
                                                              initial_state=self.initial_lstm_state,
                                                              sequence_length=self.step_size,
                                                              time_major=False,
                                                              scope=scope)

            # LSTM 的输出 唯独 (1,5,256) 用于反向传播，（1,1,256） 用于前向传播
            lstm_outputs = tf.reshape(lstm_outputs, [-1, 256])
            # 策略更新部分，定义了测率输出
            self.pi = tf.nn.softmax(tf.matmul(lstm_outputs, self.W_fc2) + self.b_fc2)

            # 之函数输出
            v_ = tf.matmul(lstm_outputs, self.W_fc3) + self.b_fc3
            self.v = tf.reshape(v_, [-1])

            # LSTM 内置的参数
            scope.reuse_variables()
            self.W_lstm = tf.get_variable("basic_lstm_cell/kernel")
            self.b_lstm = tf.get_variable("basic_lstm_cell/bias")

            self.reset_state()

    def reset_state(self):
        """
        清空 LSTM
        :return:
        """
        self.lstm_state_out = tf.contrib.rnn.LSTMStateTuple(np.zeros([1, 256]),
                                                            np.zeros([1, 256]))

    def run_policy_and_value(self, sess, s_t):
        """
        验证的时候，进行前向传播。LSTM 输出的最大的时间步是 1
        :param sess: 会话
        :param s_t: t 时刻的状态
        :return: \pi(a|s_t), 当前状态的价值
        """
        pi_out, v_out, self.lstm_state_out = sess.run([self.pi, self.v, self.lstm_state],
                                                      feed_dict={self.s: [s_t],
                                                                 self.initial_lstm_state0: self.lstm_state_out[0],
                                                                 self.initial_lstm_state1: self.lstm_state_out[1],
                                                                 self.step_size: [1]})
        # 维度， pi_out: (1,3), v_out: (1)
        return (pi_out[0], v_out[0])

    def run_policy(self, sess, s_t):
        """
        用于验证，对于当前环境给出的状态，输出一个策略知道 agent 的行动。仅进行前向传播
        :param sess: 会话
        :param s_t: t 时刻的状态
        :return: 测率
        """
        pi_out, self.lstm_state_out = sess.run([self.pi, self.lstm_state],
                                               feed_dict={self.s: [s_t],
                                                          self.initial_lstm_state0: self.lstm_state_out[0],
                                                          self.initial_lstm_state1: self.lstm_state_out[1],
                                                          self.step_size: [1]})

        return pi_out[0]

    def run_value(self, sess, s_t):
        """
        获取价值。在 local AC 网络的最大的时间步的时候，用于自举（bootstrapping），当下一个时间序列开始的时候，
        值函数会被再次以相同的状态计算，所以不需要更新 LSTM 在当前状态的输出。当下一个时间序列。见论文 Algorithm 3
        Async advantaged actor-critic
        :param sess: 会话
        :param s_t: t 时刻的状态
        :return:
        """

        prev_lstm_state_out = self.lstm_state_out
        v_out, _ = sess.run([self.v, self.lstm_state],
                            feed_dict={self.s: [s_t],
                                       self.initial_lstm_state0: self.lstm_state_out[0],
                                       self.initial_lstm_state1: self.lstm_state_out[1],
                                       self.step_size: [1]})

        # 回滚到先前的输出的状态
        self.lstm_state_out = prev_lstm_state_out
        return v_out[0]

    def get_vars(self):
        """
        获取当前 AC 网络的参数，用于同步网络的参数
        :return:
        """
        return [self.W_conv1, self.b_conv1,
                self.W_conv2, self.b_conv2,
                self.W_fc1, self.b_fc1,
                self.W_lstm, self.b_lstm,
                self.W_fc2, self.b_fc2,
                self.W_fc3, self.b_fc3]
