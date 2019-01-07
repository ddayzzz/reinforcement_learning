# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np



# should be implemented in a child class of this one
class ActorCriticNetwork(object):

    """
    Actor-Critic 基类, 需要实现策略网络和值网络。
    """

    def __init__(self,
                 action_size,
                 device="/cpu:0"):
        self._device = device
        self._action_size = action_size

    def prepare_loss(self, entropy_beta, scopes):
        # drop task id (last element) as all tasks in
        # the same scene share the same output branch
        scope_key = self._get_key(scopes[:-1])

        with tf.device(self._device):
            # taken action (input for policy)
            self.a = tf.placeholder("float", [None, self._action_size])

            # temporary difference (R-V) (input for policy)
            self.td = tf.placeholder("float", [None])

            # avoid NaN with clipping when value in pi becomes zero
            log_pi = tf.log(tf.clip_by_value(self.pi[scope_key], 1e-20, 1.0))

            # policy entropy
            self.entropy = -tf.reduce_sum(self.pi[scope_key] * log_pi, axis=1)

            # policy loss (output)
            policy_loss = - tf.reduce_sum(
                tf.reduce_sum(tf.multiply(log_pi, self.a), axis=1) * self.td + self.entropy * entropy_beta)

            # R (input for value)
            self.r = tf.placeholder("float", [None])

            # value loss (output)
            # learning rate for critic is half of actor's
            value_loss = 0.5 * tf.nn.l2_loss(self.r - self.v[scope_key])

            # gradienet of policy and value are summed up
            self.total_loss = policy_loss + value_loss

    def run_policy_and_value(self, sess, s_t, task):
        raise NotImplementedError()

    def run_policy(self, sess, s_t, task):
        raise NotImplementedError()

    def run_value(self, sess, s_t, task):
        raise NotImplementedError()

    def get_vars(self):
        raise NotImplementedError()

    def sync_from(self, src_netowrk, name=None):
        """
        从某个网络中同步参数到本网络
        :param src_netowrk: 源网络
        :param name: 指定 assign 操作的名称， 如果没有，操作命名为 ActorCriticNetwork
        :return: 返回一些列从赋值到本网络的 op
        """
        src_vars = src_netowrk.get_vars()
        dst_vars = self.get_vars()

        local_src_var_names = [self._local_var_name(x) for x in src_vars]
        local_dst_var_names = [self._local_var_name(x) for x in dst_vars]

        # keep only variables from both src and dst
        src_vars = [x for x in src_vars
                    if self._local_var_name(x) in local_dst_var_names]
        dst_vars = [x for x in dst_vars
                    if self._local_var_name(x) in local_src_var_names]

        sync_ops = []

        with tf.device(self._device):
            with tf.name_scope(name, "ActorCriticNetwork", []) as name:
                for (src_var, dst_var) in zip(src_vars, dst_vars):
                    sync_op = tf.assign(dst_var, src_var)
                    sync_ops.append(sync_op)

                return tf.group(*sync_ops, name=name)

    # variable (global/scene/task1/W_fc:0) --> scene/task1/W_fc:0
    def _local_var_name(self, var):
        return '/'.join(var.name.split('/')[1:])

    # 以下的代码是从 ALE 的经典初始 NN 的权重的代码
    # 来自 https://github.com/muupan/async-rl/blob/master/a3c_ale.py
    def _fc_weight_variable(self, shape, name='W_fc'):
        """
        用于全连接的权重网络
        :param shape: 维度信息 4-D
        :param name: 名称
        :return: tf 变量
        """
        input_channels = shape[0]
        d = 1.0 / np.sqrt(input_channels)
        initial = tf.random_uniform(shape, minval=-d, maxval=d)
        return tf.Variable(initial, name=name)

    def _fc_bias_variable(self, shape, input_channels, name='b_fc'):
        """
        全连接的偏移
        :param shape:
        :param input_channels:
        :param name:
        :return:
        """
        d = 1.0 / np.sqrt(input_channels)
        initial = tf.random_uniform(shape, minval=-d, maxval=d)
        return tf.Variable(initial, name=name)

    def _conv_weight_variable(self, shape, name='W_conv'):
        w = shape[0]
        h = shape[1]
        input_channels = shape[2]
        d = 1.0 / np.sqrt(input_channels * w * h)
        initial = tf.random_uniform(shape, minval=-d, maxval=d)
        return tf.Variable(initial, name=name)

    def _conv_bias_variable(self, shape, w, h, input_channels, name='b_conv'):
        d = 1.0 / np.sqrt(input_channels * w * h)
        initial = tf.random_uniform(shape, minval=-d, maxval=d)
        return tf.Variable(initial, name=name)

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

    def _get_key(self, scopes):
        """
        所以的 AC 网络的参数需要组合成一个完整的路径
        :param scopes:
        :return:
        """
        return '/'.join(scopes)



class ActorCriticLSTMNetwork(ActorCriticNetwork):

    """
    A3C LSTM 的实现。 LSTM 主要是实现类似于最小化 T 探索的
    """

    def __init__(self,
                 action_size,
                 device="/cpu:0",
                 network_scope="network",
                 scene_scopes=["scene"]):
        super(ActorCriticLSTMNetwork, self).__init__(action_size=action_size, device=device)

        # 定义策略和状态值函数
        self.pi = dict()
        self.v = dict()
        #
        self.W_convS1 = dict()
        self.b_convS1 = dict()

        self.W_convS2 = dict()
        self.b_convS2 = dict()

        self.W_convS3 = dict()
        self.b_convS3 = dict()

        self.W_fc1 = dict()
        self.b_fc1 = dict()

        self.W_fc2 = dict()
        self.b_fc2 = dict()

        self.W_fc3 = dict()
        self.b_fc3 = dict()

        self.W_policy = dict()
        self.b_policy = dict()

        self.W_value = dict()
        self.b_value = dict()

        with tf.device(self._device):
            # Pong 游戏的状态， 像素矩阵， RGB BHWC 格式 None x 210 x 160 x 3
            self.s = tf.placeholder(dtype="float", shape=[None, 210, 160, 3])

            with tf.variable_scope(network_scope):
                # 网络的id
                key = network_scope

                # 卷积网络的传播过程
                # Conv 1
                self.W_convS1[key], self.b_convS1[key] = self._conv_variable([8, 8, 3, 16])  # stride=4
                # self.W_convT1[key], self.b_convT1[key] = self._conv_variable([8, 8, 4, 16])  # stride=4

                self.S_conv1 = tf.nn.relu(self._conv2d(self.s, self.W_convS1[key], 4) + self.b_convS1[key])
                # self.T_conv1 = tf.nn.relu(self._conv2d(self.t,  self.W_convT1[key], 4) + self.b_convT1[key])

                # Conv 2
                self.W_convS2[key], self.b_convS2[key] = self._conv_variable([4, 4, 16, 32])  # stride=2
                # self.W_convT2[key], self.b_convT2[key] = self._conv_variable([4, 4, 16, 32])  # stride=2

                self.S_conv2 = tf.nn.relu(self._conv2d(self.S_conv1, self.W_convS2[key], 2) + self.b_convS2[key])
                # self.T_conv2 = tf.nn.relu(self._conv2d(self.T_conv1,  self.W_convT2[key], 2) + self.b_convT2[key])

                # Conv 3
                self.W_convS3[key], self.b_convS3[key] = self._conv_variable([4, 4, 32, 64])  # stride=2
                # self.W_convT3[key], self.b_convT3[key] = self._conv_variable([4, 4, 32, 64])  # stride=2

                self.S_conv3 = tf.nn.relu(self._conv2d(self.S_conv2, self.W_convS3[key], 2) + self.b_convS3[key])

                # flatten input
                self.s_flat = tf.reshape(self.S_conv3, [-1, 5632])

                # shared siamese layer
                self.W_fc1[key] = self._fc_weight_variable([5632, 512])
                self.b_fc1[key] = self._fc_bias_variable([512], 5632)

                h_s_flat = tf.nn.relu(tf.matmul(self.s_flat, self.W_fc1[key]) + self.b_fc1[key])

                h_s_reshaped = tf.reshape(h_s_flat, [1, -1, 512])

                #
                for scene_scope in scene_scopes:
                    # 指定当前网络的环境下， 进行的 off-policy 的回放操作
                    key = self._get_key([network_scope, scene_scope])

                    with tf.variable_scope(scene_scope):

                        # LSTM, 总共 512 个遗忘门， state_is_tuple 必须是 True， 返回的状态是元组，
                        # 在展开 tau（s_t,a_t,s_{t+1},a_{t+1},...） 的时候需要
                        self.lstm = tf.contrib.rnn.BasicLSTMCell(512, state_is_tuple=True)

                        # 时间推进的占位符， 用于 LSTM 的时间推进。这个需要记录时间
                        self.step_size = tf.placeholder(tf.float32, [1])
                        # LSTM 单元，表示推进的状态
                        self.initial_lstm_state0 = tf.placeholder(tf.float32, [1, 512])
                        self.initial_lstm_state1 = tf.placeholder(tf.float32, [1, 512])
                        self.initial_lstm_state = tf.contrib.rnn.LSTMStateTuple(self.initial_lstm_state0,
                                                                                self.initial_lstm_state1)

                        # Unrolling LSTM up to LOCAL_T_MAX time steps. (= 5time steps.)
                        # When episode terminates unrolling time steps becomes less than LOCAL_TIME_STEP.
                        # Unrolling step size is applied via self.step_size placeholder.
                        # When forward propagating, step_size is 1.
                        # (time_major = False, so output shape is [batch_size, max_time, cell.output_size])
                        # 展开 LSTM, 我们在 local AC网络中定义了最大的时间状态数量， 也就是公式中的 T
                        # 当一局对抗结束后，展开的所有的时间步到会小于 LOCAL_TIME_STEP
                        # dynamic_rnn 可以做到一次调用，输出 step_size 个数的记忆的信息
                        # 构建时候的参数 time_magor = False， 输出的维度为 [batch_size, max_time, cell.output_size]
                        lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(self.lstm,  # 指定的单元的格式
                                                                          h_s_reshaped,
                                                                          initial_state=self.initial_lstm_state,
                                                                          sequence_length=self.step_size,
                                                                          time_major=False)

                        lstm_outputs = tf.reshape(lstm_outputs, [-1, 512])

                        # weight for policy output layer
                        # 策略输出的权重和偏移
                        self.W_policy[key] = self._fc_weight_variable([512, action_size])
                        self.b_policy[key] = self._fc_bias_variable([action_size], 512)

                        # 新的策略，输出动作的概率
                        pi_ = tf.matmul(lstm_outputs, self.W_policy[key]) + self.b_policy[key]
                        self.pi[key] = tf.nn.softmax(pi_)

                        # 值函数的输出
                        self.W_value[key] = self._fc_weight_variable([512, 1])
                        self.b_value[key] = self._fc_bias_variable([1], 512)

                        # 输出新的值函数
                        v_ = tf.matmul(lstm_outputs, self.W_value[key]) + self.b_value[key]
                        self.v[key] = tf.reshape(v_, [-1])

                        # 清空 LSTM 的所有的数据
                        self.reset_state()

    def reset_state(self):
        """
        重置 LSTM 的记忆信息
        :return:
        """
        self.lstm_state_out = tf.contrib.rnn.LSTMStateTuple(np.zeros([1, 512]),
                                                            np.zeros([1, 512]))

    def run_policy_and_value(self, sess, state, scopes):
        """
        获得当前策略的输出以及值函数的输出
        :param sess: 会话
        :param state: 游戏的状态
        :param scopes: 运行的所在的 AC 网络的路径
        :return: 返回动作的概率，值函数的输出
        """
        k = self._get_key(scopes[:2])  #
        pi_out, v_out, self.lstm_state_out = sess.run([self.pi[k], self.v[k], self.lstm_state],
                                                      feed_dict={self.s: [state],
                                                                 self.initial_lstm_state0: self.lstm_state_out[0],
                                                                 self.initial_lstm_state1: self.lstm_state_out[1],
                                                                 self.step_size: [1]})
        return (pi_out[0], v_out[0])

    def run_policy(self, sess, state, scopes):
        """
        获得策略的输出
        :param sess: 会话
        :param state: 游戏的状态
        :param scopes: 运行的所在的 AC 网络的路径
        :return: 返回动作的概率
        """
        k = self._get_key(scopes[:2])
        # LSTM 的状态推进了
        pi_out, self.lstm_state_out = sess.run([self.pi[k], self.lstm_state],
                                               feed_dict={self.s: [state],
                                                          self.initial_lstm_state0: self.lstm_state_out[0],
                                                          self.initial_lstm_state1: self.lstm_state_out[1],
                                                          self.step_size: [1]})
        return pi_out[0]

    def run_value(self, sess, state, scopes):
        """
        获得值函数的输出
        :param sess: 会话
        :param state: 游戏的状态
        :param scopes: 运行的所在的 AC 网络的路径
        :return: 返回值函数
        """
        k = self._get_key(scopes[:2])
        prev_lstm_state_out = self.lstm_state_out
        v_out, _ = sess.run([self.v[k], self.lstm_state],
                            feed_dict={self.s: [state],
                                       self.initial_lstm_state0: self.lstm_state_out[0],
                                       self.initial_lstm_state1: self.lstm_state_out[1],
                                       self.step_size: [1]})
        # roll back lstm state
        self.lstm_state_out = prev_lstm_state_out
        return v_out[0]

    def get_vars(self):
        """
        获得当前网络的所有的参数， 用于同步参数到 目标（在 A3C 中一般是 global） 网络
        :return:
        """
        var_list = [
            self.W_fc1, self.b_fc1,
            self.W_fc2, self.b_fc2,
            self.W_fc3, self.b_fc3,
            self.W_policy, self.b_policy,
            self.W_value, self.b_value
        ]
        vs = []
        for v in var_list:
            vs.extend(v.values())
        # 全部打包为列表，用 zip 方便一对一赋值
        return vs