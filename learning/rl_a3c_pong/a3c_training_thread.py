# -*- coding: utf-8 -*-
# 定义 A3C 每个 local 网络的训练过程
import tensorflow as tf
import numpy as np
import time


from game_state import GameState
from game_state import ACTION_SIZE
from game_ac_network import GameACLSTMNetwork

# 需要的常熟
from constants import GAMMA
from constants import LOCAL_T_MAX
from constants import ENTROPY_BETA

# 输出日志的间隔
LOG_INTERVAL = 100
PERFORMANCE_LOG_INTERVAL = 1000


class A3CTrainingThread(object):

    def __init__(self,
                 thread_index,
                 global_network,
                 initial_learning_rate,
                 learning_rate_input,
                 grad_applier,
                 max_global_time_step,
                 device):
        """
        A3C 算法的 local AC 网络的训练
        :param thread_index:  线程编号，-1 是 全局的 AC 网络
        :param global_network:
        :param initial_learning_rate:
        :param learning_rate_input:
        :param grad_applier: 梯度更新器对象，论文中使用了 RMSProp
        :param max_global_time_step:
        :param device:
        """
        self.thread_index = thread_index
        self.learning_rate_input = learning_rate_input
        self.max_global_time_step = max_global_time_step
        # 初始化网络的参数
        self.local_network = GameACLSTMNetwork(ACTION_SIZE, thread_index, device)

        self.local_network.prepare_loss(ENTROPY_BETA)
        # 需要手机 loss 函数关于各个训练参数？的梯度信息
        with tf.device(device):
            var_refs = [v._ref() for v in self.local_network.get_vars()]
            self.gradients = tf.gradients(
                self.local_network.total_loss, var_refs,
                gate_gradients=False,
                aggregation_method=None,
                colocate_gradients_with_ops=False)
        # 更新梯度的 tf 操作
        self.apply_gradients = grad_applier.apply_gradients(
            global_network.get_vars(),
            self.gradients)
        # 每一个 local AC 在算法结束的时候需要从 global AC 网络同步参数
        self.sync = self.local_network.sync_from(global_network)
        # 封装游戏
        self.game_state = GameState()
        # 统计 时间步
        self.local_t = 0
        # 各色训练参参数
        self.initial_learning_rate = initial_learning_rate

        self.episode_reward = 0

        # 控制日志的输出
        self.prev_local_t = 0

    def _anneal_learning_rate(self, global_time_step):
        """
        递减学习率，主要是防止在 loss 的最小值的地方来回的震荡
        :param global_time_step: 已经玩的时间
        :return:
        """
        learning_rate = self.initial_learning_rate * (
                    self.max_global_time_step - global_time_step) / self.max_global_time_step
        if learning_rate < 0.0:
            learning_rate = 0.0
        return learning_rate

    def choose_action(self, pi_values):
        """
        这个是 epsilon-greedy， 需要指定输出行为的分布
        :param pi_values: 获得的策略
        :return: 返回一个动作
        """
        return np.random.choice(range(len(pi_values)), p=pi_values)

    def _record_score(self, sess, summary_writer, summary_op, score_input, score, global_t):
        """
        tensorboard 的可视化输出
        :param sess: 会话
        :param summary_writer:
        :param summary_op:
        :param score_input:
        :param score:
        :param global_t:
        :return:
        """
        summary_str = sess.run(summary_op, feed_dict={
            score_input: score
        })
        summary_writer.add_summary(summary_str, global_t)
        summary_writer.flush()

    def set_start_time(self, start_time):
        """
        设置开始时间
        :param start_time:
        :return:
        """
        self.start_time = start_time

    def process(self, sess, global_t, summary_writer, summary_op, score_input):
        """
        开始 local AC 网络的训练过程
        :param sess:
        :param global_t:
        :param summary_writer:
        :param summary_op:
        :param score_input:
        :return:
        """
        states = []
        actions = []
        rewards = []
        values = []

        terminal_end = False

        # 从全局的 AC 网络中获取参数
        sess.run(self.sync)

        start_local_t = self.local_t

        start_lstm_state = self.local_network.lstm_state_out

        # 必须规定 local AC 网络最大的时间步
        for i in range(LOCAL_T_MAX):
            # 已经的到了游戏的当前状态以及更新后的
            pi_, value_ = self.local_network.run_policy_and_value(sess, self.game_state.s_t)
            action = self.choose_action(pi_)
            # 保存累计的信息
            states.append(self.game_state.s_t)
            actions.append(action)
            values.append(value_)
            # 只有 global AC 网络需要在合适的时候输出日志
            if (self.thread_index == 0) and (self.local_t % LOG_INTERVAL == 0):
                print("pi={}".format(pi_))
                print(" V={}".format(value_))

            # 执行动作
            self.game_state.process(action)

            # 获取游戏的回报，这个封装了连接 连续4帧图像的过程
            reward = self.game_state.reward
            terminal = self.game_state.terminal
            # 累计的报答信息
            self.episode_reward += reward

            rewards.append(np.clip(reward, -1, 1))

            self.local_t += 1

            # 状态更新
            self.game_state.update()
            # 在附录的 Algorithm 3 中
            if terminal:
                terminal_end = True
                print("score={}".format(self.episode_reward))

                self._record_score(sess, summary_writer, summary_op, score_input,
                                   self.episode_reward, global_t)

                self.episode_reward = 0
                self.game_state.reset()
                # LSTM 的传递的装套重置
                self.local_network.reset_state()
                break
        # 分类讨论，计算的是 discounted 的 Rewards
        R = 0.0
        if not terminal_end:
            R = self.local_network.run_value(sess, self.game_state.s_t)  # 在最后的一个状态开始自举

        actions.reverse()
        states.reverse()
        rewards.reverse()
        values.reverse()

        batch_si = []
        batch_a = []
        batch_td = []
        batch_R = []

        # 这是 MDP 的四元组形式 在论文中，时间步是反的
        for (ai, ri, si, Vi) in zip(actions, rewards, states, values):
            R = ri + GAMMA * R
            td = R - Vi
            # a 用 one-hot 表示
            a = np.zeros([ACTION_SIZE])
            a[ai] = 1

            batch_si.append(si)
            batch_a.append(a)
            batch_td.append(td)
            batch_R.append(R)

        cur_learning_rate = self._anneal_learning_rate(global_t)

        batch_si.reverse()
        batch_a.reverse()
        batch_td.reverse()
        batch_R.reverse()

        sess.run(self.apply_gradients,
                 feed_dict={
                     self.local_network.s: batch_si,
                     self.local_network.a: batch_a,
                     self.local_network.td: batch_td,
                     self.local_network.r: batch_R,
                     self.local_network.initial_lstm_state: start_lstm_state,
                     self.local_network.step_size: [len(batch_a)],
                     self.learning_rate_input: cur_learning_rate})

        if (self.thread_index == 0) and (self.local_t - self.prev_local_t >= PERFORMANCE_LOG_INTERVAL):
            self.prev_local_t += PERFORMANCE_LOG_INTERVAL
            elapsed_time = time.time() - self.start_time
            steps_per_sec = global_t / elapsed_time
            print("### Performance : {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format(
                global_t, elapsed_time, steps_per_sec, steps_per_sec * 3600 / 1000000.))

        # return advanced local step size
        diff_local_t = self.local_t - start_local_t
        return diff_local_t

