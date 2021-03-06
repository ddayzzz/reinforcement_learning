#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import cv2
import tensorflow as tf
import threading
import sys
import time
import os


def MakeDir(path):
    try:
        os.makedirs(path)
    except:
        pass


lab = True
# 如果载入之前的模型，train 为 false， load_model 为 true
load_model = False
train = True
test_display = True
test_write_video = True
path_work_dir = "~/rl_3d_maze/"

# Imshow 和 Env 的分辨率以及 fps 相同
out_size_width = 320
out_size_height = 240
out_fps = 60

from env_lab import EnvLab

model_path = path_work_dir + "model_lab_a3c/"

# 学习率
learning_rate = 0.00025
# 设备
device = "/gpu:0"
# local AC 的数量
num_workers = 8
# 限制最优函数的估计公式中 k 的范围。
t_max = 30
frame_repeat = 10  # 4
gamma = 0.99
step_num = int(2.5e5)
save_each = 0.01 * step_num
step_load = 100
entropy_beta = 0.01
grad_norm_clip = 40.0

global_scope_name = "global"
step = 0
train_scores = []
lock = threading.Lock()
start_time = 0

# Global.
env = None

MakeDir(model_path)
model_name = model_path + "a3c"
# deepmid level
level = "nav_maze_random_goal_02"


def PrintStat(elapsed_time, step, step_num, train_scores):
    steps_per_s = 1.0 * step / elapsed_time
    steps_per_m = 60.0 * step / elapsed_time
    steps_per_h = 3600.0 * step / elapsed_time
    steps_remain = step_num - step
    remain_h = int(steps_remain / steps_per_h)
    remain_m = int((steps_remain - remain_h * steps_per_h) / steps_per_m)
    remain_s = int((steps_remain - remain_h * steps_per_h - remain_m * steps_per_m) / steps_per_s)
    elapsed_h = int(elapsed_time / 3600)
    elapsed_m = int((elapsed_time - elapsed_h * 3600) / 60)
    elapsed_s = int((elapsed_time - elapsed_h * 3600 - elapsed_m * 60))
    print("{}% | Steps: {}/{}, {:.2f}M step/h, {:02}:{:02}:{:02}/{:02}:{:02}:{:02}".format(
        100.0 * step / step_num, step, step_num, steps_per_h / 1e6,
        elapsed_h, elapsed_m, elapsed_s, remain_h, remain_m, remain_s), file=sys.stderr)

    mean_train = 0
    std_train = 0
    min_train = 0
    max_train = 0
    if (len(train_scores) > 0):
        train_scores = np.array(train_scores)
        mean_train = train_scores.mean()
        std_train = train_scores.std()
        min_train = train_scores.min()
        max_train = train_scores.max()
    print("Episodes: {} Rewards: mean: {:.2f}, std: {:.2f}, min: {:.2f}, max: {:.2f}".format(
        len(train_scores), mean_train, std_train, min_train, max_train), file=sys.stderr)


channels = 3
resolution = (40, 40, channels)


def Preprocess(frame):
    if (channels == 1):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像
    frame = cv2.resize(frame, (resolution[1], resolution[0]))  # 缩放的图像分辨率
    return np.reshape(frame, resolution)


class ACNet(object):
    def __init__(self, num_actions, scope, trainer):
        with tf.variable_scope(scope):
            self.inputs = tf.placeholder(shape=[None] + list(resolution),
                                         dtype=tf.float32)  # resolution 估计是实际的 observation 的数量
            # CNN 的定义
            # 步长在两个维度上分别为2，2
            conv1 = tf.contrib.layers.conv2d(self.inputs, num_outputs=16, kernel_size=[3, 3], stride=[2, 2])
            conv2 = tf.contrib.layers.conv2d(conv1, num_outputs=32, kernel_size=[3, 3], stride=[2, 2])
            """
            通常在CNN的最后一步连接到Fully Connected 的网络之前会将其展开，例如CNN的conv层输出的tensor的shape为[batch_size, height, width, channel], 刚展开会就是[batch_size, height * width * channel]。
            """
            conv2_flat = tf.contrib.layers.flatten(conv2)
            # 全连接层
            hidden = tf.contrib.layers.fully_connected(conv2_flat, 256)

            # Recurrent network for temporal dependencies

            # Introduce a "fake" batch dimension of 1 after flatten so that we can do LSTM over time dim
            rnn_in = tf.expand_dims(hidden, [0])  # 增加维度，用于 RNN 处理序列的数据

            lstm_size = 256  # LSTM 隐藏层数量
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size, state_is_tuple=True)
            # 获取输入的维度。返回值是 3
            step_size = tf.shape(self.inputs)[:1]

            c_init = np.zeros((1, lstm_cell.state_size.c), dtype=np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), dtype=np.float32)
            self.state_init = [c_init, h_init]
            self.rnn_state = self.state_init

            c_in = tf.placeholder(shape=[1, lstm_cell.state_size.c], dtype=tf.float32)
            h_in = tf.placeholder(shape=[1, lstm_cell.state_size.h], dtype=tf.float32)
            self.state_in = (c_in, h_in)

            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            # time_major == False。 The shape format of the inputs and outputs Tensors。these Tensors must be shaped [batch_size, max_time, depth]
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm_cell, rnn_in, initial_state=state_in,
                                                         sequence_length=step_size, time_major=False)
            # 分解为 c 和 h
            lstm_c, lstm_h = lstm_state
            # 重新调整大小为 （?, 256）
            rnn_out = tf.reshape(lstm_outputs, [-1, lstm_size])
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])

            # Output layers for policy and value estimations。
            # policy 是概率的分布
            self.policy = tf.contrib.layers.fully_connected(rnn_out, num_actions, activation_fn=tf.nn.softmax,
                                                            weights_initializer=self.normalized_columns_initializer(
                                                                0.01),
                                                            biases_initializer=None)
            # 值函数
            self.value = tf.contrib.layers.fully_connected(rnn_out, 1, activation_fn=None,
                                                           weights_initializer=self.normalized_columns_initializer(1.0),
                                                           biases_initializer=None)

            # Only the worker network need ops for loss functions and gradient updating.
            if (scope != global_scope_name):
                # 定义输入的行为
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
                # onehot 编码行为
                actions_onehot = tf.one_hot(self.actions, num_actions, dtype=tf.float32)
                # target_v 是 TD 中的 R(t+1) + gama * V(S(t+1))，也叫 TD 目标
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                # 优势函数的定义
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)
                # 每一个动作对应的概率，reduce_sum 把列的维度收缩了。
                responsible_outputs = tf.reduce_sum(self.policy * actions_onehot, [1])

                # Loss functions：损失函数的定义
                # Loss function 1:
                # reshape 指定 [-1] 将合并所有的维度，相当于 flatten
                # 平方的误差，计算 TD error 用的。乘以 0.5 目前不知道是怎么回事。原始的公式 Rt - V（st,theta_v)
                value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
                # Loss function2: 公式参见论文的 第4页 Entropy regularization term
                entropy = -tf.reduce_sum(self.policy * tf.log(self.policy))  # H(pi(st,theta_))。这个公式是信息熵公式
                policy_loss = -tf.reduce_sum(tf.log(
                    responsible_outputs) * self.advantages)  # log_pi(ai|si;theta_)。注意这里的 reduce_sum 是因为最终相乘的得到一个矩阵。
                self.loss = 0.5 * value_loss + policy_loss - entropy * entropy_beta  # 这个定义似乎与论文的不相同

                # Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                # 定义求梯度，参数 ys（被微分的），xs（用于微分）
                # 返回：A list of sum(dy/dx) for each x in xs.
                # https://www.tensorflow.org/api_docs/python/tf/gradients
                self.gradients = tf.gradients(self.loss, local_vars)

                if (grad_norm_clip != None):
                    grads, _ = tf.clip_by_global_norm(self.gradients, grad_norm_clip)
                else:
                    grads = self.gradients

                # Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, global_scope_name)
                self.apply_grads = trainer.apply_gradients(
                    zip(grads, global_vars))  # zip 打包为列表，返回一个对 Zip 对象，可迭代，每个元素是一个 tuple，对应的位置的一个 tuple

    # Used to initialize weights for policy and value output layers
    def normalized_columns_initializer(self, std=1.0):
        def _initializer(shape, dtype=None, partition_info=None):
            out = np.random.randn(*shape).astype(np.float32)
            out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
            return tf.constant(out)

        return _initializer

    def Train(self, sess, discounted_rewards, states, actions, advantages):
        states = states / 255.0
        self.ResetLstm()
        feed_dict = {self.target_v: discounted_rewards,
                     self.inputs: np.stack(states, axis=0),
                     self.actions: actions,
                     self.advantages: advantages,
                     self.state_in[0]: self.rnn_state[0],
                     self.state_in[1]: self.rnn_state[1]}
        _ = sess.run([self.apply_grads], feed_dict=feed_dict)

    def ResetLstm(self):
        self.rnn_state = self.state_init

    def GetAction(self, sess, state):
        # 根据当前的策略选择，选择 action。返回选择的 action 和 值
        state = state / 255.0  # 至于为何 / 255.0 我还不清楚
        # 输入的是 state
        a_dist, v, self.rnn_state = sess.run([self.policy, self.value, self.state_out],
                                             feed_dict={self.inputs: [state],
                                                        self.state_in[0]: self.rnn_state[0],
                                                        self.state_in[1]: self.rnn_state[1]})
        # 根据指定的概率选择 a_dist 中的行为。a_dist 是每一个 action 的概率，【0】是从矩阵中选择第一行。a_dist[0] 使用本身计算的概率随即选择。
        a = np.random.choice(a_dist[0], p=a_dist[0])
        # 找到根据概率选择的行为的位置。这个 == 表达式返回匹配的序列，匹配的就是 1。可能会有都是 0 的情况
        a = np.argmax(a_dist == a)
        # v 是一个 1x1 的矩阵，所以需要用 [0,0] 取值
        return a, v[0, 0]

    def GetValue(self, sess, state):
        state = state / 255.0
        v = sess.run([self.value],
                     feed_dict={self.inputs: [state],
                                self.state_in[0]: self.rnn_state[0],
                                self.state_in[1]: self.rnn_state[1]})
        return v[0][0, 0]


class Worker(object):
    def __init__(self, number, num_actions, trainer, model_name):

        self.name = "worker_" + str(number)
        self.number = number
        self.model_name = model_name

        # Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_ac = ACNet(num_actions, self.name, trainer)
        # 从全局 AC 网络中获取数据
        self.update_target_graph = self.update_target(global_scope_name, self.name)
        # 使用的环境
        self.env = EnvLab(width=out_size_width, height=out_size_height, fps=out_fps, level=level)

    # Copies one set of variables to another.
    # Used to set worker network parameters to those of global network.
    def update_target(self, from_scope, to_scope):
        # 指定了 global 和 local 的作用域
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

        op_holder = []
        # 由于 global 和 local 定义的变量一致。所以直接组合成同一个 Zip，然后从 from 中给 to 赋值
        for from_var, to_var in zip(from_vars, to_vars):
            op_holder.append(to_var.assign(from_var))
        return op_holder

    # Calculate discounted returns.
    def Discount(self, x, gamma):
        # 他是计算来的这是一个通用的函数，具体的含义只有在调用位置才有指定
        for idx in reversed(range(len(x) - 1)):
            x[idx] += x[idx + 1] * gamma
        return x

    def Start(self, session, saver, coord):
        worker_process = lambda: self.Process(session, saver, coord)
        thread = threading.Thread(target=worker_process)
        thread.start()

        global start_time
        start_time = time.time()
        return thread

    def Train(self, episode_buffer, sess, bootstrap_value):
        """
        训练数据集
        :param episode_buffer: local 缓存的数据
        :param sess: 会话
        :param bootstrap_value: 自举值，总是计算得到的当前最新的状态的
        :return:
        """
        # 定义的 local_ac 保存的计算的数据
        episode_buffer = np.array(episode_buffer)
        states = episode_buffer[:, 0]
        actions = episode_buffer[:, 1]
        rewards = episode_buffer[:, 2]
        values = episode_buffer[:, 3]

        # Here we take the rewards and values from the episode_buffer, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        # **删除** 目前的线索包括 迹函数、后观点的 TD error
        # **删除** 生成一个带有刚刚计算出来的值函数的新的 reward 列表，需要用来计算 eligibility trace。
        rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        # 这个对应于伪代码中，更新 R 的公式，目前 x 中的值是 r_i。对于 x 中最后一个元素，就是刚刚计算出来的 R。最终计算出来的 R 保存在 x[0] 中。
        # 我们不需要最后扩展出来的 R。
        # 这个可以推出来。Discount 的公式可以提炼出：X(i) = SIGMA(i=0,i<指定的上界)(r(t+i) * gama^i)。这里的t是起始的位置，这个函数默认是 0。可以对应论文章的 t_start
        discounted_rewards = self.Discount(rewards_plus, gamma)[:-1]
        # advatnages 的定义 A(at,st) = Q(at,st) - V(st)
        value_plus = np.asarray(values.tolist() + [bootstrap_value])
        # 这一个部分计算的是 TD error ：delta_t = R(t+1) + gama * V(S(t+1)) - V(S(t))
        advantages = rewards + gamma * value_plus[1:] - value_plus[:-1]
        # TD 的后向观点的计算。 更新 Q
        advantages = self.Discount(advantages, gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        self.local_ac.Train(sess, discounted_rewards, states, actions, advantages)

    def Process(self, sess, saver, coord):
        global step, train_scores, start_time, lock

        print("Starting worker " + str(self.number))
        while (not coord.should_stop()):
            sess.run(self.update_target_graph)
            episode_buffer = []
            episode_reward = 0
            # 重新设置环境
            self.env.Reset()
            s = self.env.Observation()
            # 当 channel==1 的时候，把 环境返回的 RGBD 格式转换为 RGB 格式。最后要缩放
            s = Preprocess(s)
            self.local_ac.ResetLstm()

            while (self.env.IsRunning()):
                # Take an action using probabilities from policy network output.
                a, v = self.local_ac.GetAction(sess, s)
                r = self.env.Act(a, frame_repeat)
                finished = not self.env.IsRunning()
                if (not finished):
                    s1 = self.env.Observation()
                    s1 = Preprocess(s1)
                else:
                    s1 = None

                episode_buffer.append([s, a, r, v])

                episode_reward += r
                s = s1

                lock.acquire()

                step += 1

                if (step % save_each == 0):
                    model_name_curr = self.model_name + "_{:04}".format(int(step / save_each))
                    print("\nSaving the network weigths to:", model_name_curr, file=sys.stderr)
                    saver.save(sess, model_name_curr)

                    PrintStat(time.time() - start_time, step, step_num, train_scores)

                    train_scores = []

                if (step == step_num):
                    coord.request_stop()

                lock.release()

                # If the episode hasn't ended, but the experience buffer is full, then we
                # make an update step using that experience rollout.
                if (len(episode_buffer) == t_max or (finished and len(episode_buffer) > 0)):
                    # Since we don't know what the true final return is,
                    # we "bootstrap" from our current value estimation.
                    if (not finished):
                        v1 = self.local_ac.GetValue(sess, s)
                        self.Train(episode_buffer, sess, v1)
                        episode_buffer = []
                        # 将全局 AC 中的数据更新
                        sess.run(self.update_target_graph)
                    else:
                        self.Train(episode_buffer, sess, 0.0)

            lock.acquire()
            train_scores.append(episode_reward)
            lock.release()


class Agent(object):
    def __init__(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = False
        config.allow_soft_placement = True

        self.session = tf.Session(config=config)

        with tf.device(device):
            # Global network
            self.global_net = ACNet(env.NumActions(), global_scope_name, None)

            if (train):
                # 使用训练的优化器
                trainer = tf.train.RMSPropOptimizer(learning_rate)

                workers = []
                for i in range(num_workers):
                    workers.append(Worker(i, env.NumActions(), trainer, model_name))

        saver = tf.train.Saver(max_to_keep=100)
        if (load_model):
            model_name_curr = model_name + "_{:04}".format(step_load)
            print("Loading model from: ", model_name_curr)
            saver.restore(self.session, model_name_curr)
        else:
            self.session.run(tf.global_variables_initializer())

        if (train):
            coord = tf.train.Coordinator()
            # Start the "work" process for each worker in a separate thread.
            worker_threads = []
            for worker in workers:
                thread = worker.Start(self.session, saver, coord)
                worker_threads.append(thread)
            coord.join(worker_threads)

    def Reset(self):
        self.global_net.ResetLstm()

    def Act(self, state):
        action, _ = self.global_net.GetAction(self.session, state)
        return action


def Test(agent):
    if (test_write_video):
        size = (out_size_width, out_size_height)
        fps = out_fps
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # cv2.cv.CV_FOURCC(*'XVID')
        out_video = cv2.VideoWriter(path_work_dir + "test.avi", fourcc, fps, size)

    reward_total = 0
    num_episodes = 30

    if test_display:
        cv2.namedWindow("frame-test", 0)  # 创建窗口，大小可以调整
    while (num_episodes != 0):
        if (not env.IsRunning()):
            env.Reset()
            agent.Reset()
            print("Total reward: {}".format(reward_total))
            reward_total = 0
            num_episodes -= 1

        state_raw = env.Observation()

        state = Preprocess(state_raw)
        action = agent.Act(state)

        for _ in range(frame_repeat):
            if (test_display):
                # cv2.resizeWindow("frame-test", size[0], size[1])
                cv2.imshow("frame-test", state_raw)
                cv2.waitKey(20)

            if (test_write_video):
                out_video.write(state_raw)
            # 先走一步？
            reward = env.Act(action, 1)
            reward_total += reward

            if (not env.IsRunning()):
                break

            state_raw = env.Observation()
    # 释放资源
    if test_write_video:
        out_video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # seekavoid_arena_01 的 actions：
    """
    [{'max': 512, 'min': -512, 'name': 'LOOK_LEFT_RIGHT_PIXELS_PER_FRAME'},
 {'max': 512, 'min': -512, 'name': 'LOOK_DOWN_UP_PIXELS_PER_FRAME'},
 {'max': 1, 'min': -1, 'name': 'STRAFE_LEFT_RIGHT'},
 {'max': 1, 'min': -1, 'name': 'MOVE_BACK_FORWARD'},
 {'max': 1, 'min': 0, 'name': 'FIRE'},
 {'max': 1, 'min': 0, 'name': 'JUMP'},
 {'max': 1, 'min': 0, 'name': 'CROUCH'}]
    """
    if (lab):
        env = EnvLab(width=out_size_width, height=out_size_height, fps=out_fps, level=level)

    agent = Agent()

    Test(agent)
