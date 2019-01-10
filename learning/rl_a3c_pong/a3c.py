# -*- coding: utf-8 -*-
# A3C 训练程序
import tensorflow as tf
import threading
import numpy as np

import signal
import random
import math
import os
import time

from game_ac_network import GameACLSTMNetwork
from a3c_training_thread import A3CTrainingThread
from rmsprop_applier import RMSPropApplier

from constants import ACTION_SIZE
from constants import PARALLEL_SIZE
from constants import INITIAL_ALPHA_LOW
from constants import INITIAL_ALPHA_HIGH
from constants import INITIAL_ALPHA_LOG_RATE
from constants import MAX_TIME_STEP
from constants import CHECKPOINT_DIR
from constants import LOG_FILE
from constants import RMSP_EPSILON
from constants import RMSP_ALPHA
from constants import GRAD_NORM_CLIP
from constants import USE_GPU



def log_uniform(lo, hi, rate):
    """
    计算的最佳的学习率
    :param lo:
    :param hi:
    :param rate:
    :return:
    """
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    v = log_lo * (1 - rate) + log_hi * rate
    return math.exp(v)

# 规定会话运行的设备, 默认使用 CPU
device = "/cpu:0"
if USE_GPU:
    device = "/gpu:0"

initial_learning_rate = log_uniform(INITIAL_ALPHA_LOW,
                                    INITIAL_ALPHA_HIGH,
                                    INITIAL_ALPHA_LOG_RATE)

global_t = 0

stop_requested = False

# global AC 网络
global_network = GameACLSTMNetwork(ACTION_SIZE, -1, device)

# local AC 网络
training_threads = []

learning_rate_input = tf.placeholder("float")
# 定义优化器, 论文中又 SGD 和 RMSProp
grad_applier = RMSPropApplier(learning_rate=learning_rate_input,
                              decay=RMSP_ALPHA,
                              momentum=0.0,
                              epsilon=RMSP_EPSILON,
                              clip_norm=GRAD_NORM_CLIP,
                              device=device)
# 创建 local AC 网络,需要在线程上运行
for i in range(PARALLEL_SIZE):
    training_thread = A3CTrainingThread(i, global_network, initial_learning_rate,
                                        learning_rate_input,
                                        grad_applier, MAX_TIME_STEP,
                                        device=device)
    training_threads.append(training_thread)

# 初始化会话
sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                        allow_soft_placement=True))
# 初始化所有变量
init = tf.global_variables_initializer()
sess.run(init)

# 定义 tensorboard 的可视化输出
## 第一个 local AC 的分数
score_input = tf.placeholder(tf.int32)
learning_rate_input = tf.placeholder(tf.float32)
tf.summary.scalar("score", score_input)
tf.summary.scalar("learning_rate", learning_rate_input)
summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(LOG_FILE, sess.graph)
# 定义调度器
coord = tf.train.Coordinator()
# 主要是用于保存模型以及恢复的过程
saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("加载恢复点:", checkpoint.model_checkpoint_path)
    tokens = checkpoint.model_checkpoint_path.split("-")
    # tf 的 检查点格式的最后一项是 global_step
    global_t = int(tokens[1])
    print(">>> global AC 网络时间: ", global_t)
    # 用于记录模型的起始时间，local AC 网络的运行的时间步，这个决定了算法运行的时间。这个时间是纪元时间
    wall_t_fname = CHECKPOINT_DIR + '/' + 'wall_t.' + str(global_t)
    with open(wall_t_fname, 'r') as f:
        wall_t = float(f.read())
else:
    print("没找到检查点，从头开始训练")
    # 设置时间步为 0
    wall_t = 0.0


def train_function(parallel_index, coord):
    """

    :param parallel_index:
    :return:
    """
    global global_t

    training_thread = training_threads[parallel_index]
    # 计算还能运行的世家u你
    start_time = time.time() - wall_t
    training_thread.set_start_time(start_time)

    while not coord.should_stop():
        if stop_requested:
            # 是否发出了终止信号？
            break
        if global_t > MAX_TIME_STEP:
            # global AC 已经运行完成
            break
        diff_global_t = training_thread.process(sess,
                                                global_t,
                                                summary_writer,
                                                summary_op,
                                                learning_rate_input,
                                                score_input)
        global_t += diff_global_t



def signal_handler(signal, frame):
    """
    SIGINT 的处理程序
    :param signal:
    :param frame:
    :return:
    """
    global stop_requested
    print('接收到了终止信号: SIGTERM 或 SIGINT')
    # 通知线程，结束 local AC
    stop_requested = True


train_threads = []
for i in range(PARALLEL_SIZE):
    train_threads.append(threading.Thread(target=train_function, args=(i, coord)))
# 绑定 KeyboardInterrupt 的信号，OS 会给主进程发送 SIGINT 信号
signal.signal(signal.SIGINT, signal_handler)
# 主进程可能会被终止, 接收 SIGTERM 信号
signal.signal(signal.SIGTERM, signal_handler)

# 起始时间
start_time = time.time() - wall_t

for t in train_threads:
    t.start()
coord.join(train_threads)

print('按下 Ctrl+C 退出')
if os.name != 'nt':
    signal.pause()

print('保存检查点...')
# 等待所有线程退出
for t in train_threads:
    t.join()

if not os.path.exists(CHECKPOINT_DIR):
    os.mkdir(CHECKPOINT_DIR)

# 写入 wall clock time
wall_t = time.time() - start_time
wall_t_fname = CHECKPOINT_DIR + '/' + 'wall_t.' + str(global_t)
with open(wall_t_fname, 'w') as f:
    f.write(str(wall_t))

saver.save(sess, CHECKPOINT_DIR + '/' + 'checkpoint', global_step=global_t)
sess.close()
