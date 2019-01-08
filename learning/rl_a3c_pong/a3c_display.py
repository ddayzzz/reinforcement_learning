# -*- coding: utf-8 -*-
# 定义 A3C 测试过程
import tensorflow as tf
import numpy as np

from game_state import GameState
from game_ac_network import GameACLSTMNetwork

from rmsprop_applier import RMSPropApplier

from constants import ACTION_SIZE

from constants import CHECKPOINT_DIR
from constants import RMSP_EPSILON
from constants import RMSP_ALPHA
from constants import GRAD_NORM_CLIP


def choose_action(pi_values):
    return np.random.choice(range(len(pi_values)), p=pi_values)


# 使用 CPU，可以边训练边检查
device = "/cpu:0"

global_network = GameACLSTMNetwork(ACTION_SIZE, -1, device)

learning_rate_input = tf.placeholder("float")

grad_applier = RMSPropApplier(learning_rate=learning_rate_input,
                              decay=RMSP_ALPHA,
                              momentum=0.0,
                              epsilon=RMSP_EPSILON,
                              clip_norm=GRAD_NORM_CLIP,
                              device=device)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("加载恢复点:", checkpoint.model_checkpoint_path)
else:
    print("没找到检查点")

game_state = GameState(display=True, seed=0, init_actions_max=0)

while True:

    pi_values = global_network.run_policy(sess, game_state.s_t)

    action = choose_action(pi_values)
    game_state.process(action)

    if game_state.terminal:
        game_state.reset()
    else:
        game_state.update()
