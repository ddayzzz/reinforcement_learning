# -*- coding: utf-8 -*-
# 显示 NN 的输出
import tensorflow as tf

import matplotlib.pyplot as plt

from game_ac_network import GameACLSTMNetwork

from rmsprop_applier import RMSPropApplier

from constants import ACTION_SIZE

from constants import CHECKPOINT_DIR
from constants import RMSP_EPSILON
from constants import RMSP_ALPHA
from constants import GRAD_NORM_CLIP


device = "/cpu:0"

global_network = GameACLSTMNetwork(ACTION_SIZE, -1, device)


training_threads = []

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

W_conv1 = sess.run(global_network.W_conv1)

# NN 第一个卷积层的卷积核
fig, axes = plt.subplots(4, 16, figsize=(12, 6),
                         subplot_kw={'xticks': [], 'yticks': []})
fig.subplots_adjust(hspace=0.1, wspace=0.1)

for ax, i in zip(axes.flat, range(4 * 16)):
    inch = i // 16
    outch = i % 16
    img = W_conv1[:, :, inch, outch]
    ax.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
    ax.set_title(str(inch) + "," + str(outch))

plt.show()
