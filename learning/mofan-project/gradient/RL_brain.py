import numpy as np
import tensorflow as tf

# 如果一个回合的动作获得的 reward 越大，那么 v 的作用就可以保证

class PolicyGradient:

    def __init__(self, n_actions, n_features, learning_rate=0.01,
                 reward_decay=0.95, output_graph=False):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()
        # 限制显存的使用：https://segmentfault.com/a/1190000009954640?utm_source=itdadao&utm_medium=referral
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")
        # full connected layer
        layer = tf.layers.dense(  # 建立一个紧密的连接层（全连接层）
            inputs=self.tf_obs,
            units=10,  # 10 个神经元
            activation=tf.nn.tanh,  # 激活函数
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )
        # fc2
        all_act = tf.layers.dense(
            inputs=layer,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name="fc2"
        )
        # 转换的为概率：Softmax函数实际上是有限项离散概率分布的梯度对数归一化
        # 参见 https://zh.wikipedia.org/wiki/Softmax%E5%87%BD%E6%95%B0
        self.all_act_prob = tf.nn.softmax(all_act, name="act_prob")

        with tf.name_scope('loss'):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            # 求交叉熵，介于 logits 和 labels 之间。信息上的公式 Ent(D) = sum(pk * log2(pk))
            #neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)
            # 或者使用，one_hot 类似于硬件语言的编码，一个不同状态之间只有一个不同的二进制为置1。axis=1，按照列。depth=行为的数量，编码为多少个行为
            # 注意使用了负号，因为 tensorflow 只有 minimize 但是我们需要最大，所以取负号
            # one_hot : https://www.tensorflow.org/api_docs/python/tf/one_hot
            # 交叉熵刻画的是两个概率分布之间的距离。类似于自定义损失函数，reduce_sum
            neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob) * tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            # reward guided loss，
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)
            # reduce 的操作都会减小一个维度，相当于减小括号的层数
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    # 选择行为
    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob,
                                     feed_dict={
                                         self.tf_obs: observation[np.newaxis, :]
                                     })
        # 选择动作，制订了每个 action 的概率 p。numpy.ravel() 返回一维的矩阵
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.ravel.html
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action

    # 存储回合 transition
    def store_transition(self, s, a, r):
        # 回合重存储 observation。由此知道关于环境的信息
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    # 学习参数更新
    def learn(self):
        # discount and normalize episode reward
        discount_ep_rs_norm = self._discount_and_norm_rewards()
        # 每个回合的训练
        self.sess.run(self.train_op, feed_dict={
            self.tf_obs: np.vstack(self.ep_obs),  # shape = [None, n_obs]
            self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
            self.tf_vt: discount_ep_rs_norm  # shape=[None,]
        })
        # 重新计算回合
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        return discount_ep_rs_norm

    # 衰减回合的 reward
    def _discount_and_norm_rewards(self):
        # 每个回合衰减 rewards
        discount_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discount_ep_rs[t] = running_add
        # normalize 每个回合的回报
        discount_ep_rs -= np.mean(discount_ep_rs)
        discount_ep_rs /= np.std(discount_ep_rs)
        return discount_ep_rs