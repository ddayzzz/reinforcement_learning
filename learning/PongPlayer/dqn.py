# 定义 DQN 网络
import keras
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD , Adam
from keras.callbacks import TensorBoard
# cv2
import cv2

import random
import numpy as np
from replay_memory import ReplayBuffer

class DQN(object):

    def __init__(self,
                 img_width,
                 img_height,
                 num_history,
                 num_actions,
                 init_learning_rate):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), padding='same', strides=(4, 4), input_shape=(img_width, img_height, num_history)))  # 80*80*4
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(num_actions))
        # 定义学习率
        adam = Adam(lr=init_learning_rate)
        model.compile(loss='mse', optimizer=adam)
        self.model = model


#game parameters
ACTIONS = 6 # possible actions: jump, do nothing
GAMMA = 0.99 # decay rate of past observations original 0.99
OBSERVATION = 400 # timesteps to observe before training
EXPLORE = 100000  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4
img_rows , img_cols = 80,80
img_channels = 4 #We stack 4 frames
SAVE_PER_EPISODE = 2

class DQNPlayer():

    def __init__(self,
                 model,
                 env,
                 epsilon):
        self.model = model.model
        self.env = env
        self.epsilon = epsilon

    @staticmethod
    def show_img(graphs=None):
        """
        Show images in new window
        """
        while True:
            screen = (yield)
            cv2.namedWindow("nn_input", cv2.WINDOW_NORMAL)
            cv2.imshow("nn_input", screen)
            if (cv2.waitKey(1) & 0xFF == ord('q')):
                cv2.destroyAllWindows()
                break

    def choose_action(self, s_t):
        if random.random() <= self.epsilon:  # randomly explore an action
            action_index = random.randrange(self.env.num_actions)
        else:  # predict the output
            q = self.model.predict(s_t)  # input a stack of 4 images, get the prediction
            max_q_index = np.argmax(q)
            action_index = max_q_index
        return action_index

    def play(self,
             render_observation,
             render_nn_input):
        # 定义一个显示 NN 输入的生成器
        if render_nn_input:
            display = self.show_img(None)  # display the processed image on screen using openCV, implemented using python coroutine
            display.__next__()  # initiliaze the display coroutine
        # 定义训练过程的参数
        episode = 0
        replay_buffer = ReplayBuffer(50000)
        max_episode = 50000000
        # 训练
        for i_episode in range(max_episode):
            # 清空环境, 初始化状态 S, 包含四帧图像
            x_t = self.env.reset()
            s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)  # stack 4 images to create placeholder input
            s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  # 1*20*40*4
            # 开始一局
            time_step = 0
            loss = np.inf
            scores = 0
            while True:
                # 选择并运行一个动作
                action = self.choose_action(s_t)
                # 是否更新 epsilon
                if self.epsilon > FINAL_EPSILON and time_step > OBSERVATION:
                    self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
                x_t1, reward, done = self.env.step(action)
                # 计分
                if reward >= 1.0:
                    scores += 1
                # 生成新的状态 s_t1
                x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)  # 1x20x40x1
                s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)  # 移除最遥远时刻的一帧
                # 存入buffer
                replay_buffer.add(s_t, action, reward, s_t1, done)
                # 是否更新网络
                if time_step > OBSERVATION:
                    # sample a minibatch to train on
                    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH)
                    inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))  # 32, 20, 40, 4
                    targets = np.zeros((inputs.shape[0], self.env.num_actions))  # 32, 2

                    # Now we do the experience replay
                    for i in range(0, BATCH):
                        state_t = states[i]  # 4D stack of images
                        action_t = actions[i]  # This is action index
                        reward_t = rewards[i]  # reward at state_t due to action_t
                        state_t1 = next_states[i]  # next state
                        terminated = dones[i]  # wheather the agent died or survided due the action

                        inputs[i:i + 1] = state_t

                        targets[i] = self.model.predict(state_t)  # predicted q values
                        Q_sa = self.model.predict(state_t1)  # predict q values for next step

                        if terminated:
                            targets[i, action_t] = reward_t  # if terminated, only equals reward
                        else:
                            targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)
                    loss += self.model.train_on_batch(inputs, targets)
                time_step += 1
                # 画图
                if render_observation:
                    self.env.render()
                # 是否退出
                if done:
                    # 打印相关的信息
                    print('Episode:{episode}, Scores:{score}, Loss: {loss}'.format(loss=loss, episode=episode, score=scores))
                    episode += 1
                    #
                    if episode % SAVE_PER_EPISODE:
                        self.model.save_weights('pong_dqn.json', overwrite=True)
                    break
                else:
                    s_t = s_t1


