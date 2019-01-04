# coding=utf-8
"""
定义玩的过程
"""
import gym
import time
import numpy as np
from IPython.display import clear_output
from collections import  deque
import cv2

#keras imports
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD , Adam
from keras.callbacks import TensorBoard
import random
import pickle
from io import BytesIO
import base64
import json

# game parameters
ACTIONS = 6 # possible actions: jump, do nothing
GAMMA = 0.99 # decay rate of past observations original 0.99
OBSERVATION = 1 # timesteps to observe before training
EXPLORE = 100000  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
EPSILON_DECAY_RATE = 0.25
EPSILON_DECAY_PER_EPISODE = 2
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 16 # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4
img_rows , img_cols = 80, 80
img_channels = 4 #We stack 4 frames

def process_img(image):
    # 转换为 TF 的格式
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # RGB to Grey Scale
    image = image[:300, :500]  # Crop Region of Interest(ROI)
    image = cv2.resize(image, (80, 80))
    return image


def buildmodel(env):
    print("Now we build the model")
    model = Sequential()
    model.add(
        Conv2D(32, (8, 8), padding='same', strides=(4, 4), input_shape=(img_cols, img_rows, img_channels)))  # 80*80*4
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
    model.add(Dense(ACTIONS))
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse', optimizer=adam)

    # create model file if not present
    # if not os.path.isfile(loss_file_path):
    #     model.save_weights('model.h5')
    print("We finish building the model")
    return model


''' 
main training module
Parameters:
* model => Keras Model to be trained
* game_state => Game State module with access to game environment and dino
* observe => flag to indicate wherther the model is to be trained(weight updates), else just play
'''


def trainNetwork(model, game_state, render=False, trainMode=True):

    D = deque()  # 存放的是 state, reward, action, next_state

    x_t = game_state.reset()
    # corp process
    x_t = process_img(x_t)

    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)  # stack 4 images to create placeholder input

    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  # 1*20*40*4

    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse', optimizer=adam)

    epsilon = INITIAL_EPSILON

    episode = 0
    ACTIONS = game_state.action_space.n
    while True:  # endless running
        loss = 0
        Q_sa = 0
        # a_t = np.zeros([ACTIONS])  # action at t

        # choose an action epsilon greedy
        if random.random() <= epsilon:  # randomly explore an action
            print("----------Random Action----------")
            action_index = random.randrange(ACTIONS)
        else:  # predict the output
            q = model.predict(s_t)  # input a stack of 4 images, get the prediction
            max_Q = np.argmax(q)  # chosing index with maximum q value
            action_index = max_Q
        a_t = action_index


        x_t1, r_t, terminal, _ = game_state.step(a_t)

        x_t1 = process_img(x_t1)

        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)  # 1x20x40x1
        s_t1 = np.append(x_t1, s_t[:, :, :, :3],
                         axis=3)  # append the new image to input stack and remove the first one
        # store the transition in D
        D.append((s_t, action_index, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        if render:
            game_state.render()



        if terminal:
            episode += 1
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)
            inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))  # 32, 20, 40, 4
            targets = np.zeros((inputs.shape[0], ACTIONS))  # 32, 2

            # Now we do the experience replay
            for i in range(0, len(minibatch)):
                state_t = minibatch[i][0]  # 4D stack of images
                action_t = minibatch[i][1]  # This is action index
                reward_t = minibatch[i][2]  # reward at state_t due to action_t
                state_t1 = minibatch[i][3]  # next state
                terminal = minibatch[i][4]  # wheather the agent died or survided due the action

                inputs[i:i + 1] = state_t

                targets[i] = model.predict(state_t)  # predicted q values
                Q_sa = model.predict(state_t1)  # predict q values for next step

                targets[i, action_t] = reward_t  # if terminated, only equals reward

            loss += model.train_on_batch(inputs, targets)
            # We reduced the epsilon (exploration parameter) gradually
            if epsilon > FINAL_EPSILON and episode % EPSILON_DECAY_PER_EPISODE:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
            new_x = game_state.reset()
            new_x = process_img(new_x)
            s_t = np.stack((new_x, new_x, new_x, new_x), axis=2)  # stack 4 images to create placeholder input
            s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])
        # print('fps: {0}'.format(1 / (time.time() - last_time)))  # helpful for measuring frame rate
        # last_time = time.time()
        #

        # s_t = initial_state if terminal else s_t1  # reset game to initial frame if terminate
        # t = t + 1

        # save progress every 1000 iterations

        # print info
        state = ""
        # if t <= OBSERVE:
        #     state = "observe"
        # elif t > OBSERVE and t <= OBSERVE + EXPLORE:
        #     state = "explore"
        # else:
        #     state = "train"

        print("EPISODE", episode, "/ STATE", state, "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t,
              "/ Q_MAX ", np.max(Q_sa), "/ Loss ", loss)

# main function
env = gym.make("Pong-v0")
model = buildmodel(env)
trainNetwork(model, env, render=True)