import numpy as np
import pandas as pd
import time


# 设置随机数种子
# np.random.seed(inttime.time())

# the length of the 1 dimensional world. The very beginning of the length between initial position and that of treasure
N_STATES = 6
# available states
ACTIONS = ['left', 'right']
# greedy policy
EPSILON = 0.9
# learning rate
ALPHA = 0.1
# discount factor
LAMBDA = 0.9
# maximum episodes. Only work 13 episodes
MAX_EPISODES = 13
# fresh time for one move
FRESH_TIME = 0.3

def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),  # q_table initial values
        columns=actions  # actions' name
    )
    return table


def choose_action(state, q_table):
    state_actions = q_table.iloc[state,:]
    if(np.random.uniform() > EPSILON) or (state_actions.all() == 0):
        # choose action at random
        action_name = np.random.choice(ACTIONS)
    else:
        # act greedy, choose the maximum action
        action_name = state_actions.idxmax()
    return action_name


def get_env_feedback(S, A):
    """
    Feedback, how agent will do to interact with environment
    :param S:
    :param A:
    :return:
    """
    if A == 'right':
        if S == N_STATES - 2: # reach the target
            S_ = 'terminated'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:  # move left
        R = 0
        if S == 0:
            S_ = S
        else:
            S_ = S - 1
    return S_, R


def update_env(S, episode, step_counter):
    """
    Display the environment changes
    :param S:
    :param episode:
    :param setp_counter:
    :return:
    """
    env_list = ['-'] * (N_STATES - 1) + ['T']
    if S == 'terminated':
        interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                      ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    """
    The main loop for RL
    :return:
    """
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        # initial
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:
            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A)
            q_predict = q_table.ix[S, A]
            if S_ != 'terminated':
                q_target = R + LAMBDA * q_table.iloc[S_,:].max()  # iloc： chose the specific columns based on integer
            else:
                q_target = R
                is_terminated = True
            q_table.ix[S, A] += ALPHA * (q_target - q_predict)
            # next_state <- old_state
            S = S_
            update_env(S, episode, step_counter + 1)
            step_counter += 1
    return q_table

if __name__ == '__main__':
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)