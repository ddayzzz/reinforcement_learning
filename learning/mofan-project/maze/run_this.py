# control
from maze.RL_brain import *
from maze.maze_env import Maze

def update():
    for episode in range(100):
        # initial observation
        observation = env.reset()
        # RL choose action based on observation
        action = RL.choose_action(str(observation))
        # Sarsa lambda：clean the eligibility for every episode
        RL.eligibility_trace *= 0
        while True:
            # interact with GUI environment
            env.render()
            # RL take action and get next observation, reward and terminated state
            observation_, reward, done = env.step(action)
            # Sarsa
            action_ = RL.choose_action(str(observation_))
            # RL learn from this transition
            # RL.learn(str(observation), action, reward, str(observation_)) # Q-learning
            RL.learn(str(observation), action, reward, str(observation_), action_)  # Q-learning
            # next_state <- old_state
            observation = observation_
            # Sarsa
            action = action_
            # is terminated?
            if done:
                break
    # end
    print('Game over')
    env.destroy()

if __name__ == '__main__':
    env = Maze()
    # RL = QLearningTable(
    #     actions=list(range(env.n_actions))
    # )
    # Sarsa
    # RL = SarsaTable(
    #     actions=list(range(env.n_actions)),
    #     e_greedy=0.99  # 建议调大，按照策略选择的概率会变大
    # )
    # Sarsa lambda
    RL = SarsaLambdaTable(
        actions=list(range(env.n_actions)),
        e_greedy=0.99  # 建议调大，按照策略选择的概率会变大
    )
    env.after(100, update)
    env.mainloop()