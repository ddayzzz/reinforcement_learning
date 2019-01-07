from env import PongEnv
from dqn import DQNPlayer, DQN


if __name__ == '__main__':
    env = PongEnv(crop_image_height=80, crop_image_width=80)
    model = DQN(img_width=80, img_height=80, num_history=4, num_actions=env.num_actions, init_learning_rate=1e-4)
    player = DQNPlayer(model, env, epsilon=0.25)
    player.play(True, False)
