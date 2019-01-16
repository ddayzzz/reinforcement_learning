# Pong-v0(Atair game) based on A3C
This is our end-term assignment of Machine learning in the 2018-2019-1 semester.
# Introduction
1. Our repo is derived from [miyosuda](https://github.com/miyosuda/async_deep_reinforce) and the original repo's branch named `gym`. I manually merged the pull request on this branch.
2. For more explanation of the background of Policy Gradient and Actor-Critic method and its async type, please refer to our [paper(In Chinese)](paper.pdf)
# Structures
## Directories
1. `logs`: target dir of tensorboard if you want to show the structure of neural network and scores.
2. `videos`: out video of a episode of`Pong`
3. `checkpoints`: saved model of nn.
# Dependencies
1. gym
2. tensorflow
3. python-opencv
4. python >= 3.x
# Acknowledgement
- [miyosuda](https://github.com/miyosuda/async_deep_reinforce)
- [gym branch of miyosuda](https://github.com/miyosuda/async_deep_reinforce/tree/gym) and the pull request 
[Updated for tensorflow 1.0, support for Mnih 2015 network architecture](https://github.com/miyosuda/async_deep_reinforce/pull/38/commits/5a49ec87eabc3a791a4cb2c6b3ed933b5186e983)
- [The author of our paper](https://github.com/zasle)
- All of my teammates and my adviser of Machine Learning named Liu Ding