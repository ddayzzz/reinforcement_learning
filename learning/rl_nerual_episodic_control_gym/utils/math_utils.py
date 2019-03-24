import torch
import scipy.signal as signal


def discount(x, gamma):
    """
    计算的折扣后的未来的回报
    :param x:
    :param gamma:
    :return:
    """
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def inverse_distance(h, h_i, delta=1e-3):
    """
    For the kernel function
    we chose a function that interpolates between the mean
    for short distances and weighted inverse distance for large
    distances, more precisely:
    :param h:
    :param h_i:
    :param delta: 计算 h 和 h_i
    :return:
    """
    return 1 / torch.dist(h, h_i) + delta