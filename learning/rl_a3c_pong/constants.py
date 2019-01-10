# -*- coding: utf-8 -*-
import multiprocessing


LOCAL_T_MAX = 20  # 定义 local AC 网络进行采样的最大的时间步
RMSP_ALPHA = 0.99  # RMSProp 的损失 公式（S2）
RMSP_EPSILON = 0.1  # RMSProp epsilon 公式 （S3）
CHECKPOINT_DIR = 'checkpoints'
LOG_FILE = 'logs'
# 学习率的选择，[LOW, HIGH]
INITIAL_ALPHA_LOW = 1e-4
INITIAL_ALPHA_HIGH = 1e-2

PARALLEL_SIZE = multiprocessing.cpu_count()  # local AC 的数量
ACTION_SIZE = 3  # 状态空间，这里把 gym 的 6个动作映射到 3个

INITIAL_ALPHA_LOG_RATE = 0.4226  # 计算初始学习学习率的参数
GAMMA = 0.99  # discounted 的回报
ENTROPY_BETA = 0.01  # 熵正则化的参数
MAX_TIME_STEP = 10 * 10 ** 7  # global AC 的 时间步
GRAD_NORM_CLIP = 40.0  # 梯度截取的参数
USE_GPU = True  # 是否使用 GPU
