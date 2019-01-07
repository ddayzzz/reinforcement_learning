# -*- coding: utf-8 -*-

import tensorflow as tf


class AccumTrainer(object):

    """
    定义一个训练器，用于显示
    """
    def __init__(self,
                 device="/cpu:0",
                 name="AccumTrainer"):
        self._name = name
        self._device = device

    def _create_accum_grad(self, var):
        """
        创建并返回一个累计梯度的变量
        :param var: 变量
        :return: 初始的累计梯度的变量
        """
        zero = tf.zeros(var.get_shape().as_list(), dtype=var.dtype)
        name = var.name.replace(":", "_") + "_accum_grad"
        accum_grad = tf.Variable(zero, name=name, trainable=False)
        return accum_grad

    def prepare_minimize(self, loss, var_list):
        """
        实现梯度下降
        :param loss: 损失函数
        :param var_list: 在损失函数中的求导公式中分母项（关于 var 的导）
        :return:
        """
        with tf.device(self._device):
            # 创建 var 的显式引用，再赋值的时候隐式调用，但我们的目的是求导
            # https://stackoverflow.com/questions/35175032/how-to-dereference-ref-tensor-type-in-tensorflow
            var_refs = [v._ref() for v in var_list]
            grads = tf.gradients(loss, var_refs)

            self._var_list = var_list
            self._grad_list = grads
            self._accum_grad_list = []  # 对每一个变量，都创建一个累计的梯度变量

            with tf.control_dependencies(None):
                for var in var_list:
                    accum_grad = self._create_accum_grad(var)
                    self._accum_grad_list.append(accum_grad)

    def get_accum_grad_list(self):
        return self._accum_grad_list

    def accumulate_gradients(self, name=None):
        """
        累计梯度
        :param name: 默认的名称
        :return: 返回一个累加 op
        """
        with tf.device(self._device):
            accumulate_ops = []

            with tf.name_scope(name, self._name, []) as name:
                for var, grad, accum_grad in zip(self._var_list, self._grad_list, self._accum_grad_list):
                    with tf.name_scope("accum_" + var.op.name):
                        accumulate_ops.append(tf.assign_add(accum_grad, grad))
                return tf.group(*accumulate_ops, name=name)

    def reset_gradients(self, name=None):
        """
        清空并返回处理后的op
        :param name:
        :return: 返回清空的 op
        """
        with tf.device(self._device):
            reset_ops = []

            with tf.name_scope(name, self._name, []) as name:
                for var, accum_grad in zip(self._var_list, self._accum_grad_list):
                    with tf.name_scope("reset_" + var.op.name):
                        zero = tf.zeros(accum_grad.get_shape())
                        reset = accum_grad.assign(zero)
                        reset_ops.append(reset)
                return tf.group(*reset_ops, name=name)
