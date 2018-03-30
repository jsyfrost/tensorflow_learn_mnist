import tensorflow as tf

from . import config


def gen_dnn(input_tensor, ema, w1, b1, w2, b2):
    """
    一个隐藏层的dnn，最简单的全连接，ReLU函数处理
    :param input_tensor:
    :param ema:
    :return: 返回预测结果和正则L2泛函
    """

    # layer1 = tf.nn.relu(tf.matmul(input_tensor, ema.average(w1)) + ema.average(b1))
    # predict_y = tf.matmul(layer1, ema.average(w2)) + ema.average(b2)
    layer1 = tf.nn.relu(tf.matmul(input_tensor, w1) + b1)
    predict_y = tf.matmul(layer1, w2) + b2
    print(predict_y)
    regularizer = tf.contrib.layers.l2_regularizer(config.regularization_rate)

    return predict_y, [regularizer(w1), regularizer(w2)]