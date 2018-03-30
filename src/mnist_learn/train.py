import os

import tensorflow as tf

from . import gen_dnn
from . import config


def train(mnist):
    # 输入的占位符
    x = tf.placeholder(tf.float32, [None, config.input_node], name='x-input')
    y = tf.placeholder(tf.float32, [None, config.output_node], name='y-input')
    print(x)
    global_step = tf.Variable(0, trainable=False)

    # 网络的隐藏层和输出层参数
    w1 = tf.Variable(tf.truncated_normal([config.input_node, config.layer_node], seed=0.1))
    b1 = tf.Variable(tf.constant(0.1, shape=[config.layer_node]))

    w2 = tf.Variable(tf.truncated_normal([config.layer_node, config.output_node], seed=0.1))
    b2 = tf.Variable(tf.constant(0.1, shape=[config.output_node]))

    # 尝试使用滑动平均模型，但实际使用中出了问题，所以实际并未使用
    ema = tf.train.ExponentialMovingAverage(config.moving_average_decay, global_step)
    ema_op = ema.apply(tf.trainable_variables())

    # 构建神经网络
    predict_y, l2_list = gen_dnn.gen_dnn(x, ema, w1, b1, w2, b2)
    print("build nn success")

    # 总的损失：交叉熵的损失 + L2泛函损失
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=predict_y, labels=y)
    loss = tf.reduce_mean(cross_entropy)

    for l2_w in l2_list:
        loss = loss + l2_w

    # 随着迭代次数减小的学习率
    learning_rate = tf.train.exponential_decay(config.learning_rate_base,
                                               global_step,
                                               mnist.train.num_examples / config.batch_size,
                                               config.learning_rate_decay)

    # 基本梯度下降优化器
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    # 计算验证集的准确率
    correct_predict = tf.equal(tf.argmax(predict_y, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

    # 仅保存最后3个模型
    saver = tf.train.Saver(max_to_keep=3)

    with tf.Session() as sess:
        # 初始化参数 & 读取验证数据集
        tf.global_variables_initializer().run()
        validate_feed = {x: mnist.validation.images,
                         y: mnist.validation.labels}

        for i in range(config.training_step):
            if i % 200 == 0:
                # 每200次跑一下在验证集上的准确率
                predict_y_ot, validate_acc, loss_ot, global_step_ot = sess.run([tf.argmax(predict_y, 1), accuracy, loss, global_step], feed_dict=validate_feed)
                print("after %d training step(s), validation accuracy: %f loss %f" % (i, validate_acc, loss_ot))
            if i % 1000 == 0:
                # 每1000次保存一下模型
                saver.save(sess, os.path.join(config.store_path, "mnist.ckpt"), global_step=global_step)

            # 按批次获取训练数据，然后启动一批数据的训练计算
            xs, ys = mnist.train.next_batch(config.batch_size)
            sess.run(train_op, feed_dict={x: xs, y: ys})