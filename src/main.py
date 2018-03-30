from mnist_learn import load_data
from mnist_learn import train
from mnist_learn import test


if __name__ == '__main__':
    # 加载训练数据
    mnist = load_data.load_data()
    # 在训练数据集上训练
    train.train(mnist)
    # 读取训练结果，在测试数据集上测试
    test.test(mnist)