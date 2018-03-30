import os

from tensorflow.examples.tutorials.mnist import input_data

def load_data():
    data_path = os.path.join(os.getcwd(), "train_data")
    mnist = input_data.read_data_sets(data_path, one_hot=True)

    print("train data size:%d" % mnist.train.num_examples)
    print("validation data size:%d" % mnist.validation.num_examples)
    print("test data size:%d" % mnist.test.num_examples)
    print("train image:", mnist.train.images.shape)
    print("train labels:", mnist.train.labels.shape)
    return mnist