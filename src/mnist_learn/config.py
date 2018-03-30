import os

input_node = 28 * 28
output_node = 10

layer_node = 500
batch_size = 100

learning_rate_base = 0.8
learning_rate_decay = 0.99

regularization_rate = 0.0001
training_step = 10000
moving_average_decay = 0.99

store_path = os.path.join(os.getcwd(), "model")