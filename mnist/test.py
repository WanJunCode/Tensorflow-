import tensorflow as tf
import mnist.input_data as input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print(mnist)