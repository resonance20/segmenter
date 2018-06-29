import numpy as np
import cv2
import tensorflow as tf

#Convolutional layer with batch norm and ReLU
def conv2d(x, fsize, ksize=3, strides=1):
    W = tf.Variable(tf.random_normal([ksize, ksize, x.get_shape().as_list()[3], fsize]))
    b = tf.Variable(tf.random_normal([fsize]))
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    x_norm = tf.contrib.layers.batch_norm(x)
    return tf.nn.relu(x_norm)

#Max pooling layer
def maxpool2d(x, k=2, indices=False):
    if indices:
        return tf.nn.max_pool_with_argmax(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')
    else:
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

#Up-/De-convolutional layer
def upconv2d(x, fsize, ksize=2, strides = 2):
    bsize = tf.shape(x)[0]
    channels = x.get_shape().as_list()[3]
    output = [tf.multiply(tf.shape(x)[1], strides), tf.multiply(tf.shape(x)[1], strides)]
    W = tf.Variable(tf.random_normal([ksize, ksize, fsize, channels]))
    b = tf.Variable(tf.random_normal([fsize]))
    x = tf.nn.conv2d_transpose(x, W, output_shape=[bsize, output[0], output[1], fsize], strides=[1, strides, strides, 1])
    x = tf.nn.bias_add(x, b)
    return x