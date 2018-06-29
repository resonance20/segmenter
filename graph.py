import numpy as np
import tensorflow as tf

from layers import *
from Segnet import segnet

def graph(X, Y, net = segnet, cropsize = 224, k = 4):
    pair = tf.concat([X, Y], axis = 3)
    cropped_data = tf.random_crop(pair, [tf.shape(pair)[0], cropsize, cropsize, k+3])
    crop_x, crop_y = (cropped_data[:,:,:,0:3], cropped_data[:,:,:,3:])
    y_pred =  net(k, crop_x)
    loss = tf.losses.softmax_cross_entropy(onehot_labels = crop_y, logits = y_pred)
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.005)
    train_op = optimizer.minimize(loss, name="TrainOp")
    return train_op, loss