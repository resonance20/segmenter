import numpy as np
import cv2
import tensorflow as tf

from layers import *

#Stacking two convolutional layers
def conv_module(img, fsize):
    conv1 = conv2d(img, fsize)
    conv2 = conv2d(conv1, fsize)
    return conv2

#Stacking three convolutional layers
def extended_conv_module(img, fsize):
    conv2 = conv_module(img, fsize)
    conv3 = conv2d(conv2, fsize)
    return conv3

#Application of Segnet from Cambridge Machine Intelligence Labortory
def segnet(k, img):
    
    #Encoding
    conv1 = conv_module(img, 3)
    down1 = maxpool2d(conv1)
    conv2 = conv_module(down1, 3)
    down2 = maxpool2d(conv2)
    conv3 = extended_conv_module(down2, 3)
    down3 = maxpool2d(conv3)
    conv4 = extended_conv_module(down3, 3)
    down4 = maxpool2d(conv4)
    conv5 = extended_conv_module(down4, 3)
    down5 = maxpool2d(conv5)

    #Decoding
    up6 = upconv2d(down5, 3)
    conv6 = extended_conv_module(up6, 3)
    up7 = upconv2d(conv6, 3)
    conv7 = extended_conv_module(up7, 3)
    up8 = upconv2d(conv7, 3)
    conv8 = extended_conv_module(up8, 3)
    up9 = upconv2d(conv8, 3)
    conv9 = conv_module(up9, 3)
    up10 = upconv2d(conv9, 3)
    conv10 = conv_module(up10, 3)

    #Output
    W = tf.Variable(tf.random_normal([1,1,3,k]))
    seg_im = tf.nn.conv2d(conv10, W, strides = [1,1,1,1], padding="SAME")
    return tf.nn.softmax(seg_im)