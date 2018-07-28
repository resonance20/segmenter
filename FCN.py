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

#Application of FCN from UC Berkeley
def fcn(k, img):
    
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
    conv6 = conv_module(down5, 3)

    #FCN8
    up1 = upconv2d(conv6, 4)
    up2 = upconv2d(down4, 2)
    fuse1 = down3 + up1 + up2
    fcn8 = upconv2d(fuse1, 8)

    #FCN16
    up3 = upconv2d(conv6, 2)
    fuse2 = down4 + up3
    fcn16 = upconv2d(fuse2, 16)

    #FCN32
    fcn32 = upconv2d(conv6, 32)

    #Output
    W = tf.Variable(tf.random_normal([1,1,3,k]))
    seg_im = tf.nn.conv2d(fcn8, W, strides = [1,1,1,1], padding="SAME")
    return tf.nn.softmax(seg_im)