import numpy as np
import cv2
import tensorflow as tf

from layers import *

#Stacking two convolutional layers
def conv_module(img, fsize):
    conv1 = conv2d(img, fsize)
    conv2 = conv2d(conv1, fsize)
    return conv2

#Application of U-Net from Ronnenberger et.al.
def u_net(img):

    #Descent
    mod1 = conv_module(img, 64)
    img2 = maxpool2d(mod1)
    mod2 = conv_module(img2, 128)
    img3 = maxpool2d(mod2)
    mod3 = conv_module(img3, 256)
    img4 = maxpool2d(mod3)
    mod4 = conv_module(img4, 512)
    img5 = maxpool2d(mod4)
    mod5 = conv_module(img5, 1024)

    #Ascent
    up6 = upconv2d(mod5, fsize = 512)
    img6 = tf.concat(values=[up6, mod4], axis = 3)
    mod6 = conv_module(img6, 512)
    up7 = upconv2d(mod6, fsize = 256)
    img7 = tf.concat(values=[up7, mod3], axis = 3)
    mod7 = conv_module(img7, 256)
    up8 = upconv2d(mod7, fsize = 128)
    img8 = tf.concat(values=[up8, mod2], axis = 3)
    mod8 = conv_module(img8, 128)
    up9 = upconv2d(mod8, fsize = 64)
    img9 = tf.concat(values=[up9, mod1], axis = 3)
    mod9 = conv_module(img9, 64)

    #Final layer
    W = tf.Variable(tf.random_normal([1,1,64,k]))
    seg_im = tf.nn.conv2d(mod9, W ,strides = [1,1,1,1], padding="SAME")
    output = tf.nn.softmax(seg_im, name = "OutputLayer")
    return output