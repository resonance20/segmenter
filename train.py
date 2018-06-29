import numpy as np
import tensorflow as tf
import cv2
import os

from graph import graph
from helpers import *
from Segnet import segnet#Make sure to import the net you need!

#Shuffle and train images
def train(x, y, epochNo = 300, net = segnet, sess = tf.Session(), cropsize=224, k=4, bsize = 5):
    #Start TF graph
    X = tf.placeholder(tf.float32, (None,None,None,3), name="InputImg")
    Y = tf.placeholder(tf.float32, (None,None,None,k), name="Label")
    train_op, loss = graph(X, Y, net, cropsize, k)
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(0, epochNo):
        print("Fetching data...")
        #Shuffle input image list
        x_shuf = []
        y_shuf = []
        index_shuf = np.linspace(0, len(x) - 1, len(x))
        shuffle(index_shuf)
        for ind in index_shuf:
            x_shuf.append(x[int(ind)])
            y_shuf.append(y[int(ind)])
        #Load limited images and augment
        for im_set in range(0, len(x_shuf), 4):
            x_im = []
            y_im = []
            for im in range(im_set, im_set+4):
                if im>=len(x_shuf):
                    break
                img = cv2.imread(x_shuf[im], cv2.IMREAD_COLOR)
                lab = cv2.imread(y_shuf[im], cv2.IMREAD_GRAYSCALE)
                x_im.append(img)
                y_im.append(logitise(lab, k))
            #Batch and train
            if not x_im:
                break
            print("Augmenting...")
            aug_x, aug_y = augment_data(x_im, y_im)
            train_batch, lab_batch = prepare_batches(aug_x, aug_y, bsize)
            print("Optimizing...")
            for i,_ in enumerate(train_batch):
                _, epoch_loss = sess.run([train_op,loss], feed_dict={X: train_batch[i], Y:lab_batch[i]})
        print("Epoch "+str(epoch + 1)+" complete! Loss is "+str(epoch_loss))
    print("Optimization complete after "+str(epoch + 1)+" epochs!")

    #Save session
    print("Saving session...")
    tf.train.Saver().save(sess, os.getcwd() + "/models/trained_net")
    print("Done")


imdir = "C:/Users/brain/Box Sync/Postgrad notes/Core/Master Thesis/colorCheckerPictures/train_data/"
list_files = os.listdir(imdir)
x = []
y = []
for file in list_files:
    if file.endswith(".jpg"):
        x.append(imdir + file)
        y.append(imdir+"gt_"+file[0:len(file)-3]+"png")
"""
x = ["Foundbox.jpg"]
y = ["Foundbox_gt.png"]
"""
train(x, y)
