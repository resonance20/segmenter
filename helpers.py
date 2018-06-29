import numpy as np
import cv2
from random import shuffle

#Create batches from images
def prepare_batches(x, y, bsize):
    batches = int(len(x)/bsize)
    join = list(zip(x, y))
    shuffle(join)
    x, y = zip(*join)
    x = np.stack(x)
    y = np.stack(y)
    train_batch = []
    lab_batch = []
    for i in range(0, batches):
        train_batch.append(x[bsize*i:bsize*i+bsize,:,:,:])
        lab_batch.append(y[bsize*i:bsize*i+bsize,:,:,:])
    return train_batch, lab_batch

#Convert labels to logits
def logitise(img, k):
    m = img
    rows, cols = np.shape(img)
    logits = np.zeros([rows,cols,k])
    for i in range(0, rows):
        for j in range(0, cols):
            val = img[i,j]/255
            index = int(round(val*(k-1)))
            logits[i,j,index] = 1     
    return logits

#Augment data by 15x
def augment_data(x, y):
    augmented_data = []
    augmented_labels = []
    for i in range(0, len(x)):
        im = x[i]
        lab = y[i]
        for img in (im,lab):
            shape = img.shape
            rows = shape[0]
            cols = shape[1]
            transforms = []
            #Add flips
            transforms.append(img)
            transforms.append(cv2.flip(img, 0))
            transforms.append(cv2.flip(img, 1))
            #Add rotations
            m1 = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
            m2 = cv2.getRotationMatrix2D((cols/2,rows/2),180,1)
            m3 = cv2.getRotationMatrix2D((cols/2,rows/2),270,1)
            transforms.append(cv2.warpAffine(img, m1, (rows,cols)))
            transforms.append(cv2.warpAffine(img, m2, (rows,cols)))
            transforms.append(cv2.warpAffine(img, m3, (rows,cols)))
            #Add translations
            dist_r = int(rows/10)
            dist_c = int(cols/10)
            t1 = np.float32([[1,0,dist_r],[0,1,dist_c]])
            t2 = np.float32([[1,0, -dist_r],[0,1,dist_c]])
            t3 = np.float32([[1,0,dist_r],[0,1, -dist_c]])
            t4 = np.float32([[1,0, -dist_r],[0,1, -dist_c]])
            transforms.append(cv2.warpAffine(img, t1, (rows,cols)))
            transforms.append(cv2.warpAffine(img, t2, (rows,cols)))
            transforms.append(cv2.warpAffine(img, t3, (rows,cols)))
            transforms.append(cv2.warpAffine(img, t4, (rows,cols)))
            #Add scaling
            scout1 = cv2.resize(img, None, fx=1.5, fy=1.5)
            scout2 = cv2.resize(img, None, fx=2, fy=2)
            transforms.append(scout1[int(rows/4):int(5*rows/4), int(cols/4):int(5*cols/4)])
            transforms.append(scout2[int(rows/2):int(3*rows/2), int(cols/2):int(3*cols/2)])
            if img.shape[2] is 3:
            #Add gaussian and salt and pepper noise
                noise1 = cv2.randn(np.zeros(img.shape), 0, 0.05)
                noise2 = cv2.randn(np.zeros(img.shape), 0, 0.1)
                spnoise = cv2.randu(np.zeros(img.shape), 0, 255)
                noise3 = img.copy()
                noise3[spnoise>225] = 255
                noise3[spnoise<30] = 0
                transforms.append(img + noise1)
                transforms.append(img + noise2)
                transforms.append(noise3)
                augmented_data.extend(transforms)
            else:
                transforms.append(img)
                transforms.append(img)
                transforms.append(img)
                augmented_labels.extend(transforms)
    return augmented_data, augmented_labels