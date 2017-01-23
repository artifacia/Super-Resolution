#!/usr/bin/python
#Training script for Image Super Resolution
import tensorflow as tf
import numpy as np
import cv2
from SRCNN import SRCNN
import os
import glob
import datetime

class SRCNN_Train:
    """Class file for training SRCNNs"""
    def __init__(self,input_path,nImgs,nEpochs,batch_size,print_every):
        self.input_path = input_path #Path where data and labels are stored in folders Data/ and Labels/
        self.nImgs = nImgs #Number of images to use for training
        self.nEpochs = nEpochs #Number of epochs to train for
        self.batch_size = batch_size
        self.print_every = print_every
        self.nIters = self.nImgs*self.nEpochs/self.batch_size
        self.data_files = glob.glob(self.input_path + '/Data/*.bmp')[0:nImgs]
        self.label_files = glob.glob(self.input_path + '/Labels/*.bmp')[0:nImgs]
        print "{0} files loaded".format(len(self.data_files))
        self.mean = 113.087154872
        self.stddev = 69.7176496121
        self.model = SRCNN()
        self.imgs = []
        self.target = []

    def read_batch(self):
        """ Function to read nImgs files from input_path
            Mean normalizes images and scales them by their standard deviation.
            Mean and stddev computed on full training set"""
        imgs = []
        labels = []
        idx = np.random.choice(self.nImgs,self.batch_size)
    	for i in idx:
            imgs.append(cv2.imread(self.data_files[i]))
    	    labels.append(cv2.imread(self.label_files[i]))
    	imgs,labels = np.array(imgs),np.array(labels)
        imgs = (imgs - self.mean)/self.stddev
    	labels = (labels - self.mean)/self.stddev
        return imgs,labels

    def train(self):
    	sess = tf.Session()
    	sess.run(tf.initialize_all_variables())
    	for i in range(self.nIters):
    		imgs_inp,imgs_lab = self.read_batch()
    		_,curr_loss = sess.run([self.model.optim,self.model.loss],feed_dict={self.model.X:imgs_inp,self.model.y:imgs_lab})
    		if(i%self.print_every==0):
    			print "Step {0} Training loss: {1}".format(i+1,curr_loss)
    	save = raw_input("Save the model?")
    	if(save=="y"):
    		now = datetime.datetime.now()
    		name = now.strftime("%Y-%m-%d_%H:%M")
    		saver = tf.train.Saver()
    		path = saver.save(sess,os.environ['HOME'] + '/' + name + '.ckpt')
    		print "Saved in " + path

if __name__=="__main__":
    train_model = SRCNN_Train('../Datasets/GenTrain',32,10,16,10)
    train_model.train()
