#!/usr/bin/python
#SRCNN model building script
import tensorflow as tf
import numpy as np
class SRCNN:
    def __init__(self,input_size = 33,label_size = 21,learning_rate = 1e-3,verbose = True):
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.label_size = label_size
        self.verbose = verbose
        self.parameters = []
        self.create_placeholders()
        self.build_model()
        self.create_loss()
        self.print_vars()

    def print_vars(self):
        for p in self.parameters:
            print p.name,p.get_shape()

    def bias_variable(self,shape):
        """ Function to create a bias variable of the given shape. Bias is initialized to a constant value of 0.1.
            Input:
                shape: Shape of the required bias variable
            Returns:
                bias: Required bias variable"""
        bias = tf.get_variable(dtype=tf.float32,initializer=tf.constant_initializer(0.1),shape=shape,name='bias')
        return bias

    def weight_variable(self,shape):
        """ Function to create a weight(conv filter) of the specified shape.
            Filter is initialized using tf.truncated_normal with zero mean
            and 0.001 standard deviation.
            Input:
                shape: Shape of the required weight variable
            Returns:
                W: Weight of the required shape"""
        W = tf.get_variable(dtype=tf.float32,initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.001),shape=shape,name='filter')
        return W

    def conv2d(self,x,W,padding='VALID',strides=[1,1,1,1]):
        """ Function to perform 2d convolution of a given input using a given kernel.
            Input:
                x: Input over which convolution is performed. Shape: [batch_size,h,w,nChannels]
                W: Kernel which is convolved over x
                padding: Specifies how x is to be padded before convolution(default is VALID padding)
                strides: Specifies stride on each dimension of x for convolution(default is a stride of 1)
            Returns:
                out: Result of 2d convolution of W over x"""
        out = tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='VALID',name='conv')
        return out

    def create_placeholders(self):
        """ Function to create placeholders for input and labels.
            X: placeholder for input of type float32 with variable batch size.
            y: placeholder for labels of type float32 with variable batch size."""
        self.X = tf.placeholder(dtype=tf.float32,shape=[None,self.input_size,self.input_size,3])
        self.y = tf.placeholder(dtype=tf.float32,shape=[None,self.label_size,self.label_size,3])
        if self.verbose:
            print "Created placeholders for input and labels"

    def build_model(self):
        """ Function to build the graph for our model(SRCNN)."""
        with tf.variable_scope('conv1'):
            W_conv1 = self.weight_variable([9,9,3,64])
            b_conv1 = self.bias_variable([64])
            f_conv1 = tf.nn.relu(self.conv2d(self.X,W_conv1) + b_conv1)
            self.parameters += [W_conv1,b_conv1]
        with tf.variable_scope('conv2'):
            W_conv2 = self.weight_variable([1,1,64,32])
            b_conv2 = self.bias_variable([32])
            f_conv2 = tf.nn.relu(self.conv2d(f_conv1,W_conv2) + b_conv2)
            self.parameters += [W_conv2,b_conv2]
        with tf.variable_scope('conv3'):
            W_conv3 = self.weight_variable([5,5,32,3])
            b_conv3 = self.bias_variable([3])
            self.f_out = self.conv2d(f_conv2,W_conv3) + b_conv3
            self.parameters += [W_conv3,b_conv3]

    def create_loss(self):
        """ Function to create loss op for the model.
            The model currently uses an l2 loss."""
        self.loss = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(self.f_out,self.y))))
        if self.verbose:
            print "Loss op created"

    def create_optim(self):
        """ Function to create optimizer op for the model.
            We use Adam optimizer with a learning rate of 1e-3"""
        self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        if self.verbose:
            print "Optimizer op created"

if __name__=="__main__":
    src = SRCNN()
