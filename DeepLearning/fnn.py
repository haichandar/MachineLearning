# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 17:07:43 2018

@author: Chandar_S
"""

#import numpy as np
import tensorflow as tf
import datetime
from BaseNN import BaseNNAbstract

class fnn(BaseNNAbstract):

    x = tf.placeholder("float", [None, None], name="x")
    y = tf.placeholder("float", [None, None], name="labels")
    keep_prob = tf.placeholder("float", name="keep_probability")
    learning_rate_var = 0.001

    data_path = None
    
    def __init__(self, path):
        self.data_path = path
        self.logs_path = path + 'Logs\\fnn_' + datetime.datetime.now().strftime("%d%b-%H.%M")
        tf.set_random_seed(self.random_state)

        
    ### RUN THE FNN TRAINING FOR INPUT DATA AND HIDDEN NODE SIZE AND GET TRAINED MODEL OUTPUT
    def create_model(self, input_x, input_features, n_hidden_1, n_hidden_2, output_classes, single_layer_fnn= False):
                
        ## BUILD NN STRUCTURE
        def multilayer_perceptron(input_data):
            with tf.name_scope('fnn'):
                weights = {
                    'h1': tf.Variable(tf.random_normal([input_features, n_hidden_1]), name='FNN_Layer1_Weight'),
                    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name='FNN_Layer2_Weight'),
                    'out': tf.Variable(tf.random_normal([n_hidden_2, output_classes]), name='FNN_Output_Weight')
                }
                
                biases = {
                    'b1': tf.Variable(tf.random_normal([n_hidden_1]), name='FNN_Layer1_Bias'),
                    'b2': tf.Variable(tf.random_normal([n_hidden_2]), name='FNN_Layer2_Bias'),
                    'out': tf.Variable(tf.random_normal([output_classes]), name='FNN_Output_Bias')
                }
            
                with tf.name_scope("fnn_Layer1"):
                    layer_1 = tf.add(tf.matmul(input_data, weights['h1']), biases['b1'])
                    layer_1 = tf.nn.relu(layer_1)
                    layer_1 = tf.nn.dropout(layer_1, self.keep_prob)
                with tf.name_scope("fnn_Layer2"):
                    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
                    layer_2 = tf.nn.relu(layer_2)
                    layer_2 = tf.nn.dropout(layer_2, self.keep_prob)
                with tf.name_scope("fnn_Out"):
                    out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])
            
            return out_layer

        def singlelayer_perceptron(input_data):
            with tf.name_scope('fnn'):
                weight = tf.Variable(tf.random_normal([input_features, output_classes]), name='FNN_Output_Weight')
                out = tf.Variable(tf.random_normal([output_classes]), name='FNN_Output_Bias')

                with tf.name_scope("fnn_Out"):
                    out_layer = tf.add(tf.matmul(tf.reduce_mean(input_data, [1,2]), weight), out)
            
            return out_layer
        
        if single_layer_fnn:
            model = singlelayer_perceptron(input_x)
        else:
            model = multilayer_perceptron(input_x)

        
        with tf.name_scope('xent'):
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=self.y))
        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_var).minimize(cost)
        with tf.name_scope('Accuracy'):
            # Accuracy
            acc = tf.equal(tf.argmax(model, 1), tf.argmax(self.y, 1))
            acc = tf.reduce_mean(tf.cast(acc, tf.float32))*100
             
        return optimizer, cost, acc, model

    def create_simplified_model(self, input_x, input_features, n_hidden_1, n_hidden_2, output_classes, single_layer_fnn= False):
        
        with tf.variable_scope('fnn', reuse=tf.AUTO_REUSE):
            if single_layer_fnn:
                input_x = tf.reduce_mean(input_x, [1,2])
                # define the only connected layer
                model = tf.layers.dense(inputs=input_x, units=output_classes, kernel_initializer=tf.initializers.lecun_normal(), use_bias=True, bias_initializer=tf.zeros_initializer(), name="dense_layer") 
            else:
                input_x.set_shape([None, input_features])
                # pass flattened input into the first fully connected layer
                fc1 = tf.layers.dense(inputs=input_x, units=n_hidden_1, activation=tf.nn.relu, kernel_initializer=tf.initializers.lecun_normal(), use_bias=True, bias_initializer=tf.zeros_initializer(), name="fnn_Layer1")
                dropout1 = tf.layers.dropout(fc1, rate=1-self.keep_prob)
                # pass input into the second fully connected layer
                fc2 = tf.layers.dense(inputs=dropout1, units=n_hidden_2, activation=tf.nn.relu, kernel_initializer=tf.initializers.lecun_normal(), use_bias=True, bias_initializer=tf.zeros_initializer(), name="fnn_Layer2")
                dropout2 = tf.layers.dropout(fc2, rate=1-self.keep_prob)
                # define third fully connected layer
                model = tf.layers.dense(inputs=dropout2, units=output_classes, kernel_initializer=tf.initializers.lecun_normal(), use_bias=True, bias_initializer=tf.zeros_initializer(), name="fnn_Out") 
    
        with tf.variable_scope('xent', reuse=tf.AUTO_REUSE):
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=self.y))
        with tf.variable_scope('train', reuse=tf.AUTO_REUSE):
            if single_layer_fnn:
                optimizer = tf.train.MomentumOptimizer(self.learning_rate_var, 0.9, use_nesterov=True).minimize(cost)
            else:
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_var).minimize(cost)
        with tf.variable_scope('Accuracy', reuse=tf.AUTO_REUSE):
            # Accuracy
            acc = tf.equal(tf.argmax(model, 1), tf.argmax(self.y, 1))
            acc = tf.reduce_mean(tf.cast(acc, tf.float32))*100
        
        self.fnn_outlayer = model
     
        return optimizer, cost, acc, model
        ############ END: BUILD THE DNN CODE ################
