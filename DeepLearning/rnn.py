# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 12:39:52 2018

@author: Chandar_S
"""

import tensorflow as tf
import datetime
from BaseNN import BaseNNAbstract

class rnn(BaseNNAbstract):

    # tf Graph input
    x = tf.placeholder("float", [None, None, None], name="x")
    y = tf.placeholder("float", [None, None], name="labels")
    keep_prob = tf.placeholder("float", name="keep_probability")
    learning_rate_var = 0.001

    data_path = None
    
    def __init__(self, path):
        self.data_path = path
        self.logs_path = path+ 'Logs\\rnn_'+ datetime.datetime.now().strftime("%d%b-%H.%M")
        tf.set_random_seed(self.random_state)

    def create_model(self, num_input, timesteps, num_hidden, num_classes):        

        with tf.name_scope('rnn'):
            self.x = tf.placeholder("float", [None, timesteps, num_input], name="x")

            # Define weights
            weights = {
                # Hidden layer weights => 2*n_hidden because of forward + backward cells
                'out': tf.Variable(tf.random_normal([2*num_hidden, num_classes]), name='RNN_Out_Weight')
            }
            biases = {
                'out': tf.Variable(tf.random_normal([num_classes]), name='RNN_Out_Bias')
            }
    
            # Prepare data shape to match `rnn` function requirements
            # Current data input shape: (batch_size, timesteps, n_input)
            # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)
                    
            # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
            x1 = tf.unstack(self.x, timesteps, 1)
        
            # Define a lstm cell with tensorflow
            #    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_hidden, forget_bias=1.0)
        
#            fw_cell = tf.keras.layers.SimpleRNNCell(num_hidden)
            fw_cell = tf.nn.rnn_cell.LSTMCell(num_hidden, forget_bias=1.0)
            # Backward direction cell
#            bw_cell = tf.keras.layers.SimpleRNNCell(num_hidden)
            bw_cell = tf.nn.rnn_cell.LSTMCell(num_hidden, forget_bias=1.0)
            #    # Get lstm cell output
            #    outputs, states = tf.nn.static_rnn(lstm_cell, x, dtype=tf.float32)
        
            # Get lstm cell output
            outputs, _, _ =  tf.nn.static_bidirectional_rnn(fw_cell, bw_cell, x1,
                                                  dtype=tf.float32)
            
            model = tf.matmul(outputs[-1], weights['out']) + biases['out']

        with tf.name_scope('xent'):
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=self.y))
        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_var).minimize(cost)
        with tf.name_scope('Accuracy'):
            # Accuracy
            acc = tf.equal(tf.argmax(model, 1), tf.argmax(self.y, 1))
            acc = tf.reduce_mean(tf.cast(acc, tf.float32))*100
     
        return optimizer, cost, acc, model
    