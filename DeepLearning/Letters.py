# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 16:03:41 2019

@author: Chandar_S
"""

import tensorflow as tf
from nn_utilities_py import nn_utilities
import numpy as np
import datetime
import os
from tqdm import tqdm
from tensorflow.python.ops import control_flow_ops

batch_pointer = 0
random_state = 650
#data_path = 'E:\MLData\\'
data_path = ''
logs_path = os.path.abspath(data_path + 'Logs\\AlphaDigits\\' + datetime.datetime.now().strftime("%d%b-%H.%M"))
np.random.seed(random_state)
tf.set_random_seed(random_state)

# Load the input data
input_data = nn_utilities(data_path).load_emnist_alphadigit_data_google_collab()
x_train_input = input_data["x_train"]
print (f"Number of training records {x_train_input.shape[0]}")
print (f"Shape {x_train_input.shape}")
y_train_input = input_data["y_train"]
x_validation = input_data["x_validation"]
print (f"Number of validation records {x_validation.shape[0]}")
y_validation = input_data["y_validation"]


def getTrainBatch(x_train, y_train, batch_size, batch_pointer):
    inds = np.arange(batch_pointer, batch_pointer + batch_size)
    inds = np.mod( inds , x_train.shape[0] ) #cycle through dataset
    batch_pointer += batch_size #increment counter before returning
    batch = (x_train[inds], y_train[inds], batch_pointer) #grab batch
    return batch

def conv_batch_norm(x, n_out, phase_train):
    beta_init = tf.constant_initializer(value = 0.0,
                                        dtype=tf.float32)
    gamma_init = tf.constant_initializer(value = 1.0,
                                         dtype=tf.float32)
    
    beta = tf.get_variable("beta", [n_out], initializer=beta_init)
    gamma = tf.get_variable("gamma", [n_out], initializer=gamma_init)
    
    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    ema_apply_op = ema.apply([batch_mean, batch_var])
    ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)
    
    def mean_var_with_update():
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    
    mean, var = control_flow_ops.cond(phase_train, mean_var_with_update, 
                                      lambda: (ema_mean, ema_var))
    
    normed = tf.nn.batch_norm_with_global_normalization(x, mean, var, beta, 
                                                        gamma, 1e-3, True)
    return normed

def fc_batch_norm(x, n_out, phase_train):
    beta_init = tf.constant_initializer(value=0.0,
                                        dtype=tf.float32)
    
    gamma_init = tf.constant_initializer(value=1.0,
                                        dtype=tf.float32)
    
    beta = tf.get_variable("beta", [n_out], initializer=beta_init)
    gamma = tf.get_variable("gamma", [n_out], initializer=gamma_init)
    
    batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    ema_apply_op = ema.apply([batch_mean, batch_var])
    ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)

    def mean_var_with_update():
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = control_flow_ops.cond(phase_train, mean_var_with_update, 
                                      lambda: (ema_mean, ema_var))
    x_r = tf.reshape(x, [-1, 1, 1, n_out])
    normed = tf.nn.batch_norm_with_global_normalization(x_r, mean, var, beta, 
                                                        gamma, 1e-3, True)
 
    return tf.reshape(normed, [-1, n_out])

''' BUILD THE MODEL '''

# INPUT SHAPE OF -1, 28, 28, 1
# single channel image
cnn_layer1_input_channels = 1
cnn_layer1_filter_count = 32
cnn_layer1_filter_shape = [5, 5]
#cnn_layer1_pooling_shape = [2, 2]
#cnn_layer1_pooling_stride = [2, 2]

# layer 1.5
cnn_layer15_filter_count = 64
cnn_layer15_filter_shape = [5, 5]
cnn_layer15_pooling_shape = [2, 2]
cnn_layer15_pooling_stride = [2, 2]

# layer 2
cnn_layer2_filter_count = 128
cnn_layer2_filter_shape = [5, 5]

# layer 2.5
cnn_layer25_filter_count = 256
cnn_layer25_filter_shape = [5, 5]
cnn_layer25_pooling_shape = [2, 2]
cnn_layer25_pooling_stride = [2, 2]


# will be 1 as we are using the batch normalization
keep_prob_val = 1
n_hidden_1 = 1500
n_hidden_2 = 500
output_classes = y_train_input.shape[1]

x = tf.placeholder("float", [None, None], name="input")
y = tf.placeholder("float", [None, output_classes], name="labels")
keep_prob = tf.placeholder("float", name="keep_probability")
phase_train = tf.placeholder(tf.bool)
learning_rate_var = 0.001

def adjustImage(frame):
    frame1 = tf.image.per_image_standardization(frame)
    frame2 = tf.image.random_brightnes(frame1, 50, seed = tf.set_random_seed(random_state))
    return frame2

# if the input image was flattenned, make it to 4 D image
x_shaped = tf.reshape(x, [-1, int(x_train_input.shape[1] ** 0.5), int(x_train_input.shape[1] ** 0.5), cnn_layer1_input_channels])
  
x_adjusted_image = tf.map_fn(lambda frame: adjustImage, x_shaped)

with tf.variable_scope('cnn10', reuse=tf.AUTO_REUSE):
   # first convolutional layer
    conv1 = tf.layers.conv2d(inputs=x_adjusted_image, filters=cnn_layer1_filter_count, 
                             kernel_size=cnn_layer1_filter_shape, 
                             kernel_initializer=tf.initializers.lecun_normal(), 
                             use_bias=True, bias_initializer=tf.zeros_initializer(), 
                             padding = 'SAME',
                             activation=None,
                             name='CNN_Layer1')
    
    conv1_relu = tf.nn.relu(conv_batch_norm(conv1, cnn_layer1_filter_count, 
                                            phase_train))

with tf.variable_scope('cnn15', reuse=tf.AUTO_REUSE):

    conv15 = tf.layers.conv2d(inputs=conv1_relu, filters=cnn_layer15_filter_count, 
                             kernel_size=cnn_layer15_filter_shape, 
                             kernel_initializer=tf.initializers.lecun_normal(), 
                             use_bias=True, bias_initializer=tf.zeros_initializer(), 
                             padding = 'SAME',
                             activation=None,
                             name='CNN_Layer15')
    

    conv15_relu = tf.nn.relu(conv_batch_norm(conv15, cnn_layer15_filter_count, 
                                            phase_train))
    # first pooling layer
    conv15_pool = tf.layers.max_pooling2d(inputs=conv15_relu, 
                                    pool_size=cnn_layer15_pooling_shape, 
                                    strides=cnn_layer15_pooling_stride, 
                                    name='CNN_Layer15_pooling')

with tf.variable_scope('cnn20', reuse=tf.AUTO_REUSE):
    # second convolutional layer
    conv2 = tf.layers.conv2d(inputs=conv15_pool, 
                             filters=cnn_layer2_filter_count, 
                             kernel_size=cnn_layer2_filter_shape, 
                             kernel_initializer=tf.initializers.lecun_normal(), 
                             use_bias=True,
                             bias_initializer=tf.zeros_initializer(), 
                             padding = 'SAME',
                             activation=None,
                             name='CNN_Layer2')

    conv2_relu = tf.nn.relu(conv_batch_norm(conv2, cnn_layer2_filter_count, 
                                            phase_train))

with tf.variable_scope('cnn25', reuse=tf.AUTO_REUSE):

    conv25 = tf.layers.conv2d(inputs=conv2_relu, filters=cnn_layer25_filter_count, 
                             kernel_size=cnn_layer25_filter_shape, 
                             kernel_initializer=tf.initializers.lecun_normal(), 
                             use_bias=True, bias_initializer=tf.zeros_initializer(), 
                             padding = 'SAME',
                             activation=None,
                             name='CNN_Layer25')

    conv25_relu = tf.nn.relu(conv_batch_norm(conv25, cnn_layer25_filter_count, 
                                            phase_train))
    # second pooling layer
    cnn_output = tf.layers.max_pooling2d(inputs=conv25_relu, 
                                    pool_size=cnn_layer25_pooling_shape, 
                                    strides=cnn_layer25_pooling_stride, 
                                    name='CNN_Layer25_pooling')        


with tf.variable_scope('fnn10', reuse=tf.AUTO_REUSE):
    # pass flattened input into the first fully connected layer
    fc1 = tf.layers.dense(inputs=tf.layers.flatten(cnn_output), 
                          units=n_hidden_1, activation=None, 
                          kernel_initializer=tf.initializers.lecun_normal(), 
                          use_bias=True, bias_initializer=tf.zeros_initializer(), 
                          name="fnn_Layer1")
    
    fc1_relu = tf.nn.relu(fc_batch_norm(fc1, n_hidden_1, 
                                            phase_train))
    
    fc1_dropout = tf.layers.dropout(fc1_relu, rate= 1 - keep_prob)

with tf.variable_scope('fnn20', reuse=tf.AUTO_REUSE):
    # pass input into the second fully connected layer
    fc2 = tf.layers.dense(inputs=fc1_dropout, units=n_hidden_2, 
                          activation=tf.nn.relu, 
                          kernel_initializer=tf.initializers.lecun_normal(), 
                          use_bias=True, bias_initializer=tf.zeros_initializer(), 
                          name="fnn_Layer2")

    fc2_relu = tf.nn.relu(fc_batch_norm(fc2, n_hidden_2, 
                                            phase_train))

    fc2_dropout = tf.layers.dropout(fc2_relu, rate= 1 - keep_prob)

    # define third fully connected layer
    model = tf.layers.dense(inputs=fc2_dropout, 
                            units=output_classes, 
                            kernel_initializer=tf.initializers.lecun_normal(), 
                            use_bias=True, bias_initializer=tf.zeros_initializer(), 
                            name="fnn_Out") 

       
with tf.variable_scope('xent', reuse=tf.AUTO_REUSE):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=y))
with tf.variable_scope('train', reuse=tf.AUTO_REUSE):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_var).minimize(cost)
with tf.variable_scope('Accuracy', reuse=tf.AUTO_REUSE):
    # Accuracy
    acc = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(acc, tf.float32))*100
    

''' TRAIN THE MODEL '''
batch_size = 120
training_epochs = 5
display_step = 1000
run_validation_accuracy = True

summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
# Create a summary to monitor cost tensor
tf.summary.scalar("Model_Loss", cost)
# Create a summary to monitor accuracy tensor
tf.summary.scalar("Training_Accuracy", acc)
# Add input images
#            tf.summary.image('input',x_train_input_4D,max_outputs=10)

# Create summaries to visualize weights
for var in tf.trainable_variables():
    tf.summary.histogram(var.name, var)

# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

# Create a summary to monitor accuracy tensor
validation_acc_summary = tf.summary.scalar('Validation_Accuracy', acc)  # intended to run on validation set


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    batch_steps = int(len(x_train_input) / batch_size)
    x_batches = np.array_split(x_train_input, batch_steps)
    y_batches = np.array_split(y_train_input, batch_steps)
                
    for epoch in tqdm(range(training_epochs)):
        avg_cost = 0.0
    
        (batch_x, batch_y, batch_pointer) = getTrainBatch(x_train_input, y_train_input, batch_steps, batch_pointer)
        _, c, summary, train_accr = sess.run([optimizer, cost, merged_summary_op, acc],
                        feed_dict={
                            x: batch_x,
                            y: batch_y,
                            keep_prob: keep_prob_val,
                            phase_train: True
                        })
        # Write logs at every iteration
        summary_writer.add_summary(summary, epoch * batch_steps + 1)
                
        if run_validation_accuracy == True:
            val_summary, val_accr = sess.run([validation_acc_summary, acc],
                        feed_dict={
                            x: x_validation,
                            y: y_validation,
                            keep_prob: 1,
                            phase_train: False
                        })
            print (f"Training Accuracy = {int(train_accr)}%   Validation Accuracy = {int(val_accr)}%")
            summary_writer.add_summary(val_summary, epoch * batch_steps + 1)
    #                        
        
        avg_cost += c / batch_steps
    
        ''' Save the model '''
        saver = tf.train.Saver()
        save_path = saver.save(sess, data_path + "SavedModel/LettersModel.ckpt")
            
    
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=",
                  "{:.9f}".format(avg_cost))
    

print("Optimization Finished!")