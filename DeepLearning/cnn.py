# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 19:33:12 2018

@author: Chandar_S
"""


from fnn import fnn
import tensorflow as tf
import datetime

class cnn(fnn):

    data_path = None
    
    def __init__(self, path):
        self.data_path = path
        self.logs_path = path+ 'Logs\\cnn_'+ datetime.datetime.now().strftime("%d%b-%H.%M")
        tf.set_random_seed(self.random_state)
       
    def create_model(self, input_x, input_x_shape, n_hidden_1, n_hidden_2, output_classes, single_layer_fnn=False):

        ### CREATE A CNN LAYER ###
        def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
           with tf.name_scope(name):
    
                # setup the filter input shape for tf.nn.conv_2d
                conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]
            
                # initialise weights and bias for the filter
                weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03), name=name+'_Weight')
                bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_Bias')
    
                # TIPS: to display the 32/64 convolution filters, re-arrange the
                # weigths to look like 32/64 images with a transposition.
                a = tf.reshape(weights, [filter_shape[0] * filter_shape[1] * num_input_channels, num_filters])
                b = tf.transpose(a)
                c = tf.reshape( b, [num_filters, filter_shape[0], filter_shape[1] * num_input_channels, 1])
                tf.summary.image(name + "_filter", c, num_filters)
    
            
                # setup the convolutional layer operation
                out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')
            
                # add the bias
                out_layer += bias
            
                # apply a ReLU non-linear activation
                out_layer = tf.nn.relu(out_layer)
            
                # now perform max pooling
                # ksize is the argument which defines the size of the max pooling window (i.e. the area over which the maximum is
                # calculated).  It must be 4D to match the convolution - in this case, for each image we want to use a 2 x 2 area
                # applied to each channel
                ksize = [1, pool_shape[0], pool_shape[1], 1]
                # strides defines how the max pooling area moves through the image - a stride of 2 in the x direction will lead to
                # max pooling areas starting at x=0, x=2, x=4 etc. through your image.  If the stride is 1, we will get max pooling
                # overlapping previous max pooling areas (and no reduction in the number of parameters).  In this case, we want
                # to do strides of 2 in the x and y directions.
                strides = [1, 2, 2, 1]
                out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, padding='SAME')
    
                # convoluted image
                a1 = tf.reshape(out_layer, [-1, out_layer.shape[1], out_layer.shape[2] * out_layer.shape[3], 1])
                tf.summary.image(name + "_convimage", a1, 5)
                
            
                return out_layer
            
        ############## START: BUILD CNN MODEL ##################
        # layer 1
        cnn_layer1_input_channels = 1
        cnn_layer1_filter_count = 32
        cnn_layer1_filter_shape = [5, 5]
        cnn_layer1_pooling_shape = [2, 2]

        # layer 2
        cnn_layer2_input_channels = cnn_layer1_filter_count
        cnn_layer2_filter_count = 64
        cnn_layer2_filter_shape = [5, 5]
        cnn_layer2_pooling_shape = [2, 2]

        # reshape the input data so that it is a 4D tensor.  The first value (-1) tells function to dynamically shape that
        # dimension based on the amount of data passed to it.  The two middle dimensions are set to the image size (i.e. 28
        # x 28).  The final dimension is 1 as there is only a single colour channel i.e. grayscale.  If this was RGB, this
        # dimension would be 3        
        x_shaped = tf.reshape(self.x, [-1, input_x_shape[0], input_x_shape[1], cnn_layer1_input_channels])
        x_standardized = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame),x_shaped)

        with tf.name_scope('cnn'):
            # create some convolutional layers
            layer1 = create_new_conv_layer(x_standardized, cnn_layer1_input_channels, cnn_layer1_filter_count, cnn_layer1_filter_shape, cnn_layer1_pooling_shape, name='CNN_Layer1')
            layer2 = create_new_conv_layer(layer1, cnn_layer2_input_channels, cnn_layer2_filter_count, cnn_layer2_filter_shape, cnn_layer2_pooling_shape, name='CNN_Layer2')
            self.cnn_output = layer2
    
        # flatten the output ready for the fully connected output stage - after two layers of stride 2 pooling, we go
        # from 28 x 28, to 14 x 14 to 7 x 7 x,y co-ordinates, but with 64 output channels.  To create the fully connected,
        # "dense" layer, the new shape needs to be [-1, 7 x 7 x 64]
        #cnn_output_shape = [-1, 7 * 7 * cnn_layer2_filter_count]
        #cnn_output = tf.reshape(layer2, cnn_output_shape)
        ############## END: BUILD CNN MODEL ##################
  

        
        
        ############### START: BUILD FNN MODEL ###################
        if single_layer_fnn:
            optimizer, cost, acc, cnn_fnn_model = super().create_model(layer2, cnn_layer2_filter_count, n_hidden_1, n_hidden_2, output_classes, single_layer_fnn)
        else:
            output = tf.layers.flatten(layer2)
            #conv1_height = (input_x_shape[0] - cnn_layer1_filter_shape[0] + 2*0) + 1
            pool1_height = input_x_shape[0] / cnn_layer1_pooling_shape[0]
            #conv2_height = (pool1_height - cnn_layer2_filter_shape[0] + 2*0) + 1 
            cnn_pool2_height = pool1_height / cnn_layer2_pooling_shape[0] #TODO
    #        optimizer, cost, acc, cnn_fnn_model = super().create_simplified_model(cnn_output, int(cnn_pool2_height*cnn_pool2_height)* cnn_layer2_filter_count, n_hidden_1, n_hidden_2, output_classes)
            optimizer, cost, acc, cnn_fnn_model = super().create_model(output, int(cnn_pool2_height*cnn_pool2_height)* cnn_layer2_filter_count, n_hidden_1, n_hidden_2, output_classes)
        ############### END: BUILD FNN MODEL ###################
            
        return optimizer, cost, acc, cnn_fnn_model 


    def create_simplified_model(self, input_x_shape, n_hidden_1, n_hidden_2, output_classes, single_layer_fnn):
    
       ############### START: BUILD CNN MODEL ###################
        # layer 1
        cnn_layer1_input_channels = 1
        cnn_layer1_filter_count = 32
        cnn_layer1_filter_shape = [5, 5]
        cnn_layer1_pooling_shape = [2, 2]
        cnn_layer1_pooling_stride = [2, 2]

        # layer 2
        cnn_layer2_filter_count = 48
        cnn_layer2_filter_shape = [5, 5]
        cnn_layer2_pooling_shape = [2, 2]
        cnn_layer2_pooling_stride = [2, 2]
        
        x_shaped = tf.reshape(self.x, [-1, input_x_shape[0], input_x_shape[1], cnn_layer1_input_channels])
        x_standardized = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame),x_shaped)


        if single_layer_fnn:
            self.cnn_output, filters = self.PneumothoraxDetectionModel(x_standardized, output_classes)
            optimizer, cost, acc, cnn_fnn_model = super().create_simplified_model(self.cnn_output, filters, n_hidden_1, n_hidden_2, output_classes, single_layer_fnn)
           # first convolutional layer
#            conv1 = tf.layers.conv2d(inputs=x_standardized, filters=cnn_layer1_filter_count, kernel_size=cnn_layer1_filter_shape, kernel_initializer=tf.initializers.lecun_normal(), use_bias=True, bias_initializer=tf.zeros_initializer(), activation=tf.nn.relu, name='CNN_Layer1')
#            # first pooling layer
#            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=cnn_layer1_pooling_shape, strides=cnn_layer1_pooling_stride, name='CNN_Layer1_pooling')
#            # second convolutional layer
#            conv2 = tf.layers.conv2d(inputs=pool1, filters=cnn_layer2_filter_count, kernel_size=cnn_layer2_filter_shape, kernel_initializer=tf.initializers.lecun_normal(), use_bias=True, bias_initializer=tf.zeros_initializer(), activation=tf.nn.relu, name='CNN_Layer2')
#            # second pooling layer
#            self.cnn_output = tf.layers.max_pooling2d(inputs=conv2, pool_size=cnn_layer2_pooling_shape, strides=cnn_layer2_pooling_stride, name='CNN_Layer2_pooling')
#            optimizer, cost, acc, cnn_fnn_model = super().create_simplified_model(self.cnn_output, cnn_layer2_filter_count, n_hidden_1, n_hidden_2, output_classes, single_layer_fnn)

        else:
            with tf.name_scope('cnn'):
               # first convolutional layer
                conv1 = tf.layers.conv2d(inputs=x_standardized, filters=cnn_layer1_filter_count, kernel_size=cnn_layer1_filter_shape, kernel_initializer=tf.initializers.lecun_normal(), use_bias=True, bias_initializer=tf.zeros_initializer(), activation=tf.nn.relu, name='CNN_Layer1')
                # first pooling layer
                pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=cnn_layer1_pooling_shape, strides=cnn_layer1_pooling_stride, name='CNN_Layer1_pooling')
            
                # second convolutional layer
                conv2 = tf.layers.conv2d(inputs=pool1, filters=cnn_layer2_filter_count, kernel_size=cnn_layer2_filter_shape, kernel_initializer=tf.initializers.lecun_normal(), use_bias=True, bias_initializer=tf.zeros_initializer(), activation=tf.nn.relu, name='CNN_Layer2')
                # second pooling layer
                self.cnn_output = tf.layers.max_pooling2d(inputs=conv2, pool_size=cnn_layer2_pooling_shape, strides=cnn_layer2_pooling_stride, name='CNN_Layer2_pooling')
            flat = tf.layers.flatten(self.cnn_output)
            conv1_height = (input_x_shape[0] - cnn_layer1_filter_shape[0] + 2*0) + 1
            pool1_height = conv1_height / cnn_layer1_pooling_shape[0]
            conv2_height = (pool1_height - cnn_layer2_filter_shape[0] + 2*0) + 1 
            cnn_pool2_height = conv2_height / cnn_layer2_pooling_shape[0] #TODO
            optimizer, cost, acc, cnn_fnn_model = super().create_simplified_model(flat, int(cnn_pool2_height*cnn_pool2_height)* cnn_layer2_filter_count, n_hidden_1, n_hidden_2, output_classes)
        ############### END: BUILD CNN MODEL ###################
            
        return optimizer, cost, acc, cnn_fnn_model 

    def PneumothoraxDetectionModel(self, x_image, output_classes):


        conv1 = tf.layers.conv2d(inputs=x_image, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
        skip1 = tf.layers.max_pooling2d(conv1+conv2, 2, 2)

        conv3 = tf.layers.conv2d(inputs=skip1, filters=128, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
        conv4 = tf.layers.conv2d(inputs=conv3, filters=128, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
        skip2 = tf.layers.max_pooling2d(conv3+conv4, 2, 2)

        conv5 = tf.layers.conv2d(inputs=skip2, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        conv6 = tf.layers.conv2d(inputs=conv5, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        skip3 = tf.layers.max_pooling2d(conv5+conv6, 2, 2)

        conv7 = tf.layers.conv2d(inputs=skip3, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        conv8 = tf.layers.conv2d(inputs=conv7, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        self.cnn_output = tf.layers.max_pooling2d(conv7+conv8, 2, 2)

        return self.cnn_output, 512
