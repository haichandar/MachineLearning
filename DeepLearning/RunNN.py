# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 22:05:38 2018

@author: Chandar_S
"""

from cnn import cnn
from fnn import fnn
from rnn import rnn
from nn_utilities_py import nn_utilities
import tensorflow as tf
from scipy.misc import imread
import os
import numpy as np
import pylab
import matplotlib.pyplot as plt
import sys

data_path = 'E:\MLData\\'
nn_utilities_obj = nn_utilities(data_path)

def run_test():
    nn_utilities_obj.load_PneumothoraxDataset()
#    nn_utilities_obj.load_fashion_data()
#    nn_utilities_obj.load_mnist_digit_data()
#    nn_utilities_obj.prepare_digits_image_inputs()

def run_fnn():
    fnn_obj = fnn(data_path)

    # Flag makes it run with new simplified code and does not run validation accuracy for quicker response
    legacy_run = False

    ## GET INPUT  DATA
#    input_data = nn_utilities_obj.prepare_digits_image_inputs()
    input_data = nn_utilities_obj.load_mnist_digit_data()
#    input_data = nn_utilities_obj.load_fashion_data()

    ## 2 LAYER FNN INPUTS
    hiddenlayer_1_width = 256
    hiddenlayer_2_width = 256

    ## Override the default learning rate
    fnn_obj.learning_rate_var = 0.001

    if legacy_run == True:
        ## CREATE FNN MODEL
        optimizer, cost,  accuracy, fnn_model = fnn_obj.create_model(fnn_obj.x, input_data["x_train"].shape[1], hiddenlayer_1_width, hiddenlayer_2_width, input_data["y_train"].shape[1])
    else:
        ## CREATE FNN MODEL
        optimizer, cost,  accuracy, fnn_model = fnn_obj.create_simplified_model(fnn_obj.x, input_data["x_train"].shape[1], hiddenlayer_1_width, hiddenlayer_2_width, input_data["y_train"].shape[1] )

    ## TRAIN THE MODEL AND TEST PREDICTION
    run_nn(fnn_obj, input_data, optimizer, cost, accuracy, fnn_model, "fnn/"+input_data["name"])


def run_cnn():
    cnn_obj = cnn(data_path)
    
    # Flag makes it run with new simplified code and does not run validation accuracy for quicker response
    legacy_run = False

    ''' WE NEED THIS FOR LOOKING AT HEAT MAP OVER IMAGE'''
    single_layer_fnn = True

    ## Override the default learning rate
    cnn_obj.learning_rate_var = 0.0001
    
    ## GET INPUT  DATA
#    input_data = nn_utilities_obj.prepare_digits_image_inputs()
    input_data = nn_utilities_obj.load_mnist_digit_data()
#    input_data = nn_utilities_obj.load_fashion_data()
#    input_data = nn_utilities_obj.load_PneumothoraxDataset()

    ## 2 LAYER FNN INPUTS
    hiddenlayer_1_width = 500
    hiddenlayer_2_width = 500

    ## Assuming it's a SQUARE IMAGE
    image_height = int(np.sqrt(input_data["x_train"].shape[1]))
    image_width = image_height
    
    if legacy_run == True:
        ## CREATE CNN & DNN MODEL
        optimizer, cost, accuracy, cnn_fnn_model = cnn_obj.create_model([image_height, image_width], hiddenlayer_1_width, hiddenlayer_2_width, input_data["y_train"].shape[1], single_layer_fnn)
    else:
        ## CREATE CNN & DNN MODEL
        optimizer, cost, accuracy, cnn_fnn_model = cnn_obj.create_simplified_model([image_height, image_width], hiddenlayer_1_width, hiddenlayer_2_width, input_data["y_train"].shape[1], single_layer_fnn)

    ## TRAIN THE MODEL AND TEST PREDICTION
    run_nn(cnn_obj, input_data, optimizer, cost, accuracy, cnn_fnn_model, "cnn\\"+input_data["name"], False)


def run_rnn():
    rnn_obj = rnn(data_path)

    ## GET INPUT  DATA
#    input_data = nn_utilities_obj.prepare_digits_image_inputs()
    input_data = nn_utilities_obj.load_fashion_data()

    ## Override the default learning rate
    rnn_obj.learning_rate_var = 0.05

    ## Assuming it's a SQUARE IMAGE
    image_height = int(np.sqrt(input_data["x_train"].shape[1]))
    image_width = image_height

    # Network Parameters
    num_input = image_height # MNIST data input (img shape: 28*28)
    timesteps = image_width # timesteps
    num_hidden = 128 # hidden layer num of features
    num_classes = 10 # MNIST total classes (0-9 digits)

    ## CREATE RNN MODEL
    optimizer, cost,  accuracy, rnn_model = rnn_obj.create_model(num_input, timesteps, num_hidden, num_classes)

    input_data["x_train"] = np.reshape(input_data["x_train"],[input_data["x_train"].shape[0], timesteps,num_input])
    input_data["x_validation"] = np.reshape(input_data["x_validation"],[input_data["x_validation"].shape[0], timesteps,num_input])

    ## TRAIN THE MODEL AND TEST PREDICTION
    run_nn(rnn_obj, input_data, optimizer, cost, accuracy, rnn_model, "rnn/"+input_data["name"])


def run_nn(obj, input_data, optimizer, cost, accuracy, model, model_name=None, run_validation_accuracy=True):
    # Python optimisation variables
    training_epochs = 10
    display_step = 100
    batch_size = 10
    quick_training = True

    print ("Starting session")
    #### TRAIN AND TEST NN
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # TRAIN
        trained_model = obj.train_model(sess, model, training_epochs, display_step, batch_size, optimizer, cost, accuracy, input_data["x_train"], input_data["x_train_4D"], input_data["y_train"], input_data["x_validation"], input_data["y_validation"], quick_training, model_name, run_validation_accuracy)

        ## TEST
        test = input_data["test"]
        if (test is not None):
            data_dir = input_data["data_dir"]

            img_name = obj.rng.choice(test.filename)
            filepath = os.path.join(data_dir, 'Numbers', 'Images', 'test', img_name)
            img = imread(filepath, flatten=True)
            # convert list to ndarray and PREP AS PER INPUT FORMAT
            x_test = np.stack(img)
            if len(input_data["x_train"].shape) == 2:
                x_test = x_test.reshape(-1, input_data["x_train"].shape[1])
            else:
                x_test = x_test.reshape(-1, input_data["x_train"].shape[1], input_data["x_train"].shape[2])

            ## PREDICT AND VALIDATE
            predicted_test = obj.predictvalue(trained_model, x_test)

            print("Prediction is: ", predicted_test[0])
            pylab.imshow(img, cmap='gray')
            pylab.axis('off')
            pylab.show()
    print ("Ending session")

    ## DO MIT CAM Analysis to print the Heatmap
    CAM_analysis = False
    if (CAM_analysis == True):
        load_saved_model(model_name, obj, input_data)


def load_saved_model(model_name, obj, input_data):
    with tf.Session() as sess:
        saver = tf.train.Saver()
        print ("Restoring Model")
        saver.restore(sess, data_path + "SavedModel\\"+model_name+".ckpt")
        
        print ("Starting with CAM Analysis")
        """DOING CAM Heatmaps Analysis"""
        
        '''extract the features and weights using the function defined directly above '''
        (feature_maps, dense_weights) = extract_features_weights(sess, obj) #TODO

#        print("Feature Maps: "+str(feature_maps))
#        print("Dense Weights: "+str(dense_weights))

        '''TODO: compute the CAM for a pneumothorax detection using the function above'''
        WHICH_OPTION_INDEX = 1
        cam = compute_cam(WHICH_OPTION_INDEX, feature_maps, dense_weights)
       
        ## Assuming it's a SQUARE IMAGE
        image_height = int(np.sqrt(input_data["x_train"].shape[1]))
        image_width = image_height

        ''' upsample the CAM Tensor to a 28\times 28 image '''
        cam_upsampled =  tf.image.resize_bilinear(cam,  [image_height,image_width])

       
        inds = []
        for check_index in range (1,20):
            if np.argmax(input_data["y_validation"][check_index]) == WHICH_OPTION_INDEX:
                inds.extend([check_index])
        print (inds)
#        inds= [79, 31]
        input_data["y_validation"] = np.stack(input_data["y_validation"])
#        print (type(input_data["x_validation"][1]))
#        print (input_data["y_validation"][1])
        
        for im, cl in zip(input_data["x_validation"][inds], input_data["y_validation"][inds]):
            heatmap = sess.run(
                cam_upsampled,
                feed_dict={
                    obj.x: im[np.newaxis,:],
                })

            vis_cam(im, np.squeeze(heatmap), input_data)
        """DOING CAM Heatmaps Analysis"""


''' Extract the last Layer weights of CNN and FNN for CAM manipulation'''
def extract_features_weights(sess, cnn_obj):
    #access feature map activations directly from the model declaration
    feature_maps = cnn_obj.cnn_output

#    graph = tf.get_default_graph()
#    for op in graph.get_operations():
#        print(op.name)
    
    # we have implemented 2 different methods, so handling both scenarios
    try:
        #access the weights by searching by name
        dense_weights = sess.graph.get_tensor_by_name('fnn/FNN_Output_Weight:0')
    except:
        #access the weights by searching by name
        dense_weights = sess.graph.get_tensor_by_name('dense_layer/kernel:0')

    return (feature_maps, dense_weights)


''' Forms a CAM operation given a class name, feature maps, and weights
   
    Params: 
        - class_index: index of the class to measure
        - fmap: (1 x h x w x d) tf.Tensor of activations from the final convolutional layer
        - weights: (features x #ofoutputclasses) tf.Tensor with the learned weights of the final FC layer
    
    Returns: 
        - (16 x 16) tf.Tensor of downscaled CAMs  
    '''
def compute_cam(class_index, fmap, weights):
    w_vec = tf.expand_dims(weights[:, class_index], 1)
    _, h, w, c = fmap.shape.as_list()
    fmap = tf.squeeze(fmap) # remove batch dim
    fmap = tf.reshape(fmap, [h * w, c])
    # compute the CAM! Remeber to look at the equation defining CAMs above to do this 
    CAM = tf.matmul(fmap, w_vec) # TODO
    CAM = tf.reshape(CAM, [1, h, w, 1])

    return CAM


""" Visualize class activation heatmap, overlaying on image."""
def vis_cam(image, cam, input_data, save_file=None):
#    print (cam)
    
    if (cam.min() != cam.max()):
        cam = (cam - cam.min()) / (cam.max() - cam.min()) # TODO: check
    ## Assuming it's a SQUARE IMAGE
    image_height = int(np.sqrt(input_data["x_train"].shape[1]))
    image_width = image_height

    image = image.reshape(image_height, image_width, 1 )
    plt.imshow(255-image.squeeze(), cmap=plt.cm.gray)
    plt.imshow(1-cam, cmap=plt.cm.jet, alpha=0.5, interpolation='nearest', vmin=0, vmax=1)

    if save_file:
        plt.savefig(save_file)

    plt.show()
    plt.close()


if __name__ == "__main__":
    if len (sys.argv) != 2 :
        print ("Usage: python RunNN.py <cnn/fnn/rnn/test>")
        sys.exit (1)

    if (sys.argv[1] == "cnn"):
        print ("Running CNN model")
        run_cnn()
    elif (sys.argv[1] == "fnn"):
        print ("Running FNN model")
        run_fnn()
    elif (sys.argv[1] == "rnn"):
        print ("Running RNN model")
        run_rnn()
    elif (sys.argv[1] == "test"):
        print ("Running Test")
        run_test()


    #tensorboard --logdir=E:\\MLData\\Logs

            ## PREDICT AND TEST
#        x_test = input_data["x_test"]
#        predicted_test = nn_utilities_obj.predictvalue(trained_model, x_test)
#
#    print ("Ending session")
#
#    test = input_data["test"]
#    data_dir = input_data["data_dir"]
#
#    img_name = fnn_obj.rng.choice(test.filename)
#    filepath = os.path.join(data_dir, 'Numbers', 'Images', 'test', img_name)
#    img = imread(filepath, flatten=True)
#    test_index = int(img_name.split('.')[0]) - 49000
#    print("Prediction is: ", predicted_test[test_index])
#
#    pylab.imshow(img, cmap='gray')
#    pylab.axis('off')
#    pylab.show()