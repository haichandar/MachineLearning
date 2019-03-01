# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 06:40:07 2019

@author: Chandar_S
"""
import tensorflow as tf
import os
import numpy as np
import PIL
from PIL import Image



data_path = 'E:\MLData\\'
#data_path = ''
alphadigit = { 0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 
              9:'9', 10:'A', 11:'B', 12:'C', 13:'D', 14:'E', 15:'F', 16:'G', 
              17:'H', 18:'I', 19:'J', 20:'K', 21:'L', 22:'M', 23:'N', 24:'O', 
              25:'P', 26:'Q', 27:'R', 28:'S', 29:'T', 30:'U', 31:'V', 32:'W', 
              33:'X', 34:'Y', 35:'Z', 36:'a', 37:'b', 38:'d', 39:'e', 40:'f', 
              41:'g', 42:'h', 43:'n', 44:'q', 45:'r', 46:'t'}

def PredictUsingModel(img_name):
    filepath = os.path.join(data_path, 'Test', 'GeneratedLetters',  img_name)
#    filepath = os.path.join(data_path, img_name)
    
#    basewidth = 28
    img = Image.open(filepath).convert('L')
#    wpercent = (basewidth / float(img.size[0]))
#    hsize = int((float(img.size[1]) * float(wpercent)))
#    img_resized = 255 - np.array(img.resize((basewidth, hsize), PIL.Image.ANTIALIAS))

    # convert list to ndarray and PREP AS PER INPUT FORMAT
    x_test = np.array(img)

    imported_meta = tf.train.import_meta_graph(data_path + "SavedModel/Letters/LettersModel.ckpt.meta")  

    print ("Restoring Model")
    sess=tf.Session()
    imported_meta.restore(sess, data_path + "SavedModel/Letters/LettersModel.ckpt")
    graph = tf.get_default_graph()
    
 #   for op in graph.get_operations():
 #       print(op.name)

    x = graph.get_tensor_by_name("input:0")
    keep_prob = graph.get_tensor_by_name("keep_probability:0")
    phase_train = graph.get_tensor_by_name("phase_train:0")
    #Now, access the op that you want to run. 
    model = graph.get_tensor_by_name("dense/BiasAdd:0")

    ## PREDICT AND VALIDATE
    x_test_1 = x_test.reshape(-1, x_test.shape[0] * x_test.shape[1])
    feed_dict ={x:x_test_1, keep_prob:1.0, phase_train:False}

    predicted_pct = tf.nn.softmax(model) * 100
    with  sess:
       predicted_confidence =  np.round(predicted_pct.eval(feed_dict), 0)
       predicted_test = np.argmax(predicted_confidence, 1)
    return predicted_test[0], np.max(predicted_confidence)


print ("Starting to predict")
answer, confidence = PredictUsingModel("TestSheet2_Section1_SubSection5_Image0_predictionF.jpg")
print (f"Answer is { str(alphadigit[answer])} and Confidence is {confidence}%")