# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 09:58:53 2019

@author: Chandar_S
"""

import cv2
import sys
import numpy as np
import copy
from ReadLines import ReadLines
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.ndimage import zoom

sys.path.insert(0, '..\DeepLearning')
from cnn import cnn

readLineObj = ReadLines()
img = cv2.imread("C:\\Users\\chandar_s\\Pictures\\TestSheet2.tif")
FileName = "TestSheet2"


def clipped_zoom(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out


def PredictImages(x_test):
    # convert list to ndarray and PREP AS PER INPUT FORMAT
#    x_test = []
#    for image in images:
#        x_test.append(np.array(image))
#    x_test = np.stack(x_test)

    sess=tf.Session()
    graph = tf.get_default_graph()
    saver = tf.train.Saver()
    model_path="E:\\MLData\\SavedModel\\cnn\\emnist_alpha_digit_data.ckpt"
    saver.restore(sess, model_path)
    print ("Restored Model")

    #Now, access the op that you want to run.
    x = graph.get_tensor_by_name("x:0")
    keep_prob = graph.get_tensor_by_name("keep_probability:0")
    model = graph.get_tensor_by_name("fnn/fnn_Out/BiasAdd:0")


    ## PREDICT AND VALIDATE
    try:
        x_test_1 = x_test.reshape(-1, x_test.shape[0] * x_test.shape[1])
        feed_dict ={x:x_test_1, keep_prob:1.0}
    except:
        x_test_2 = x_test.reshape(-1, x_test.shape[0], x_test.shape[1])
        feed_dict ={x:x_test_2, keep_prob:1.0}

    predicted_pct = tf.nn.softmax(model) * 100
    with sess:
       predicted_confidence =  np.round(predicted_pct.eval(feed_dict), 0)
       predicted_output = np.argmax(predicted_confidence, 1)

    return predicted_output, np.amax(predicted_confidence, axis=1)
    

''' BEGIN: INITIATE TENSOR SESSION '''
alphadigit = { 0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 
              9:'9', 10:'A', 11:'B', 12:'C', 13:'D', 14:'E', 15:'F', 16:'G', 
              17:'H', 18:'I', 19:'J', 20:'K', 21:'L', 22:'M', 23:'N', 24:'O', 
              25:'P', 26:'Q', 27:'R', 28:'S', 29:'T', 30:'U', 31:'V', 32:'W', 
              33:'X', 34:'Y', 35:'Z', 36:'a', 37:'b', 38:'d', 39:'e', 40:'f', 
              41:'g', 42:'h', 43:'n', 44:'q', 45:'r', 46:'t'}
cnn_obj = cnn("")
single_layer_fnn = False

## Override the default learning rate
cnn_obj.learning_rate_var = 0.0001

## 2 LAYER FNN INPUTS
hiddenlayer_1_width = 500
hiddenlayer_2_width = 500

## Assuming it's a SQUARE IMAGE
image_height = 28
image_width = 28

## CREATE CNN & DNN MODEL
optimizer, cost, accuracy, cnn_fnn_model = cnn_obj.create_simplified_model(
        [image_height, image_width], hiddenlayer_1_width, hiddenlayer_2_width, 
        len(alphadigit), single_layer_fnn)
''' END: INITIATE TENSOR SESSION '''

height, width = img.shape[:2]
## divide the entire sheet by 5 x 2 starting from 150 y position
w_start = 0
h_start = 150
h_step_size = int(height/5)
w_step_size = int(width/2)

image_count = 1
for w_current in range(w_start, width, w_step_size) :
    ## BREAK BY EACH BOX
    for h_current in range(h_start, height, h_step_size) :
#        print ("Height : " + str(j+h_steps) + "width : " + str(i+w_steps))
        crop_img = img[h_current:h_current + h_step_size,
                       w_current  :w_current   + w_step_size]

        cv2.imwrite( f"images/{FileName}_Section{image_count}.jpg", crop_img );

        # Find the vertical crop co-ordinates
        img_subsection, img_subsection_withlines, lines = readLineObj.DetectEdgesAndLines(None, copy.deepcopy(crop_img))
        vert_coords, return_area = readLineObj.Find_Vertical_Lines(img_subsection_withlines, lines)

        print (f"Vertical Areas --> {return_area}")
        q75, q25 = np.percentile(return_area, [75 ,25])
        minimum_vertical_area_threshold = q25 - (q75 - q25) * 1.5
        minimum_vertical_area_threshold = 20000 if  minimum_vertical_area_threshold < 0 else minimum_vertical_area_threshold
        print (f"minimum_vertical_area_threshold {minimum_vertical_area_threshold}")

        image_Vsection_count = 0
        print (" ~~START VERTICAL AREA~~")
        for x1, y1, x2, y2, area in vert_coords:
            print (f'     Vertical split -> x1:{x1}, y1:{y1}, x2:{x2}, y2:{y2}, area:{area}')
            if area < minimum_vertical_area_threshold:
                print (f"    Area not meeting minimum threshold of {minimum_vertical_area_threshold}. Skipping..")
                continue
            cropped_vertical_image = crop_img[y1: y2, x1:x2]
            # Find the horizontal crop co-ordinates
            img_subsection, img_subsection_withlines, lines = readLineObj.DetectEdgesAndLines(None, copy.deepcopy(cropped_vertical_image))

#            cv2.imshow('rects', cropped_vertical_image)
#            cv2.waitKey()

            print ("  ~~START HORIZONTAL AREA~~")

            # proceed if basic lines are detected
            if not lines is None:
                horiz_coords, return_area = readLineObj.Find_Horizontal_Lines(img_subsection_withlines, lines)
                print (f"    Horizondal Areas --> {return_area}")
                q75, q25 = np.percentile(return_area, [75 ,25])
                minimum_horizontal_area_threshold = q25 - (q75 - q25) * 1.5
                print (f"    minimum_horizontal_area_threshold {minimum_horizontal_area_threshold}")
                for x1, y1, x2, y2, area in horiz_coords:
                    if area == [] or area < minimum_horizontal_area_threshold:
                        print (f"     Area not meeting minimum threshold of {minimum_horizontal_area_threshold}. Skipping..")
                        continue
                    print (f'    Horizontal split -> x1:{x1}, y1:{y1}, x2:{x2}, y2:{y2}, area:{area}')
                    cropped_horizondal_image = cropped_vertical_image[y1: y2 + 10, x1:x2]
                    cv2.imwrite( f"images/{FileName}_Section{image_count}_SubSection{image_Vsection_count}.jpg", cropped_horizondal_image)
                    image_Vsection_count+=1
#                    FindContour(cropped_horizondal_image)
                    ''' Break images into sections and send for prediction'''
#                    images_for_analysis = []
                    images_for_analysis = readLineObj.AnalyzeAndCropImages(None, cropped_horizondal_image)
                    
                    ''' BEGIN: PREDICT AND DISPLAY RESULTS '''
                    if len(images_for_analysis) > 0:
#                        print (images_for_analysis[0])
                        predicted_output, predicted_confidence = PredictImages(images_for_analysis)

                        cols_count = int(len(images_for_analysis)/2) + (len(images_for_analysis) - int(len(images_for_analysis)/2)*2)
                    
                        f, a = plt.subplots(nrows=2, ncols=cols_count, figsize=(8, 3),
                                                        sharex=True, sharey=True, squeeze=False)
                        img_nbr = 0
                        i = 0
                        for image in images_for_analysis:
                            a[i][img_nbr].imshow(image, cmap=plt.cm.gray)
                            a[i][img_nbr].axis('off')
                    
                            title = str(alphadigit[predicted_output[img_nbr]]) + " (" + str(int(predicted_confidence[img_nbr])) + "%)"
                            a[i][img_nbr].set_title(title, fontsize=5)
                            cv2.imwrite( f"E://MLData//Test//GeneratedLetters//{FileName}_Section{image_count}_SubSection{image_Vsection_count}_Image{img_nbr}_prediction{str(alphadigit[predicted_output[img_nbr]])}.jpg", clipped_zoom(image, 1.5))

                            if (predicted_confidence[img_nbr] > 70):
                                a[i][img_nbr].set_facecolor('xkcd:mint green')
                            else:
                                a[i][img_nbr].set_facecolor('xkcd:salmon')
                            img_nbr += 1
                    
                            ''' New row'''
                            if (img_nbr == cols_count):
                                i = i + 1
                                img_nbr = 0
#                            break
                        f.show()
                        plt.draw()
                        plt.waitforbuttonpress()
                        ''' END: PREDICT AND DISPLAY RESULTS '''

            else:
                print("    No Horizontal lines detected. Skipping...")
        break
        image_count += 1;

    break
