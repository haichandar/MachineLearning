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

sys.path.insert(0, '..\DeepLearning')
from cnn import cnn

readLineObj = ReadLines()
img = cv2.imread("C:\\Users\\chandar_s\\Pictures\\TestSheet1.tif")
FileName = "TestSheet1"

def FindContour(img):
    rgb = cv2.pyrDown(img)
    small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)
    
    
    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    # using RETR_EXTERNAL instead of RETR_CCOMP
    im2, contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    mask = np.zeros(bw.shape, dtype=np.uint8)
    
    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        mask[y:y+h, x:x+w] = 0
        cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
        r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)
    
        if r > 0.45 and w > 8 and h > 8:
            cv2.rectangle(rgb, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)
    
    cv2.imshow('rects', rgb)
    cv2.waitKey()


def DoLetterPrediction(image_array):
    data_path = 'E:\MLData\\'
    cnn_obj = cnn(data_path)

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
    optimizer, cost, accuracy, cnn_fnn_model = cnn_obj.create_simplified_model([image_height, image_width], hiddenlayer_1_width, hiddenlayer_2_width, None, single_layer_fnn)
    
        
    images = ["input1.jpg", "input2.jpg", "input3.jpg", 'TestSheet1_Section1_VSubSection1_HSubSection8.jpg', 'TestSheet1_Section1_VSubSection1_HSubSection9.jpg', 'TestSheet1_Section1_VSubSection1_HSubSection10.jpg', 'TestSheet1_Section1_VSubSection1_HSubSection11.jpg','TestSheet1_Section1_VSubSection1_HSubSection12.jpg','TestSheet1_Section1_VSubSection1_HSubSection13.jpg','TestSheet1_Section1_VSubSection1_HSubSection14.jpg','TestSheet1_Section1_VSubSection1_HSubSection15.jpg','TestSheet1_Section1_VSubSection1_HSubSection16.jpg']
    
    cols_count = int(len(images)/2) + (len(images) - int(len(images)/2)*2)
    f, a = plt.subplots(nrows=2, ncols=cols_count, figsize=(8, 3),
                                    sharex=True, sharey=True, squeeze=False)
    img_nbr = 0
    i = 0
    for image_name in images:
        img, prediction, prediction_confidence = test_mnist_model(model_name="cnn\\"+input_data["name"],img_name=image_name)
        a[i][img_nbr].imshow(img, cmap='gray')
        a[i][img_nbr].axis('off')
        
#            title = str(prediction)
#            print(prediction)
        title = str(alphadigit[prediction]) + " (" + str(int(prediction_confidence)) + "%)" 
        a[i][img_nbr].set_title(title, fontsize=10)
        img_nbr += 1
        
        ''' New row'''
        if (img_nbr == cols_count):
            i = i + 1
            img_nbr = 0
            
    f.show()
    plt.draw()
    plt.waitforbuttonpress()

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

        cv2.imwrite( f"{FileName}_Section{image_count}.jpg", crop_img );
        
        # Find the vertical crop co-ordinates
        img_subsection, img_subsection_withlines, lines = readLineObj.DetectEdgesAndLines(None, copy.deepcopy(crop_img))
        vert_coords, return_area = readLineObj.Normalize_vertical_lines(img_subsection_withlines, lines)
        
        print (f"Vertical Areas --> {return_area}")
        q75, q25 = np.percentile(return_area, [75 ,25])
        minimum_vertical_area_threshold = q25 - (q75 - q25) * 1.5
        minimum_vertical_area_threshold = 20000 if  minimum_vertical_area_threshold < 0 else minimum_vertical_area_threshold
        print (f"minimum_vertical_area_threshold {minimum_vertical_area_threshold}")

        image_Vsection_count = 0
        print ("   ~~START VERTICAL AREA~~")
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
                horiz_coords, return_area = readLineObj.Normalize_horizontal_lines(img_subsection_withlines, lines)
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
                    cv2.imwrite( f"{FileName}_Section{image_count}_SubSection{image_Vsection_count}.jpg", cropped_horizondal_image)
                    image_Vsection_count+=1
#                    FindContour(cropped_horizondal_image)
                    ''' Break images into sections and send for prediction'''
                    images_for_analysis = readLineObj.AnalyzeHorizondalEdges(None, cropped_horizondal_image)
            else:
                print("    No Horizontal lines detected. Skipping...")
        break
        image_count += 1;

    break



#      ''' BREAK BY ROWS - VERTICAL SUBSECTION '''
#        image_Vsection_count = 1
#        h_Vsubsection_start = 0
#        h_Vsubsection_step_size= int(h_step_size/6)
#        for h_Vsubsection_current in range(h_Vsubsection_start, h_Vsubsection_start + h_step_size, h_Vsubsection_step_size) :
#             crop_Vsection_img = crop_img[h_Vsubsection_current : h_Vsubsection_current + h_Vsubsection_step_size, 
#                               0                 : w_step_size]
#             
##             cv2.imwrite( "TestSheet1_Section"+str(image_count) + "_VSubSection" + str(image_Vsection_count) + ".jpg", crop_Vsection_img );
#            
#             ''' BREAK BY COLUMNS - HORIZONDAL SUBSECTION '''
#             image_Hsection_count = 1
#             h_Hsubsection_start = 50
#             print (crop_Vsection_img.shape)
#             ''' MAKE A SQUARE IMAGE BY MATCHING HEIGHT TO WIDTH'''
#             h_Hsubsection_step_size= 70 #crop_Vsection_img.shape[0]
#             for h_Hsubsection_current in range(h_Hsubsection_start, h_Hsubsection_start + w_step_size, h_Hsubsection_step_size) :
#                 crop_Hsection_img = crop_Vsection_img[40 : 40 + h_Hsubsection_step_size, 
#                                              h_Hsubsection_current : h_Hsubsection_current + h_Hsubsection_step_size]
#             
##                 cv2.imwrite( "TestSheet1_Section"+str(image_count) + "_VSubSection" + str(image_Vsection_count) + "_HSubSection" + str(image_Hsection_count) +".jpg", crop_Hsection_img )
#                 image_Hsection_count += 1
#            
#             image_Vsection_count += 1
#             break
#        image_count += 1;
##        break