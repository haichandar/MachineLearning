# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 09:58:53 2019

@author: Chandar_S
"""

import cv2
import pytesseract

img = cv2.imread("C:\\Users\\chandar_s\\Pictures\\TestSheet2.tif")
height, width = img.shape[:2]

w_start = 0
h_start = 150
h_step_size = int(height/5)
w_step_size = int(width/2)

# Define config parameters.
# '-l eng'  for using the English language
# '--oem 1' for using LSTM OCR Engine
config = ('-l eng --oem 1 --psm 3')
image_count = 1
for w_current in range(w_start, width, w_step_size) :
    ## BREAK BY EACH BOX
    for h_current in range(h_start, height, h_step_size) :
#        print ("Height : " + str(j+h_steps) + "width : " + str(i+w_steps))
        crop_img = img[h_current:h_current + h_step_size, 
                       w_current  :w_current   + w_step_size]

#        cv2.imshow("cropped", crop_img)
#        cv2.waitKey(0)
        cv2.imwrite( "TestSheet2_Section"+str(image_count) +".jpg", crop_img );
        
        ''' BREAK BY ROWS - VERTICAL SUBSECTION '''
        image_Vsection_count = 1
        h_Vsubsection_start = 0
        h_Vsubsection_step_size= int(h_step_size/6)
        for h_Vsubsection_current in range(h_Vsubsection_start, h_Vsubsection_start + h_step_size, h_Vsubsection_step_size) :
             crop_Vsection_img = crop_img[h_Vsubsection_current : h_Vsubsection_current + h_Vsubsection_step_size, 
                               0                 : w_step_size]
             
#             cv2.imwrite( "TestSheet1_Section"+str(image_count) + "_VSubSection" + str(image_Vsection_count) + ".jpg", crop_Vsection_img );
            
             ''' BREAK BY COLUMNS - HORIZONDAL SUBSECTION '''
             image_Hsection_count = 1
             h_Hsubsection_start = 50
             print (crop_Vsection_img.shape)
             ''' MAKE A SQUARE IMAGE BY MATCHING HEIGHT TO WIDTH'''
             h_Hsubsection_step_size= 70 #crop_Vsection_img.shape[0]
             for h_Hsubsection_current in range(h_Hsubsection_start, h_Hsubsection_start + w_step_size, h_Hsubsection_step_size) :
                 crop_Hsection_img = crop_Vsection_img[40 : 40 + h_Hsubsection_step_size, 
                                              h_Hsubsection_current : h_Hsubsection_current + h_Hsubsection_step_size]
             
#                 cv2.imwrite( "TestSheet1_Section"+str(image_count) + "_VSubSection" + str(image_Vsection_count) + "_HSubSection" + str(image_Hsection_count) +".jpg", crop_Hsection_img )
                 image_Hsection_count += 1
            
             image_Vsection_count += 1
             break
        image_count += 1;
#        break
        # Run tesseract OCR on image
#       text = pytesseract.image_to_string(crop_img, config=config)
        # Print recognized text
#       print(text)
    break   
