# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 09:58:53 2019

@author: Chandar_S
"""

import cv2
import pytesseract

img = cv2.imread("C:\\Users\\chandar_s\\Pictures\\TestSheet1.tif")
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
    for h_current in range(h_start, height, h_step_size) :
#        print ("Height : " + str(j+h_steps) + "width : " + str(i+w_steps))
        crop_img = img[h_current:h_current + h_step_size, 
                       w_current  :w_current   + w_step_size]
#        cv2.imshow("cropped", crop_img)
        cv2.imwrite( "TestSheet1_Image"+str(image_count) +".jpg", crop_img );
#        cv2.waitKey(0)
        image_count = image_count + 1;

        # Run tesseract OCR on image
#        text = pytesseract.image_to_string(crop_img, config=config)
        # Print recognized text
#        print(text)
 
