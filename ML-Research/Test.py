# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 12:52:35 2018

@author: Chandar_S
"""

"""
@file hough_lines.py
@brief This program demonstrates line finding with the Hough transform
"""
import sys
import math
import cv2 as cv
import numpy as np
def main(argv):
    
    default_file =  "C://Users//chandar_s//Pictures//square.jpg"
    filename = argv[0] if len(argv) > 0 else default_file
    # Loads an image
    src = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
        return -1
    
    
    dst = cv.Canny(src, 50, 300, None, 3)
    
    # Copy edges to the images that will display the results in BGR
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)
    
    lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 50,10)
#    lines = lines_temp
#    rho_threshold= 15;
#    theta_threshold = 3*DEGREES_TO_RADIANS;
#    cv.contourCluster(lines_t, rho_max, theta_max, lines);)
    
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv.line(cdst, pt1, pt2, (0,0,255), 3, cv.LINE_AA)
            print ("a=", a, "b=",b)
            print ("PT1=", pt1, "PT2=",pt2)
    
    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 150, None, 50, 10)
#    
#    dst: Output of the edge detector. It should be a grayscale image (although in fact it is a binary one)
#lines: A vector that will store the parameters (x_{start}, y_{start}, x_{end}, y_{end}) of the detected lines
#rho : The resolution of the parameter r in pixels. We use 1 pixel.
#theta: The resolution of the parameter \theta in radians. We use 1 degree (CV_PI/180)
#threshold: The minimum number of intersections to “detect” a line
#minLinLength: The minimum number of points that can form a line. Lines with less than this number of points are disregarded.
#maxLineGap: The maximum gap between two points to be considered in the same line.
    
    
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
            
    
    #cv.imshow("Source", src)
    #cv.imshow("Edges", dst)
    #cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    #cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
    
    cv.imwrite('C://Users//chandar_s//Pictures//Edges.png',dst)
    #cv.imwrite('C://Users//chandar_s//Pictures//Standard Hough Line Transform.png',cdst)
    cv.imwrite('C://Users//chandar_s//Pictures//Probabilistic Line Transform.png',cdstP)
    
    #cv.waitKey()
    return 0
    
if __name__ == "__main__":
    main(sys.argv[1:])