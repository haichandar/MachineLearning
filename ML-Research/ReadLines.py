import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy 

class ReadLines:

    ''' DETECT EDGES USING CANNY METHOD AND BASIC LINES USING HoughLinesP method '''
    def DetectEdgesAndLines(self, image_name, img):
        if not image_name is None:
            # Loading image contains lines
            img = cv2.imread(image_name)

        # Convert to grayscale
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        # Apply Canny edge detection, return will be a binary image
        edges = cv2.Canny(gray,100,200)
        
        # Apply HoughLines Probablistic detection of lines
        minLineLength = 100
        maxLineGap = 10
        lines = cv2.HoughLinesP(edges,1,np.pi/180,50,minLineLength,maxLineGap)

        ## Deep copy to draw detected lines
        img_org = copy.deepcopy(img)

        ## Draw initial detected lines and then finetune
        if not lines is None:
            for line in lines:
                x1,y1,x2,y2 = line[0]
                cv2.line(img,(x1,y1),(x2,y2),(255,0,0),2)
        
        return img, img_org, lines
    
    ''' NORMALIZE THE LINES DETECTED - MERGE CLOSER LINES AND DRAW ONE SINGLE HORIZONTAL LINE INSTEAD OF MULTIPLE SMALL LINES'''
    def Normalize_horizontal_lines(self, img_org, lines):

        ''' Hyper Parameters'''
        v_step = 3
        #Line_count_merge_threshold = 7
        Line_count_merge_threshold = 5
        ## what should be the minimum distance between 2 lines which 
        #minimum_distance_threshold_default = 20 
        minimum_distance_threshold_default = 20
        ## what should be the average length of the line which should be considered as a true line
        #minimum_length_threshold_default = 10
        minimum_length_threshold_default = 10

        new_coord = []
        length_array = []
        return_coord = []
        return_area = []
       
        # see merging of lines along x axis as we convolve in Y axis
        for current_height in range (0, img_org.shape[0], v_step):
            new_x1 = -1
            new_x2 = -1
            new_y = current_height + int(v_step/2)
            length = 0
            lines_merged = 0
            ## SCAN EACH LINE AND SEE IF IT CAN BE MERGED
            for line in lines:
                x1,y1,x2,y2 = line[0]
                # CHECK IF FALLS INTO HORIZONTAL RANGE
                if (int(np.mean([y1, y2])) >= current_height and int(np.mean([y1, y2])) < current_height + v_step):
#                    new_x1 = min(x1, x2) if (new_x1 == -1) else min(new_x1, x1, x2)
#                    new_x2 = max(new_x1, x1, x2)
                    length += np.sqrt(np.square(x2 - x1))
                    new_x1 = 1
                    new_x2 = img_org.shape[1]
                    lines_merged += 1
            
            # This confirms merging happened
            if (new_x1 > 0):
                length_array.append(int(length/lines_merged))
                new_coord.append([new_x1, new_y, new_x2, new_y, lines_merged, int(length/lines_merged)]) if lines_merged > Line_count_merge_threshold else None
        #        new_coord.append([new_x1, new_y, new_x2, new_y, lines_merged, int(distance/lines_merged)])
                
        # Find minimun outlier based on Interquartile Range and Outliers model
        if length_array != []:
            q75, q25 = np.percentile(length_array, [75 ,25])
            minimum_length_threshold = q25 - (q75 - q25) * 1.5
        else:
            minimum_length_threshold = -1
        minimum_length_threshold= -1
        minimum_length_threshold = minimum_length_threshold_default if minimum_length_threshold < 0 else minimum_length_threshold
        
        prev_y = 0
        distance = 0
        updated_new_coord = []
        distance_array = []
        
        ## MERGE NEAR BY LINES TO MAKE IT 1 HORIZAONTAL LINE
        for rec in new_coord:
            x1, y1, x2, y2, lines_merged, avg_length = rec
            distance = minimum_distance_threshold_default + 1 if prev_y == 0 else np.abs(y1 - prev_y)
            prev_y = y1
            distance_array.append(distance)
            updated_new_coord.append([x1, y1, x2, y2, lines_merged, avg_length, distance])
        
        
        # Find minimun outlier based on Interquartile Range and Outliers model
        if distance_array != []:
            q75, q25 = np.percentile(distance_array, [75 ,25])
            minimum_distance_threshold = q25 - (q75 - q25) * 1.5
        else:
            minimum_distance_threshold = -1
        minimum_distance_threshold = minimum_distance_threshold_default if minimum_distance_threshold <= 0 else minimum_distance_threshold
        
        prev_coord = {"x1":0, "y1":0, "x2":img_org.shape[1], "y2":0}
        # loop through to plot
        for x1, y1, x2, y2, lines_merged, avg_length, distance in updated_new_coord:
        #    cv2.line(img_org,(x1,y1),(x2,y2),(255,0,0),2) if avg_length > minimum_length_threshold and distance > minimum_distance_threshold else None
        #    cv2.line(img_org,(x1,y1),(x2,y2),(255,0,0),2)
            if distance > minimum_distance_threshold:
                  cv2.line(img_org,(x1,y1),(x2,y2),(255,0,0),2) 
                  area = np.abs(y2 - prev_coord["y1"]) * np.abs(x2 - x1)
                  return_area.append(area)
                  return_coord.append([x1,prev_coord["y1"],x2,y2,area])
                  prev_coord = {"x1":x1,"y1":y1,"x2":x2,"y2":y2}

        # Account for remaining space                
        area = np.abs(img_org.shape[0] - prev_coord["y1"]) * np.abs(prev_coord["x2"] - prev_coord["x1"])
        return_coord.append([prev_coord["x1"],prev_coord["y1"],prev_coord["x2"],img_org.shape[0],area])
        return_area.append(area)

        return return_coord, return_area
    
    ''' NORMALIZE THE LINES DETECTED - MERGE CLOSER LINES AND DRAW ONE SINGLE HORIZONTAL LINE INSTEAD OF MULTIPLE SMALL LINES'''
    def Normalize_vertical_lines(self, img_org, lines):

        ''' Hyper Parameters'''
        h_step = 3
        #Line_count_merge_threshold = 7
        Line_count_merge_threshold = 3
        ## what should be the minimum distance between 2 lines which 
        #minimum_distance_threshold_default = 20 
        minimum_distance_threshold_default = 20
        ## what should be the average length of the line which should be considered as a true line
        #minimum_length_threshold_default = 10
        minimum_length_threshold_default = 15

        new_coord = []
        length_array = []
        return_coord = []
        return_area = []
        
        # see merging of lines along x axis as we convolve in Y axis
        for current_width in range (0, img_org.shape[1], h_step):
            new_y1 = -1
            new_y2 = -1
            new_x = current_width + int(h_step/2)
            length = 0
            lines_merged = 0
            ## SCAN EACH LINE AND SEE IF IT CAN BE MERGED
            for line in lines:
                x1,y1,x2,y2 = line[0]
                # CHECK IF FALLS INTO HORIZONTAL RANGE
                if (int(np.abs(np.mean([x1, x2]))) >= current_width and int(np.abs(np.mean([x1, x2]))) < current_width + h_step):
                    length += np.sqrt(np.square(y2 - y1))
                    new_y1 = 1
                    new_y2 = img_org.shape[0]
                    lines_merged += 1
            
            # This confirms merging happened
            if (new_y1 > 0):
                length_array.append(int(length/lines_merged))
                new_coord.append([new_x, new_y1, new_x, new_y2, lines_merged, int(length/lines_merged)]) if lines_merged > Line_count_merge_threshold else None
#                new_coord.append([new_x, new_y1, new_x, new_y2, lines_merged, int(length/lines_merged)])
                
        # Find minimun outlier based on Interquartile Range and Outliers model
        q75, q25 = np.percentile(length_array, [75 ,25])
        minimum_length_threshold = q25 - (q75 - q25) * 1.5
        minimum_length_threshold = minimum_length_threshold_default if minimum_length_threshold <= 0 else minimum_length_threshold
        
        prev_x = 0
        distance = 0
        updated_new_coord = []
        distance_array = []
        
        ## MERGE NEAR BY LINES TO MAKE IT 1 VERTICAL LINE
        for rec in new_coord:
            x1, y1, x2, y2, lines_merged, avg_length = rec
            distance = minimum_distance_threshold_default + 1 if prev_x == 0 else np.abs(x1 - prev_x)
            prev_x = x1
            distance_array.append(distance)
            updated_new_coord.append([x1, y1, x2, y2, lines_merged, avg_length, distance])
        
        
        # Find minimun outlier based on Interquartile Range and Outliers model
        q75, q25 = np.percentile(distance_array, [75 ,25])
        minimum_distance_threshold = q25 - (q75 - q25) * 1.5
        minimum_distance_threshold =  minimum_distance_threshold_default if  minimum_distance_threshold < 0 else  minimum_distance_threshold
        
        prev_coord = {"x1":0}
        # loop through to plot
        for x1, y1, x2, y2, lines_merged, avg_length, distance in updated_new_coord:
            if avg_length > minimum_length_threshold and distance > minimum_distance_threshold:
                cv2.line(img_org,(x1,y1),(x2,y2),(255,0,0),2)
                area = np.abs(y2 - y1) * np.abs(x2 - prev_coord["x1"])
                return_area.append(area)
                return_coord.append([prev_coord["x1"],y1,x2,y2,area])
                prev_coord = {"x1":x1,"y1":y1,"x2":x2,"y2":y2}
#            cv2.line(img_org,(x1,y1),(x2,y2),(255,0,0),2)
#            cv2.line(img_org,(x1,y1),(x2,y2),(255,0,0),2) if distance > minimum_distance_threshold else None

        # Account for remaining space                
        area = np.abs(prev_coord["y2"] - prev_coord["y1"]) * np.abs(img_org.shape[1] - prev_coord["x1"])
        return_coord.append([prev_coord["x1"],prev_coord["y1"],img_org.shape[1],prev_coord["y2"] ,area])
        return_area.append(area)

        return  return_coord, return_area      
    
    '''' WRITE THE DATA TO OUTPUT IMAGE AND SHOW THE DETAILS IN PLOT'''    
    def Draw_plots(self, img, img_org):
        cv2.imwrite('houghlines-output.jpg',img_org)

        # display results
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3),
                                            sharex=True, sharey=True)
        
        ax1.imshow(img_org, cmap=plt.cm.gray)
        ax1.axis('off')
        ax1.set_title('Enhanced Detection', fontsize=20)
        
        ax2.imshow(img, cmap=plt.cm.gray)
        ax2.axis('off')
        ax2.set_title('Original Detection', fontsize=20)
        
        #fig.tight_layout()
        
        plt.show()
#        quit()

if __name__ == "__main__":
    readLineObj = ReadLines()
    
#    img_array = ['TestSheet1_Section1.jpg', 'TestSheet1_Section2.jpg', 'TestSheet1_Section3.jpg','TestSheet1_Section4.jpg','TestSheet1_Section5.jpg',
#                 'TestSheet2_Section1.jpg', 'TestSheet2_Section2.jpg', 'TestSheet2_Section3.jpg','TestSheet2_Section4.jpg','TestSheet2_Section5.jpg']
    img_array = ['TestSheet1_Section3.jpg']
    
    for img_name in img_array:
        img, img_org, lines = readLineObj.DetectEdgesAndLines(img_name, None)
        horiz_coords = readLineObj.Normalize_horizontal_lines(img_org, lines)
#        vert_coords = readLineObj.Normalize_vertical_lines(img_org, lines)
        readLineObj.Draw_plots(img, img_org)