import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image, ImageOps
import copy 

class ReadLines:

    ''' DETECT EDGES USING CANNY METHOD AND BASIC LINES USING HoughLinesP method '''
    def DetectEdgesAndLines(self, image_name, img):
        if not image_name is None:
            # Loading image contains lines
            img = cv2.imread(image_name)

        ## Deep copy to draw detected lines
        img_original = copy.deepcopy(img)

        # Convert to grayscale
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#        ret,thresh_img = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
        thresh_img = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        img_edges = copy.deepcopy(thresh_img)
        
        # Apply Canny edge detection, return will be a binary image
        edges = cv2.Canny(thresh_img,100,200)
        
        # Apply HoughLines Probablistic detection of lines
        minLineLength = 100
        maxLineGap = 10
        lines = cv2.HoughLinesP(edges,1,np.pi/180,50,minLineLength,maxLineGap)


        ## Draw initial detected lines and then finetune
        if not lines is None:
            for line in lines:
                x1,y1,x2,y2 = line[0]
                cv2.line(img_edges,(x1,y1),(x2,y2),(255,0,0),2)
        
        return img_edges, img_original, lines

       
    ''' OPTIMIZE THE LINES DETECTED - MERGE CLOSER LINES AND DRAW ONE SINGLE HORIZONTAL LINE INSTEAD OF MULTIPLE SMALL LINES'''
    def Find_Horizontal_Lines(self, img_org, lines):

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
    
    ''' OPTIMIZE THE LINES DETECTED - MERGE CLOSER LINES AND DRAW ONE SINGLE HORIZONTAL LINE INSTEAD OF MULTIPLE SMALL LINES'''
    def Find_Vertical_Lines(self, img_org, lines):

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
    def Draw_plots(self, img_withEdges, img_withLines):
        cv2.imwrite('houghlines-output.jpg',img_org)

        # display results
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3),
                                            sharex=True, sharey=True)
        
        ax1.imshow(img_withLines, cmap=plt.cm.gray)
        ax1.axis('off')
        ax1.set_title('Enhanced Detection', fontsize=20)
        
        ax2.imshow(img_withEdges, cmap=plt.cm.gray)
        ax2.axis('off')
        ax2.set_title('Original Detection', fontsize=20)
        
        #fig.tight_layout()
        
        plt.show()
#        quit()

    ''' THIS FUNCTION CROPS THE IMAGE HORIZONTALLY TO ELIMINATE UNNECESSARY PADDING BUT JUST TEXT'''
    def AnalyzeAndCropImages(self, image_name, img):
        if not image_name is None:
            # Loading image contains lines
            img = cv2.imread(image_name)

        # Convert to grayscale
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#        ret,thresh_img = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
        thresh_img = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

        
        # Apply Canny edge detection, return will be a binary image
        edges = cv2.Canny(thresh_img,100,200)
        edges_density = np.sum(edges,axis=1)

        # Find minimun outlier based on Interquartile Range and Outliers model
        q75, q25 = np.percentile(edges_density, [75 ,25])
        max_threshold = q75 + (q75 - q25) * 1.5
        
        new_arr = []
        for arr in edges_density:
           new_arr.append(0 if  arr > max_threshold else arr)
         
        edges_density = np.round(new_arr/np.sum(new_arr)*100,0)
        cumulative_pixel_denst = np.cumsum(edges_density)

        ## This section is to identify where the text is. where the slope changes from flat to curve        
        previous_num  = 0
        nochangeseen = 0
        selected_num = 0
        selected_index = 0
        selected_nochangeseen = 0
        for curr_nbr in cumulative_pixel_denst:
            if (previous_num != curr_nbr and nochangeseen > selected_nochangeseen):
                selected_num = curr_nbr
                selected_index = np.where(cumulative_pixel_denst==selected_num)[0][0]
                selected_nochangeseen = nochangeseen
            
            nochangeseen += 1 if (previous_num == curr_nbr) else -nochangeseen
            previous_num = curr_nbr
          
#        print (f"Selected num {selected_num/10} and index is {selected_index}")
        
        cropped_new_image = img[selected_index - 10:]
        
#        y = range(1, gray.shape[0]+1)
#        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4,  figsize=(8, 3))
#        fig, (ax3, ax4) = plt.subplots(nrows=1, ncols=2,  figsize=(8, 3))

#        ax1.imshow(edges, cmap=plt.cm.hot)
#        ax1.set_title('Edge Detection', fontsize=20)
        
#        ax2.invert_yaxis()
#        ax2.plot(edges_density, y)
#        ax2.plot(cumulative_pixel_denst/10, y)
#        ax2.set_title('Density of pixels', fontsize=20)
        
#        ax3.imshow(img, cmap=plt.cm.gray)
#        ax3.set_title('Original Image', fontsize=20)
#        
#        ax4.imshow(cropped_new_image, cmap=plt.cm.gray)
#        ax4.set_title('New Image', fontsize=20)
#        
#        fig.tight_layout()
        
#        plt.show()
        
#        plt.imshow(cropped_new_image)
#        plt.show()
        
        ''' HOW TO CROP THE IMAGE INTO LETTERS '''
        images_for_prediction = self.DetectLettersInImages(None, cropped_new_image)
        return images_for_prediction


    ''' ANALYZE THE VERTICAL EDGES AND SPLIT THE IMAGE INTO TEXT - IDENTIFY CONTOURS'''
    def DetectLettersInImages(self, image_name, img):
        if not image_name is None:
            # Loading image contains lines
            img = cv2.imread(image_name)

        # Convert to grayscale and remove noise
        nose_removed_img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
        
        gray = cv2.cvtColor(nose_removed_img,cv2.COLOR_BGR2GRAY)
#        ret,gray = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

        # Apply Canny edge detection, return will be a binary image
        edges = cv2.Canny(gray,100,200)
        
        edges_density = np.sum(edges,axis=0)

        # Find minimun outlier based on Interquartile Range and Outliers model
        q75, q25 = np.percentile(edges_density, [75 ,25])
        max_threshold = q75 + (q75 - q25) * 1.5
        min_threshold = q25 - (q75 - q25) * 1.5
        
#        print (f'min_threshold:{min_threshold} q25:{q25} q75:{q75}')
        new_arr = []
        med = np.mean(edges_density, axis=0)
        contour_area = []
        start_x = -1
        end_x = -1
        y = img.shape[0]
        analyzing = False
        images_for_prediction = []

        '''BEGIN: HYPER PARAMETERS HARDCODED TODAY TO FILTER AND MAKE DECISIONS'''
        mean_gap = 0
        max_whitespace_density = 1000
        min_contour_area = 500
        '''END: HYPER PARAMETERS HARDCODED TODAY TO FILTER AND MAKE DECISIONS'''
        
        ''' ADJUST THE DENSITY TO AN ACCEPTABLE RANGE'''
        for arr in edges_density:
           new_val = 0.
           if  arr > max_threshold:
                new_val = med
           else:
                ''' TRY TO DO MEAN OF NEIGHBORING POINTS TO STRENGTHEN THE WEAK DENSITY POINTS 
                AND TO AVOID MARKING AS WHITE SPACE'''
                index = len(new_arr)
                
                sum_arr = [] 
                for i in range(index - mean_gap, index + mean_gap + 1):
                    if i >= 0 and i < len(edges_density):
                        sum_arr.append(edges_density[i])
                new_val = int(np.mean(sum_arr))
#                print (f"oldval:{arr} and newval:{new_val}")

           new_arr.append(new_val if new_val > arr else arr)
           current_index = len(new_arr) - 1
           
           ''' Trying to check if my total density is below threshold(1000), if so, consider it as a white space'''
           if new_arr[current_index] < max_whitespace_density:
#               cv2.line(img,(current_index,0),(current_index,img.shape[0]),(255,0,0),2)
               
               ''' Mark the starting and ending points'''
               if analyzing:
                   area = (end_x - start_x) * y
#                   print (area)
                   ''' Check if area is more than threshold (500)'''
                   if area > min_contour_area:
                       contour_area.append([start_x - 1, 5,end_x + 1,y - 10])
                       cv2.rectangle(img, (start_x - 1, 5), (end_x + 1, y - 10), (255, 0, 0), 2)

                       ''' START THE PROCESS TO MAKE IT A SQUARE BY PADDING '''
                       destination_image_size = 28
                       pre_image = 255 - gray[5:y - 10, start_x - 1:end_x + 1]
                       height, width = pre_image.shape
                       size = height if height > width else width
                       size = destination_image_size if size < destination_image_size else destination_image_size*int(size/destination_image_size + 1)
                       
                       padding_top = int((size - height) / 2)
                       padding_bottom = size - height - padding_top
                       padding_left = int((size - width) / 2)
                       padding_right = size - width - padding_left
                       padded_image = np.pad(pre_image, ((padding_top,padding_bottom),(padding_left,padding_right )), 'constant')
                       
                        
                       # normalize image to range [0,255]
#                       minv = np.amin(padded_image)
#                       maxv = np.amax(padded_image)
#                       padded_image = (255 * (padded_image - minv) / (maxv - minv)).astype(np.uint8)
#
#                       padded_image[padded_image > 150] = 255
#                       ret,padded_image = cv2.threshold(padded_image,200,255,cv2.THRESH_BINARY)

                       
                       ''' Resizing to desired size '''
                       padded_image = Image.fromarray(padded_image)
                       fit_and_resized_image = ImageOps.fit(padded_image, (destination_image_size, destination_image_size), Image.ANTIALIAS)
                       
                       ''' CONVERT THE PIL IMAGE TO ARRAY FOR EASY MANIPULATION AND CONVERT TO BINARY'''
#                       fit_and_resized_image = cv2.adaptiveThreshold \
#                       (np.array(fit_and_resized_image),255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,22,2)
#                       _, fit_and_resized_image  = cv2.threshold(np.array(fit_and_resized_image),150,255,cv2.THRESH_BINARY)

                       images_for_prediction.append(fit_and_resized_image)
                   analyzing = False

               end_x = -1
               start_x  = current_index
           else:
               ''' Mark the starting and ending points'''
               analyzing = True
               ''' Reset variables '''
               end_x = current_index
        
        images_return = np.stack(images_for_prediction) if len(images_for_prediction) > 0 else []
        return images_return

        y = range(1, gray.shape[1]+1)
        
#        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3,  figsize=(8, 3))
        fig, (ax2, ax3, ax4) = plt.subplots(nrows=3, ncols=1,  figsize=(100, 30))

#        ax1.imshow(edges, cmap=plt.cm.hot)
#        ax1.set_title('Edge Detection', fontsize=20)
#        
#        
        ax2.plot(y, new_arr)
##        ax2.plot(cumulative_pixel_denst, y)
        ax2.set_title('Density of pixels', fontsize=20)
#        
        ax3.imshow(img, cmap=plt.cm.gray)
        ax3.set_title('Detected contours', fontsize=20)
        
        img_temp = gray if len(images_return) == 0 else images_return[0]
#        print (img_temp)
        ax4.imshow(img_temp, cmap=plt.cm.gray)
        ax4.set_title('Original Gray Image', fontsize=20)
        
#        fig.tight_layout()
        plt.show()
        return images_return

            
if __name__ == "__main__":
    readLineObj = ReadLines()
    
#    img_array = ['Images/TestSheet2_Section1.jpg', 'Images/TestSheet2_Section2.jpg', 'Images/TestSheet2_Section3.jpg','Images/TestSheet2_Section4.jpg','Images/TestSheet2_Section5.jpg',
#                 'Images/TestSheet2_Section1.jpg', 'Images/TestSheet2_Section2.jpg', 'Images/TestSheet2_Section3.jpg','Images/TestSheet2_Section4.jpg','Images/TestSheet2_Section5.jpg']
    img_array = ['Images/TestSheet2_Section1_SubSection1.jpg', 'Images/TestSheet2_Section1_SubSection2.jpg', 'Images/TestSheet2_Section1_SubSection3.jpg', 'Images/TestSheet2_Section1_SubSection4.jpg', 'Images/TestSheet2_Section1_SubSection5.jpg']
#    img_array = ['Images/TestSheet1_Section1_SubSection1.jpg', 'Images/TestSheet1_Section1_SubSection2.jpg', 'Images/TestSheet1_Section1_SubSection3.jpg', 'Images/TestSheet1_Section1_SubSection4.jpg', 'Images/TestSheet1_Section1_SubSection5.jpg']

#    img_array = ['Images/TestSheet2_Section1_SubSection1.jpg']

    for img_name in img_array:
#        img_withEdges, img_org, lines = readLineObj.DetectEdgesAndLines(img_name, None)
#        horiz_coords = readLineObj.Find_Horizontal_Lines(img_org, lines)
#        vert_coords = readLineObj.Find_Vertical_Lines(img_org, lines)
#        readLineObj.Draw_plots(img_withEdges, img_org)
        
        cropped_images = readLineObj.AnalyzeAndCropImages(img_name, None)
