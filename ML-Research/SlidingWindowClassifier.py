# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 12:36:04 2019

@author: Chandar_S
"""

# import the necessary packages
import imutils
import time
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
import numpy as np
from ReadLines import ReadLines


def pyramid(image, scale, minSize=(30, 30)):
	# yield the original image
#	yield image

	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)

		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break

		# yield the next image in the pyramid
		yield image

def sliding_window(image, vStepSize, hStepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], vStepSize):
		for x in range(0, image.shape[1], hStepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
            # import the necessary packages

def loadInceptionModel():
    # load json and create model
    json_file = open('E:\\MLData\\Savedmodel\\inception\\alpha_numeric\\inception_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("E:\\MLData\\Savedmodel\\inception\\alpha_numeric\\inception_model.h5")
    print("Loaded model from disk")
    return loaded_model
    
def PredictImagesUsingInceptionModel(X_test):

    X_test = X_test[:, :, :, np.newaxis]

    generator = ImageDataGenerator(featurewise_center=True, 
                                   featurewise_std_normalization=True)
    generator.fit(X_test)


    batchsize = 1
    predicted_confidence = loaded_model.predict_generator(generator.flow(X_test, batch_size=batchsize, shuffle=False, seed=1), steps=len(X_test)//batchsize)

    return np.argmax(predicted_confidence[0], axis=1), np.amax(predicted_confidence[0]*100, axis=1)


#print (np.array(predicted_output).shape)
alphadigit = { 0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 
              9:'9', 10:'A', 11:'B', 12:'C', 13:'D', 14:'E', 15:'F', 16:'G', 
              17:'H', 18:'I', 19:'J', 20:'K', 21:'L', 22:'M', 23:'N', 24:'O', 
              25:'P', 26:'Q', 27:'R', 28:'S', 29:'T', 30:'U', 31:'V', 32:'W', 
              33:'X', 34:'Y', 35:'Z', 36:'a', 37:'b', 38:'d', 39:'e', 40:'f', 
              41:'g', 42:'h', 43:'n', 44:'q', 45:'r', 46:'t'}

# construct the argument parser and parse the arguments
image = "Images/SchoolData_Section1.jpg"
# load the image and define the window width and height
image = cv2.imread(image)

readLineObj = ReadLines()
#image = readLineObj.AnalyzeAndCropImages(image, None)

(winW, winH) = (20, 28)
loaded_model = loadInceptionModel()

count = 1
# loop over the image pyramid
for resized in pyramid(image, scale=1.5):
    resized_copy = resized.copy()
	# loop over the sliding window for each layer of the pyramid
    for (x, y, window) in sliding_window(resized_copy, vStepSize=45, hStepSize=5, windowSize=(winW, winH)):
		# if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

		# THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
		# MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
		# WINDOW

        # Convert to grayscale and remove noise
        nose_removed_img = cv2.fastNlMeansDenoisingColored(window,None,10,10,7,21)
        gray = 255 - cv2.cvtColor(nose_removed_img, cv2.COLOR_BGR2GRAY)
#        gray = 255 - cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        
        destination_image_size = 28
        height, width = gray.shape
        size = height if height > width else width
        size = destination_image_size if size <= destination_image_size else destination_image_size*int(size/destination_image_size + 1)
       
        padding_top = int((size - height) / 2)
        padding_bottom = size - height - padding_top
        padding_left = int((size - width) / 2)
        padding_right = size - width - padding_left
        padded_image = np.pad(gray, ((padding_top,padding_bottom),(padding_left,padding_right )), 'constant')

        cv2.imshow("Segment", padded_image)
        images_for_analysis = [padded_image]
        predicted_output, predicted_confidence = PredictImagesUsingInceptionModel(np.array(images_for_analysis))
#        print ("predicted_output", predicted_output, "predicted_confidence", predicted_confidence, "%")
        
		# since we do not have a classifier, we'll just draw the window
        clone = resized_copy.copy()
        if predicted_confidence > 90:
            cv2.putText(clone, alphadigit[predicted_output[0]], (x, y + int(winH/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), lineType=cv2.LINE_AA)
#            cv2.putText(resized_copy, alphadigit[predicted_output[0]], (x, y + int(winH/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), lineType=cv2.LINE_AA) 
            cv2.imwrite( f"TestImages//image{count}_prediction{alphadigit[predicted_output[0]]}_confidence{predicted_confidence}.jpg", images_for_analysis[0])
            count += 1
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        cv2.imshow("Window", clone)
        cv2.waitKey(1)
        time.sleep(0.025)