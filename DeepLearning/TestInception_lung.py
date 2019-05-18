# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 09:20:00 2019

@author: Chandar_S
"""

from keras.models import model_from_json

from nn_utilities_py import nn_utilities
#import os
from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import matplotlib.pyplot as plt

data_path =  'E:\MLData\\'

nn_utilities_obj = nn_utilities(data_path)
input_data = nn_utilities_obj.load_PneumothoraxDataset()
#input_data = nn_utilities_obj.load_emnist_alphadigit_data()


X_train_2D, y_train, X_test_2D, y_test = input_data["x_train"][:5,:], input_data["y_train"][:5,:], input_data["x_validation"][:10,:], input_data["y_validation"][:10,:]

image_size = 256
num_of_color_channels = 1

X_train = X_train_2D.reshape(X_train_2D.shape[0], image_size, image_size, num_of_color_channels)
X_test = X_test_2D.reshape(X_test_2D.shape[0], image_size, image_size, num_of_color_channels)

'''
#path = "E:\\MLData\\Test\\GeneratedLetters\\"
#img_array = ['TestSheet2_Section1_SubSection3_Image13_predictiong.jpg', 'TestSheet2_Section1_SubSection3_Image6_predictionM.jpg', 'TestSheet2_Section1_SubSection4_Image1_predictionM.jpg', 'TestSheet2_Section1_SubSection3_Image2_predictionG.jpg', 'TestSheet2_Section1_SubSection3_Image1_predictiong.jpg', 'TestSheet2_Section1_SubSection4_Image3_predictiong.jpg', 'TestSheet2_Section1_SubSection4_Image1_predictionM.jpg', 'TestSheet2_Section1_SubSection4_Image0_predictionM.jpg']

path = "C:\\Users\\chandar_s\\.spyder-py3\\MachineLearning\\ML-Research\\TestImages\\"
img_array = ['image24_predictionT.jpg', 'image25_predictionJ.jpg']

X_test = []
for img_name in img_array:
#    print (path + img_name)
    img = cv2.imread(path + img_name)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray_expanded = gray[:, :, np.newaxis]
#    cv2.imshow('Original image',img)
#    cv2.imshow('Gray image', gray)
    X_test.append(gray_expanded)
#    print (gray.shape)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()

X_test = np.array(X_test)
print ("X", X_test.shape)
'''
#%%

#print ("Y", y_test.shape)



generator = ImageDataGenerator(featurewise_center=True, 
                               featurewise_std_normalization=True
                               )
generator.fit(X_test)

def generator_multiple_data(X, y, batch_size):
    genX = generator.flow(X, y,  batch_size=batch_size, seed=1)
    while True:
        X_NEW = genX.next()
        yield X_NEW[0], [X_NEW[1], X_NEW[1], X_NEW[1]]



# load json and create model
json_file = open(data_path + '\\Savedmodel\\inception\\lung\\inception_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(data_path + "\\Savedmodel\\inception\\lung\\inception_model.h5")
print("Loaded model from disk")

#%%
# evaluate loaded model on test data
#loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#score = loaded_model.evaluate(X_train[0:50000],  [y_train[0:50000], y_train[0:50000], y_train[0:50000]], verbose=1)
#score = loaded_model.evaluate_generator(generator_multiple_data(X_test, y_test, 100), steps=len(X_test), verbose=1)

#print("%s: %.2f%%" % (loaded_model.metrics_names[4], score[4]*100))
#print("%s: %.2f%%" % (loaded_model.metrics_names[5], score[5]*100))
#print("%s: %.2f%%" % (loaded_model.metrics_names[6], score[6]*100))


#output = loaded_model.predict(X_test[51:100,:])


batchsize = 1
#input_data = X_test[50:55,:]
input_data = X_test
predicted_confidence = loaded_model.predict_generator(generator.flow(input_data, batch_size=batchsize, shuffle=False, seed=1), steps=len(input_data)//batchsize)


predicted_output, predicted_confidence = np.argmax(predicted_confidence[0], axis=1), np.amax(predicted_confidence[0]*100, axis=1)

#print (np.array(predicted_output).shape)
alphadigit = { 0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 
              9:'9', 10:'A', 11:'B', 12:'C', 13:'D', 14:'E', 15:'F', 16:'G', 
              17:'H', 18:'I', 19:'J', 20:'K', 21:'L', 22:'M', 23:'N', 24:'O', 
              25:'P', 26:'Q', 27:'R', 28:'S', 29:'T', 30:'U', 31:'V', 32:'W', 
              33:'X', 34:'Y', 35:'Z', 36:'a', 37:'b', 38:'d', 39:'e', 40:'f', 
              41:'g', 42:'h', 43:'n', 44:'q', 45:'r', 46:'t'}


#output_number = [alphadigit[predicted_output[img_nbr]] for img_nbr in np.argmax(predicted_output[0], axis=1)]

cols_count = int(len(input_data)/2) + (len(input_data) - int(len(input_data)/2)*2)

f, a = plt.subplots(nrows=2, ncols=cols_count, figsize=(8, 3),
                                sharex=True, sharey=True, squeeze=False)
img_nbr = 0
i = 0
count = 0
for image in input_data:
    a[i][img_nbr].imshow(image.reshape(256,256))
    a[i][img_nbr].axis('off')

    title = str(alphadigit[predicted_output[count]]) + " (" + str(int(predicted_confidence[count])) + "%)" + str(np.argmax(y_test[count], axis=0))
    a[i][img_nbr].set_title(title, fontsize=10)

    if (predicted_confidence[img_nbr] > 70):
        a[i][img_nbr].set_facecolor('xkcd:mint green')
    else:
        a[i][img_nbr].set_facecolor('xkcd:salmon')
    img_nbr += 1
    count += 1

    ''' New row'''
    if (img_nbr == cols_count):
        i = i + 1
        img_nbr = 0
#                            break
f.show()
#plt.draw()
#plt.waitforbuttonpress()
''' END: PREDICT AND DISPLAY RESULTS '''

#print (np.array(output).shape)
#print (np.argmax(output[0], axis=1))
#print (output_number)
#print (np.argmax(output[2], axis=1))
#print (np.argmax(y_test[50:100], axis=1))
