# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 12:17:34 2018

@author: Chandar_S
"""

import pandas as pd
import os
from scipy.misc import imread
import numpy as np
import h5py
from urllib.request import urlopen
#from tensorflow.examples.tutorials.mnist import input_data

class nn_utilities:

    data_path = None
    
    def __init__(self, path):
        self.data_path = path

    def convert_to_onehot(self, series):
        return pd.get_dummies(series).values


    ##### START: PREP DATA ######
    def prepare_digits_image_inputs(self):
        data_dir = os.path.abspath(self.data_path + 'Image')
        
        # check for existence
        os.path.exists(data_dir)
        
        train = pd.read_csv(os.path.join(data_dir, 'Numbers_Train_Mapping-5000.csv'))
        test = pd.read_csv(os.path.join(data_dir, 'Numbers_Test_Mapping.csv'))
        
        
        # GET THE TEST AND VALIDATION DATA
        temp = []
        for img_name in train.filename:
            image_path = os.path.join(data_dir, 'Numbers', 'Images', 'train', img_name)
            img = imread(image_path, flatten=True)
            img = img.astype('float32')
            temp.append(img)
        
        # convert list to ndarray and PREP AS PER INPUT FORMAT
        x_train = np.stack(temp)
        x_train = x_train.reshape(-1, x_train.shape[1] * x_train.shape[2])
        
              
        ## GET THE TEST DATA
        temp = []
        for img_name in test.filename:
            image_path = os.path.join(data_dir, 'Numbers', 'Images', 'test', img_name)
            img = imread(image_path, flatten=True)
            img = img.astype('float32')
            temp.append(img)
        
        # convert list to ndarray and PREP AS PER INPUT FORMAT
        x_test = np.stack(temp)
        x_test = x_test.reshape(-1, x_test.shape[1] * x_test.shape[2])

        return self.prep_returndata(x_train, train.label, None, None, "local_digits_data", 1,
                                    x_test, test, data_dir)
    ##### END : PREP DATA #######

    def load_mnist(self, path, kind='train'):
        import gzip
    
        """Load MNIST data from `path`"""
        labels_path = os.path.join(path,
                                   '%s-labels-idx1-ubyte.gz'
                                   % kind)
        images_path = os.path.join(path,
                                   '%s-images-idx3-ubyte.gz'
                                   % kind)
    
        with gzip.open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                                   offset=8)
    
        with gzip.open(images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                                   offset=16).reshape(len(labels), 784)

        return images, labels

    def load_fashion_data(self):
            x_train, y_train = self.load_mnist(self.data_path + 'Image\Fashion', kind='train')
            x_validation, y_validation = self.load_mnist(self.data_path + 'Image\Fashion', kind='t10k')
            return self.prep_returndata(x_train, y_train, x_validation, y_validation, "mnist_fashion_data")
        
    def load_mnist_digit_data(self):
            x_train, y_train = self.load_mnist(self.data_path + 'Image\MNIST_Digit_data', kind='train')
            x_validation, y_validation = self.load_mnist(self.data_path + 'Image\MNIST_Digit_data', kind='t10k')
            return self.prep_returndata(x_train, y_train, x_validation, y_validation, "mnist_digit_data")
    
    def load_emnist_alphadigit_data(self):
            train = pd.read_csv(self.data_path + 'Image\emnist_alphadigit_data\emnist-balanced-train.csv', header=None)
            test = pd.read_csv(self.data_path + 'Image\emnist_alphadigit_data\emnist-balanced-test.csv', header=None)

            x_train_data, y_train = train.iloc[:, 1:].values, train.iloc[:, 0].values
            x_validation_data, y_validation  = pd.get_dummies(test.iloc[:, 1:]), pd.get_dummies(test.iloc[:, 0])
            x_train = np.apply_along_axis(self.rotate, 1, x_train_data)
            x_validation = np.apply_along_axis(self.rotate, 1, x_validation_data)
            del x_train_data, x_validation_data
            return self.prep_returndata(x_train, y_train, x_validation, y_validation, "emnist_alpha_digit_data")
     
    def load_emnist_alphadigit_data_google_collab(self):
            train = pd.read_csv(self.data_path + 'emnist-balanced-train.csv', header=None)
            test = pd.read_csv(self.data_path + 'emnist-balanced-test.csv', header=None)

            x_train_data, y_train = train.iloc[:, 1:].values, train.iloc[:, 0].values
            x_validation_data, y_validation  = pd.get_dummies(test.iloc[:, 1:]), pd.get_dummies(test.iloc[:, 0])
            x_train = np.apply_along_axis(self.rotate, 1, x_train_data)
            x_validation = np.apply_along_axis(self.rotate, 1, x_validation_data)
            del x_train_data, x_validation_data
            return self.prep_returndata(x_train, y_train, x_validation, y_validation, "emnist_alpha_digit_data")        

    def load_emnist_letters_data(self):
            train = pd.read_csv(self.data_path + 'Image\EMINIST_EnglishLetters\emnist-letters-train.csv', header=None)
            test = pd.read_csv(self.data_path + 'Image\EMINIST_EnglishLetters\emnist-letters-test.csv', header=None)

            x_train_data, y_train = train.iloc[:, 1:].values, train.iloc[:, 0].values
            x_validation_data, y_validation  = pd.get_dummies(test.iloc[:, 1:]), pd.get_dummies(test.iloc[:, 0])
            x_train = np.apply_along_axis(self.rotate, 1, x_train_data)
            x_validation = np.apply_along_axis(self.rotate, 1, x_validation_data)
            del x_train_data, x_validation_data
            return self.prep_returndata(x_train, y_train, x_validation, y_validation, "emnist_EnglishLetters")
  
    def rotate(self, image):
        image = image.reshape([28, 28])
        image = np.fliplr(image)
        image = np.rot90(image)
        return image.reshape([28 * 28])
      
    def prep_returndata(self, x_train, y_train, x_validation, y_validation, name="unnamed_dataset", num_of_color_channels=1,
                        x_test=None, test=None, data_dir=data_path):
            
            # Num of samples x [height * width * no of channels]
            if (len(x_train.shape) == 2):
                # Assume it's a square and try to split it equally. Will break if it's not a square
                size = int(np.sqrt(x_train.shape[1]/num_of_color_channels))
                # Reshape to a format where it can be displayed 
                x_train_4D = x_train.reshape(x_train.shape[0], size, size, num_of_color_channels)
            elif(len(x_train.shape) == 4):
                x_train_4D = x_train
                x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2] * x_train.shape[3])
                if len(x_validation.shape) == 4:
                    x_validation = x_validation.reshape(x_validation.shape[0], x_validation.shape[1] * x_validation.shape[2] * x_validation.shape[3])
            
            # GET THE TEST AND VALIDATION DATA
           
            # IF VALIDATION IS NOT SENT, SPLIT THE TRAINING DATA BY 70-30 AND USE 30% FOR VALIDATION
            # Convert labels to one hot values
            if (x_validation is None):
                split_size = int(x_train.shape[0]*0.7)
                x_train, x_validation = x_train[:split_size], x_train[split_size:]
                trainLabels = self.convert_to_onehot(y_train)
                y_train, y_validation = trainLabels[:split_size], trainLabels[split_size:]
            else:
                if (len(y_train.shape) == 1):
                    y_train =  self.convert_to_onehot(y_train)
                    y_validation = self.convert_to_onehot(y_validation)

            x_train, y_train, x_validation, y_validation = x_train, y_train, x_validation, y_validation
            
            return { "x_train" : x_train, 
                     "y_train" : y_train,
                     "x_validation": x_validation,
                     "y_validation": y_validation,
                     "num_of_color_channels": num_of_color_channels,
                     "x_test" : x_test,
                     "x_train_4D" : x_train_4D,
                     "test": test,
                     "data_dir": data_dir,
                     "name":name}
    

    def load_PneumothoraxDataset(self):
        urls = {'pneumothorax_test':'https://www.dropbox.com/s/x74ykyivipwnozs/pneumothorax_test.h5?dl=1',
                'pneumothorax_train':'https://www.dropbox.com/s/pnwf67qzztd1slc/pneumothorax_train.h5?dl=1'}
        
        data_dir =  os.path.abspath(self.data_path + 'Image\Lung_Data\\')
        
        for (name,url) in urls.items():
            if not os.path.isfile(data_dir+name+'.h5'):
                print('Downloading '+name+'...')
                u = urlopen(url)
                data = u.read()
                u.close()
        
                with open(data_dir+name+'.h5', "wb") as f :
                    f.write(data)
            else:
                print("Looks to be available")
        print('Files have been downloaded.')
        print("Loading X-Ray Dataset!")

        train = h5py.File(data_dir+'pneumothorax_train.h5','r')
        validation = h5py.File(data_dir+'pneumothorax_test.h5','r')

        x_train = train['image'][:]
        x_validation = validation['image'][:500]
        y_train = train['label'][:]
        y_validation = validation['label'][:500]

        return self.prep_returndata(x_train, y_train, x_validation, y_validation, "Pneumothorax_data")