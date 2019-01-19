import os
import numpy as np
import pandas as pd
from scipy.misc import imread
import tensorflow as tf
#from tensorflow.python import debug as tfdbg
#import tensorflow.contrib.eager as tfe
#tfe.enable_eager_execution()

random_state = 650
np.random.seed(random_state)
rng = np.random.RandomState(random_state)
tf.set_random_seed(random_state)


##### START: PREP DATA ######
root_dir = os.path.abspath('E:\\MLData\\')
data_dir = os.path.join(root_dir, 'Image')

# check for existence
os.path.exists(root_dir)
os.path.exists(data_dir)

train = pd.read_csv(os.path.join(data_dir, 'Numbers_Train_Mapping-5000.csv'))
test = pd.read_csv(os.path.join(data_dir, 'Numbers_Test_Mapping.csv'))

def convert_to_onehot(series):
  return pd.get_dummies(series.astype(str))

# GET THE TEST AND VALIDATION DATA
temp = []
for img_name in train.filename:
    image_path = os.path.join(data_dir, 'Numbers', 'Images', 'train', img_name)
    img = imread(image_path, flatten=True)
    img = img.astype('float32')
    temp.append(img)

# convert list to ndarray
x_train = np.stack(temp)


# PREP AS PER INPUT FORMAT
x_train = x_train.reshape(-1, x_train.shape[1] * x_train.shape[2])
split_size = int(x_train.shape[0]*0.7)
x_train, x_validation = x_train[:split_size], x_train[split_size:]
trainLabels = convert_to_onehot(train.label)
y_train, y_validation = trainLabels[:split_size], trainLabels[split_size:]


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



##### END : PREP DATA #######


############ START: BUILD THE DNN CODE ################

## BUILD NN STRUCTURE

def multilayer_perceptron(input_data, weights, biases, keep_prob):
    layer_1 = tf.add(tf.matmul(input_data, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.dropout(layer_1, keep_prob)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    layer_2 = tf.nn.dropout(layer_2, keep_prob)
    out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])
    # out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer


###  TRAIN NN BASED ON STRUCTURE CREATED
def train_nn_model(sess, training_epochs_input, display_step_input, batch_size_input, x_train_input, y_train_input):
        sess.run(tf.global_variables_initializer())

        for epoch in range(training_epochs_input):
            avg_cost = 0.0
            batch_steps = int(len(x_train) / batch_size_input)
            x_batches = np.array_split(x_train_input, batch_steps)
            y_batches = np.array_split(y_train_input, batch_steps)
            for i in range(batch_steps):
                batch_x, batch_y = x_batches[i], y_batches[i]
                #batch_x, batch_y = batch_creator(batch_size_input, x_train.shape[0], 'train')
                _, c = sess.run([optimizer, cost],
                                feed_dict={
                                    x: batch_x,
                                    y: batch_y,
                                    keep_prob: 0.8
                                })
                avg_cost += c / batch_steps
                #avg_cost = c
            if epoch % display_step_input == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=",
                      "{:.9f}".format(avg_cost))
                
                ## CHECK TO MAKE SURE IT'S NOT OVER FITTING BY COMPARING TESTING AND VALIDATION ACCURACY
                train_accuracy = testprediction(x_train, y_train)
                print("Train Accuracy of %:", train_accuracy)
                validation_accuracy = testprediction(x_validation, y_validation)
                print("Validation Accuracy of %:", validation_accuracy)

        print("Optimization Finished!")


## TEST NN MODEL WITH TEST DATA
def testprediction( x_test_input, y_test_input):
        correct_prediction = tf.equal(tf.argmax(fnn_model, 1), tf.argmax(y, 1))
        predict_accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))*100
        return predict_accuracy.eval({x: x_test_input, y: y_test_input, keep_prob: 1.0})


## PREDICT OUTPUT BASED ON MODEL CREATED
def predictvalue(x_test_input):
        #print (fnn_model)
        predict = tf.argmax(fnn_model, 1)
        return predict.eval({x: x_test_input, keep_prob: 1.0})


## PREP THE STRCUTURE OF NN
n_input = x_train.shape[1]
n_hidden_1 = 500
n_hidden_2 = 500
n_classes = y_train.shape[1]

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    #'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

keep_prob = tf.placeholder("float")

training_epochs = 1
display_step = 1

batch_size = 128
learning_rate_var = 0.01

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

fnn_model = multilayer_perceptron(x, weights, biases, keep_prob)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fnn_model, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_var).minimize(cost)


############ END: BUILD THE DNN CODE ################

print ("Starting session")
#### TRAIN AND TEST NN
with tf.Session() as sess:
    #sess = tfdbg.TensorBoardDebugWrapperSession(sess,  'HYDHTC130784L:7007')
    ## TRAIN
    train_nn_model(sess, training_epochs, display_step, batch_size, x_train, y_train)

    ## TEST
    #tested_accuracy = testprediction(x_validation, y_validation)
    #print("Validation Accuracy%:", tested_accuracy)

    ## PREDICT AND TEST
    predict = predictvalue(x_test)

    img_name = rng.choice(test.filename)
    filepath = os.path.join(data_dir, 'Numbers', 'Images', 'test', img_name)
    img = imread(filepath, flatten=True)
    test_index = int(img_name.split('.')[0]) - 49000
    print("Prediction is: ", predict[test_index])

    #pylab.imshow(img, cmap='gray')
    #pylab.axis('off')
    #pylab.show()