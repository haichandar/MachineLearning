import os
import numpy as np
import pandas as pd
from scipy.misc import imread
import pylab
import tensorflow as tf

random_state = 65
np.random.seed(random_state)
rng = np.random.RandomState(random_state)
tf.set_random_seed(random_state)


##### START: PREP DATA ######
root_dir = os.path.abspath('E:\\MLData\\')
data_dir = os.path.join(root_dir, 'Image')
#sub_dir = os.path.join(root_dir, 'sub')

# check for existence
os.path.exists(root_dir)
os.path.exists(data_dir)
#os.path.exists(sub_dir)

train = pd.read_csv(os.path.join(data_dir, 'Numbers_Train_Mapping.csv'))
test = pd.read_csv(os.path.join(data_dir, 'Numbers_Test_Mapping.csv'))

#sample_submission = pd.read_csv(os.path.join(data_dir, 'Sample_Submission.csv'))

# img_name = rng.choice(train.filename)
# filepath = os.path.join(data_dir, 'Numbers', 'Images', 'train', img_name)
#
# img = imread(filepath, flatten=True)

# pylab.imshow(img, cmap='gray')
# pylab.axis('off')
# pylab.show()
# print (train)

temp = []
for img_name in train.filename:
    image_path = os.path.join(data_dir, 'Numbers', 'Images', 'train', img_name)
    img = imread(image_path, flatten=True)
    img = img.astype('float32')
    temp.append(img)

x_train = np.stack(temp)

temp = []
for img_name in test.filename:
    image_path = os.path.join(data_dir, 'Numbers', 'Images', 'test', img_name)
    img = imread(image_path, flatten=True)
    img = img.astype('float32')
    temp.append(img)

x_test = np.stack(temp)



split_size = int(x_train.shape[0]*0.7)

x_train, x_validation = x_train[:split_size], x_train[split_size:]
y_train, y_validation = train.label.values[:split_size], train.label.values[split_size:]


##### END : PREP DATA #######


############ START: BUILD THE DNN CODE ################

## BUILD NN STRUCTURE

## Convert class labels from scalars to one-hot vectors
def dense_to_one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot


## Convert values to range 0-1"""
def preproc(unclean_batch_x):

    temp_batch = unclean_batch_x / unclean_batch_x.max()

    return temp_batch


## Create batch with random samples and return appropriate format
def batch_creator(batch_size, dataset_length, dataset_name):
    batch_mask = rng.choice(dataset_length, batch_size)

    batch_x = eval('x_' + dataset_name)[[tuple(batch_mask)]].reshape(-1, n_input)
    batch_x = preproc(batch_x)

    if dataset_name == 'train':
        batch_y = eval(dataset_name).ix[batch_mask, 'label'].values
        batch_y = dense_to_one_hot(batch_y)

    return batch_x, batch_y


def multilayer_perceptron(input_data, weights, biases, keep_prob):
    layer_1 = tf.add(tf.matmul(input_data, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.dropout(layer_1, keep_prob)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    layer_2 = tf.nn.dropout(layer_2, keep_prob)
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    # out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer


###  TRAIN NN BASED ON STRUCTURE CREATED
def train_nn_model(sess, training_epochs_input, display_step_input, batch_size_input, x_train_input, y_train_input):
        sess.run(tf.global_variables_initializer())

        for epoch in range(training_epochs_input):
            avg_cost = 0.0
            batch_steps = int(len(x_train) / batch_size_input)
            #x_batches = np.array_split(x_train_input, batch_steps)
            #y_batches = np.array_split(y_train_input, batch_steps)
            for i in range(batch_steps):
                #batch_x, batch_y = x_batches[i], y_batches[i]
                batch_x, batch_y = batch_creator(batch_size_input, x_train.shape[0], 'train')
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
        print("Optimization Finished!")


## TEST NN MODEL WITH TEST DATA
def testprediction( x_test_input, y_test_input):
        correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))*100
        return accuracy.eval({x: x_test_input, y: y_test_input, keep_prob: 1.0})


## PREDICT OUTPUT BASED ON MODEL CREATED
def predictvalue(x_test_input):
        predict = tf.argmax(predictions, 1)
        return predict.eval({x: x_test_input.reshape(-1, n_input), keep_prob: 1.0})


## PREP THE STRCUTURE OF NN
n_input = 28 * 28
n_hidden_1 = 500
n_hidden_2 = 500
n_classes = 10

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

training_epochs = 5
display_step = 1

batch_size = 128
learning_rate_var = 0.01

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

predictions = multilayer_perceptron(x, weights, biases, keep_prob)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predictions, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_var).minimize(cost)

############ END: BUILD THE DNN CODE ################

#### TRAIN AND TEST NN
with tf.Session() as sess:
    ## TRAIN
    train_nn_model(sess, training_epochs, display_step, batch_size, x_train, y_train)

    ## TEST
    tested_accuracy = testprediction(x_validation.reshape(-1, n_input), dense_to_one_hot(y_validation))
    print("Validation Accuracy%:", tested_accuracy)

    ## PREDICT AND TEST
    predict = predictvalue(x_test)

    img_name = rng.choice(test.filename)
    filepath = os.path.join(data_dir, 'Numbers', 'Images', 'test', img_name)
    img = imread(filepath, flatten=True)
    test_index = int(img_name.split('.')[0]) - 49000
    print("Prediction is: ", predict[test_index])

    pylab.imshow(img, cmap='gray')
    pylab.axis('off')
    pylab.show()