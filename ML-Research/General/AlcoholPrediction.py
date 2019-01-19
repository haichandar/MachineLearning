import tensorflow as tf
import numpy as np
import pandas as pd
from math import floor, ceil

random_state = 42
np.random.seed(random_state)
tf.set_random_seed(random_state)

## EXTRACT DATA+
math_df = pd.read_csv("E:\\MLData\\General\\student-alcohol-consumption\\student-mat.csv", sep=",")
port_df = pd.read_csv("E:\\MLData\\General\\student-alcohol-consumption\\student-por.csv", sep=",")
math_df["course"] = "math"
port_df["course"] = "portuguese"
merged_df = math_df.append(port_df)
merge_vector = ["school","sex","age","address",
                "famsize","Pstatus","Medu","Fedu",
                "Mjob","Fjob","reason","nursery","internet"]
duplicated_mask = merged_df.duplicated(keep=False, subset=merge_vector)
duplicated_df = merged_df[duplicated_mask]
unique_df = merged_df[~duplicated_mask]
both_courses_mask = duplicated_df.duplicated(subset=merge_vector)
both_courses_df = duplicated_df[~both_courses_mask].copy()
both_courses_df["course"] = "both"
students_df = unique_df.append(both_courses_df)
students_df = students_df.sample(frac=1)
students_df['alcohol'] = (students_df.Walc * 2 + students_df.Dalc * 5) / 7
students_df['alcohol'] = students_df.alcohol.map(lambda x: ceil(x))
students_df['drinker'] = students_df.alcohol.map(lambda x: "yes" if x > 2 else "no")


### PREP DATA FOR NN input
def encode(series):
  return pd.get_dummies(series.astype(str))

train_x = pd.get_dummies(students_df.school)
#print(train_x)
train_x['age'] = students_df.age
train_x['absences'] = students_df.absences
train_x['g1'] = students_df.G1
train_x['g2'] = students_df.G2
train_x['g3'] = students_df.G3

train_x = pd.concat([train_x, encode(students_df.sex), encode(students_df.Pstatus),
                     encode(students_df.Medu), encode(students_df.Fedu),
                     encode(students_df.guardian), encode(students_df.studytime),
                     encode(students_df.failures), encode(students_df.activities),
                     encode(students_df.higher), encode(students_df.romantic),
                     encode(students_df.reason), encode(students_df.paid),
                     encode(students_df.goout), encode(students_df.health),
                     encode(students_df.famsize), encode(students_df.course)
                    ], axis=1)
train_y = encode(students_df.drinker)


### SPLIT DATA FOR TRAINING AND TESTING
train_size = 0.9
train_cnt = floor(train_x.shape[0] * train_size)
x_train = train_x.iloc[0:train_cnt].values
y_train = train_y.iloc[0:train_cnt].values
x_test = train_x.iloc[train_cnt:].values
y_test = train_y.iloc[train_cnt:].values

print (x_train.shape)

## BUILD NN STRUCTURE
def multilayer_perceptron(input_data, weights, biases, keep_prob):
    layer_1 = tf.add(tf.matmul(input_data, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.dropout(layer_1, keep_prob)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    layer_2 = tf.nn.dropout(layer_2, keep_prob)
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


###  TRAIN NN BASED ON STRUCTURE CREATED
def train_nn_model(sess, training_epochs_input, display_step_input, batch_size_input ,x_train_input, y_train_input):
        sess.run(tf.global_variables_initializer())

        for epoch in range(training_epochs_input):
            avg_cost = 0.0
            batch_steps = int(len(x_train) / batch_size_input)
            x_batches = np.array_split(x_train_input, batch_steps)
            y_batches = np.array_split(y_train_input, batch_steps)
            for i in range(batch_steps):
                batch_x, batch_y = x_batches[i], y_batches[i]
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

def predictvalue(x_predict_input):
        predict = tf.argmax(predictions, 1)
        return predict.eval({x: x_predict_input, keep_prob: 1.0})


## PREP THE STRCUTURE OF NN
n_input = train_x.shape[1]
n_hidden_1 = 38
n_hidden_2 = 15
n_classes = train_y.shape[1]

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

keep_prob = tf.placeholder("float")

training_epochs = 5000
display_step = 1000
batch_size = 32
learning_rate_var = 0.0001

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

predictions = multilayer_perceptron(x, weights, biases, keep_prob)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predictions, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_var).minimize(cost)


#### TRAIN AND TEST NN
with tf.Session() as sess:
    ## TRAIN
    train_nn_model(sess, training_epochs, display_step, batch_size, x_train, y_train)

    ## VALIDATION
    tested_accuracy = testprediction(x_test, y_test)
    print("Validation Accuracy%:", tested_accuracy)

    ## PREDICT AND TEST
    pred = predictvalue(x_test)
    a = 0
    for i in range(len(x_test)):
         a += y_test[i, pred[i]]
    print("Validation Accuracy%", a/len(x_test)*100)
