# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 12:14:53 2018

@author: Chandar_S
"""

import tensorflow as tf
import numpy as np
from abc import ABC
from tqdm import tqdm

class BaseNNAbstract(ABC):
    random_state = 650
    logs_path = None
    np.random.seed(random_state)
    rng = np.random.RandomState(random_state)
    tf.set_random_seed(random_state)
    batch_pointer = 0

    ### OVERRIDE THIS METHOD TO IMPLEMENT YOUR CLASS ###
    def create_model(self):
        pass
    
    def getTrainBatch(self, x_train, y_train, batch_size):
        inds = np.arange(self.batch_pointer,self.batch_pointer+batch_size)
        inds = np.mod( inds , x_train.shape[0] ) #cycle through dataset
        batch = (x_train[inds], y_train[inds]) #grab batch
    
        self.batch_pointer += batch_size #increment counter before returning
        return batch

    ###  TRAIN NN BASED ON STRUCTURE CREATED
    def train_model(self, sess, model, training_epochs_input, display_step_input, batch_size_input, optimizer, cost, accuracy, x_train_input, x_train_input_4D, y_train_input, x_validation, y_validation, quick_training = False, save_model_name=None, run_validation_accuracy=True):
            # op to write logs to Tensorboard
            summary_writer = tf.summary.FileWriter(self.logs_path, graph=tf.get_default_graph())
            # Create a summary to monitor cost tensor
            tf.summary.scalar("Model Loss", cost)
            # Create a summary to monitor accuracy tensor
            tf.summary.scalar("Training Accuracy", accuracy)
            # Add input images
#            tf.summary.image('input',x_train_input_4D,max_outputs=10)

            # Create summaries to visualize weights
            for var in tf.trainable_variables():
                tf.summary.histogram(var.name, var)

            # Merge all summaries into a single op
            merged_summary_op = tf.summary.merge_all()

            # Create a summary to monitor accuracy tensor
            validation_acc_summary = tf.summary.scalar('Validation Accuracy', accuracy)  # intended to run on validation set
    
            batch_steps = int(len(x_train_input) / batch_size_input)
            x_batches = np.array_split(x_train_input, batch_steps)
            y_batches = np.array_split(y_train_input, batch_steps)
                        
            for epoch in tqdm(range(training_epochs_input)):
                avg_cost = 0.0

                if (quick_training):
                    (batch_x, batch_y) = self.getTrainBatch(x_train_input, y_train_input, batch_steps)
                    _, c, summary = sess.run([optimizer, cost, merged_summary_op],
                                    feed_dict={
                                        self.x: batch_x,
                                        self.y: batch_y,
                                        self.keep_prob: 0.8
                                    })
                    # Write logs at every iteration
                    summary_writer.add_summary(summary, epoch * batch_steps + 1)
                else:
                    for i in tqdm(range(batch_steps)):
                        batch_x, batch_y  = x_batches[i], y_batches[i]
    
                        _, c, summary = sess.run([optimizer, cost, merged_summary_op],
                                        feed_dict={
                                            self.x: batch_x,
                                            self.y: batch_y,
                                            self.keep_prob: 0.8
                                        })
                        # Write logs at every iteration
                        summary_writer.add_summary(summary, epoch * batch_steps + i + 1)
                        
                if run_validation_accuracy == True:
                    val_summary = sess.run(validation_acc_summary,
                                feed_dict={
                                    self.x: x_validation,
                                    self.y: y_validation,
                                    self.keep_prob: 1
                                })
                    summary_writer.add_summary(val_summary, epoch * batch_steps + 1)
#                        
                
                avg_cost += c / batch_steps

                ''' Save the model '''
                if save_model_name is not None:
                    self.saveModel(sess, save_model_name)

                if epoch % display_step_input == 0:
                    print("Epoch:", '%04d' % (epoch + 1), "cost=",
                          "{:.9f}".format(avg_cost))
                    
                    ## CHECK TO MAKE SURE IT'S NOT OVER FITTING BY COMPARING TESTING AND VALIDATION ACCURACY
#                    train_accuracy = self.predictAccuracy(model, x_train_input[:1000], y_train_input[:1000])
#                    print("Train Accuracy of %:", train_accuracy)
#                    validation_accuracy = self.predictAccuracy(model, x_validation[:1000], y_validation[:1000])
#                    print("Validation Accuracy of %:", validation_accuracy)
    

            print("Optimization Finished!")
                                
            return model
    
    ### Save the model ###
    def saveModel(self, sess, model_name):
        saver = tf.train.Saver()
        save_path = saver.save(sess, self.data_path + "SavedModel\\" + model_name + ".ckpt")
#        print("Model saved in file: %s" % save_path)
        
    ### TEST NN MODEL WITH DATA
    def predictAccuracy(self, trained_model, x_test_input, y_test_input):
        correct_prediction = tf.equal(tf.argmax(trained_model, 1), tf.argmax(self.y, 1))
        predict_accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))*100
        return predict_accuracy.eval({self.x: x_test_input, self.y: y_test_input, self.keep_prob: 1.0})
    
    
    ### PREDICT OUTPUT BASED ON MODEL CREATED
    def predictvalue(self, trained_model, x_test_input):
            predict = tf.argmax(trained_model, 1)
            return predict.eval({self.x: x_test_input, self.keep_prob: 1.0})
