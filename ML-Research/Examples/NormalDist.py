# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 17:27:07 2018

@author: Chandar_S
"""

import tensorflow as tf

k = tf.placeholder(tf.float32)

# Make a normal distribution, with a shifting mean
mean_moving_normal = tf.random_normal(shape=[1000], mean=5, stddev=(1+k))
# Record that distribution into a histogram summary
tf.summary.histogram("normal/moving_mean", mean_moving_normal)

# Setup a session and summary writer
sess = tf.Session()
writer = tf.summary.FileWriter("E:\\MLData\\Logs\\histogram_example\\new")

summaries = tf.summary.merge_all()

# Setup a loop and write the summaries to disk
N = 400
for step in range(N):
#  k_val = step/float(N)
  k_val=step
  summ = sess.run(summaries, feed_dict={k: k_val})
  writer.add_summary(summ, global_step=step)