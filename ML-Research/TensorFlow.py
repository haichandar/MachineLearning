# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 12:57:37 2018

@author: Chandar_S
"""

import tensorflow as tf

const1 = tf.constant([[1,2,3], 
                      [1,2,3]]);
const2 = tf.constant([[3,4,5],
                      [6,7,8]]);

result = tf.add(const1, const2);

with tf.Session() as sess:
  output = sess.run(result)
  print(output)